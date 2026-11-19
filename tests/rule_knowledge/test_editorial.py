import copy
import pytest
from fastapi import HTTPException
from src import database as db
from src import editorial_api as api


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "rules.db")
    monkeypatch.setenv("KNOWLEDGE_ADMIN_KEY", "test-admin")
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(api, "check_game", lambda _: None)
    db.init_db()
    from src.models import Game
    db.create_game(Game(id="game", name="測試遊戲"))
    api.init_editorial_db()


def doc(text="支付 2 資源"):
    return {"blocks": [
        {"kind": "規則", "id": "R1", "fields": {"大項": "進行遊戲", "標題": "支付資源", "內容": text, "角色": "全部", "人數": "全部", "擴充": "全部", "狀態": "已確認", "條件": "回合中", "例外": "無", "來源": "S1", "PDF頁碼": "1", "原文": "Pay resources"}},
        {"kind": "文件", "id": "META", "fields": {"遊戲": "測試", "版本": "1"}},
        {"kind": "來源", "id": "S1", "fields": {"名稱": "規則書", "版本": "1", "PDF頁數": "1"}},
        {"kind": "術語", "id": "T1", "fields": {"中文": "資源", "原文": "Resource"}},
        {"kind": "涵蓋", "id": "C1", "fields": {"章節": "行動", "來源": "S1", "PDF頁碼": "1", "規則": "R1", "狀態": "已整理"}},
    ]}


def publish(document=None, base=""):
    draft = api.save("game", api.Draft(document=document or doc(), base_version=base), "test-admin")
    api.publish("game", api.Publish(version=draft["version"], expected_version=base, confirmed=True), "test-admin")
    return draft["version"]


def test_draft_does_not_publish_and_is_idempotent():
    one = api.save("game", api.Draft(document=doc()), "test-admin")
    assert api.current("game")["document"] is None
    assert one == api.save("game", api.Draft(document=doc()), "test-admin")


def test_publish_stale_and_rollback():
    first = publish()
    assert db.get_game("game").status.value == "published"
    second = publish(doc("支付 3 資源"), first)
    with pytest.raises(HTTPException) as caught:
        api.publish("game", api.Publish(version=first, expected_version=first, confirmed=True), "test-admin")
    assert caught.value.status_code == 409
    assert api.current("game")["version"] == second
    api.publish("game", api.Publish(version=first, expected_version=second, confirmed=True), "test-admin")
    assert api.current("game")["version"] == first


def test_admin_and_confirmation_required():
    with pytest.raises(HTTPException) as caught:
        api.save("game", api.Draft(document=doc()), "wrong")
    assert caught.value.status_code == 401
    draft = api.save("game", api.Draft(document=doc()), "test-admin")
    with pytest.raises(HTTPException):
        api.publish("game", api.Publish(version=draft["version"]), "test-admin")
    assert api.current("game")["document"] is None


def test_shared_admin_session_expiry_revocation_and_password_rotation(monkeypatch):
    from src import editorial_sessions as sessions
    with pytest.raises(HTTPException):
        api.create_admin_session('game', 'wrong')
    token = api.create_admin_session('game', 'test-admin')['token']
    assert api.read_admin_session('game', token) == {'authenticated': True}
    conn = db._connect()
    try:
        row = dict(conn.execute('SELECT * FROM editorial_sessions').fetchone())
        assert token not in str(row) and 'test-admin' not in str(row)
    finally:
        conn.close()
    # A fresh module/connection must authenticate the same session.
    import importlib
    importlib.reload(sessions)
    api.admin(token)
    with pytest.raises(HTTPException):
        api.create_admin_session('game', token)
    with pytest.raises(HTTPException):
        api.admin(token[:-1] + ('0' if token[-1] != '0' else '1'))
    monkeypatch.setenv('KNOWLEDGE_ADMIN_KEY', 'rotated')
    with pytest.raises(HTTPException):
        api.admin(token)
    monkeypatch.setenv('KNOWLEDGE_ADMIN_KEY', 'test-admin')
    api.delete_admin_session('game', token)
    with pytest.raises(HTTPException):
        api.admin(token)
    token = api.create_admin_session('game', 'test-admin')['token']
    monkeypatch.setattr(sessions.time, 'time', lambda: row['expires'] + 7300)
    with pytest.raises(HTTPException):
        api.admin(token)


def test_faq_does_not_use_model(monkeypatch):
    manuscript = doc()
    manuscript["blocks"].append({"kind": "FAQ", "id": "F1", "fields": {"問題": "支付多少？", "答案": "支付 2 資源", "狀態": "已確認", "引用": "R1", "角色": "全部", "人數": "全部", "擴充": "全部"}})
    version = publish(manuscript)
    monkeypatch.setenv("EDITORIAL_DAILY_AI_LIMIT", "0")
    answer = api.ask("game", api.Question(question="支付多少？", version=version))
    assert answer["kind"] == "faq"
    assert answer["citations"] == ["R1"]


def test_local_qa_has_sources_cache_and_no_cloud_fallback(monkeypatch):
    from src import local_qa
    monkeypatch.setenv('EDITORIAL_QA_PROVIDER', 'local')
    monkeypatch.setenv('LOCAL_QA_DAILY_LIMIT', '5')
    monkeypatch.setattr(api, 'generate', lambda *a, **k: pytest.fail('cloud called'))
    calls = []
    def local(*args, **kwargs):
        calls.append(kwargs)
        return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{"answer":"支付 2 資源","rule_explanation":"規則寫明須支付資源。","scope_limit":"","citations":["R1"],"cannot_determine":false,"evidence":[{"id":"R1","quote":"Pay resources"}]}'}]}}]}
    monkeypatch.setattr(local_qa, 'generate', local)
    version = publish()
    question = api.Question(question='支付資源', version=version)
    answer = api.ask('game', question)
    assert answer['provider'] == 'local'
    assert answer['sources'] == [{'id': 'R1', 'source': 'S1', 'page': '1'}]
    assert api.ask('game', question)['kind'] == 'cache'
    assert len(calls) == 1
    def failed(*args, **kwargs):
        raise TimeoutError('offline')
    monkeypatch.setattr(local_qa, 'generate', failed)
    with pytest.raises(HTTPException) as caught:
        api.ask('game', api.Question(question='如何支付資源', version=version))
    assert caught.value.status_code == 502

    def invalid(*args, **kwargs):
        return {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{"answer":"錯誤引用","citations":["MISSING"],"cannot_determine":false}'}]}}]}
    monkeypatch.setattr(local_qa, 'generate', invalid)
    answer = api.ask('game', api.Question(question='如何支付資源', version=version))
    assert answer['kind'] == 'abstained' and answer['citations'] == []


def test_context_and_version_required():
    manuscript = doc()
    manuscript["blocks"][0]["fields"]["角色"] = "工人"
    version = publish(manuscript)
    assert api.ask("game", api.Question(question="支付多少", version=version))["kind"] == "clarification"
    with pytest.raises(HTTPException) as caught:
        api.ask("game", api.Question(question="支付多少", version="stale"))
    assert caught.value.status_code == 409


def test_cap_and_exact_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(api, 'generate', lambda *a, **kw: calls.append(kw))
    monkeypatch.setenv('EDITORIAL_QA_PROVIDER', 'gemini')
    monkeypatch.setenv('EDITORIAL_DAILY_AI_LIMIT', '100')
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only')
    version = publish()
    result = api.ask('game', api.Question(question='支付資源', version=version))
    assert result['kind'] == 'disabled'
    assert calls == []


def test_google_docs_rejects_arbitrary_urls():
    with pytest.raises(HTTPException) as caught:
        api.google_doc("game", api.Docs(url="https://example.com/private"), "test-admin")
    assert caught.value.status_code == 422


def test_busy_local_gpu_is_service_wait_not_answer_failure(monkeypatch):
    from src import local_qa
    from src.local_inference_queue import LocalInferenceBusy
    monkeypatch.setenv("EDITORIAL_QA_PROVIDER", "local")
    def busy(*args, **kwargs):
        raise LocalInferenceBusy("busy")
    monkeypatch.setattr(local_qa, "generate", busy)
    version = publish()
    with pytest.raises(HTTPException) as error:
        api.ask("game", api.Question(question="支付資源", version=version))
    assert error.value.status_code == 503
    assert error.value.headers == {"Retry-After": "5"}
    assert "尚未送出推論" in error.value.detail

    conn = db._connect()
    try:
        assert conn.execute("SELECT calls FROM editorial_usage").fetchone()["calls"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM editorial_answers").fetchone()["n"] == 0
    finally:
        conn.close()
