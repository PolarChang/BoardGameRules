import copy

import pytest
from fastapi import HTTPException
from src import database as db, editorial_api as api
from src.editorial_flow import validate_flows
from src.editorial_validation import validate_draft
from .test_editorial import doc


def flow_blocks():
    return [
        {"kind": "流程", "id": "FLOW", "fields": {"名稱": "回合流程", "範圍": "回合", "起點": "N1", "狀態": "已確認"}},
        {"kind": "流程節點", "id": "N1", "fields": {"流程": "FLOW", "標題": "行動", "順序": "1", "規則": "R1", "角色": "全部", "人數": "全部", "擴充": "全部", "結束": "否", "狀態": "已確認"}},
        {"kind": "流程節點", "id": "N2", "fields": {"流程": "FLOW", "標題": "結算", "順序": "2", "規則": "R1", "角色": "全部", "人數": "全部", "擴充": "全部", "結束": "是", "狀態": "已確認"}},
        {"kind": "流程連線", "id": "E1", "fields": {"流程": "FLOW", "起點": "N1", "終點": "N2", "條件": "沒有剩餘行動", "順序": "1", "循環": "否", "狀態": "已確認"}},
        {"kind": "流程連線", "id": "E2", "fields": {"流程": "FLOW", "起點": "N1", "終點": "N1", "條件": "還有行動", "順序": "2", "循環": "是", "狀態": "已確認"}},
    ]


def manuscript():
    result = doc()
    result["blocks"].extend(flow_blocks())
    return result


def test_flow_accepts_marked_cycle_with_reachable_end():
    assert validate_flows(manuscript()) == []


@pytest.mark.parametrize("index,key,value,message", [
    (4, "循環", "否", "循環連線須明確標記"),
    (1, "規則", "missing", "流程引用的規則不存在於此版本"),
    (3, "終點", "foreign-node", "連線端點不存在於此流程"),
    (1, "流程", "foreign-flow", "所屬流程不存在"),
    (2, "結束", "否", "流程須有結束節點"),
    (2, "順序", "1", "節點順序不可重複"),
    (4, "條件", "沒有剩餘行動", "分支條件不可重複"),
])
def test_flow_rejects_invalid_structure(index, key, value, message):
    document = manuscript()
    document["blocks"][5 + index]["fields"][key] = value
    assert any(message in issue for issue in validate_flows(document))


def test_flow_publication_is_authoritative_and_versioned(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "rules.db")
    for key in ["DB_HOST", "DATABASE_URL", "RULEBOOK_STORAGE_BUCKET"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KNOWLEDGE_ADMIN_KEY", "test-admin")
    db.init_db()
    from src.models import Game
    db.create_game(Game(id="game", name="流程測試"))
    api.init_editorial_db()
    document = manuscript()
    document["blocks"][5]["fields"]["狀態"] = "待確認"
    assert validate_draft(document) == []
    pending = api.save("game", api.Draft(document=document), "test-admin")
    with pytest.raises(HTTPException) as caught:
        api.publish("game", api.Publish(version=pending["version"], confirmed=True), "test-admin")
    assert caught.value.status_code == 422
    document["blocks"][5]["fields"]["狀態"] = "已確認"
    first = api.save("game", api.Draft(document=document), "test-admin")
    api.publish("game", api.Publish(version=first["version"], confirmed=True), "test-admin")
    revised = copy.deepcopy(document)
    revised["blocks"][6]["fields"]["標題"] = "新的行動標題"
    second = api.save("game", api.Draft(document=revised, base_version=first["version"]), "test-admin")
    assert second["version"] != first["version"]
    assert api.get_version("game", first["version"], "test-admin")["document"] == document
    assert api.current("game")["version"] == first["version"]
