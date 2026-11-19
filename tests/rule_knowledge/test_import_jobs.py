import io
import fitz
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src import database as db, editorial_api, editorial_visuals, import_jobs as jobs
from src.models import Game


@pytest.fixture
def client(tmp_path, monkeypatch):
    from src import local_inference_queue
    monkeypatch.setattr(local_inference_queue, 'QUEUE_PATH', tmp_path / 'local-inference-queue.db')
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'rules.db')
    monkeypatch.setattr(editorial_visuals, 'DATA_DIR', tmp_path)
    monkeypatch.setenv('KNOWLEDGE_ADMIN_KEY', 'test-admin')
    for key in ('DB_HOST', 'DATABASE_URL', 'RULEBOOK_STORAGE_BUCKET'):
        monkeypatch.delenv(key, raising=False)
    db.init_db()
    for game_id in ('game', 'other'):
        db.create_game(Game(id=game_id, name=game_id))
    app = FastAPI()
    app.include_router(editorial_api.router)
    return TestClient(app)


BASE = '/api/knowledge/games/game/editorial/imports'
HEADERS = {'X-Knowledge-Admin': 'test-admin'}


def upload(client):
    with fitz.open() as pdf:
        pdf.new_page().insert_text((50, 100), 'Pay 2 resources. Never pay twice.')
        data = pdf.tobytes()
    result = client.post(BASE, files={'file': ('rules.pdf', data)}, headers=HEADERS)
    assert result.status_code == 200
    return result.json(), data


def test_upload_auth_scope_and_deduplication(client):
    assert client.get(BASE).status_code == 401
    job, data = upload(client)
    repeat = client.post(BASE, files={'file': ('renamed.pdf', data)}, headers=HEADERS)
    assert repeat.json()['id'] == job['id']
    assert client.get(BASE.replace('/game/', '/other/') + '/' + job['id'] + '/source', headers=HEADERS).status_code == 404
    assert client.post(BASE, files={'file': ('bad.pdf', b'invalid')}, headers=HEADERS).status_code == 422


def test_shared_queue_leases_games_without_cross_game_access(client):
    first, data = upload(client)
    second = client.post(BASE.replace('/game/', '/other/'), files={'file': ('rules.pdf', data)}, headers=HEADERS).json()
    assert first['id'] != second['id']
    a, b = jobs.claim_next(), jobs.claim_next()
    assert {a['job']['game_id'], b['job']['game_id']} == {'game', 'other'}
    assert a['job']['id'] != b['job']['id']
    assert jobs.claim_next()['job'] is None
    assert client.post(BASE + '/' + second['id'] + '/heartbeat', json={'token': b['token']}, headers=HEADERS).status_code == 409


def test_document_links_pin_parent_and_reject_cross_game_changes(client):
    base, _ = upload(client)
    with fitz.open() as pdf:
        pdf.new_page().insert_text((50, 100), 'Expansion changes scoring.')
        data = pdf.tobytes()
    expansion = client.post(BASE, files={'file': ('expansion.pdf', data)}, headers=HEADERS).json()
    path = BASE + '/' + expansion['id'] + '/document'
    link = {'kind': 'expansion', 'edition': '1', 'parent_job_id': base['id']}
    assert client.post(path, json=link).status_code == 401
    assert client.post(path, json=link, headers=HEADERS).status_code == 422
    assert client.post(BASE + '/' + base['id'] + '/document', json={'kind': 'base', 'edition': '1'}, headers=HEADERS).status_code == 200
    first = client.post(path, json=link, headers=HEADERS)
    assert first.status_code == 200
    assert first.json()['document']['parent_digest'] == base['digest']
    assert client.post(path, json=link, headers=HEADERS).status_code == 200
    assert client.post(path, json={**link, 'edition': '2'}, headers=HEADERS).status_code == 409
    assert client.get(path.replace('/game/', '/other/'), headers=HEADERS).status_code == 404


def test_source_crop_roundtrip_requires_auth_and_verified_pixels(client):
    import hashlib
    job, data = upload(client)
    lease = client.post(BASE + '/claim', json={}, headers=HEADERS).json()
    with fitz.open(stream=data, filetype='pdf') as pdf:
        box = [40, 80, 300, 130]
        image = pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(box)).tobytes('png')
    body = {'token': lease['token'], 'page': 1, 'rules': [{'title': '規則', 'translation': '支付 2 資源', 'original': '圖片來源', 'category': '進行遊戲'}], 'warnings': [],
            'locations': [{'block': 1, 'kind': 'image', 'bbox': box, 'scale': 2, 'image_sha256': hashlib.sha256(image).hexdigest()}]}
    bad = {**body, 'locations': [{**body['locations'][0], 'image_sha256': '0'*64}]}
    assert client.post(BASE + '/' + job['id'] + '/pages', json=bad, headers=HEADERS).status_code == 422
    assert client.post(BASE + '/' + job['id'] + '/pages', json=body, headers=HEADERS).status_code == 200
    url = BASE + '/' + job['id'] + '/pages/1/blocks/1/image'
    assert client.get(url).status_code == 401
    response = client.get(url, headers=HEADERS)
    assert response.content == image
    assert response.headers['cache-control'] == 'private, no-store'
    assert client.get(url.replace('/game/', '/other/'), headers=HEADERS).status_code == 404


def test_lease_reclaim_and_stale_worker_fencing(client):
    job, _ = upload(client)
    first = client.post(BASE + '/claim', json={}, headers=HEADERS).json()
    assert client.post(BASE + '/claim', json={}, headers=HEADERS).json()['job'] is None
    conn = db._connect()
    conn.execute('UPDATE editorial_import_jobs SET lease_until=0')
    conn.commit()
    conn.close()
    second = client.post(BASE + '/claim', json={}, headers=HEADERS).json()
    assert second['token'] != first['token']
    path = BASE + '/' + job['id']
    assert client.post(path + '/heartbeat', json={'token': first['token']}, headers=HEADERS).status_code == 409
    assert client.post(path + '/pause', json={'token': second['token']}, headers=HEADERS).status_code == 200
    assert client.post(path + '/resume', json={}, headers=HEADERS).status_code == 200


def test_completed_draft_remains_unreviewed(client):
    job, _ = upload(client)
    lease = client.post(BASE + '/claim', json={}, headers=HEADERS).json()
    path = BASE + '/' + job['id']
    result = client.post(path + '/pages', json={'token': lease['token'], 'page': 1, 'warnings': [], 'rules': [
        {'title': '支付資源', 'translation': '支付 2 資源，不可重複支付。', 'original': 'Pay 2 resources. Never pay twice.', 'category': '進行遊戲'}]}, headers=HEADERS)
    assert result.status_code == 200
    assert result.json()['status'] == 'review'
    draft = client.get(path + '/draft', headers=HEADERS).json()
    assert '狀態：待確認' in draft['text']
    assert '狀態：已確認' not in draft['text']
    assert draft['completed_pages'] == 1


def test_usage_is_idempotent_and_records_unknown_attempts(client):
    from src.local_usage import summary
    job, _ = upload(client)
    path = BASE + '/' + job['id'] + '/usage'
    event = {'id': 'a' * 32, 'model': 'qwen3:8b'}
    assert client.post(path, json=event, headers=HEADERS).status_code == 200
    assert summary('game')['models'][0]['unknown_requests'] == 1
    event.update(prompt_tokens=120, output_tokens=30)
    for _ in range(2):
        assert client.post(path, json=event, headers=HEADERS).status_code == 200
    assert client.post(path, json={'id': 'a' * 32, 'model': 'qwen3:8b'}, headers=HEADERS).status_code == 200
    model = summary('game')['models'][0]
    assert (model['requests'], model['input_tokens'], model['output_tokens'], model['unknown_requests']) == (1, 120, 30, 0)
    assert client.post(path.replace('/game/', '/other/'), json=event, headers=HEADERS).status_code == 404
    assert client.post(path, json={**event, 'output_tokens': -1}, headers=HEADERS).status_code == 422


def test_image_upload_converts_to_private_pdf_and_deduplicates(client):
    from PIL import Image
    image = io.BytesIO()
    Image.new('RGB', (120, 80), 'white').save(image, format='PNG')
    payload = image.getvalue()
    first = client.post(BASE, files={'file': ('table.png', payload, 'image/png')}, headers=HEADERS)
    second = client.post(BASE, files={'file': ('table-copy.png', payload, 'image/png')}, headers=HEADERS)
    assert first.status_code == 200
    assert first.json()['total'] == 1
    assert second.json()['id'] == first.json()['id']
    source = client.get(BASE + '/' + first.json()['id'] + '/source', headers=HEADERS)
    assert source.headers['cache-control'] == 'private, no-store'
    assert source.content.startswith(b'%PDF-')


def test_completed_job_reports_unresolved_blocks(client):
    job, _ = upload(client)
    lease = client.post(BASE + '/claim', json={}, headers=HEADERS).json()
    result = client.post(BASE + '/' + job['id'] + '/pages', json={'token': lease['token'], 'page': 1, 'warnings': ['須補充'], 'rules': [{'title': '待辨識區塊 1', 'translation': '未確認', 'original': 'source', 'category': '其他規則'}]}, headers=HEADERS)
    assert result.json()['status'] == 'review'
    assert result.json()['unresolved_blocks'] == 1
    assert result.json()['warning_pages'] == 1

def test_retry_preserves_successful_blocks_and_archives_previous_page(client):
    import json
    job, _ = upload(client)
    path = BASE + '/' + job['id']
    lease = client.post(BASE + '/claim', json={}, headers=HEADERS).json()
    good = {'title': '支付', 'translation': '支付 2', 'original': 'Pay 2', 'category': '進行遊戲'}
    failed = {**good, 'title': '待辨識區塊 2', 'translation': '未確認'}
    payload = {'token': lease['token'], 'page': 1, 'warnings': [], 'rules': [good, failed]}
    assert client.post(path + '/pages', json=payload, headers=HEADERS).status_code == 200
    assert client.post(path + '/retry', json={'blocks': [{'page': 1, 'block': 1}]}, headers=HEADERS).status_code == 422
    assert client.get(BASE, headers=HEADERS).json()['jobs'][0]['status'] == 'review'
    assert client.post(path + '/retry', json={}, headers=HEADERS).json()['pending_retry_blocks'] == 1
    assert client.post(path + '/retry', json={}, headers=HEADERS).status_code == 409
    assert client.get(path.replace('/game/', '/other/') + '/retries', headers=HEADERS).status_code == 404
    details = client.get(path + '/retries', headers=HEADERS).json()['pages']['1']
    assert details['blocks'] == [2]
    conn = db._connect()
    saved = conn.execute('SELECT payload FROM editorial_import_page_history').fetchone()
    conn.close()
    assert json.loads(saved['payload'])['rules'] == [good, failed]
    lease2 = client.post(BASE + '/claim', json={}, headers=HEADERS).json()
    assert client.post(path + '/pages', json=payload, headers=HEADERS).status_code == 409
    payload['token'] = lease2['token']
    payload['rules'] = [{**good, 'translation': 'changed'}, good]
    assert client.post(path + '/pages', json=payload, headers=HEADERS).status_code == 409
    payload['rules'] = [good, good]
    result = client.post(path + '/pages', json=payload, headers=HEADERS)
    assert result.status_code == 200
    assert result.json()['status'] == 'review'
    assert result.json()['pending_retry_blocks'] == 0
    assert result.json()['unresolved_blocks'] == 0


def test_preview_read_version_and_private_ask(client, monkeypatch):
    job, _ = upload(client)
    path = BASE + '/' + job['id']
    assert client.get(path + '/draft').status_code == 401
    snapshot = client.get(path + '/draft', headers=HEADERS).json()
    assert len(snapshot['version']) == 64
    assert snapshot['document']['blocks'][0]['kind'] == '文件'
    body = {'question': '最終計分', 'version': snapshot['version']}
    assert client.post(path + '/ask', json=body).status_code == 401
    assert client.post(path.replace('/game/', '/other/') + '/ask', json=body, headers=HEADERS).status_code == 404
    assert client.post(path + '/ask', json={**body, 'version': '0' * 64}, headers=HEADERS).status_code == 409
    def answer(game_id, question, cache_allowed, preview_release):
        assert cache_allowed is False
        assert preview_release['version'] == snapshot['version']
        return {'answer': '尚未有完成頁面', 'citations': [], 'kind': 'abstained'}
    monkeypatch.setattr(editorial_api, '_ask', answer)
    result = client.post(path + '/ask', json=body, headers=HEADERS)
    assert result.status_code == 200 and result.json()['preview'] is True


def test_preview_retrieval_preserves_source_and_multiple_pages():
    from src.preview_retrieval import retrieve
    blocks = [{'id': str(i), 'kind': '規則', 'fields': {'原文': text, 'PDF頁碼': str(i), '狀態': '待確認'}} for i, text in enumerate([
        'Endgame\nScore 2 points per building.\nNever count empty buildings.',
        'Game End\nEach loan costs 5 points.',
        'Setup\nPlace the board on the table.'
    ])]
    result = retrieve(blocks, '最終計分與貸款')
    assert set(result) == {'0', '1'}
    for key, block in result.items():
        assert block['fields']['原文'] in blocks[int(key)]['fields']['原文']
    assert 'Never count empty buildings.' in result['0']['fields']['原文']


def test_quality_checks_are_generic_and_do_not_certify_clean_text():
    from src.import_quality import findings
    document = {'blocks': [{'id': 'X', 'kind': '規則', 'fields': {'PDF頁碼': '1', '原文': 'Table\n1\n2\n3\n4\nGain 2     / 3 points', '內容': '獲得 99 分'}}]}
    issues = findings(document)
    assert {i['code'] for i in issues} == {'table_layout', 'missing_symbols', 'number_mismatch'}
    assert all(i['id'] == 'X' and i['page'] == '1' for i in issues)
    assert findings({'blocks': []}) == []


def test_feedback_binds_stored_answer_and_repair_queue_is_bounded(client, monkeypatch):
    from src import import_quality
    job, _ = upload(client)
    path = BASE + '/' + job['id']
    snapshot = client.get(path + '/draft', headers=HEADERS).json()
    identifier = import_quality.save_answer('game', job['id'], snapshot['version'], '問題', {'answer': '答案', 'citations': []}, snapshot['document'])
    body = {'answer_id': identifier, 'kind': 'missing_condition', 'comment': '漏掉最後一步'}
    assert client.post(path + '/feedback', json=body).status_code == 401
    first = client.post(path + '/feedback', json=body, headers=HEADERS)
    assert first.status_code == 200
    assert client.post(path + '/feedback', json=body, headers=HEADERS).json()['id'] == first.json()['id']
    assert client.post(path.replace('/game/', '/other/') + '/feedback', json=body, headers=HEADERS).status_code == 404
    assert client.post(path + '/repair', json={'version': '0' * 64}, headers=HEADERS).status_code == 409
    assert client.get(path + '/quality').status_code == 401
    assert client.get(path + '/quality', headers=HEADERS).json()['status'] == 'no_detected_issue'
    blocks = [{'id': str(i), 'kind': '規則', 'fields': {'PDF頁碼': str(i), '原文': '1\n2\n3\n4\n'}} for i in range(1, 5)]
    monkeypatch.setattr(jobs, 'draft', lambda *args: {**snapshot, 'document': {'blocks': blocks}})
    queued = client.post(path + '/repair', json={'version': snapshot['version']}, headers=HEADERS).json()['queued']
    assert len(queued) == 2
    assert client.post(path + '/repair', json={'version': snapshot['version']}, headers=HEADERS).json()['queued'] == queued


def test_answer_coverage_flags_missing_branches():
    from src.import_quality import answer_gaps
    selected = {'R': {'fields': {'原文': 'Count the section of their choice. Tied players share the victory.', 'PDF頁碼': '4'}}}
    assert len(answer_gaps('平手如何判定？', {'kind': 'ai', 'answer': '比較數量', 'citations': ['R']}, selected)) == 2
    assert answer_gaps('平手如何判定？', {'kind': 'ai', 'answer': '比較各自對應區域，仍平手則共享勝利', 'citations': ['R']}, selected) == []


def test_phase_enumeration_flags_unreferenced_list_and_exception():
    from src.import_quality import answer_gaps
    selected = {
        'list': {'fields': {'原文': 'Phases:\n• Planning Phase\n• Cleanup Phase', 'PDF頁碼': '2'}},
        'exception': {'fields': {'原文': 'This phase is skipped in the first round.', 'PDF頁碼': '2'}},
    }
    answer = {'kind':'ai','answer':'規劃之後清理。','citations':[]}
    assert len(answer_gaps('每回合依序有哪些階段？', answer, selected)) == 2
    # A citation check does not infer whether a translated answer is correct.
    assert answer_gaps('每回合依序有哪些階段？', {**answer,'citations':['list','exception']}, selected) == []
    assert answer_gaps('行動階段要支付多少？', answer, selected) == []
    assert answer_gaps('有哪些階段？', {**answer,'kind':'abstained'}, selected) == []


def test_repair_worker_rejects_title_only_and_preserves_source(client, monkeypatch):
    from src import import_quality, local_repair_worker, local_import_worker
    job, _ = upload(client)
    path = BASE + '/' + job['id']
    snap = client.get(path + '/draft', headers=HEADERS).json()
    monkeypatch.setattr(jobs, 'draft', lambda *args: {**snap, 'document': {'blocks': [{'id': 'R', 'kind': '規則', 'fields': {'PDF頁碼': '1', '原文': '1\n2\n3\n4\n'}}]}})
    client.post(path + '/repair', headers=HEADERS, json={'version': snap['version']})
    monkeypatch.setattr(local_import_worker, 'local_chat', lambda *args, **kwargs: ({'title': '標題', 'translation': '標題', 'category': '其他規則'}, {'prompt_tokens': 1, 'output_tokens': 1}))
    assert local_repair_worker.run_one() is True
    assert local_repair_worker.run_one() is False
    c = db._connect()
    try:
        row = c.execute('SELECT state,result FROM import_repair_candidates').fetchone()
        assert row['state'] == 'rejected'
        assert 'image_sha256' in row['result']
        assert c.execute('SELECT COUNT(*) FROM editorial_import_pages').fetchone()[0] == 0
    finally:
        c.close()


def test_budget_pause_preserves_source_and_explains_limit(client):
    job,original=upload(client)
    claim=jobs.claim_next()
    path=BASE+'/'+job['id']
    response=client.post(path+'/pause',json={'token':claim['token'],'reason':'document_tokens_limit'},headers=HEADERS)
    assert response.status_code==200
    paused=client.get(BASE,headers=HEADERS).json()['jobs'][0]
    assert paused['status']=='paused'
    assert '此文件Token預算上限' in paused['error']
    assert client.get(path+'/source',headers=HEADERS).content==original
    assert client.post(path+'/pause',json={'token':claim['token'],'reason':'private path or secret'},headers=HEADERS).status_code==422


@pytest.mark.parametrize('reason,delay',[('local_service',60),('day_tokens_limit',86400),('document_tokens_limit',None),('work_tokens_limit',None)])
def test_import_wait_resumes_only_after_its_due_time(client,monkeypatch,reason,delay):
    clock=[86400.0]
    monkeypatch.setattr(jobs.time,'time',lambda:clock[0])
    job,_=upload(client)
    claim=jobs.claim_next()
    path=BASE+'/'+job['id']
    assert client.post(path+'/pause',json={'token':claim['token'],'reason':reason},headers=HEADERS).status_code==200
    assert jobs.claim_next()['job'] is None
    saved=client.get(BASE,headers=HEADERS).json()['jobs'][0]
    assert saved['resume_at']==(86400+delay if delay else 0)
    if delay:
        clock[0]+=delay-1
        assert jobs.claim_next()['job'] is None
        clock[0]+=1
        resumed=jobs.claim_next()
        assert resumed['job']['id']==job['id']
        assert resumed['token']!=claim['token']
    else:
        clock[0]+=86400*3
        assert jobs.claim_next()['job'] is None


def test_original_page_image_requires_auth_scope_and_immutable_source(client, monkeypatch):
    import hashlib
    job, data = upload(client)
    digest = hashlib.sha256(data).hexdigest()
    url = f"{BASE}/{job['id']}/pages/1/image?digest={digest}"
    assert client.get(url).status_code == 401
    assert client.get(url.replace('/game/', '/other/'), headers=HEADERS).status_code == 404
    assert client.get(url.replace(digest, '0' * 64), headers=HEADERS).status_code == 409
    assert client.get(url.split('?')[0], headers=HEADERS).status_code == 422
    assert client.get(url.replace('/pages/1/', '/pages/0/'), headers=HEADERS).status_code == 404
    assert client.get(url.replace('/pages/1/', '/pages/2/'), headers=HEADERS).status_code == 404
    result = client.get(url, headers=HEADERS)
    assert result.status_code == 200
    assert result.headers['content-type'] == 'image/png'
    assert result.headers['cache-control'] == 'private, no-store'
    assert result.headers['x-source-digest'] == digest
    image = fitz.Pixmap(result.content)
    assert max(image.width, image.height) <= 2200
    assert image.width * image.height <= 4_010_000
    monkeypatch.setattr(jobs, 'read_bytes', lambda key: b'changed')
    assert client.get(url, headers=HEADERS).status_code == 409


def test_original_page_image_bounds_unusually_large_page(client):
    import hashlib
    with fitz.open() as pdf:
        pdf.new_page(width=10000, height=10000)
        data = pdf.tobytes()
    job = client.post(BASE, files={'file': ('large-page.pdf', data)}, headers=HEADERS).json()
    digest = hashlib.sha256(data).hexdigest()
    result = client.get(f"{BASE}/{job['id']}/pages/1/image?digest={digest}", headers=HEADERS)
    assert result.status_code == 200
    image = fitz.Pixmap(result.content)
    assert max(image.width, image.height) <= 2200
    assert image.width * image.height <= 4_010_000
