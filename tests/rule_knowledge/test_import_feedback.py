from tests.rule_knowledge.test_import_jobs import client, upload, BASE, HEADERS
from src import import_quality, database


def test_feedback_keeps_original_evidence_without_queueing_models(client):
    job, _ = upload(client)
    path = BASE + '/' + job['id']
    document = {'blocks': [{'id': 'R', 'fields': {'原文': 'Pay 5 coins.', 'PDF頁碼': '1'}}]}
    answer = {'answer': '支付6枚', 'citations': ['R'], 'model': 'fixture', 'answer_engine': 'v1', 'confirmed_context': {'players': '2'}}
    identifier = import_quality.save_answer('game', job['id'], 'a' * 64, '費用？', answer, document)
    body = {'answer_id': identifier, 'kind': 'wrong_answer', 'comment': '數量不符'}
    saved = client.post(path + '/feedback', json=body, headers=HEADERS)
    assert saved.status_code == 200
    with database._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM import_repair_candidates').fetchone()[0] == 0
    assert client.get(path + '/feedback').status_code == 401
    assert client.get(path.replace('/game/', '/other/') + '/feedback', headers=HEADERS).status_code == 404
    other, _ = upload(client)
    assert client.post(BASE + '/' + other['id'] + '/feedback', json=body, headers=HEADERS).status_code == 404
    assert client.get(BASE + '/' + other['id'] + '/feedback', headers=HEADERS).json()['items'] == []
    assert client.get(path + '/feedback?offset=-1', headers=HEADERS).status_code == 422
    # Fill daily cap; retry of the already saved item must still succeed.
    for i in range(49):
        assert client.post(path + '/feedback', json={**body, 'comment': str(i)}, headers=HEADERS).status_code == 200
    assert client.post(path + '/feedback', json=body, headers=HEADERS).json()['id'] == saved.json()['id']
    assert client.post(path + '/feedback', json={**body, 'comment': 'new'}, headers=HEADERS).status_code == 429
    items, offset = [], 0
    while offset is not None:
        page = client.get(path + f'/feedback?offset={offset}', headers=HEADERS).json()
        items.extend(page['items']); offset = page['next_offset']
    assert len(items) == len({r['id'] for r in items}) == 50
    original = next(r for r in items if r['id'] == saved.json()['id'])
    assert original['record'] == {'question': '費用？', 'answer': answer, 'blocks': document['blocks']}
    assert original['version'] == 'a' * 64


def test_saved_answers_reopen_with_original_version_and_feedback_identity(client):
    job, _ = upload(client)
    path = BASE + '/' + job['id']
    answer = {'answer': '舊答案', 'citations': [], 'settings_version': 'b' * 64, 'confirmed_context': {'players': '2'}}
    identifier = import_quality.save_answer('game', job['id'], 'a' * 64, '原問題', answer, {'blocks': []})
    assert client.get(path + '/answers').status_code == 401
    assert client.get(path.replace('/game/', '/other/') + '/answers', headers=HEADERS).status_code == 404
    assert client.get(path + '/answers?offset=-1', headers=HEADERS).status_code == 422
    item = client.get(path + '/answers', headers=HEADERS).json()['jobs'][0]
    assert item['question'] == {'question': '原問題', 'version': 'a' * 64, 'settings_version': 'b' * 64, 'confirmed_context': {'players': '2'}}
    assert item['result'] == {**answer, 'answer_id': identifier}
    other, _ = upload(client)
    assert client.get(BASE + '/' + other['id'] + '/answers', headers=HEADERS).json()['jobs'] == []
    assert client.get(path + '/answers?offset=50', headers=HEADERS).json()['jobs'] == []
    assert client.post(path + '/feedback', json={'answer_id': item['result']['answer_id'], 'kind': 'unclear'}, headers=HEADERS).status_code == 200
