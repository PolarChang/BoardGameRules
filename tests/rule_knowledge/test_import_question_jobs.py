from tests.rule_knowledge.test_import_jobs import client, upload, BASE, HEADERS
from src import import_question_jobs as jobs


def setup_question(client):
    job, _ = upload(client)
    path = BASE + '/' + job['id']
    version = client.get(path + '/draft', headers=HEADERS).json()['version']
    body = {'request_id': 'request-000000001', 'version': version, 'question': '需要多少資源？'}
    return path + '/questions', body


def test_submit_is_durable_idempotent_and_does_not_dispatch(client):
    path, body = setup_question(client)
    first = client.post(path, json=body, headers=HEADERS)
    assert first.status_code == 202
    row = first.json()
    assert row['status'] == 'queued' and row['question']['settings_version']
    assert client.post(path, json=body, headers=HEADERS).json() == row
    assert client.get(path + '/' + row['id'], headers=HEADERS).json() == row
    assert client.get(path, headers=HEADERS).json()['jobs'] == [row]
    assert client.post(path, json={**body, 'question': '另一個問題？'}, headers=HEADERS).status_code == 409
    assert client.get(path, headers=HEADERS).json()['jobs'] == [row]


def test_repeated_request_preserves_settings_after_correction(client):
    path, body = setup_question(client)
    first = client.post(path, json=body, headers=HEADERS).json()
    settings_path = path.removesuffix('/questions') + '/document'
    correction = {'kind':'base','edition':'second','expected_settings_version':first['question']['settings_version']}
    assert client.post(settings_path, json=correction, headers=HEADERS).status_code == 200
    assert client.post(path, json=body, headers=HEADERS).json() == first
    second = client.post(path, json={**body,'request_id':'request-000000002'}, headers=HEADERS).json()
    assert second['question']['settings_version'] != first['question']['settings_version']


def test_private_scope_and_missing_snapshot(client):
    path, body = setup_question(client)
    assert client.post(path, json=body).status_code == 401
    assert client.post(path.replace('/game/','/other/'), json=body, headers=HEADERS).status_code == 404
    assert client.post(path, json={**body,'version':'0'*64}, headers=HEADERS).status_code == 409
    row = client.post(path, json=body, headers=HEADERS).json()
    other = path.replace('/game/','/other/') + '/' + row['id']
    assert client.get(other, headers=HEADERS).status_code == 404
    assert client.post(other + '/cancel', headers=HEADERS).status_code == 404
    assert 'token' not in row


def test_cancelled_queued_work_is_never_claimed(client):
    path, body = setup_question(client)
    row = client.post(path, json=body, headers=HEADERS).json()
    endpoint = path + '/' + row['id']
    assert client.post(endpoint + '/cancel', headers=HEADERS).json()['status'] == 'cancelled'
    assert jobs.claim_next() is None
    assert client.post(path, json=body, headers=HEADERS).json()['status'] == 'cancelled'


def test_single_dispatch_lease_and_terminal_result_survive_reads(client):
    path, body = setup_question(client)
    row = client.post(path, json=body, headers=HEADERS).json()
    lease = jobs.claim_next()
    assert jobs.claim_next() is None
    assert not jobs.finish(row['id'], 'wrong-token', result={'answer':'wrong'})
    assert jobs.finish(row['id'], lease['token'], result={'answer':'支付 2 資源'})
    assert not jobs.finish(row['id'], lease['token'], error='late_error')
    saved = client.get(path + '/' + row['id'], headers=HEADERS).json()
    assert saved['status'] == 'completed'
    assert saved['result'] == {'answer':'支付 2 資源'}
    assert client.post(path, json=body, headers=HEADERS).json() == saved


def test_deadline_includes_queue_wait_and_rejects_late_results(client, monkeypatch):
    path, body = setup_question(client)
    first = client.post(path, json=body, headers=HEADERS).json()
    lease = jobs.claim_next()
    second = client.post(path, json={**body,'request_id':'request-000000002'}, headers=HEADERS).json()
    monkeypatch.setattr(jobs.time, 'time', lambda: max(first['deadline'],second['deadline']) + 1)
    assert not jobs.finish(first['id'], lease['token'], result={'answer':'late'})
    assert jobs.claim_next() is None
    rows = client.get(path, headers=HEADERS).json()['jobs']
    assert all(row['status'] == 'timed_out' and row['result'] is None for row in rows)


def test_failed_work_is_saved_without_automatic_retry(client):
    path, body = setup_question(client)
    row = client.post(path, json=body, headers=HEADERS).json()
    lease = jobs.claim_next()
    assert jobs.finish(row['id'], lease['token'], error='local_service_unavailable')
    assert jobs.claim_next() is None
    assert client.get(path + '/' + row['id'], headers=HEADERS).json()['status'] == 'failed'


def test_worker_uses_pinned_question_and_saves_result(client, monkeypatch):
    from src import local_question_worker, local_preview_qa
    path, body = setup_question(client)
    row = client.post(path, json=body, headers=HEADERS).json()
    monkeypatch.setenv('LOCAL_M3_QA_ENABLED', 'true')
    seen = []
    def answer(snapshot, question, game_id, **kwargs):
        seen.append((snapshot['version'],question,game_id))
        return {'status':'complete','answer':'支付 2 資源','citations':[]}
    monkeypatch.setattr(local_preview_qa, 'ask', answer)
    monkeypatch.setattr(local_question_worker, 'supervise', lambda lease: jobs.finish(lease['id'], lease['token'], result=local_question_worker.execute(lease)))
    assert local_question_worker.run_once()
    assert not local_question_worker.run_once()
    assert seen == [(body['version'],body['question'],'game')]
    saved = client.get(path + '/' + row['id'], headers=HEADERS).json()
    assert saved['status'] == 'completed'
    assert saved['result']['settings_version'] == row['question']['settings_version']
    assert saved['result']['answer_id']


def test_worker_requires_admin_key_before_claim(client, monkeypatch):
    import pytest
    from src import local_question_worker
    path, body = setup_question(client)
    client.post(path, json=body, headers=HEADERS)
    monkeypatch.setenv('LOCAL_M3_QA_ENABLED', 'false')
    from src import gemini_usage
    with monkeypatch.context() as settings:
        settings.setattr(gemini_usage, 'setting', lambda name, default='': default)
        with pytest.raises(RuntimeError, match='KNOWLEDGE_ADMIN_KEY'):
            local_question_worker.run_once()
    assert client.get(path, headers=HEADERS).json()['jobs'][0]['status'] == 'queued'


def test_worker_retains_partial_result_on_pipeline_failure(client, monkeypatch):
    from src import local_question_worker, local_preview_qa
    path, body = setup_question(client)
    client.post(path, json=body, headers=HEADERS)
    monkeypatch.setenv('LOCAL_M3_QA_ENABLED', 'true')
    monkeypatch.setattr(local_preview_qa, 'ask', lambda *a,**kw: {'status':'insufficient','answer':'來源仍可閱讀','failure_type':'TimeoutError','citations':[]})
    def supervised(lease):
        result = local_question_worker.execute(lease)
        jobs.finish(lease['id'], lease['token'], result=result, error=result['failure_type'])
    monkeypatch.setattr(local_question_worker, 'supervise', supervised)
    assert local_question_worker.run_once()
    row = client.get(path, headers=HEADERS).json()['jobs'][0]
    assert row['status'] == 'failed' and row['error'] == 'TimeoutError'
    assert row['result']['answer'] == '來源仍可閱讀'
