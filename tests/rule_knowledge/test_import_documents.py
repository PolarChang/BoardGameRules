import json

from src.database import _connect
from tests.rule_knowledge.test_import_jobs import client,upload,BASE,HEADERS


def test_correction_creates_new_setting_and_old_version_stays_readable(client):
    job,_=upload(client)
    path=BASE+'/'+job['id']+'/document'
    unknown=client.get(path,headers=HEADERS).json()
    assert unknown['document'] is None
    first=client.post(path,json={'kind':'base','edition':'1'},headers=HEADERS).json()
    corrected=client.post(path,json={'kind':'base','edition':'2','expected_settings_version':first['settings_version']},headers=HEADERS)
    assert corrected.status_code==200
    assert corrected.json()['settings_version']!=first['settings_version']
    assert client.get(path,headers=HEADERS).json()['document']['edition']=='2'
    for old in (unknown,first):
        assert client.get(path,params={'version':old['settings_version']},headers=HEADERS).json()['document']==old['document']
    assert client.post(path,json={'kind':'base','edition':'3','expected_settings_version':first['settings_version']},headers=HEADERS).status_code==409
    assert client.post(path,json={'kind':'base','edition':'3'},headers=HEADERS).status_code==409
    assert client.get(path.replace('/game/','/other/'),headers=HEADERS).status_code==404
    assert client.get(path,params={'version':first['settings_version']}).status_code==401


def test_question_keeps_setting_when_corrected_during_generation(client,monkeypatch):
    from src import local_preview_qa
    monkeypatch.setenv('LOCAL_M3_QA_ENABLED','true')
    job,_=upload(client)
    path=BASE+'/'+job['id']
    first=client.post(path+'/document',json={'kind':'base','edition':'1'},headers=HEADERS).json()
    snapshot=client.get(path+'/draft',headers=HEADERS).json()
    def answer(*args,**kwargs):
        response=client.post(path+'/document',json={'kind':'base','edition':'2','expected_settings_version':first['settings_version']},headers=HEADERS)
        assert response.status_code==200
        return {'status':'insufficient','kind':'abstained','answer':'No evidence','citations':[]}
    monkeypatch.setattr(local_preview_qa,'ask',answer)
    response=client.post(path+'/ask',json={'question':'規則？','version':snapshot['version'],'settings_version':first['settings_version']},headers=HEADERS)
    assert response.status_code==200
    result=response.json()
    assert result['settings_version']==first['settings_version']
    assert result['source_settings']['edition']=='1'
    conn=_connect()
    try:
        row=conn.execute('SELECT payload FROM import_answer_records WHERE id=?',(result['answer_id'],)).fetchone()
        assert json.loads(row['payload'])['answer']['source_settings']['edition']=='1'
    finally: conn.close()
    assert client.get(path+'/document',headers=HEADERS).json()['document']['edition']=='2'


def test_unknown_setting_version_never_dispatches_inference(client,monkeypatch):
    from src import local_preview_qa
    monkeypatch.setenv('LOCAL_M3_QA_ENABLED','true')
    monkeypatch.setattr(local_preview_qa,'ask',lambda *a,**k: (_ for _ in ()).throw(AssertionError('must not dispatch')))
    job,_=upload(client)
    path=BASE+'/'+job['id']
    snapshot=client.get(path+'/draft',headers=HEADERS).json()
    assert client.post(path+'/ask',json={'question':'規則？','version':snapshot['version'],'settings_version':'f'*64},headers=HEADERS).status_code==409


def test_confirmed_context_is_separate_from_question_and_saved_with_answer(client,monkeypatch):
    from src import local_preview_qa
    monkeypatch.setenv('LOCAL_M3_QA_ENABLED','true')
    job,_=upload(client)
    path=BASE+'/'+job['id']
    snapshot=client.get(path+'/draft',headers=HEADERS).json()
    context={'role':'工人','players':'2','stage':'第一回合'}
    def answer(source,question,game_id,*,confirmed_context):
        assert question=='如何準備？' and confirmed_context==context
        assert all(b.get('fields',{}).get('原文')!='工人' for b in source['document']['blocks'])
        return {'status':'needs_context','answer':'請核對版本','citations':[],'kind':'clarification'}
    monkeypatch.setattr(local_preview_qa,'ask',answer)
    result=client.post(path+'/ask',json={'question':'如何準備？','version':snapshot['version'],'confirmed_context':context},headers=HEADERS)
    assert result.status_code==200
    conn=_connect()
    try:
        saved=json.loads(conn.execute('SELECT payload FROM import_answer_records WHERE id=?',(result.json()['answer_id'],)).fetchone()['payload'])
        assert saved['question']=='如何準備？' and saved['answer']['confirmed_context']==context
    finally: conn.close()
    bad=client.post(path+'/ask',json={'question':'如何準備？','version':snapshot['version'],'confirmed_context':{'previous_model_answer':'自行編造的規則'}},headers=HEADERS)
    assert bad.status_code==422
