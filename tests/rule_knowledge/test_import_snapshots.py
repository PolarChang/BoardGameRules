import json
import fitz
from tests.rule_knowledge.test_import_jobs import client, upload, BASE, HEADERS
from src import database as db, import_snapshots, import_jobs, local_source_worker
from src.import_alignment import align


def test_old_snapshot_and_index_survive_new_translation(client, monkeypatch):
    job,_=upload(client)
    path=BASE+'/'+job['id']
    old=client.get(path+'/draft',headers=HEADERS).json()
    lease=client.post(BASE+'/claim',json={},headers=HEADERS).json()
    client.post(path+'/pages',headers=HEADERS,json={'token':lease['token'],'page':1,'rules':[{'title':'Cost','translation':'支付兩個資源','original':'Pay 2 resources. Never pay twice.','category':'其他規則'}],'warnings':[]})
    new=client.get(path+'/draft',headers=HEADERS).json()
    assert old['version']!=new['version']
    assert client.get(path+'/draft?version='+old['version'],headers=HEADERS).json()==old
    def answer(game_id, body, **kwargs):
        assert kwargs['preview_release']['document']==old['document']
        return {'answer':'Pay 2 resources.','citations':['NATIVE-P001-R001'],'kind':'ai'}
    from src import editorial_api
    monkeypatch.setattr(editorial_api,'_ask',answer)
    assert client.post(path+'/ask',headers=HEADERS,json={'version':old['version'],'question':'resources'}).status_code==200
    result=client.get(path+'/draft',params={'version':old['version'],'q':'Never pay'},headers=HEADERS).json()
    assert result['search']['ids']==['NATIVE-P001-R001']
    assert client.get(path.replace('/game/','/other/')+'/draft?version='+old['version'],headers=HEADERS).status_code==404
    assert client.get(path+'/draft?version='+old['version']).status_code==401
    assert client.get(path+'/draft?q=%',headers=HEADERS).json()['search']['total']==0


def test_source_worker_extracts_without_reader_and_resumes_page_cache(client, monkeypatch):
    with fitz.open() as pdf:
        for text in ('First page condition.','Second page exception.'):
            pdf.new_page().insert_text((50,100),text)
        data=pdf.tobytes()
    job=client.post(BASE,files={'file':('two.pdf',data)},headers=HEADERS).json()
    assert local_source_worker.run_batch()==1
    conn=db._connect()
    snapshot=json.loads(conn.execute('SELECT payload FROM import_snapshots').fetchone()['payload'])
    conn.close()
    assert snapshot['readable_pages']==1
    # A new invocation resumes using durable page results without extracting page 1.
    original=fitz.Page.get_text
    def guard(page,*args,**kwargs):
        if page.number==0 and args and args[0]=='blocks':
            raise AssertionError('Completed native page was re-extracted')
        return original(page,*args,**kwargs)
    # Categories also reads text; isolate only that independent consumer.
    from src import import_categories
    monkeypatch.setattr(import_categories,'single_column',lambda page:False)
    monkeypatch.setattr(fitz.Page,'get_text',guard)
    assert local_source_worker.run_batch()==1
    assert local_source_worker.run_batch()==0
    conn=db._connect()
    rows=[json.loads(r['payload']) for r in conn.execute('SELECT payload FROM import_snapshots')]
    conn.close()
    assert sorted(r['readable_pages'] for r in rows)==[1,2]


def test_snapshot_save_is_immutable_and_index_covers_all_usable_blocks(client):
    job,_=upload(client)
    snapshot=import_jobs.build_draft('game',job['id'])
    initial=json.loads(json.dumps(snapshot))
    snapshot['document']['blocks'].append({'id':'BAD','kind':'規則','fields':{'原文':'injected'}})
    import_snapshots.save('game',job['id'],snapshot)
    assert import_snapshots.load('game',job['id'],initial['version'])==initial
    assert import_snapshots.search('game',job['id'],initial['version'],'injected')['total']==0
    assert import_snapshots.search('game',job['id'],initial['version'],'')['total']==1


def test_alignment_does_not_merge_different_conditions_or_ambiguous_matches():
    def block(identifier,text,native=False):
        return {'kind':'規則','id':identifier,'fields':{'來源':'S','PDF頁碼':'1','原文':text,'內容':text},**({'extraction':'native'} if native else {'derivation':'model-extraction'})}
    blocks=[block('N1','Pay 2 unless exempt.',True),block('L1','Pay 2.')]
    result,aliases=align(blocks)
    assert len(result)==2 and not aliases
    result,aliases=align([block('N1','Pay 2.',True),block('N2','Pay 2.',True),block('L1','Pay 2.')])
    assert len(result)==3 and not aliases

def test_local_api_lifespan_processes_upload_without_opening_reader(client, monkeypatch):
    import threading
    import os
    monkeypatch.setenv('EDITORIAL_QA_PROVIDER', os.environ.get('EDITORIAL_QA_PROVIDER', 'gemini'))
    from fastapi.testclient import TestClient
    from src.local_knowledge_server import app
    ready, finished = threading.Event(), threading.Event()
    original = local_source_worker.run_batch
    def scheduled():
        if ready.wait(5):
            original()
            finished.set()
    monkeypatch.setattr(local_source_worker,'run_batch',scheduled)
    with TestClient(app) as api:
        job,_=upload(api)
        ready.set()
        assert finished.wait(5)
        conn=db._connect()
        row=conn.execute('SELECT payload FROM import_snapshots WHERE job_id=?',(job['id'],)).fetchone()
        conn.close()
        assert json.loads(row['payload'])['readable_pages']==1

def test_worker_recovers_after_native_cache_completed_before_snapshot(client):
    job,_=upload(client)
    from src.import_native import source_pages
    conn=db._connect()
    row=dict(conn.execute('SELECT * FROM editorial_import_jobs WHERE id=?',(job['id'],)).fetchone())
    conn.close()
    source_pages(row)
    assert local_source_worker.run_batch()==1
    assert local_source_worker.run_batch()==0
    conn=db._connect()
    assert conn.execute('SELECT COUNT(*) FROM import_snapshots').fetchone()[0]==1
    conn.close()
