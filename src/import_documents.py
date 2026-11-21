"""Explicit, immutable relationships between private imported rulebooks."""
import json
import hashlib
from typing import Literal
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from src.local_question_context import ConfirmedContext
from src.database import _connect
from src.import_jobs import authorize, get_job

router = APIRouter()


class SavedContext(BaseModel):
    model_config = ConfigDict(extra='forbid')
    settings_version: str = Field(pattern=r'^[a-f0-9]{64}$')
    expected_revision: int = Field(ge=0)
    confirmed_context: ConfirmedContext


def init_contexts(cur):
    cur.execute('CREATE TABLE IF NOT EXISTS import_confirmed_contexts (job_id TEXT NOT NULL, settings_version TEXT NOT NULL, revision INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id,settings_version))')


@router.get('/imports/{job_id}/context')
def read_context(game_id: str, job_id: str, settings_version: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        get_job(cur, game_id, job_id)
        settings(cur, job_id, settings_version)
        init_contexts(cur)
        cur.execute('SELECT revision,payload FROM import_confirmed_contexts WHERE job_id=? AND settings_version=?', (job_id, settings_version))
        row = cur.fetchone()
        conn.commit()
        return {'settings_version': settings_version, 'revision': row['revision'] if row else 0,
                'confirmed_context': json.loads(row['payload']) if row else ConfirmedContext().model_dump()}
    finally:
        conn.close()


@router.post('/imports/{job_id}/context')
def save_context(game_id: str, job_id: str, body: SavedContext, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        get_job(cur, game_id, job_id)
        settings(cur, job_id, body.settings_version)
        init_contexts(cur)
        context = {key: value.strip() for key, value in body.confirmed_context.model_dump().items()}
        encoded = json.dumps(context, ensure_ascii=False, sort_keys=True)
        if body.expected_revision == 0:
            cur.execute('INSERT INTO import_confirmed_contexts VALUES (?,?,1,?) ON CONFLICT DO NOTHING', (job_id, body.settings_version, encoded))
        else:
            cur.execute('UPDATE import_confirmed_contexts SET revision=revision+1,payload=? WHERE job_id=? AND settings_version=? AND revision=?', (encoded, job_id, body.settings_version, body.expected_revision))
        changed = cur.rowcount == 1
        cur.execute('SELECT revision,payload FROM import_confirmed_contexts WHERE job_id=? AND settings_version=?', (job_id, body.settings_version))
        row = cur.fetchone()
        # A lost response may be retried only if it is the immediate same write.
        if not changed and not (row and row['revision'] == body.expected_revision + 1 and row['payload'] == encoded):
            raise HTTPException(409, '情境已由另一頁更新，請重新讀取後再保存')
        conn.commit()
        return {'settings_version': body.settings_version, 'revision': row['revision'], 'confirmed_context': context}
    finally:
        conn.close()


class DocumentLink(BaseModel):
    kind: Literal['base', 'expansion', 'errata']
    edition: str = Field(min_length=1, max_length=120)
    parent_job_id: str = Field(default='', max_length=64)
    expected_settings_version: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')


def init_links(cur):
    cur.execute('CREATE TABLE IF NOT EXISTS editorial_import_documents (job_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    cur.execute('CREATE TABLE IF NOT EXISTS import_document_settings (job_id TEXT NOT NULL, version TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id,version))')
    cur.execute('CREATE TABLE IF NOT EXISTS import_document_settings_heads (job_id TEXT PRIMARY KEY, version TEXT NOT NULL)')


def settings(cur, job_id, version=None):
    """Resolve an immutable setting; legacy unknowns remain explicitly unknown."""
    init_links(cur)
    if version is None:
        cur.execute('SELECT version FROM import_document_settings_heads WHERE job_id=?',(job_id,))
        head=cur.fetchone()
        if head: version=head['version']
    if version:
        cur.execute('SELECT payload FROM import_document_settings WHERE job_id=? AND version=?',(job_id,version))
        row=cur.fetchone()
        if row: return {'settings_version':version,'document':json.loads(row['payload'])}
    cur.execute('SELECT payload FROM editorial_import_documents WHERE job_id=?',(job_id,))
    row=cur.fetchone()
    payload=json.loads(row['payload']) if row else None
    encoded=json.dumps(payload,ensure_ascii=False,sort_keys=True)
    legacy_version=hashlib.sha256(json.dumps([job_id,payload],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    if version and version!=legacy_version: raise HTTPException(409,'找不到指定適用設定版本')
    cur.execute('INSERT INTO import_document_settings VALUES (?,?,?) ON CONFLICT DO NOTHING',(job_id,legacy_version,encoded))
    cur.execute('INSERT INTO import_document_settings_heads VALUES (?,?) ON CONFLICT DO NOTHING',(job_id,legacy_version))
    return {'settings_version':legacy_version,'document':payload}


def question_settings(game_id, job_id, version=None):
    conn=_connect()
    try:
        cur=conn.cursor()
        get_job(cur,game_id,job_id)
        result=settings(cur,job_id,version)
        conn.commit()
        return result
    finally:
        conn.close()


@router.get('/imports/{job_id}/document')
def document(game_id: str, job_id: str, x_knowledge_admin: str = Header(default=''), version: str | None = None):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        job = get_job(cur, game_id, job_id)
        result=settings(cur,job_id,version)
        conn.commit()
        return {'job_id': job_id, 'digest': job['digest'], **result}
    finally:
        conn.close()


@router.post('/imports/{job_id}/document')
def link_document(game_id: str, job_id: str, body: DocumentLink, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    if not body.edition.strip() or (body.kind == 'base') != (not body.parent_job_id) or body.parent_job_id == job_id:
        raise HTTPException(422, '基本版不可有父文件；擴充與勘誤必須指定另一份來源文件')
    conn = _connect()
    try:
        cur = conn.cursor()
        get_job(cur, game_id, job_id)
        current=settings(cur,job_id)
        payload = body.model_dump(exclude={'expected_settings_version'})
        if body.parent_job_id:
            parent = get_job(cur, game_id, body.parent_job_id)
            parent_settings=settings(cur,body.parent_job_id)
            if not parent_settings['document'] or parent_settings['document']['kind'] not in (['base'] if body.kind == 'expansion' else ['base', 'expansion']):
                raise HTTPException(422, '請先分類父文件；擴充須連基本版，勘誤須連基本版或擴充')
            payload['parent_digest'] = parent['digest']
            payload['parent_settings_version'] = parent_settings['settings_version']
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if current['document']==payload:
            conn.commit()
            return {'job_id':job_id,**current}
        expected=body.expected_settings_version
        if (expected is not None and expected!=current['settings_version']) or (current['document'] is not None and expected is None):
            raise HTTPException(409,'適用設定已變更；請重新讀取目前版本再更正')
        identifier=hashlib.sha256(json.dumps([job_id,payload],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        cur.execute('INSERT INTO import_document_settings VALUES (?,?,?) ON CONFLICT DO NOTHING',(job_id,identifier,encoded))
        cur.execute('UPDATE import_document_settings_heads SET version=? WHERE job_id=? AND version=?',(identifier,job_id,current['settings_version']))
        if cur.rowcount!=1: raise HTTPException(409,'適用設定已變更；請重新讀取目前版本再更正')
        # Preserve the initial legacy record for old clients and rollback.
        cur.execute('INSERT INTO editorial_import_documents VALUES (?,?) ON CONFLICT (job_id) DO NOTHING', (job_id, encoded))
        conn.commit()
        return {'job_id': job_id, 'document': payload,'settings_version':identifier}
    finally:
        conn.close()
