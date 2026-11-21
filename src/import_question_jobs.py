"""Durable private question jobs; inference is never dispatched by submission."""
import hashlib
import json
import secrets
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import Field
from src.database import _connect
from src.import_jobs import PreviewQuestion, authorize, get_job

router = APIRouter()


class QuestionJob(PreviewQuestion):
    request_id: str = Field(min_length=16, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    timeout_seconds: int = Field(default=600, ge=60, le=1800)


def init(cur):
    cur.execute('CREATE TABLE IF NOT EXISTS import_question_jobs (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, job_id TEXT NOT NULL, fingerprint TEXT NOT NULL, payload TEXT NOT NULL, status TEXT NOT NULL, created_at DOUBLE PRECISION NOT NULL, deadline DOUBLE PRECISION NOT NULL, token TEXT NOT NULL, result TEXT, error TEXT NOT NULL)')


def expire(cur, now):
    # Interrupted executions are not automatically replayed: their usage may be unknown.
    cur.execute("UPDATE import_question_jobs SET status='timed_out', token='', error='deadline_exceeded' WHERE status IN ('queued','running','cancelling') AND deadline<=?", (now,))


def public(row):
    return {key: row[key] for key in ('id','game_id','job_id','status','created_at','deadline','error')} | {
        'question': json.loads(row['payload']),
        'result': json.loads(row['result']) if row['result'] else None,
    }


def find(cur, game_id, job_id, identifier):
    cur.execute('SELECT * FROM import_question_jobs WHERE id=? AND game_id=? AND job_id=?', (identifier,game_id,job_id))
    row = cur.fetchone()
    if row is None:
        raise HTTPException(404, '找不到問答工作')
    return row


@router.post('/imports/{job_id}/questions', status_code=202)
def submit(game_id: str, job_id: str, body: QuestionJob, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    requested = body.model_dump(exclude={'request_id'})
    fingerprint = hashlib.sha256(json.dumps(requested, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    identifier = hashlib.sha256(json.dumps([game_id,job_id,body.request_id]).encode()).hexdigest()
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur)
        get_job(cur, game_id, job_id)
        expire(cur, time.time())
        cur.execute('SELECT * FROM import_question_jobs WHERE id=?', (identifier,))
        row = cur.fetchone()
        conn.commit()
        if row is None:
            from src.import_snapshots import load
            from src.import_documents import question_settings
            load(game_id, job_id, body.version)
            scope = question_settings(game_id, job_id, body.settings_version)
            pinned = {**requested, 'settings_version': scope['settings_version']}
            now = time.time()
            cur.execute('INSERT INTO import_question_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING',
                        (identifier,game_id,job_id,fingerprint,json.dumps(pinned,ensure_ascii=False),'queued',now,now+body.timeout_seconds,'',None,''))
            conn.commit()
            row = find(cur, game_id, job_id, identifier)
        if row['fingerprint'] != fingerprint:
            raise HTTPException(409, '相同 request_id 已用於不同問題或設定')
        return public(row)
    finally:
        conn.close()


@router.get('/imports/{job_id}/questions')
def history(game_id: str, job_id: str, x_knowledge_admin: str = Header(default=''), offset: int = 0):
    authorize(game_id, x_knowledge_admin)
    if offset < 0:
        raise HTTPException(422, 'offset 不可為負數')
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); get_job(cur, game_id, job_id)
        expire(cur, time.time())
        cur.execute('SELECT * FROM import_question_jobs WHERE game_id=? AND job_id=? ORDER BY created_at DESC,id LIMIT 50 OFFSET ?', (game_id,job_id,offset))
        rows = [public(row) for row in cur.fetchall()]
        conn.commit()
        return {'jobs': rows, 'offset': offset, 'limit': 50}
    finally:
        conn.close()


@router.get('/imports/{job_id}/questions/{identifier}')
def get(game_id: str, job_id: str, identifier: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); expire(cur, time.time())
        row = find(cur, game_id, job_id, identifier)
        conn.commit()
        return public(row)
    finally:
        conn.close()


@router.post('/imports/{job_id}/questions/{identifier}/cancel')
def cancel(game_id: str, job_id: str, identifier: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); expire(cur, time.time())
        row = find(cur, game_id, job_id, identifier)
        cur.execute("UPDATE import_question_jobs SET status=CASE WHEN status='queued' THEN 'cancelled' ELSE 'cancelling' END WHERE id=? AND status IN ('queued','running')", (identifier,))
        conn.commit()
        return public(find(cur, game_id, job_id, identifier))
    finally:
        conn.close()


def claim_next():
    """Internal worker lease. A CAS prevents duplicate dispatch across consumers."""
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); expire(cur, time.time())
        cur.execute("SELECT * FROM import_question_jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1")
        row = cur.fetchone()
        if row is None:
            conn.commit()
            return None
        token = secrets.token_hex(32)
        cur.execute("UPDATE import_question_jobs SET status='running',token=? WHERE id=? AND status='queued' AND deadline>?", (token,row['id'],time.time()))
        claimed = cur.rowcount == 1
        conn.commit()
        return {**public(row), 'status': 'running', 'token': token} if claimed else None
    finally:
        conn.close()


def finish(identifier, token, *, result=None, error=''):
    if not token or (result is None and not error):
        raise ValueError('Supply a lease token and a result or error')
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); expire(cur, time.time())
        cur.execute("UPDATE import_question_jobs SET status=?,result=?,error=?,token='' WHERE id=? AND status='running' AND token=? AND deadline>?",
                    ('failed' if error else 'completed',json.dumps(result,ensure_ascii=False) if result is not None else None,error,identifier,token,time.time()))
        accepted = cur.rowcount == 1
        conn.commit()
        return accepted
    finally:
        conn.close()


def execution_status(identifier, token):
    """Private supervisor check; never disclose lease tokens through the API."""
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); expire(cur, time.time())
        cur.execute('SELECT status,token FROM import_question_jobs WHERE id=?', (identifier,))
        row = cur.fetchone()
        conn.commit()
        return row['status'] if row and secrets.compare_digest(row['token'], token) else 'revoked'
    finally:
        conn.close()


def acknowledge_cancel(identifier, token):
    """Call only after the supervisor has joined the terminated child process."""
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur)
        cur.execute("UPDATE import_question_jobs SET status='cancelled',token='',error='python_execution_stopped' WHERE id=? AND token=? AND status='cancelling'", (identifier,token))
        accepted = cur.rowcount == 1
        conn.commit()
        return accepted
    finally:
        conn.close()


def timeout_execution(identifier, token):
    """Supervisor's monotonic deadline remains authoritative if wall time moves."""
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur)
        cur.execute("UPDATE import_question_jobs SET status='timed_out',token='',error='deadline_exceeded' WHERE id=? AND token=? AND status IN ('running','cancelling')", (identifier,token))
        conn.commit()
    finally:
        conn.close()
