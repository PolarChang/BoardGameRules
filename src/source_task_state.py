"""Durable CPU extraction leases and retry timing, independent of model quota."""
import hashlib
import time
import uuid
from src.database import _connect
from src.import_native import ENGINE


def identity(job):
    return hashlib.sha256(f"{job['id']}:{job['digest']}:{ENGINE}".encode()).hexdigest()


def init(cur):
    cur.execute('CREATE TABLE IF NOT EXISTS import_source_tasks (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, state TEXT NOT NULL, lease TEXT NOT NULL, lease_until DOUBLE PRECISION NOT NULL, attempts INTEGER NOT NULL, retry_at DOUBLE PRECISION NOT NULL, error TEXT NOT NULL)')
    cur.execute('CREATE TABLE IF NOT EXISTS import_source_retry_counts (id TEXT PRIMARY KEY, failures INTEGER NOT NULL)')


def claim(job,now=None):
    now=time.time() if now is None else now
    conn=_connect()
    try:
        cur=conn.cursor(); init(cur)
        key=identity(job); token=uuid.uuid4().hex
        cur.execute("INSERT INTO import_source_tasks VALUES (?,?,'queued','',0,0,0,'') ON CONFLICT DO NOTHING",(key,job['id']))
        cur.execute("UPDATE import_source_tasks SET state='extracting',lease=?,lease_until=?,attempts=attempts+1 WHERE id=? AND state NOT IN ('failed','done') AND retry_at<=? AND lease_until<=?",(token,now+300,key,now,now))
        acquired=cur.rowcount==1
        conn.commit()
        return token if acquired else None
    finally: conn.close()


def finish(job,token,*,complete=False,error=None,now=None):
    now=time.time() if now is None else now
    conn=_connect()
    try:
        cur=conn.cursor(); init(cur); key=identity(job)
        cur.execute('SELECT attempts FROM import_source_tasks WHERE id=? AND lease=? AND lease_until>?',(key,token,now))
        row=cur.fetchone()
        if row is None: return False
        name=type(error).__name__ if error is not None else ''
        fatal=name in {'FileDataError','EmptyFileError'}
        state='failed' if fatal else 'waiting_retry' if error is not None else 'done' if complete else 'queued'
        cur.execute('SELECT failures FROM import_source_retry_counts WHERE id=?',(key,))
        previous=cur.fetchone()
        failures=(previous['failures'] if previous else 0)+1 if error is not None else 0
        retry_at=now+min(3600,60*2**min(failures-1,6)) if error is not None and not fatal else 0
        cur.execute("UPDATE import_source_tasks SET state=?,lease='',lease_until=0,retry_at=?,error=? WHERE id=? AND lease=? AND lease_until>?",(state,retry_at,name,key,token,now))
        changed=cur.rowcount==1
        if changed:
            cur.execute('INSERT INTO import_source_retry_counts VALUES (?,?) ON CONFLICT (id) DO UPDATE SET failures=excluded.failures',(key,failures))
            cur.execute('CREATE TABLE IF NOT EXISTS import_source_errors (job_id TEXT PRIMARY KEY, error TEXT NOT NULL)')
            cur.execute('CREATE TABLE IF NOT EXISTS import_source_materialized (id TEXT PRIMARY KEY)')
            if error is not None:
                cur.execute('INSERT INTO import_source_errors VALUES (?,?) ON CONFLICT (job_id) DO UPDATE SET error=excluded.error',(job['id'],name))
            else:
                cur.execute('DELETE FROM import_source_errors WHERE job_id=?',(job['id'],))
                if complete:
                    cur.execute('INSERT INTO import_source_materialized VALUES (?) ON CONFLICT DO NOTHING',(key,))
        conn.commit()
        return changed
    finally: conn.close()
