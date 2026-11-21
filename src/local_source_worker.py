"""CPU-only source extraction, discovered from durable upload rows on restart."""
import hashlib
import logging
from src.database import _connect
from src.import_native import ENGINE


def run_batch():
    from src.import_jobs import init_jobs, build_draft
    init_jobs()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS import_native_sources (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        cur.execute('CREATE TABLE IF NOT EXISTS import_source_errors (job_id TEXT PRIMARY KEY, error TEXT NOT NULL)')
        cur.execute('CREATE TABLE IF NOT EXISTS import_source_materialized (id TEXT PRIMARY KEY)')
        cur.execute('SELECT * FROM editorial_import_jobs ORDER BY created')
        jobs = [dict(row) for row in cur.fetchall()]
        pending = []
        for job in jobs:
            key = hashlib.sha256(f"{job['id']}:{job['digest']}:{ENGINE}".encode()).hexdigest()
            cur.execute('SELECT id FROM import_native_sources WHERE id=?', (key,))
            finished = cur.fetchone()
            cur.execute('SELECT id FROM import_source_materialized WHERE id=?',(key,))
            materialized = cur.fetchone()
            if not finished or not materialized:
                pending.append(job)
        conn.commit()
    finally:
        conn.close()
    for job in pending:
        from src.source_task_state import claim, finish
        token=claim(job)
        if token is None: continue
        try:
            build_draft(job['game_id'],job['id'],native_limit=1)
            conn = _connect()
            try:
                cur = conn.cursor()
                key = hashlib.sha256(f"{job['id']}:{job['digest']}:{ENGINE}".encode()).hexdigest()
                cur.execute('SELECT id FROM import_native_sources WHERE id=?',(key,))
                complete=bool(cur.fetchone())
                conn.commit()
            finally:
                conn.close()
            finish(job,token,complete=complete)
        except Exception as error:
            if not finish(job,token,error=error): continue
            logging.getLogger(__name__).warning('Native source extraction failed for job %s: %s',job['id'],type(error).__name__)
    return len(pending)
