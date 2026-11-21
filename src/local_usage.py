"""Local inference accounting, separate from cloud billing and quota."""
import json
import hashlib
from datetime import datetime, timezone
import math
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from src.database import _connect

_document_context=ContextVar('local_usage_document',default=None)
_execution_context=ContextVar('local_usage_execution',default=None)


@contextmanager
def execution_scope(game_id, identifier):
    token = _execution_context.set((game_id, identifier))
    try:
        yield
    finally:
        _execution_context.reset(token)


def document_work_id(game_id,input_digest):
    """Identify repeated inputs only when their document/version is known."""
    scope=_document_context.get()
    if scope is None or scope[0]!=game_id:
        return None
    return hashlib.sha256(json.dumps([scope,input_digest]).encode()).hexdigest()


@contextmanager
def output_validation(identifier,game_id,model,operation):
    """Preserve measured usage when the returned output cannot be consumed."""
    try:
        yield
    except Exception:
        record(identifier,game_id,model,operation,outcome='failed')
        raise


@contextmanager
def document_scope(game_id,document_id,version):
    token=_document_context.set((game_id,document_id,version))
    try:
        yield
    finally:
        _document_context.reset(token)


def cache_hit(game_id,model,operation,items=1):
    """Record reuse separately; cache items are not inference calls or tokens."""
    if type(items) is not int or items<1: return
    scope=_document_context.get()
    document_id=scope[1] if scope is not None and scope[0]==game_id else None
    conn=_connect()
    try:
        cur=conn.cursor(); _init_cache_events(cur)
        cur.execute('INSERT INTO local_cache_events VALUES (?,?,?,?,?,?,?)',
                    (uuid.uuid4().hex,game_id,document_id,model,operation,items,datetime.now(timezone.utc).isoformat()))
        conn.commit()
    finally: conn.close()


def _init_cache_events(cur):
    cur.execute('CREATE TABLE IF NOT EXISTS local_cache_events (id TEXT PRIMARY KEY,game_id TEXT NOT NULL,document_id TEXT,model TEXT NOT NULL,operation TEXT NOT NULL,items INTEGER NOT NULL,created_at TEXT NOT NULL)')


def _request_attempt(cur,identifier,game_id,operation,work_id):
    cur.execute('CREATE TABLE IF NOT EXISTS local_work_attempts (game_id TEXT NOT NULL,operation TEXT NOT NULL,work_id TEXT NOT NULL,attempts INTEGER NOT NULL,PRIMARY KEY(game_id,operation,work_id))')
    cur.execute('CREATE TABLE IF NOT EXISTS local_attempt_events (id TEXT PRIMARY KEY,game_id TEXT NOT NULL,operation TEXT NOT NULL,work_id TEXT NOT NULL,attempt INTEGER NOT NULL)')
    cur.execute('INSERT INTO local_attempt_events VALUES (?,?,?,?,0) ON CONFLICT DO NOTHING RETURNING id',(identifier,game_id,operation,work_id))
    inserted=cur.fetchone() is not None
    if inserted:
        cur.execute('INSERT INTO local_work_attempts VALUES (?,?,?,1) ON CONFLICT (game_id,operation,work_id) DO UPDATE SET attempts=local_work_attempts.attempts+1 RETURNING attempts',(game_id,operation,work_id))
        value=cur.fetchone()['attempts']
        cur.execute('UPDATE local_attempt_events SET attempt=? WHERE id=?',(value,identifier))
        return value
    cur.execute('SELECT attempt FROM local_attempt_events WHERE id=? AND game_id=? AND operation=? AND work_id=?',(identifier,game_id,operation,work_id))
    row=cur.fetchone()
    return row['attempt'] if row else None


def record(identifier, game_id, model, operation, result=None, *, document_id=None, outcome=None, attempt=None, work_id=None):
    if result is not None and all(result.get(k) is None for k in ('prompt_tokens', 'output_tokens')):
        result = None
    metadata={k:v for k,v in {'document_id':document_id,'outcome':outcome,'attempt':attempt}.items() if v is not None}
    execution = _execution_context.get()
    if execution and execution[0] == game_id:
        metadata['execution_id'] = execution[1]
    scope=_document_context.get()
    if scope is not None and scope[0]==game_id:
        metadata.setdefault('document_id',scope[1])
        metadata['source_version']=scope[2]
    completed_at = datetime.now(timezone.utc)
    conn = _connect()
    try:
        cur = conn.cursor()
        if work_id is not None:
            sequence=_request_attempt(cur,identifier,game_id,operation,work_id)
            if sequence is not None:
                metadata.update(work_id=work_id,attempt=sequence)
        cur.execute('CREATE TABLE IF NOT EXISTS local_model_requests (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, model TEXT NOT NULL, operation TEXT NOT NULL, created_at TEXT NOT NULL, usage_json TEXT)')
        # Keep the first timestamp: it is the start of the request.  A
        # completion update is measured from that timestamp, and its measured
        # duration is persisted so repeated updates remain idempotent.
        cur.execute('SELECT created_at,usage_json FROM local_model_requests WHERE id=? AND game_id=?', (identifier, game_id))
        existing = cur.fetchone()
        if outcome is None:
            if result is not None: metadata['outcome']='received'
            elif existing is None: metadata['outcome']='started'
        stored_result = result
        data = _decode_usage(existing['usage_json']) if existing is not None else {}
        if existing is not None and (result is not None or outcome=='failed'):
            if _known_duration(data.get('duration_seconds')):
                stored_result={**(result or {}),'duration_seconds':data['duration_seconds']}
            elif existing['usage_json'] is None or data.get('outcome') == 'started':
                started_at = _parse_timestamp(existing['created_at'])
                if started_at is not None:
                    elapsed = (completed_at - started_at).total_seconds()
                    if elapsed >= 0 and math.isfinite(elapsed):
                        stored_result = dict(result or {})
                        stored_result['duration_seconds'] = elapsed
        if data or metadata:
            stored_result={**data,**(stored_result or {}),**metadata}
        cur.execute('INSERT INTO local_model_requests VALUES (?,?,?,?,?,?) ON CONFLICT (id) DO UPDATE SET usage_json=COALESCE(excluded.usage_json,local_model_requests.usage_json) WHERE local_model_requests.game_id=excluded.game_id',
                    (identifier, game_id, model, operation, completed_at.isoformat(), json.dumps(stored_result) if stored_result is not None else None))
        conn.commit()
    finally:
        conn.close()


def interrupt_execution(game_id, execution_id):
    """After confirmed child exit, preserve unknown tokens and model duration."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS local_model_requests (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, model TEXT NOT NULL, operation TEXT NOT NULL, created_at TEXT NOT NULL, usage_json TEXT)')
        cur.execute('SELECT id,usage_json FROM local_model_requests WHERE game_id=?', (game_id,))
        changed = 0
        for row in cur.fetchall():
            data = _decode_usage(row['usage_json'])
            if data.get('execution_id') != execution_id or data.get('outcome') != 'started':
                continue
            updated = {**data, 'outcome':'interrupted', 'interrupted_at':datetime.now(timezone.utc).isoformat()}
            # CAS protects a measured completion if a separate writer raced us.
            cur.execute('UPDATE local_model_requests SET usage_json=? WHERE id=? AND game_id=? AND usage_json=?',
                        (json.dumps(updated),row['id'],game_id,row['usage_json']))
            changed += cur.rowcount
        conn.commit()
        return changed
    finally:
        conn.close()


def summary(game_id):
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS local_model_requests (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, model TEXT NOT NULL, operation TEXT NOT NULL, created_at TEXT NOT NULL, usage_json TEXT)')
        conn.commit()
        cur.execute('SELECT model,operation,usage_json,created_at FROM local_model_requests WHERE game_id=?', (game_id,))
        rows = cur.fetchall()
        _init_cache_events(cur)
        cur.execute('SELECT document_id,model,operation,COUNT(*) AS events,SUM(items) AS items FROM local_cache_events WHERE game_id=? GROUP BY document_id,model,operation',(game_id,))
        caches=[dict(row) for row in cur.fetchall()]
        conn.commit()
    finally:
        conn.close()
    models = {}
    operations = {}
    documents = {}
    for row in rows:
        data = _decode_usage(row['usage_json'])
        group = models.setdefault(row['model'], _new_group('model', row['model']))
        operation = operations.setdefault((row['operation'],row['model']), {**_new_group('operation', row['operation']),'model':row['model']})
        document_id=data.get('document_id')
        document=documents.setdefault((document_id,row['operation'],row['model']), {**_new_group('document_id',document_id),'operation':row['operation'],'model':row['model'],'failed_requests':0,'retry_requests':0,'unknown_outcome_requests':0,'unknown_attempt_requests':0,'interrupted_requests':0})
        document['interrupted_requests'] += int(data.get('outcome') == 'interrupted')
        document['failed_requests']+=int(data.get('outcome')=='failed')
        document['unknown_outcome_requests']+=int(data.get('outcome') not in {'received','failed'})
        attempt=data.get('attempt')
        document['unknown_attempt_requests']+=int(type(attempt) is not int or attempt<1)
        document['retry_requests']+=int(type(attempt) is int and attempt>1)
        values = [data.get('prompt_tokens'), data.get('output_tokens')]
        if any(type(v) is not int or v < 0 for v in values):
            group['unknown_requests'] += 1
            operation['unknown_requests'] += 1
            document['unknown_requests'] += 1
        for key, value in zip(('input_tokens', 'output_tokens'), values):
            if type(value) is int and value >= 0:
                group[key] += value
                operation[key] += value
                document[key] += value
        group['requests'] += 1
        operation['requests'] += 1
        document['requests'] += 1
        duration = data.get('duration_seconds')
        if _known_duration(duration):
            group['duration_seconds'] += duration
            operation['duration_seconds'] += duration
            document['duration_seconds'] += duration
        else:
            group['unknown_duration_requests'] += 1
            operation['unknown_duration_requests'] += 1
            document['unknown_duration_requests'] += 1
    return {'models': list(models.values()), 'operations': list(operations.values()), 'documents':list(documents.values()), 'cache_hits':caches, 'cloud_api_cost': 0}


def _decode_usage(value):
    if not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _parse_timestamp(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _known_duration(value):
    return type(value) in (int, float) and value >= 0 and math.isfinite(value)


def _new_group(label, value):
    return {label: value, 'requests': 0, 'input_tokens': 0, 'output_tokens': 0,
            'unknown_requests': 0, 'duration_seconds': 0.0,
            'unknown_duration_requests': 0}
