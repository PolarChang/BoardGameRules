"""Cross-process scheduling for one local GPU; QA precedes pending imports."""
from contextlib import contextmanager
import sqlite3
import time
import uuid
from src.config import DB_DIR

QUEUE_PATH = DB_DIR / 'local-inference-queue.db'


class LocalInferenceBusy(TimeoutError):
    """No inference was dispatched because the local GPU remained occupied."""
    pass


@contextmanager
def connect():
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(QUEUE_PATH), timeout=10)
    conn.execute('CREATE TABLE IF NOT EXISTS tickets (id TEXT PRIMARY KEY, priority INTEGER, created REAL, expires REAL, active INTEGER)')
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def acquire(identifier, priority, expires):
    with connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        now = time.time()
        conn.execute('DELETE FROM tickets WHERE expires<?', (now,))
        conn.execute('INSERT OR IGNORE INTO tickets VALUES (?,?,?,?,0)', (identifier, priority, now, expires))
        if conn.execute('SELECT 1 FROM tickets WHERE active=1').fetchone():
            return False
        first = conn.execute('SELECT id FROM tickets ORDER BY priority,created,id LIMIT 1').fetchone()
        if first and first[0] == identifier:
            conn.execute('UPDATE tickets SET active=1,expires=? WHERE id=?', (now + 300, identifier))
            return True
        return False


def release(identifier):
    with connect() as conn:
        conn.execute('DELETE FROM tickets WHERE id=?', (identifier,))


@contextmanager
def inference_slot(kind, wait_seconds=None):
    if kind not in {'qa', 'import'}:
        raise ValueError('Invalid inference priority')
    identifier = uuid.uuid4().hex
    timeout = wait_seconds if wait_seconds is not None else (5 if kind == 'qa' else 600)
    deadline = time.monotonic() + timeout
    expires = time.time() + timeout + 1
    try:
        while not acquire(identifier, 0 if kind == 'qa' else 1, expires):
            if time.monotonic() >= deadline:
                raise LocalInferenceBusy('Local GPU busy; retry later')
            time.sleep(.1)
        yield
    finally:
        release(identifier)
