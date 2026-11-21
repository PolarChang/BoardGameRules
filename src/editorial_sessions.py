"""Shared, revocable administrative sessions; only token hashes are persisted."""
import hashlib
import hmac
import re
import secrets
import time
from src.database import _connect


def _digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _generation(key):
    return hmac.new(key.encode(), b'editorial-session-generation-v1', hashlib.sha256).hexdigest()


def _connection():
    conn = _connect()
    conn.cursor().execute('CREATE TABLE IF NOT EXISTS editorial_sessions (id TEXT PRIMARY KEY, generation TEXT NOT NULL, expires DOUBLE PRECISION NOT NULL)')
    conn.commit()
    return conn


def create(key):
    token = 'ksession.' + secrets.token_hex(32)
    conn = _connection()
    try:
        cur = conn.cursor()
        cur.execute('DELETE FROM editorial_sessions WHERE expires<=?', (time.time(),))
        cur.execute('INSERT INTO editorial_sessions (id,generation,expires) VALUES (?,?,?)', (_digest(token), _generation(key), time.time() + 7200))
        conn.commit()
        return token
    finally:
        conn.close()


def valid(token, key):
    if not key or not re.fullmatch(r'ksession\.[a-f0-9]{64}', token or ''):
        return False
    conn = _connection()
    try:
        cur = conn.cursor()
        cur.execute('SELECT generation,expires FROM editorial_sessions WHERE id=?', (_digest(token),))
        row = cur.fetchone()
        return bool(row and row['expires'] > time.time() and hmac.compare_digest(row['generation'], _generation(key)))
    finally:
        conn.close()


def revoke(token):
    conn = _connection()
    try:
        conn.cursor().execute('DELETE FROM editorial_sessions WHERE id=?', (_digest(token),))
        conn.commit()
    finally:
        conn.close()
