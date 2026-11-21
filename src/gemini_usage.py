"""Durable per-attempt Gemini usage; stores metadata only, never source content."""
import json
import os
import re
import uuid
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from dotenv import dotenv_values
from src.database import _connect

FIELDS = {'input_tokens': 'promptTokenCount', 'output_tokens': 'candidatesTokenCount',
          'thinking_tokens': 'thoughtsTokenCount', 'cached_tokens': 'cachedContentTokenCount',
          'total_tokens': 'totalTokenCount'}


def setting(name, default=''):
    if name in os.environ:
        return os.environ[name]
    from src.config import LOCAL_ROOT
    from src.local_runtime_settings import DEFAULTS, read
    restored = LOCAL_ROOT / 'runtime-settings.json'
    if os.environ.get('KNOWLEDGE_LOCAL_ROOT') and name in DEFAULTS and restored.is_file():
        return read(restored)[name]
    if os.environ.get('KNOWLEDGE_DISABLE_LOCAL_ENV') == 'true':
        return default
    return dotenv_values(Path(__file__).resolve().parent.parent / '.env').get(name, default) or default


def init_usage():
    conn = _connect()
    try:
        conn.cursor().execute('CREATE TABLE IF NOT EXISTS gemini_requests (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, model TEXT NOT NULL, operation TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, usage_json TEXT)')
        conn.commit()
    finally:
        conn.close()


def generate(model, payload, *, game_id='', operation='pdf', timeout=180):
    raise RuntimeError('Gemini has been removed from Boardgame; cloud inference is disabled.')


def usage_summary(game_id):
    init_usage()
    conn = _connect()
    try:
        rows = conn.cursor().execute('SELECT * FROM gemini_requests WHERE game_id=? ORDER BY created_at DESC', (game_id,)).fetchall()
    finally:
        conn.close()
    totals = {key: 0 for key in FIELDS}
    unknown = {key: 0 for key in FIELDS}
    recent = []
    for row in rows:
        usage = json.loads(row['usage_json']) if row['usage_json'] else {}
        for key in FIELDS:
            if usage.get(key) is None:
                unknown[key] += 1
            else:
                totals[key] += usage[key]
        if len(recent) < 20:
            recent.append({key: row[key] for key in ('id', 'model', 'operation', 'created_at', 'status')} | {'usage': usage})
    return {'configured': False, 'disabled': True, 'model': setting('RULE_EXTRACTION_MODEL', 'gemini-3.8-flash'),
            'free_tier_confirmed': False,
            'requests': len(rows), 'totals': totals, 'unknown': unknown, 'recent': recent}
