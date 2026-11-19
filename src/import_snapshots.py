"""Immutable import snapshots and a source-preserving literal search index."""
import json
from fastapi import HTTPException
from src.database import _connect


def init(cur):
    cur.execute('CREATE TABLE IF NOT EXISTS import_snapshots (game_id TEXT NOT NULL, job_id TEXT NOT NULL, version TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(game_id,job_id,version))')
    cur.execute('CREATE TABLE IF NOT EXISTS import_source_index (game_id TEXT NOT NULL, job_id TEXT NOT NULL, version TEXT NOT NULL, block_id TEXT NOT NULL, text TEXT NOT NULL, PRIMARY KEY(game_id,job_id,version,block_id))')


def save(game_id, job_id, snapshot):
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur)
        version = snapshot['version']
        cur.execute('SELECT version FROM import_snapshots WHERE game_id=? AND job_id=? AND version=?', (game_id,job_id,version))
        if cur.fetchone():
            conn.commit()
            return
        for block in snapshot['document']['blocks']:
            if block['kind'] != '規則' or block.get('evidence_usable') is False:
                continue
            fields = block['fields']
            text = '\n'.join(fields.get(k,'') for k in ('標題','原文','內容'))
            cur.execute('INSERT INTO import_source_index VALUES (?,?,?,?,?) ON CONFLICT DO NOTHING', (game_id,job_id,version,block['id'],text))
        cur.execute('INSERT INTO import_snapshots VALUES (?,?,?,?) ON CONFLICT DO NOTHING', (game_id,job_id,version,json.dumps(snapshot,ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()


def load(game_id, job_id, version):
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur)
        cur.execute('SELECT payload FROM import_snapshots WHERE game_id=? AND job_id=? AND version=?', (game_id,job_id,version))
        row = cur.fetchone(); conn.commit()
        if not row:
            raise HTTPException(409, '找不到指定匯入版本，請重新載入')
        return json.loads(row['payload'])
    finally:
        conn.close()


def search(game_id, job_id, version, query, offset=0, limit=50):
    # Literal substring search works for CJK and preserves numbers; SQL wildcard
    # characters are escaped. Semantic ranking is a separate M3 concern.
    value = query.strip().replace('\\','\\\\').replace('%','\\%').replace('_','\\_')
    conn = _connect()
    try:
        cur=conn.cursor(); init(cur)
        args=(game_id,job_id,version,'%'+value+'%')
        clause="game_id=? AND job_id=? AND version=? AND LOWER(text) LIKE LOWER(?) ESCAPE '\\'"
        cur.execute('SELECT COUNT(*) AS n FROM import_source_index WHERE '+clause,args)
        total=cur.fetchone()['n']
        cur.execute('SELECT block_id FROM import_source_index WHERE '+clause+' ORDER BY block_id LIMIT ? OFFSET ?',(*args,limit,offset))
        ids=[row['block_id'] for row in cur.fetchall()]; conn.commit()
        return {'ids':ids,'total':total,'offset':offset,'limit':limit}
    finally:
        conn.close()
