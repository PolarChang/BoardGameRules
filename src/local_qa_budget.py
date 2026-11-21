"""Durable reservations: reclaim unused calls without refunding dispatched work."""
from datetime import datetime, timezone
import sqlite3
import time
import uuid
from fastapi import HTTPException
from src.database import _connect
from src.gemini_usage import setting


def _begin(conn):
    if isinstance(conn,sqlite3.Connection): conn.execute('BEGIN IMMEDIATE')
    cur=conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS editorial_usage (day TEXT PRIMARY KEY,calls INTEGER NOT NULL)')
    cur.execute('CREATE TABLE IF NOT EXISTS local_qa_reservations (id TEXT PRIMARY KEY,day TEXT NOT NULL,used INTEGER NOT NULL,expires DOUBLE PRECISION NOT NULL,state TEXT NOT NULL)')
    cur.execute('CREATE TABLE IF NOT EXISTS local_qa_execution_reservations (reservation_id TEXT PRIMARY KEY,game_id TEXT NOT NULL,execution_id TEXT NOT NULL)')
    return cur


def _refund(cur,row):
    unused=3-row['used']
    cur.execute('UPDATE editorial_usage SET calls=calls-? WHERE day=? AND calls>=?',(unused,row['day'],unused))


def reserve(now=None):
    now=time.time() if now is None else now
    cap=max(0,int(setting('LOCAL_QA_DAILY_LIMIT','100')))
    lifetime=max(60,min(3600,int(setting('LOCAL_QA_RESERVATION_SECONDS','900'))))
    day='local:'+datetime.fromtimestamp(now,timezone.utc).date().isoformat()
    conn=_connect()
    try:
        cur=_begin(conn)
        cur.execute("UPDATE local_qa_reservations SET state='expired' WHERE state='active' AND expires<=? RETURNING day,used",(now,))
        expired=cur.fetchall()
        for row in expired: _refund(cur,row)
        cur.execute('INSERT INTO editorial_usage VALUES (?,0) ON CONFLICT DO NOTHING',(day,))
        cur.execute('UPDATE editorial_usage SET calls=calls+3 WHERE day=? AND calls+3<=?',(day,cap))
        if cur.rowcount!=1:
            conn.commit()
            raise HTTPException(429,'今日本機問答預算不足；來源閱讀仍可使用。')
        identifier=uuid.uuid4().hex
        cur.execute("INSERT INTO local_qa_reservations VALUES (?,?,0,?,'active')",(identifier,day,now+lifetime))
        from src.local_usage import _execution_context
        execution = _execution_context.get()
        if execution:
            cur.execute('INSERT INTO local_qa_execution_reservations VALUES (?,?,?)', (identifier,*execution))
        conn.commit()
        return identifier
    finally: conn.close()


def consume(identifier,now=None):
    now=time.time() if now is None else now
    conn=_connect()
    try:
        cur=_begin(conn)
        cur.execute("UPDATE local_qa_reservations SET used=used+1 WHERE id=? AND state='active' AND expires>? AND used<3",(identifier,now))
        accepted=cur.rowcount==1
        conn.commit()
        if not accepted:
            raise HTTPException(409,'本次問答預算租約已到期或用完，請重新提問；來源仍可閱讀。')
    finally: conn.close()


def finish(identifier):
    conn=_connect()
    try:
        cur=_begin(conn)
        cur.execute("UPDATE local_qa_reservations SET state='finished' WHERE id=? AND state='active' RETURNING day,used",(identifier,))
        row=cur.fetchone()
        if row is not None: _refund(cur,row)
        conn.commit()
    finally: conn.close()


def finish_execution(game_id, execution_id):
    """Refund only unused slots after the associated process has exited."""
    conn = _connect()
    try:
        cur = _begin(conn)
        cur.execute("UPDATE local_qa_reservations SET state='finished' WHERE state='active' AND id IN (SELECT reservation_id FROM local_qa_execution_reservations WHERE game_id=? AND execution_id=?) RETURNING day,used", (game_id,execution_id))
        for row in cur.fetchall():
            _refund(cur, row)
        conn.commit()
    finally:
        conn.close()
