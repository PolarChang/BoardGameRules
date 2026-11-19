"""Conservative dispatch budgets for local import and repair inference."""
from datetime import datetime,timezone
import math
import sqlite3
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from src.database import _connect
from src.gemini_usage import setting
from src.local_usage import _document_context

_work_context=ContextVar('local_budget_work',default=None)


@contextmanager
def work_scope(game_id,family,work_id):
    token=_work_context.set((game_id,family,work_id))
    try:
        yield
    finally:
        _work_context.reset(token)


class BudgetExceeded(RuntimeError):
    def __init__(self,reason):
        self.reason=reason
        scope,metric,_=reason.split('_')
        self.user_message=f"已達{ {'day':'每日','document':'此文件','work':'此工作'}[scope]}{ {'calls':'呼叫次數','tokens':'Token','seconds':'推論時間'}[metric]}預算上限；來源仍可閱讀，不改用雲端模型。"
        super().__init__(reason)


FAMILIES = {
    'import': ('local_import_dispatches', (256,2_000_000,7200), (512,4_000_000,14400)),
    'qa': ('local_qa_dispatches', (100,500_000,3600), (300,1_500_000,10800)),
    'category': ('local_category_dispatches', (256,1_000_000,7200), (256,1_000_000,7200)),
    'cpu': ('local_cpu_dispatches', (1000,2_000_000,7200), (2000,4_000_000,14400)),
}


def reserve(game_id,operation,max_tokens,max_seconds,*,family='import',document_id=None,work_id=None):
    table,day_defaults,document_defaults=FAMILIES[family]
    work=_work_context.get()
    if work_id is None and work is not None and work[:2]==(game_id,family):
        work_id=work[2]
    scope=_document_context.get()
    document=document_id or (scope[1] if scope and scope[0]==game_id else 'unscoped')
    day=datetime.now(timezone.utc).date().isoformat()
    conn=_connect()
    try:
        if isinstance(conn,sqlite3.Connection): conn.execute('BEGIN IMMEDIATE')
        cur=conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS local_import_budget_lock (id INTEGER PRIMARY KEY,generation INTEGER NOT NULL)')
        cur.execute('INSERT INTO local_import_budget_lock VALUES (1,1) ON CONFLICT (id) DO UPDATE SET generation=local_import_budget_lock.generation+1')
        cur.execute(f'CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY,game_id TEXT NOT NULL,document_id TEXT NOT NULL,day TEXT NOT NULL,operation TEXT NOT NULL,tokens INTEGER NOT NULL,seconds DOUBLE PRECISION NOT NULL,state TEXT NOT NULL)')
        cur.execute('CREATE TABLE IF NOT EXISTS local_dispatch_work (dispatch_id TEXT PRIMARY KEY,family TEXT NOT NULL,work_id TEXT NOT NULL)')
        limits = [
            ('DAY','day=?',(day,),day_defaults),
            ('DOCUMENT','game_id=? AND document_id=?',(game_id,document),document_defaults)]
        if work_id is not None:
            limits.append(('WORK','game_id=? AND document_id=? AND id IN (SELECT dispatch_id FROM local_dispatch_work WHERE family=? AND work_id=?)',
                           (game_id,document,family,work_id),(8,100_000,1920)))
        for label,clause,args,defaults in limits:
            cur.execute(f'SELECT COUNT(*) AS calls,COALESCE(SUM(tokens),0) AS tokens,COALESCE(SUM(seconds),0) AS seconds FROM {table} WHERE '+clause,args)
            used=cur.fetchone()
            for metric,increment,default in zip(('calls','tokens','seconds'),(1,max_tokens,max_seconds),defaults):
                cap=max(0,int(setting(f'LOCAL_{family.upper()}_{label}_{metric.upper()}',str(default))))
                if used[metric]+increment>cap:
                    raise BudgetExceeded(f'{label.lower()}_{metric}_limit')
        identifier=uuid.uuid4().hex
        cur.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?,'reserved')",(identifier,game_id,document,day,operation,max_tokens,max_seconds))
        if work_id is not None:
            cur.execute('INSERT INTO local_dispatch_work VALUES (?,?,?)',(identifier,family,work_id))
        conn.commit()
        return identifier
    finally: conn.close()


def settle(identifier,prompt_tokens,output_tokens,seconds,*,family='import'):
    table=FAMILIES[family][0]
    # An interrupted or incomplete report retains the original reservation.
    if any(type(v) is not int or v<0 for v in (prompt_tokens,output_tokens)): return
    if not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0: return
    conn=_connect()
    try:
        conn.cursor().execute(f"UPDATE {table} SET tokens=?,seconds=?,state='measured' WHERE id=? AND state='reserved'",
                              (prompt_tokens+output_tokens,seconds,identifier))
        conn.commit()
    finally: conn.close()
