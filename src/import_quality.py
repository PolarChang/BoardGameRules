"""Deterministic import checks and version-bound feedback. No game-specific facts."""
import hashlib
import json
import re
import time
import uuid
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from src.database import _connect

router = APIRouter()


def findings(document):
    result = []
    for block in document['blocks']:
        if block['kind'] != '規則':
            continue
        fields = block['fields']
        source, content = fields.get('原文', ''), fields.get('內容', '')
        issues = []
        if re.search(r'(?:^\s*\d+(?:\.\d+)?\s*\n){4,}', source, re.M):
            issues.append(('table_layout', '連續獨立數字，表格欄列可能遺失', 'vision'))
        if re.search(r'\d+\s{3,}[/／]', source):
            issues.append(('missing_symbols', '數量之間出現空白，可能遺失資源圖示', 'vision'))
        if '圖片區域' in source or fields.get('標題', '').startswith('待辨識區塊'):
            issues.append(('unread_source', '尚無可靠來源轉寫', 'vision'))
        numbers = lambda text: set(re.findall(r'(?<![\w])\d+(?:\.\d+)?', text))
        if source and content and '圖片區域' not in source and numbers(content) - numbers(source):
            issues.append(('number_mismatch', '譯文出現來源文字沒有的數字，需回查原圖', 'text'))
        for code, message, route in issues:
            result.append({'id': block['id'], 'page': fields.get('PDF頁碼', ''), 'code': code,
                           'message': message, 'route': route,
                           'source_hash': hashlib.sha256(source.encode()).hexdigest()})
    return result


def init(cur):
    cur.execute('CREATE TABLE IF NOT EXISTS import_answer_records (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, job_id TEXT NOT NULL, version TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL)')
    cur.execute('CREATE TABLE IF NOT EXISTS import_feedback (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, job_id TEXT NOT NULL, answer_id TEXT NOT NULL, kind TEXT NOT NULL, comment TEXT NOT NULL, created REAL NOT NULL)')
    cur.execute('CREATE TABLE IF NOT EXISTS import_repair_candidates (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, job_id TEXT NOT NULL, version TEXT NOT NULL, block_id TEXT NOT NULL, payload TEXT NOT NULL, state TEXT NOT NULL, result TEXT NOT NULL, created REAL NOT NULL)')


def save_answer(game_id, job_id, version, question, answer, document):
    identifier = uuid.uuid4().hex
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur)
        selected = set(answer.get('retrieved_ids', [])) | set(answer.get('citations', []))
        payload = {'question': question, 'answer': answer, 'blocks': [b for b in document['blocks'] if b['id'] in selected]}
        cur.execute('INSERT INTO import_answer_records VALUES (?,?,?,?,?,?)',
                    (identifier, game_id, job_id, version, json.dumps(payload, ensure_ascii=False), time.time()))
        conn.commit()
    finally:
        conn.close()
    return identifier


class Feedback(BaseModel):
    model_config = ConfigDict(extra='forbid')
    answer_id: str = Field(pattern=r'^[a-f0-9]{32}$')
    kind: Literal['missing_condition', 'wrong_number', 'wrong_source', 'wrong_answer', 'unclear', 'other']
    comment: str = Field(default='', max_length=2000)


@router.get('/imports/{job_id}/answers')
def answer_history(game_id: str, job_id: str, offset: int = 0, x_knowledge_admin: str = Header(default='')):
    from src.import_jobs import authorize, get_job
    authorize(game_id, x_knowledge_admin)
    if offset < 0:
        raise HTTPException(422, '分頁參數無效')
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); get_job(cur, game_id, job_id)
        cur.execute('SELECT id,version,payload,created FROM import_answer_records WHERE game_id=? AND job_id=? ORDER BY created DESC,id DESC LIMIT 50 OFFSET ?', (game_id, job_id, offset))
        items = []
        for row in cur.fetchall():
            record = json.loads(row['payload'])
            answer = record['answer']
            items.append({'id': row['id'], 'status': 'completed', 'created_at': row['created'],
                          'question': {'question': record['question'], 'version': row['version'],
                                       'settings_version': answer.get('settings_version', ''),
                                       'confirmed_context': answer.get('confirmed_context', {})},
                          'result': {**answer, 'answer_id': row['id']}, 'error': ''})
        conn.commit()
        return {'jobs': items}
    finally:
        conn.close()


@router.get('/imports/{job_id}/quality')
def quality(game_id: str, job_id: str, x_knowledge_admin: str = Header(default='')):
    from src.import_jobs import draft
    snapshot = draft(game_id, job_id, x_knowledge_admin)
    issues = findings(snapshot['document'])
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur)
        cur.execute('SELECT id,version,block_id,state,result FROM import_repair_candidates WHERE game_id=? AND job_id=? ORDER BY created DESC LIMIT 50', (game_id, job_id))
        candidates = [dict(row) for row in cur.fetchall()]
        conn.commit()
    finally:
        conn.close()
    return {'version': snapshot['version'], 'findings': issues, 'candidates': candidates,
            'status': 'needs_repair' if issues else 'no_detected_issue',
            'notice': '自動檢查未發現問題不代表語意完全正確；候選修訂不直接覆寫來源。'}


@router.post('/imports/{job_id}/feedback')
def feedback(game_id: str, job_id: str, body: Feedback, x_knowledge_admin: str = Header(default='')):
    from src.import_jobs import authorize, get_job
    authorize(game_id, x_knowledge_admin)
    identifier = hashlib.sha256(json.dumps([game_id, job_id, body.model_dump()], sort_keys=True).encode()).hexdigest()
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); get_job(cur, game_id, job_id)
        cur.execute('SELECT * FROM import_answer_records WHERE id=? AND game_id=? AND job_id=?', (body.answer_id, game_id, job_id))
        answer = cur.fetchone()
        if not answer:
            raise HTTPException(404, '找不到此版本的回答紀錄')
        cur.execute('SELECT id FROM import_feedback WHERE id=?', (identifier,))
        if cur.fetchone():
            return {'id': identifier, 'state': 'recorded', 'message': '此回饋已保存，沒有重複建立。'}
        cur.execute('SELECT COUNT(*) AS n FROM import_feedback WHERE game_id=? AND created>?', (game_id, time.time() - 86400))
        if cur.fetchone()['n'] >= 50:
            raise HTTPException(429, '今日回饋上限已達，請稍後再試')
        cur.execute('INSERT INTO import_feedback VALUES (?,?,?,?,?,?,?) ON CONFLICT (id) DO NOTHING',
                    (identifier, game_id, job_id, body.answer_id, body.kind, body.comment.strip(), time.time()))
        conn.commit()
    finally:
        conn.close()
    return {'id': identifier, 'state': 'recorded', 'message': '已保存問題、回答版本與來源供後續查證；不會自動執行模型修復。'}


@router.get('/imports/{job_id}/feedback')
def feedback_history(game_id: str, job_id: str, offset: int = 0, x_knowledge_admin: str = Header(default='')):
    from src.import_jobs import authorize, get_job
    authorize(game_id, x_knowledge_admin)
    if offset < 0:
        raise HTTPException(422, '分頁參數無效')
    conn = _connect()
    try:
        cur = conn.cursor(); init(cur); get_job(cur, game_id, job_id)
        cur.execute('SELECT f.id,f.answer_id,f.kind,f.comment,f.created,a.version,a.payload FROM import_feedback f JOIN import_answer_records a ON a.id=f.answer_id AND a.game_id=f.game_id AND a.job_id=f.job_id WHERE f.game_id=? AND f.job_id=? ORDER BY f.created DESC,f.id DESC LIMIT 21 OFFSET ?', (game_id, job_id, offset))
        rows = cur.fetchall()
        items = [{**{k: row[k] for k in ('id', 'answer_id', 'kind', 'comment', 'created', 'version')},
                  'record': json.loads(row['payload'])} for row in rows[:20]]
        conn.commit()
        return {'items': items, 'next_offset': offset + 20 if len(rows) > 20 else None}
    finally:
        conn.close()


class RepairRequest(BaseModel):
    version: str = Field(pattern=r'^[a-f0-9]{64}$')


@router.post('/imports/{job_id}/repair')
def queue_repairs(game_id: str, job_id: str, body: RepairRequest, x_knowledge_admin: str = Header(default='')):
    from src.import_jobs import draft, get_job
    snapshot = draft(game_id, job_id, x_knowledge_admin)
    if body.version != snapshot['version']:
        raise HTTPException(409, '稿件已更新，請重新讀取檢查結果')
    issues = findings(snapshot['document'])
    blocks = {b['id']: b for b in snapshot['document']['blocks']}
    conn = _connect()
    queued = []
    try:
        cur = conn.cursor(); init(cur)
        job = get_job(cur, game_id, job_id)
        cur.execute('SELECT id FROM import_repair_candidates WHERE game_id=? AND job_id=? AND version=?', (game_id, job_id, body.version))
        existing = [row['id'] for row in cur.fetchall()]
        if existing:
            conn.commit()
            return {'queued': existing, 'message': '此版本已有補辨識紀錄，不重複呼叫模型。'}
        # At most two unique source pages per version, even after repeated clicks.
        seen = set()
        for issue in sorted(issues, key=lambda item: (blocks[item['id']]['fields'].get('大項') != '計分與結束', {'table_layout': 0, 'missing_symbols': 1, 'number_mismatch': 2, 'unread_source': 3}[item['code']])):
            if issue['page'] in seen:
                continue
            seen.add(issue['page'])
            if len(seen) > 2:
                break
            identifier = hashlib.sha256(f"{game_id}:{job_id}:{body.version}:{issue['page']}".encode()).hexdigest()
            payload = {'issue': issue, 'block': blocks[issue['id']], 'source_key': job['source_key'], 'digest': job['digest']}
            cur.execute('INSERT INTO import_repair_candidates VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT (id) DO NOTHING',
                        (identifier, game_id, job_id, body.version, issue['id'], json.dumps(payload, ensure_ascii=False), 'queued', '', time.time()))
            queued.append(identifier)
        conn.commit()
    finally:
        conn.close()
    return {'queued': queued, 'message': '已排入局部補辨識，每版本最多兩頁；原稿保持可查詢。'}


def answer_gaps(question, answer, selected):
    if answer.get('kind') != 'ai':
        return []
    gaps = []
    text = answer.get('answer', '')
    source = '\n'.join(block['fields'].get('原文', '') for block in selected.values())
    phase_question = re.search(r'階段|\b(?:phases?|stages?|fasi|fase)\b', question, re.I)
    enumeration = re.search(r'哪些|依序|順序|列出|全部|完整|\b(?:which|what|list|order|all)\b', question, re.I)
    if phase_question and enumeration:
        for identifier, block in selected.items():
            original = block['fields'].get('原文', '')
            phase_list = len(re.findall(r'^\s*(?:[•\-\d.)]+\s*)?[A-Za-z][A-Za-z -]{1,60}\bPhase\s*$', original, re.M | re.I)) >= 2
            phase_exception = re.search(r'\bphase\b', original, re.I) and re.search(r'\b(?:skip\w*|except\w*|unless)\b', original, re.I)
            if (phase_list or phase_exception) and identifier not in answer.get('citations', []):
                gaps.append('階段清單或例外尚未引用 ' + identifier + '（第 ' + block['fields'].get('PDF頁碼', '') + ' 頁），需逐項核對，不能視為完整涵蓋')
    if any(term in question for term in ('各角色', '各階級', '全部', '完整')):
        for identifier, block in selected.items():
            if identifier not in answer.get('citations', []):
                gaps.append('尚未引用檢索項目 ' + identifier + '（第 ' + block['fields'].get('PDF頁碼', '') + ' 頁），不能視為完整涵蓋')
    if '平手' in question or 'tie' in question.lower():
        if re.search(r'share (?:the )?victory', source, re.I) and not re.search(r'共享|共同獲勝|共同勝利|share', text, re.I):
            gaps.append('來源包含共享勝利的最後分支，回答尚未提及')
        if re.search(r'section of their choice', source, re.I) and not re.search(r'偏好|各自|對應|選擇|choice|section', text, re.I):
            gaps.append('來源要求比較各自對應的政策區域，回答尚未說明比較範圍')
    return gaps
