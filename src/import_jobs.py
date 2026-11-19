"""Persistent editorial imports with fenced leases; model work runs on a local worker."""
import hashlib
import io
import json
import secrets
import time
import uuid

import fitz
from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.database import _connect
from src.editorial_visuals import read_bytes, store_bytes

router = APIRouter()
LEASE_SECONDS = 300


def init_jobs():
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS editorial_import_jobs (id TEXT PRIMARY KEY, game_id TEXT NOT NULL, filename TEXT NOT NULL, source_key TEXT NOT NULL, digest TEXT NOT NULL, total INTEGER NOT NULL, status TEXT NOT NULL, lease TEXT NOT NULL, lease_until DOUBLE PRECISION NOT NULL, error TEXT NOT NULL, created DOUBLE PRECISION NOT NULL)')
        cur.execute('CREATE TABLE IF NOT EXISTS editorial_import_pages (job_id TEXT NOT NULL, page INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(job_id,page))')
        cur.execute('CREATE TABLE IF NOT EXISTS editorial_import_retries (job_id TEXT NOT NULL, page INTEGER NOT NULL, blocks TEXT NOT NULL, PRIMARY KEY(job_id,page))')
        cur.execute('CREATE TABLE IF NOT EXISTS editorial_import_page_history (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, page INTEGER NOT NULL, payload TEXT NOT NULL, created DOUBLE PRECISION NOT NULL)')
        cur.execute('CREATE TABLE IF NOT EXISTS import_source_errors (job_id TEXT PRIMARY KEY, error TEXT NOT NULL)')
        from src.source_task_state import init as init_source_tasks
        init_source_tasks(cur)
        cur.execute('CREATE TABLE IF NOT EXISTS import_native_sources (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        from src.local_category_worker import init as init_categories
        init_categories(cur)
        from src.import_quality import init as init_quality
        init_quality(cur)
        cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS editorial_import_source ON editorial_import_jobs (game_id,digest)')
        conn.commit()
    finally:
        conn.close()


def authorize(game_id, key):
    from src.editorial_api import admin, check_game
    admin(key)
    check_game(game_id)
    init_jobs()


def get_job(cur, game_id, job_id):
    cur.execute('SELECT * FROM editorial_import_jobs WHERE id=? AND game_id=?', (job_id, game_id))
    row = cur.fetchone()
    if not row:
        raise HTTPException(404, '找不到匯入任務')
    return dict(row)


def public_job(cur, row):
    cur.execute('SELECT page,payload FROM editorial_import_pages WHERE job_id=? ORDER BY page', (row['id'],))
    completed = cur.fetchall()
    payloads = [json.loads(r['payload']) for r in completed]
    cur.execute('SELECT blocks FROM editorial_import_retries WHERE job_id=?', (row['id'],))
    retry_count = sum(len(json.loads(r['blocks'])) for r in cur.fetchall())
    cur.execute('SELECT error FROM import_source_errors WHERE job_id=?',(row['id'],))
    source_error = cur.fetchone()
    from src.source_task_state import identity as source_identity
    cur.execute('SELECT state,retry_at,attempts,error FROM import_source_tasks WHERE id=?',(source_identity(row),))
    source_task=cur.fetchone()
    source_state=dict(source_task) if source_task else {'state':'queued','retry_at':0,'attempts':0,'error':''}
    native_key=source_identity(row)
    cur.execute('SELECT payload FROM import_native_sources WHERE id=?',(native_key,))
    all_native=cur.fetchone()
    native={}
    if all_native:
        native=json.loads(all_native['payload'])
    else:
        cur.execute('SELECT id,payload FROM import_native_sources WHERE id LIKE ?',(native_key+':%',))
        native={saved['id'].rsplit(':',1)[1]:json.loads(saved['payload']) for saved in cur.fetchall()}
    readable={int(page) for page,data in native.items() if data.get('rules')}
    readable.update(r['page'] for r in completed if any(rule.get('original') or rule.get('translation') for rule in json.loads(r['payload'])['rules'] if not rule.get('title','').startswith('待辨識區塊')))
    from src.local_category_worker import ENGINE as category_engine
    cur.execute('SELECT state,payload,result FROM import_category_tasks WHERE job_id=?',(row['id'],))
    category_rows=[r for r in cur.fetchall() if json.loads(r['payload']).get('engine')==category_engine]
    categories=[r['state'] for r in category_rows]
    cur.execute('SELECT state,COUNT(*) AS n FROM import_repair_candidates WHERE job_id=? AND game_id=? GROUP BY state',(row['id'],row['game_id']))
    repairs={r['state']:r['n'] for r in cur.fetchall()}
    from src.import_lifecycle import describe
    lifecycle=describe(source_state=source_state['state'],source_complete=bool(all_native),readable_pages=len(readable),total=row['total'],model_state=row['status'],
                       repair_pending=sum(repairs.get(s,0) for s in ('queued','processing','waiting_service','waiting_budget')),repair_gaps=sum(repairs.get(s,0) for s in ('failed','rejected','candidate')),
                       category_pending=sum(s in {'queued','processing','waiting_budget'} for s in categories),category_waiting=categories.count('waiting_service'),retries=retry_count,
                       gaps=any(p['warnings'] or any(rule['title'].startswith('待辨識區塊') for rule in p['rules']) for p in payloads) or 'unresolved' in categories or any('unassigned' in json.loads(r['result']) for r in category_rows))
    if 'waiting_budget' in categories: lifecycle['wait_reason']='budget'
    return {k: row[k] for k in ('id', 'filename', 'digest', 'total', 'status', 'error', 'created')} | {
        'source_error': source_error['error'] if source_error else '',
        'source_task':source_state,
        'lifecycle':lifecycle,
        'resume_at':row['lease_until'] if row['status']=='paused' else 0,
        'completed_pages': [r['page'] for r in completed],
        'unresolved_blocks': sum(rule['title'].startswith('待辨識區塊') for p in payloads for rule in p['rules']),
        'warning_pages': sum(bool(p['warnings']) for p in payloads),
        'pending_retry_blocks': retry_count,
    }


def enqueue(game_id, filename, data):
    if not data or len(data) > 50_000_000:
        raise HTTPException(413, '來源檔案須介於 1 byte 與 50 MB')
    try:
        if not data.startswith(b'%PDF-'):
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {'PNG', 'JPEG'} or image.width * image.height > 20_000_000:
                    raise ValueError()
                image.verify()
            with fitz.open(stream=data) as image:
                data = image.convert_to_pdf()
        with fitz.open(stream=data, filetype='pdf') as pdf:
            if pdf.needs_pass or not 1 <= len(pdf) <= 100:
                raise ValueError()
            total = len(pdf)
    except Exception:
        raise HTTPException(422, '請提供未加密的 PDF（1–100 頁）、PNG 或 JPEG 圖片') from None
    if len(data) > 50_000_000:
        raise HTTPException(413, '轉換後來源檔案不可超過 50 MB')
    init_jobs()
    digest = hashlib.sha256(data).hexdigest()
    source_key = f'editorial-imports/{hashlib.sha256(game_id.encode()).hexdigest()}/{digest}.pdf'
    store_bytes(source_key, data, 'application/pdf')
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('SELECT * FROM editorial_import_jobs WHERE game_id=? AND digest=? ORDER BY created DESC', (game_id, digest))
        existing = cur.fetchone()
        if existing:
            return public_job(cur, existing)
        identifier = uuid.uuid4().hex
        cur.execute('INSERT INTO editorial_import_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT (game_id,digest) DO NOTHING',
                    (identifier, game_id, filename[:240], source_key, digest, total, 'queued', '', 0, '', time.time()))
        conn.commit()
        cur.execute('SELECT * FROM editorial_import_jobs WHERE game_id=? AND digest=?', (game_id, digest))
        return public_job(cur, cur.fetchone())
    finally:
        conn.close()


@router.post('/imports')
async def upload(game_id: str, file: UploadFile | None = File(None), object_name: str = Form(''), filename: str = Form('rulebook'), x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    if file:
        return enqueue(game_id, file.filename or 'rulebook', await file.read(50_000_001))
    from src.config import get_cloud_settings
    import re
    settings = get_cloud_settings()
    prefix = f'editorial-uploads/{hashlib.sha256(game_id.encode()).hexdigest()}/'
    if not settings.uses_cloud_storage or not re.fullmatch(re.escape(prefix) + r'[a-f0-9]{32}', object_name):
        raise HTTPException(422, '無效的直傳檔案')
    from google.cloud import storage
    blob = storage.Client().bucket(settings.storage_bucket).blob(object_name)
    blob.reload()
    if not blob.size or blob.size > 50_000_000:
        raise HTTPException(413, '來源檔案不可超過 50 MB')
    return enqueue(game_id, filename, blob.download_as_bytes(if_generation_match=blob.generation))


@router.get('/imports')
def list_jobs(game_id: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute('SELECT * FROM editorial_import_jobs WHERE game_id=? ORDER BY created DESC LIMIT 30', (game_id,))
        return {'jobs': [public_job(cur, row) for row in cur.fetchall()]}
    finally:
        conn.close()


class Lease(BaseModel):
    token: str = Field(min_length=32, max_length=64)


class PageResult(Lease):
    page: int = Field(ge=1, le=100)
    rules: list[dict] = Field(max_length=200)
    warnings: list[str] = Field(max_length=100)
    usage: list[dict] = Field(default_factory=list, max_length=200)
    locations: list[dict] = Field(default_factory=list, max_length=200)


def validate_locations(locations, rules, data, page_number):
    import math
    if len(locations) != len(rules):
        raise HTTPException(422, '來源定位數量與區塊不符')
    with fitz.open(stream=data, filetype='pdf') as pdf:
        bounds = pdf[page_number - 1].rect
        for index, item in enumerate(locations, 1):
            box = item.get('bbox')
            scale = item.get('scale')
            if (item.get('block') != index or item.get('kind') not in {'text', 'image'} or
                not isinstance(box, list) or len(box) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
                type(scale) not in (float, int) or not math.isfinite(scale) or not 0 < scale <= 3):
                raise HTTPException(422, '來源定位格式無效')
            rect = fitz.Rect(box)
            if rect.is_empty or not bounds.contains(rect):
                raise HTTPException(422, '來源定位超出頁面')
            if item['kind'] == 'image':
                if rect.width * rect.height * scale * scale > 20_000_000:
                    raise HTTPException(422, '裁切圖片超過像素上限')
                image = pdf[page_number - 1].get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect).tobytes('png')
                if hashlib.sha256(image).hexdigest() != item.get('image_sha256'):
                    raise HTTPException(422, '來源圖片雜湊不符')


@router.get('/imports/{job_id}/pages/{page_number}/blocks/{block}/image')
def block_image(game_id: str, job_id: str, page_number: int, block: int, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        job = get_job(cur, game_id, job_id)
        cur.execute('SELECT payload FROM editorial_import_pages WHERE job_id=? AND page=?', (job_id, page_number))
        row = cur.fetchone()
        locations = json.loads(row['payload']).get('locations', []) if row else []
        item = next((p for p in locations if p['block'] == block and p['kind'] == 'image'), None)
        if item is None:
            raise HTTPException(404, '此區塊沒有已保存的圖片定位；舊資料不可猜測位置')
        with fitz.open(stream=read_bytes(job['source_key']), filetype='pdf') as pdf:
            image = pdf[page_number - 1].get_pixmap(matrix=fitz.Matrix(item['scale'], item['scale']), clip=fitz.Rect(item['bbox'])).tobytes('png')
        if hashlib.sha256(image).hexdigest() != item['image_sha256']:
            raise HTTPException(409, '來源重繪與原始裁切不符')
        return Response(image, media_type='image/png', headers={'Cache-Control': 'private, no-store'})
    finally:
        conn.close()


@router.get('/imports/{job_id}/pages/{page_number}/image')
def source_page_image(game_id: str, job_id: str, page_number: int, digest: str = '', x_knowledge_admin: str = Header(default='')):
    """Read-only original-page rendering, bound to the immutable import source."""
    import re
    authorize(game_id, x_knowledge_admin)
    if not re.fullmatch(r'[a-f0-9]{64}', digest):
        raise HTTPException(422, '缺少有效的原件版本')
    conn = _connect()
    try:
        job = get_job(conn.cursor(), game_id, job_id)
        if digest != job['digest']:
            raise HTTPException(409, '原件版本不符，請重新開啟來源')
        if page_number < 1 or page_number > job['total']:
            raise HTTPException(404, '來源頁次不存在')
        data = read_bytes(job['source_key'])
        if hashlib.sha256(data).hexdigest() != digest:
            raise HTTPException(409, '原件雜湊不符，停止顯示來源')
        try:
            with fitz.open(stream=data) as pdf:
                if page_number > len(pdf):
                    raise HTTPException(404, '來源頁次不存在')
                page = pdf[page_number - 1]
                bounds = page.rect
                if bounds.is_empty or bounds.is_infinite:
                    raise HTTPException(422, '來源頁面尺寸無效')
                # Bound both edge length and pixel area; no model or database writes.
                scale = min(2, 2200 / max(bounds.width, bounds.height), (4_000_000 / (bounds.width * bounds.height)) ** 0.5)
                image = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False).tobytes('png')
        except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError, ValueError) as error:
            raise HTTPException(422, '來源無法繪製，請改開原始文件') from error
        return Response(image, media_type='image/png', headers={
            'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff',
            'X-Source-Digest': digest, 'X-PDF-Page': str(page_number),
        })
    finally:
        conn.close()


@router.get('/imports/{job_id}/locations')
def source_locations(game_id: str, job_id: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        job = get_job(cur, game_id, job_id)
        cur.execute('SELECT page,payload FROM editorial_import_pages WHERE job_id=? ORDER BY page', (job_id,))
        return {'digest': job['digest'], 'locations': [dict(item, page=row['page']) for row in cur.fetchall()
                for item in json.loads(row['payload']).get('locations', [])]}
    finally:
        conn.close()


class UsageEvent(BaseModel):
    id: str = Field(pattern=r'^[a-f0-9]{32}$')
    model: str = Field(pattern=r'^(qwen3:8b|qwen3-vl:4b-instruct)$')
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class RetryBlock(BaseModel):
    page: int = Field(ge=1, le=100)
    block: int = Field(ge=1, le=200)


class RetryRequest(BaseModel):
    blocks: list[RetryBlock] = Field(default_factory=list, max_length=200)


@router.post('/imports/{job_id}/retry')
def retry_blocks(game_id: str, job_id: str, body: RetryRequest, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        get_job(cur, game_id, job_id)
        cur.execute("UPDATE editorial_import_jobs SET status='queued',error='' WHERE id=? AND game_id=? AND status='review'", (job_id, game_id))
        if cur.rowcount != 1:
            raise HTTPException(409, '請等待目前處理完成，或先續跑暫停的任務')
        cur.execute('SELECT page,payload FROM editorial_import_pages WHERE job_id=?', (job_id,))
        pages = {r['page']: r['payload'] for r in cur.fetchall()}
        available = {(page, i + 1) for page, payload in pages.items() for i, rule in enumerate(json.loads(payload)['rules']) if rule['title'].startswith('待辨識區塊')}
        selected = {(b.page, b.block) for b in body.blocks} if body.blocks else available
        if not selected or not selected <= available:
            raise HTTPException(422, '只能重試尚未取得可靠結果的區塊')
        for page in sorted({p for p, _ in selected}):
            blocks = sorted(b for p, b in selected if p == page)
            cur.execute('INSERT INTO editorial_import_page_history VALUES (?,?,?,?,?)', (uuid.uuid4().hex, job_id, page, pages[page], time.time()))
            cur.execute('INSERT INTO editorial_import_retries VALUES (?,?,?)', (job_id, page, json.dumps(blocks)))
        conn.commit()
        return public_job(cur, get_job(cur, game_id, job_id))
    finally:
        conn.close()


@router.get('/imports/{job_id}/retries')
def retry_details(game_id: str, job_id: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        get_job(cur, game_id, job_id)
        cur.execute('SELECT r.page,r.blocks,p.payload FROM editorial_import_retries r JOIN editorial_import_pages p ON p.job_id=r.job_id AND p.page=r.page WHERE r.job_id=? ORDER BY r.page', (job_id,))
        return {'pages': {str(r['page']): {'blocks': json.loads(r['blocks']), 'result': json.loads(r['payload'])} for r in cur.fetchall()}}
    finally:
        conn.close()


@router.post('/imports/{job_id}/usage')
def usage_event(game_id: str, job_id: str, body: UsageEvent, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        get_job(conn.cursor(), game_id, job_id)
    finally:
        conn.close()
    from src.local_usage import record
    record(body.id, game_id, body.model, f'import:{job_id}', body.model_dump())
    return {'ok': True}


def fenced(cur, game_id, job_id, token):
    # Conditional UPDATE both fences stale workers and locks this row on PostgreSQL.
    cur.execute('UPDATE editorial_import_jobs SET lease_until=? WHERE id=? AND game_id=? AND status=? AND lease=? AND lease_until>?',
                (time.time() + LEASE_SECONDS, job_id, game_id, 'processing', token, time.time()))
    if cur.rowcount != 1:
        raise HTTPException(409, '任務租約已失效；請停止此工作並重新領取')
    return get_job(cur, game_id, job_id)


@router.post('/imports/claim')
def claim(game_id: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    return claim_next(game_id)


def claim_next(game_id=None):
    """Internal lease allocator; callers must authenticate before invoking."""
    conn = _connect()
    try:
        cur = conn.cursor()
        scope = 'game_id=? AND ' if game_id is not None else ''
        now=time.time()
        params = (game_id, now, now) if game_id is not None else (now,now)
        eligible="(status='queued' OR (status='processing' AND lease_until<?) OR (status='paused' AND lease_until>0 AND lease_until<=?))"
        cur.execute("SELECT * FROM editorial_import_jobs WHERE " + scope + eligible + " ORDER BY created,id LIMIT 1", params)
        row = cur.fetchone()
        if not row:
            return {'job': None}
        token = secrets.token_hex(24)
        cur.execute("UPDATE editorial_import_jobs SET status='processing',lease=?,lease_until=?,error='' WHERE id=? AND "+eligible, (token, now + LEASE_SECONDS, row['id'], now,now))
        if cur.rowcount != 1:
            return {'job': None}
        conn.commit()
        return {'job': {**public_job(cur, get_job(cur, row['game_id'], row['id'])), 'game_id': row['game_id']}, 'token': token}
    finally:
        conn.close()


@router.post('/imports/{job_id}/heartbeat')
def heartbeat(game_id: str, job_id: str, body: Lease, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        fenced(conn.cursor(), game_id, job_id, body.token)
        conn.commit()
        return {'ok': True}
    finally:
        conn.close()


@router.get('/imports/{job_id}/source')
def source(game_id: str, job_id: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        row = get_job(conn.cursor(), game_id, job_id)
        return Response(read_bytes(row['source_key']), media_type='application/pdf', headers={'Cache-Control': 'private, no-store'})
    finally:
        conn.close()


@router.post('/imports/{job_id}/pages')
def complete_page(game_id: str, job_id: str, body: PageResult, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    for rule in body.rules:
        if (set(rule) != {'title', 'translation', 'original', 'category'} or
                any(not isinstance(v, str) or not v.strip() or len(v) > 16000 for v in rule.values())):
            raise HTTPException(422, '規則區塊結構無效')
    if len(json.dumps(body.model_dump())) > 2_000_000:
        raise HTTPException(413, '單頁結果過大')
    conn = _connect()
    try:
        cur = conn.cursor()
        row = fenced(cur, game_id, job_id, body.token)
        if body.page > row['total']:
            raise HTTPException(422, '頁碼超出範圍')
        if body.locations:
            validate_locations(body.locations, body.rules, read_bytes(row['source_key']), body.page)
        encoded = json.dumps({'rules': body.rules, 'warnings': body.warnings, 'locations': body.locations}, ensure_ascii=False)
        cur.execute('SELECT blocks FROM editorial_import_retries WHERE job_id=? AND page=?', (job_id, body.page))
        retry = cur.fetchone()
        if retry:
            selected = set(json.loads(retry['blocks']))
            cur.execute('SELECT payload FROM editorial_import_pages WHERE job_id=? AND page=?', (job_id, body.page))
            old_payload = json.loads(cur.fetchone()['payload'])
            old = old_payload['rules']
            if old_payload.get('locations') and old_payload['locations'] != body.locations:
                raise HTTPException(409, '重試不可移除或改動已保存的來源定位')
            if len(old) != len(body.rules) or any(new['original'] != old[i]['original'] or (i+1 not in selected and new != old[i]) for i, new in enumerate(body.rules)):
                raise HTTPException(409, '重試不可改動來源、區塊編號或未選取的內容')
            cur.execute('UPDATE editorial_import_pages SET payload=? WHERE job_id=? AND page=?', (encoded, job_id, body.page))
            cur.execute('DELETE FROM editorial_import_retries WHERE job_id=? AND page=?', (job_id, body.page))
        else:
            cur.execute('INSERT INTO editorial_import_pages VALUES (?,?,?) ON CONFLICT (job_id,page) DO NOTHING', (job_id, body.page, encoded))
        cur.execute('SELECT COUNT(*) AS n FROM editorial_import_pages WHERE job_id=?', (job_id,))
        all_pages = cur.fetchone()['n'] == row['total']
        cur.execute('SELECT COUNT(*) AS n FROM editorial_import_retries WHERE job_id=?', (job_id,))
        if all_pages and cur.fetchone()['n'] == 0:
            cur.execute("UPDATE editorial_import_jobs SET status='review',lease='',lease_until=0 WHERE id=?", (job_id,))
        conn.commit()
        result = public_job(cur, get_job(cur, game_id, job_id))
    finally:
        conn.close()
    from src.local_usage import record
    for event in body.usage:
        if (set(event) == {'id', 'model', 'prompt_tokens', 'output_tokens'} and
                isinstance(event['id'], str) and len(event['id']) == 32 and
                event['model'] in {'qwen3:8b', 'qwen3-vl:4b-instruct'} and
                all(type(event[k]) is int and event[k] >= 0 for k in ('prompt_tokens', 'output_tokens'))):
            record(event['id'], game_id, event['model'], f'import:{job_id}:page:{body.page}', event)
    return result


class PauseRequest(Lease):
    reason: str | None = Field(default=None,pattern=r'^((day|document|work)_(calls|tokens|seconds)_limit|local_service)$')


@router.post('/imports/{job_id}/pause')
def pause(game_id: str, job_id: str, body: PauseRequest, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        fenced(cur, game_id, job_id, body.token)
        message='本機處理失敗；已保留完成頁面，請查看 worker 紀錄後續跑'
        retry_at=0
        if body.reason=='local_service':
            retry_at=time.time()+60
            message='等待本機模型服務；60秒後自動重試，已完成來源仍可閱讀。'
        elif body.reason:
            scope,metric,_=body.reason.split('_')
            message=f"已達{'每日' if scope=='day' else '此文件'}{ {'calls':'呼叫次數','tokens':'Token','seconds':'推論時間'}[metric]}預算上限；已保留完成來源，不會改用雲端模型。"
            if scope=='day': retry_at=(int(time.time()//86400)+1)*86400
        cur.execute("UPDATE editorial_import_jobs SET status='paused',lease='',lease_until=?,error=? WHERE id=?", (retry_at,message, job_id))
        conn.commit()
        return {'ok': True}
    finally:
        conn.close()


@router.post('/imports/{job_id}/resume')
def resume(game_id: str, job_id: str, x_knowledge_admin: str = Header(default='')):
    authorize(game_id, x_knowledge_admin)
    conn = _connect()
    try:
        cur = conn.cursor()
        row = get_job(cur, game_id, job_id)
        if row['status'] != 'paused':
            raise HTTPException(409, '只有暫停的任務可續跑')
        cur.execute("UPDATE editorial_import_jobs SET status='queued',error='' WHERE id=? AND status='paused'", (job_id,))
        conn.commit()
        return {'ok': True}
    finally:
        conn.close()


@router.get('/imports/{job_id}/draft')
def draft(game_id: str, job_id: str, x_knowledge_admin: str = Header(default=''), version: str = '', q: str = '', offset: int = 0):
    authorize(game_id, x_knowledge_admin)
    from src.import_snapshots import load, search
    conn = _connect()
    try:
        get_job(conn.cursor(), game_id, job_id)
    finally:
        conn.close()
    result = load(game_id, job_id, version) if version else build_draft(game_id, job_id)
    if q:
        if len(q) > 500 or offset < 0:
            raise HTTPException(422, '搜尋參數無效')
        result = {**result, 'search': search(game_id, job_id, result['version'], q, offset)}
    return result


def build_draft(game_id, job_id, native_limit=None):
    """Internal snapshot builder; HTTP callers must authorize before invoking."""
    conn = _connect()
    try:
        cur = conn.cursor()
        row = get_job(cur, game_id, job_id)
        cur.execute('SELECT page,payload FROM editorial_import_pages WHERE job_id=? ORDER BY page', (job_id,))
        completed = {p['page']: json.loads(p['payload']) for p in cur.fetchall()}
    finally:
        conn.close()
    from src.database import get_game
    game = get_game(game_id)
    from src.import_native import source_pages
    native = source_pages(row, max_new_pages=native_limit)
    blocks = [{'kind': '文件', 'id': 'META', 'fields': {'遊戲': game.name, '版本': '本機自動擷取・待人工校對'}},
              {'kind': '來源', 'id': 'LOCAL-SOURCE', 'fields': {'名稱': row['filename'], '版本': row['digest'][:12], 'PDF頁數': str(row['total'])}}]
    categories = {'開始遊戲', '進行遊戲', '卡牌與配件', '計分與結束', '例外與 FAQ', '其他規則'}
    readable = set()
    from src.local_category_worker import has_source
    for page in range(1, row['total'] + 1):
        ids, warnings = [], []
        # Native source is independent of model completion. Keep legacy model
        # IDs and results as derivatives; never replace source with a page status.
        for origin, result in [('NATIVE', native.get(str(page))), ('LOCAL', completed.get(page))]:
            if not result:
                continue
            warnings.extend(result.get('warnings', []))
            for i, rule in enumerate(result['rules']):
                identifier = f'{origin}-P{page:03d}-R{i+1:03d}'
                block = {'kind': '規則', 'id': identifier, 'fields': {
                    '大項': rule['category'] if rule['category'] in categories else '其他規則',
                    '標題': rule['title'], '內容': rule['translation'] or rule['original'], '原文': rule['original'],
                    '角色': '未確認', '人數': '未確認', '擴充': '未確認', '條件': '未確認，請核對內容', '例外': '未確認，請核對原文',
                    '來源': 'LOCAL-SOURCE', 'PDF頁碼': str(page), '狀態': '待確認'}}
                if origin == 'NATIVE':
                    block.update(extraction=rule['extraction'], source_location=rule['source_location'])
                else:
                    block['derivation'] = 'model-extraction'
                block['evidence_usable'] = has_source(block)
                if block['evidence_usable']:
                    readable.add(page)
                else:
                    warnings.append(f'{identifier}：尚無可靠來源轉寫，保留缺漏，不作問答證據')
                ids.append(identifier)
                blocks.append(block)
        blocks.append({'kind': '涵蓋', 'id': f'LOCAL-C{page:03d}', 'fields': {
            '章節': f'PDF 第 {page} 頁', '來源': 'LOCAL-SOURCE', 'PDF頁碼': str(page), '規則': '、'.join(ids), '狀態': '待確認',
            '說明': '；'.join(dict.fromkeys(warnings)) or ('已完成機器處理，尚未校對完整性' if ids else '尚未完成擷取')}})
    from src.editorial_import import markdown
    from src.pdf_layout_tables import cached_supplement
    blocks.extend(cached_supplement(job_id, row['source_key'], row['digest']))
    from src.import_categories import categories as source_categories
    from src.import_alignment import align
    blocks, aliases = align(blocks)
    document = {'blocks': blocks, 'aliases': aliases, 'categories': source_categories(row, blocks)}
    from src.import_categories import chapter_manifest
    document['chapters'] = chapter_manifest(blocks)
    import hashlib
    version = hashlib.sha256(json.dumps(document, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    from src.import_quality import findings
    readable_pages = len(readable)
    from src.local_category_worker import progress
    classification = progress(job_id, document['categories'])
    snapshot = {'classification': classification, 'readable_pages': readable_pages, 'native_pages': len(native), 'layout_tables': sum(bool(b.get('layout_evidence')) for b in blocks), 'quality_findings': findings(document), 'text': markdown(document), 'document': document, 'version': version, 'completed_pages': len(completed), 'total': row['total']}
    from src.import_snapshots import save
    save(game_id, job_id, snapshot)
    return snapshot


from src.local_question_context import ConfirmedContext


class PreviewQuestion(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    version: str = Field(pattern=r'^[a-f0-9]{64}$')
    settings_version: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    confirmed_context: ConfirmedContext = Field(default_factory=ConfirmedContext)


@router.post('/imports/{job_id}/ask')
def preview_ask(game_id: str, job_id: str, body: PreviewQuestion, x_knowledge_admin: str = Header(default='')):
    result = draft(game_id, job_id, x_knowledge_admin, version=body.version)
    from src.import_documents import question_settings
    pinned_settings=question_settings(game_id,job_id,body.settings_version)
    context={k:v.strip() for k,v in body.confirmed_context.model_dump().items() if v.strip()}
    answer_scope={'settings_version':pinned_settings['settings_version'],'source_settings':pinned_settings['document'],'confirmed_context':context}
    from src.local_usage import document_scope
    from src.local_import_budget import work_scope
    import hashlib
    work_id=hashlib.sha256(json.dumps([job_id,result['version'],pinned_settings['settings_version'],body.question,context],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    with document_scope(game_id,job_id,result['version']), work_scope(game_id,'qa',work_id):
        from src.gemini_usage import setting
        if setting('LOCAL_M3_QA_ENABLED','false')=='true':
            from src.local_preview_qa import ask
            from src.import_quality import save_answer
            response={**ask(result,body.question,game_id,**({'confirmed_context':context} if context else {})),'preview':True,**answer_scope}
            response['answer_id']=save_answer(game_id,job_id,result['version'],body.question,response,result['document'])
            return response
        from src.editorial_api import Question, _ask
        from src.pdf_layout_tables import table_answer
        answer = table_answer(result['document']['blocks'], body.question) if not context else None
        if answer is None:
            answer = _ask(game_id, Question(question=body.question, version=body.version,
                          role=context.get('role',''),players=context.get('players',''),stage=context.get('stage','')), cache_allowed=False,
                      preview_release=result, **({'confirmed_context':context} if context else {}))
        from src.preview_retrieval import retrieve
        response = {**answer, 'preview': True, 'version': result['version'], **answer_scope, 'retrieved_ids': list(retrieve(result['document']['blocks'], body.question))}
        from src.import_quality import save_answer, answer_gaps
        response['coverage_gaps'] = answer_gaps(body.question, answer, retrieve(result['document']['blocks'], body.question))
        if response['coverage_gaps']:
            response['kind'] = 'partial'
        response['answer_id'] = save_answer(game_id, job_id, result['version'], body.question, response, result['document'])
        return response
