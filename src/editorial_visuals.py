"""Immutable source crops. No image is public until a reviewed release references it."""
import hashlib
import io
import json
import re
import secrets
import zipfile
from datetime import timedelta
from pathlib import Path

from fastapi import APIRouter, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from src.database import _connect
from src.config import DATA_DIR, get_cloud_settings

router = APIRouter()


class UploadRequest(BaseModel):
    content_type: str
    size: int = Field(gt=0, le=50_000_000)


class ExportRequest(BaseModel):
    document: dict


@router.post("/visuals/export")
def export_bundle(game_id: str, body: ExportRequest, x_knowledge_admin: str = Header(default="")):
    from src.editorial_api import admin
    from src.editorial_validation import validate_draft
    admin(x_knowledge_admin)
    if validate_draft(body.document):
        raise HTTPException(422, "請先修正段落結構再備份")
    output, total, added = io.BytesIO(), 0, set()
    conn = _connect()
    try:
        cur = conn.cursor()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("document.json", json.dumps({"schema_version": 2, **body.document}, ensure_ascii=False, indent=2))
            for block in body.document["blocks"]:
                if block["kind"] != "圖片":
                    continue
                cur.execute("SELECT * FROM editorial_visual_assets WHERE game_id=? AND id=?", (game_id, block["fields"].get("資產", "")))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(422, "圖片資產不存在或不屬於此遊戲")
                for key, name in [(asset_path(game_id, row["id"]), "assets/" + row["id"] + ".png"), (asset_path(game_id, row["source_hash"], "source"), "sources/" + row["source_hash"] + ".source")]:
                    if key in added:
                        continue
                    data = read_bytes(key)
                    total += len(data)
                    if total > 200_000_000:
                        raise HTTPException(413, "完整備份超過 200 MB，請分批備份")
                    archive.writestr(name, data)
                    added.add(key)
            archive.writestr("README.txt", "document.json preserves image metadata. assets/ contains original crops; sources/ contains original PDF, PNG or JPEG bytes. Source filenames are content hashes. Keep this archive private.\n")
    finally:
        conn.close()
    settings = get_cloud_settings()
    if settings.uses_cloud_storage:
        key = f"editorial-exports/{hashlib.sha256(game_id.encode()).hexdigest()}/{secrets.token_hex(16)}.zip"
        store_bytes(key, output.getvalue(), "application/zip")
        import google.auth
        from google.auth.transport.requests import Request
        from google.cloud import storage
        credentials, _ = google.auth.default(); credentials.refresh(Request())
        email = getattr(credentials, "service_account_email", "")
        if not email:
            raise HTTPException(503, "下載簽名服務尚未設定")
        url = storage.Client(credentials=credentials).bucket(settings.storage_bucket).blob(key).generate_signed_url(version="v4", expiration=timedelta(minutes=15), method="GET", service_account_email=email, access_token=credentials.token)
        return {"download_url": url}
    return Response(output.getvalue(), media_type="application/zip", headers={"Cache-Control": "private, no-store", "Content-Disposition": 'attachment; filename="rules-v2.zip"'})


@router.post("/visuals/upload-url")
def upload_url(game_id: str, body: UploadRequest, x_knowledge_admin: str = Header(default="")):
    from src.editorial_api import admin, check_game
    admin(x_knowledge_admin)
    check_game(game_id)
    if body.content_type not in {"application/pdf", "image/png", "image/jpeg"}:
        raise HTTPException(422, "不支援的來源格式")
    settings = get_cloud_settings()
    if not settings.uses_cloud_storage:
        raise HTTPException(503, "大檔案直傳尚未設定，請先使用小於 4 MB 的來源")
    import google.auth
    from google.auth.transport.requests import Request
    from google.cloud import storage
    credentials, _ = google.auth.default()
    credentials.refresh(Request())
    email = getattr(credentials, "service_account_email", "")
    if not email:
        raise HTTPException(503, "上傳簽名服務尚未設定")
    key = f"editorial-uploads/{hashlib.sha256(game_id.encode()).hexdigest()}/{secrets.token_hex(16)}"
    blob = storage.Client(credentials=credentials).bucket(settings.storage_bucket).blob(key)
    url = blob.generate_signed_url(version="v4", expiration=timedelta(minutes=15), method="PUT", content_type=body.content_type, service_account_email=email, access_token=credentials.token)
    return {"upload_url": url, "object_name": key}


def init_visual_db():
    conn = _connect()
    try:
        conn.cursor().execute("CREATE TABLE IF NOT EXISTS editorial_visual_assets (id TEXT NOT NULL, game_id TEXT NOT NULL, source_id TEXT NOT NULL, source_hash TEXT NOT NULL, page INTEGER NOT NULL, crop TEXT NOT NULL, public_allowed INTEGER NOT NULL, PRIMARY KEY (game_id,id))")
        conn.commit()
    finally:
        conn.close()


def asset_path(game_id, asset_id, suffix="png"):
    game_hash = hashlib.sha256(game_id.encode()).hexdigest()
    return f"editorial-visuals/{game_hash}/{asset_id}.{suffix}"


def store_bytes(key, data, content_type):
    settings = get_cloud_settings()
    if settings.uses_cloud_storage:
        from google.cloud import storage
        storage.Client().bucket(settings.storage_bucket).blob(key).upload_from_string(data, content_type=content_type)
    else:
        path = DATA_DIR / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def read_bytes(key):
    settings = get_cloud_settings()
    if settings.uses_cloud_storage:
        from google.cloud import storage
        return storage.Client().bucket(settings.storage_bucket).blob(key).download_as_bytes()
    return (DATA_DIR / key).read_bytes()


@router.post("/visuals/crop")
async def crop(game_id: str, file: UploadFile | None = File(None), source_id: str = Form(...), page: int = Form(...),
               bounds: str = Form("[0,0,1,1]"), public_allowed: bool = Form(False), x_knowledge_admin: str = Header(default=""), object_name: str = Form("")):
    from src.editorial_api import admin, check_game
    admin(x_knowledge_admin)
    check_game(game_id)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
        raise HTTPException(422, "來源編號無效")
    if file:
        data = await file.read(50_000_001)
    else:
        prefix = f"editorial-uploads/{hashlib.sha256(game_id.encode()).hexdigest()}/"
        settings = get_cloud_settings()
        if not isinstance(object_name, str) or not re.fullmatch(re.escape(prefix) + r"[a-f0-9]{32}", object_name) or not settings.uses_cloud_storage:
            raise HTTPException(422, "無效的上傳來源")
        from google.cloud import storage
        blob = storage.Client().bucket(settings.storage_bucket).blob(object_name)
        blob.reload()
        if not blob.size or blob.size > 50_000_000:
            raise HTTPException(413, "來源檔案不可超過 50 MB")
        data = blob.download_as_bytes(if_generation_match=blob.generation)
    if len(data) > 50_000_000:
        raise HTTPException(413, "來源檔案不可超過 50 MB")
    try:
        coords = json.loads(bounds)
        if len(coords) != 4 or any(type(v) not in {int, float} for v in coords):
            raise ValueError()
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError()
    except (TypeError, ValueError):
        raise HTTPException(422, "裁切範圍須為 0 到 1 的左、上、右、下座標")
    import fitz
    try:
        if data.startswith(b"%PDF-"):
            document = fitz.open(stream=data, filetype="pdf")
        else:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = 20_000_000
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in {"JPEG", "PNG"} or source.width * source.height > 20_000_000:
                    raise ValueError()
                source.verify()
            image = fitz.open(stream=data)
            document = fitz.open(stream=image.convert_to_pdf(), filetype="pdf")
            image.close()
        with document:
            if document.needs_pass or not 1 <= page <= len(document) <= 100:
                raise ValueError()
            source_page = document[page - 1]
            rect = source_page.rect
            scale = min(2, 2000 / max(rect.width, rect.height))
            clip = fitz.Rect(x0 * rect.width, y0 * rect.height, x1 * rect.width, y1 * rect.height)
            png = source_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).tobytes("png")
            while len(png) > 4_000_000:
                scale *= 0.75
                png = source_page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False).tobytes("png")
    except Exception:
        raise HTTPException(422, "無法讀取來源；請檢查頁碼、檔案格式、密碼或圖片大小")
    source_hash = hashlib.sha256(data).hexdigest()
    crop_json = json.dumps(coords, separators=(",", ":"))
    asset_id = hashlib.sha256((source_id + source_hash + str(page) + crop_json + str(public_allowed)).encode() + png).hexdigest()
    store_bytes(asset_path(game_id, source_hash, "source"), data, "application/octet-stream")
    store_bytes(asset_path(game_id, asset_id), png, "image/png")
    conn = _connect()
    try:
        conn.cursor().execute("INSERT INTO editorial_visual_assets (id,game_id,source_id,source_hash,page,crop,public_allowed) VALUES (?,?,?,?,?,?,?) ON CONFLICT (game_id,id) DO NOTHING", (asset_id, game_id, source_id, source_hash, page, crop_json, int(public_allowed)))
        conn.commit()
    finally:
        conn.close()
    return {"block": {"id": "VIS-" + asset_id[:16], "kind": "圖片", "fields": {"名稱": "", "別名": "", "替代文字": "", "類型": "元件", "來源": source_id, "PDF頁碼": str(page), "資產": asset_id, "裁切範圍": crop_json, "相關規則": "", "順序": "0", "公開": "是" if public_allowed else "否", "狀態": "待確認"}}}


def validate_assets(cur, game_id, document):
    for block in document["blocks"]:
        if block["kind"] != "圖片":
            continue
        f = block["fields"]
        cur.execute("SELECT * FROM editorial_visual_assets WHERE game_id=? AND id=?", (game_id, f.get("資產", "")))
        asset = cur.fetchone()
        if not asset or asset["source_id"] != f.get("來源") or str(asset["page"]) != f.get("PDF頁碼") or asset["crop"] != f.get("裁切範圍") or (f.get("公開") == "是" and not asset["public_allowed"]):
            raise HTTPException(422, "圖片來源、裁切或公開權限不符，請重新匯入來源")


@router.get("/visuals/{asset_id}")
def image(game_id: str, asset_id: str, version: str = "", x_knowledge_admin: str = Header(default="")):
    from src.editorial_api import admin, version as get_version
    if not re.fullmatch(r"[a-f0-9]{64}", asset_id):
        raise HTTPException(404, "找不到圖片")
    conn = _connect()
    try:
        cur = conn.cursor()
        allowed = False
        if version:
            release = get_version(cur, game_id, version)
            if release["published_at"]:
                allowed = any(b["kind"] == "圖片" and b["fields"].get("資產") == asset_id and b["fields"].get("公開") == "是" and b["fields"].get("狀態") == "已確認" for b in release["document"]["blocks"])
        if not allowed:
            admin(x_knowledge_admin)
        cur.execute("SELECT public_allowed FROM editorial_visual_assets WHERE game_id=? AND id=?", (game_id, asset_id))
        asset = cur.fetchone()
        if not asset or (allowed and not asset["public_allowed"]):
            raise HTTPException(404, "找不到圖片")
    finally:
        conn.close()
    try:
        return Response(read_bytes(asset_path(game_id, asset_id)), media_type="image/png", headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
    except (OSError, ValueError):
        raise HTTPException(404, "圖片暫時無法讀取")


def visual_answer(game_id, version, blocks, question, scope=None):
    if not re.search(r"長什麼樣|長怎樣|樣子|外觀|圖片|圖示|look like|picture|image", question, re.I):
        return None
    from src.editorial_resolution import normalize
    from src.editorial_validation import refs
    rules = {b["id"]: b["fields"] for b in blocks if b["kind"] == "規則"}
    assets = []
    for b in blocks:
        f = b["fields"]
        if b["kind"] != "圖片" or f.get("狀態") != "已確認" or f.get("公開") != "是":
            continue
        if scope and not any(rid in rules and all(not value or rules[rid].get(key) == "全部" or value in refs(rules[rid].get(key)) for key, value in scope.items()) for rid in refs(f.get("相關規則"))):
            continue
        names = [f.get("名稱", ""), f.get("替代文字", ""), *re.split(r"[,，、\n]", f.get("別名", ""))]
        if any(len(n.strip()) >= 2 and normalize(n) in normalize(question) for n in names):
            assets.append({"asset_id": f["資產"], "block_id": b["id"], "version": version, "source_id": f["來源"], "page": int(f["PDF頁碼"]), "caption": f["名稱"], "alt": f["替代文字"]})
    return {"answer": "以下是原書中已確認的元件圖片。" if len(assets) == 1 else "找到多個圖片，請選擇你指的元件。" if assets else "目前沒有符合名稱的已確認公開圖片。請用規則中的名稱或別名查找；系統不會生成或猜測外觀。", "citations": [], "kind": "visual" if len(assets) == 1 else "clarification" if assets else "abstained", "visual_citations": assets, "missing_context": ["referent"] if len(assets) > 1 else []}
