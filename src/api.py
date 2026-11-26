"""
api.py — Board Game Rules Knowledge Base API

提供完整的 RESTful API：
- Game CRUD
- RuleDocument 管理
- RuleNode 檢視
- PDF Import
- Rule Retrieval
- AI Q&A
- AI Teaching
- Review System
"""
import hashlib
import hmac
import logging
import os
import secrets
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from src.editorial_api import admin


def authorize_rulebook_write(
    game_id: str,
    method: str,
    action: str,
    admin_key: str,
    timestamp: str = "",
    signature: str = "",
) -> None:
    """Authorize rulebook writes via either a local admin/session credential or a proxy-signed private-session proof."""
    if not signature:
        admin(admin_key)
        return
    secret = os.getenv("KNOWLEDGE_RELEASE_SYNC_KEY", "") or os.getenv("COMMUNITY_PROXY_SECRET", "")
    try:
        fresh = abs(time.time() - int(timestamp)) <= 60
    except ValueError:
        fresh = False
    message = f"rulebook-admin\n{timestamp}\n{method}\n{game_id}\n{action}"
    expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()
    if not secret or not fresh or not hmac.compare_digest(expected.encode(), signature.encode()):
        raise HTTPException(status_code=401, detail="無法驗證規則書管理權限")

from src.database import (
    create_game,
    create_review_items_bulk,
    delete_game,
    get_all_games,
    get_document,
    get_documents_by_game,
    get_game,
    get_rule_node,
    get_rule_nodes_by_game,
    get_rule_stats,
    init_db,
    update_document_status,
    update_game,
    update_review_item_status,
)
from src.import_service import import_rulebook
from src.storage import create_direct_upload, download_uploaded_rulebook
from src.models import (
    Game,
    GameStatus,
    ImportResult,
    QARequest,
    QAResponse,
    ReviewAction,
    RuleNode,
    RuleSearchRequest,
    RuleSearchResponse,
    TeachingSessionRequest,
    TeachingSessionResponse,
)
from src.qa_service import answer_question
from src.retrieval import search_rules, search_with_exceptions
from src.teaching_service import create_teaching_session
from src.validator import get_review_priority, review_rule

logger = logging.getLogger(__name__)

# Keep the structured knowledge-base contract separate from the legacy RAG
# endpoints in app.py (/api/games and /api/query). Both services can now
# evolve without changing the existing public rules-search experience.
router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


class DirectUploadRequest(BaseModel):
    filename: str


class DirectImportRequest(BaseModel):
    document_id: Optional[str] = None
    object_name: str
    filename: str
    version: Optional[str] = None
    language: str = "en"


# ── 初始化 ──────────────────────────────────────────────────────────

@router.on_event("startup")
async def startup():
    """初始化資料庫。"""
    init_db()


# ── Game API ────────────────────────────────────────────────────────

from src.game_catalog_api import router as game_catalog_router
router.include_router(game_catalog_router)


@router.put("/games/{game_id}")
async def api_update_game(game_id: str, game: Game, x_knowledge_admin: str = Header(default="")):
    """更新遊戲。"""
    admin(x_knowledge_admin)
    existing = get_game(game_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="遊戲不存在")
    game.id = game_id
    updated = update_game(game)
    return updated.model_dump()


@router.delete("/games/{game_id}")
async def api_delete_game(game_id: str, x_knowledge_admin: str = Header(default="")):
    """刪除遊戲。"""
    admin(x_knowledge_admin)
    deleted = delete_game(game_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="遊戲不存在")
    return {"deleted": True}


# ── RuleDocument API ────────────────────────────────────────────────

@router.post("/games/{game_id}/rulebooks/upload-url")
async def api_rulebook_upload_url(
    game_id: str,
    request: DirectUploadRequest,
    x_knowledge_admin: str = Header(default=""),
    x_knowledge_admin_timestamp: str = Header(default=""),
    x_knowledge_admin_signature: str = Header(default=""),
):
    authorize_rulebook_write(
        game_id, "POST", "rulebooks/upload-url",
        x_knowledge_admin, x_knowledge_admin_timestamp, x_knowledge_admin_signature,
    )
    if get_game(game_id) is None:
        raise HTTPException(status_code=404, detail="遊戲不存在")
    if not request.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支援 PDF 檔案")
    try:
        return create_direct_upload(game_id, request.filename)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@router.post("/games/{game_id}/rulebooks/import")
async def api_import_direct_rulebook(
    game_id: str,
    request: DirectImportRequest,
    x_knowledge_admin: str = Header(default=""),
    x_knowledge_admin_timestamp: str = Header(default=""),
    x_knowledge_admin_signature: str = Header(default=""),
):
    authorize_rulebook_write(
        game_id, "POST", "rulebooks/import",
        x_knowledge_admin, x_knowledge_admin_timestamp, x_knowledge_admin_signature,
    )
    if get_game(game_id) is None:
        raise HTTPException(status_code=404, detail="遊戲不存在")
    if not request.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支援 PDF 檔案")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    try:
        stored_path = download_uploaded_rulebook(game_id, request.object_name, tmp_path)
        result = import_rulebook(game_id, tmp_path, request.filename, request.version, request.language, stored_path, document_id=request.document_id, max_pages=2)
        if isinstance(result, dict):
            return JSONResponse(result, status_code=202)
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Direct rulebook import failed")
        raise HTTPException(status_code=500, detail=f"匯入失敗: {error}") from error
    finally:
        Path(tmp_path).unlink(missing_ok=True)

@router.get("/games/{game_id}/documents")
async def api_list_documents(game_id: str):
    """列出遊戲的所有規則文件。"""
    docs = get_documents_by_game(game_id)
    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "version": d.version,
                "language": d.language,
                "page_count": d.page_count,
                "processing_status": d.processing_status.value,
                "created_at": d.created_at.isoformat(),
            }
            for d in docs
        ]
    }


@router.post("/games/{game_id}/rulebooks", response_model=ImportResult)
async def api_import_rulebook(
    game_id: str,
    file: UploadFile = File(...),
    version: Optional[str] = None,
    language: str = "en",
    x_knowledge_admin: str = Header(default=""),
    x_knowledge_admin_timestamp: str = Header(default=""),
    x_knowledge_admin_signature: str = Header(default=""),
):
    """
    匯入規則書 PDF。

    流程：
    Upload PDF → Extract Text → Detect Pages → Detect Sections → Create Chunks → AI Extraction → Create RuleNodes → Validation → Review Queue
    """
    authorize_rulebook_write(
        game_id, "POST", "rulebooks",
        x_knowledge_admin, x_knowledge_admin_timestamp, x_knowledge_admin_signature,
    )
    # 檢查遊戲存在
    game = get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="遊戲不存在")

    # 檢查檔案類型
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只支援 PDF 檔案")

    # 儲存上傳的 PDF 到暫存目錄
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = import_rulebook(
            game_id=game_id,
            pdf_path=tmp_path,
            filename=file.filename,
            version=version,
            language=language,
        )
        return result
    except Exception as e:
        logger.error(f"❌ 匯入失敗: {e}")
        raise HTTPException(status_code=500, detail=f"匯入失敗: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/documents/{doc_id}/status")
async def api_document_status(doc_id: str):
    """取得文件處理狀態。"""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="文件不存在")
    return {
        "document_id": doc.id,
        "game_id": doc.game_id,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "processing_status": doc.processing_status.value,
        "created_at": doc.created_at.isoformat(),
    }


# ── RuleNode API ────────────────────────────────────────────────────

@router.get("/games/{game_id}/rules")
async def api_list_rules(
    game_id: str,
    category: Optional[str] = None,
    verification_status: Optional[str] = None,
    limit: int = 100,
):
    """列出遊戲的規則。"""
    rules = get_rule_nodes_by_game(
        game_id,
        category=category,
        verification_status=verification_status,
        limit=limit,
    )
    return {
        "rules": [
            {
                "id": r.id,
                "category": r.category.value,
                "type": r.type.value,
                "title": r.title,
                "rule_text": r.rule_text,
                "simplified_text": r.simplified_text,
                "confidence": r.confidence.value,
                "verification_status": r.verification_status.value,
                "source": {
                    "page": r.source.page,
                    "paragraph": r.source.paragraph,
                    "document_id": r.source.document_id,
                    "section": r.source.section,
                },
            }
            for r in rules
        ]
    }


@router.get("/rules/{rule_id}")
async def api_get_rule(rule_id: str):
    """取得單一規則詳細資訊。"""
    rule = get_rule_node(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="規則不存在")
    return rule.model_dump()


@router.get("/games/{game_id}/rules/stats")
async def api_rule_stats(game_id: str):
    """取得遊戲的規則統計。"""
    stats = get_rule_stats(game_id)
    return stats


# ── Rule Retrieval API ──────────────────────────────────────────────

@router.post("/games/{game_id}/rules/search", response_model=RuleSearchResponse)
async def api_search_rules(game_id: str, req: RuleSearchRequest):
    """規則檢索。"""
    results = search_rules(
        game_id=game_id,
        query=req.query,
        limit=req.limit,
    )
    return RuleSearchResponse(rules=results)


@router.post("/games/{game_id}/rules/search-with-exceptions")
async def api_search_rules_with_exceptions(game_id: str, req: RuleSearchRequest):
    """規則檢索（含例外）。"""
    result = search_with_exceptions(
        game_id=game_id,
        query=req.query,
        limit=req.limit,
    )
    return result


# ── AI Q&A API ──────────────────────────────────────────────────────

@router.post("/games/{game_id}/qa", response_model=QAResponse)
async def api_qa(game_id: str, req: QARequest):
    """AI 規則問答。"""
    game = get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="遊戲不存在")

    response = answer_question(game_id, req.question)
    return response


# ── AI Teaching API ─────────────────────────────────────────────────

@router.post("/games/{game_id}/teaching/session", response_model=TeachingSessionResponse)
async def api_teaching_session(game_id: str, req: TeachingSessionRequest):
    """建立 AI 教學 Session。"""
    game = get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="遊戲不存在")

    session = create_teaching_session(game_id)
    return TeachingSessionResponse(
        session_id=session["session_id"],
        script=session["script"],
    )


# ── Review System API ───────────────────────────────────────────────

@router.get("/games/{game_id}/review")
async def api_review_queue(game_id: str):
    """取得 Review Queue（依優先度排序）。"""
    items = get_review_priority(game_id)
    return {"items": items}


@router.post("/rules/{rule_id}/review")
async def api_review_rule(rule_id: str, action: str):
    """
    人工 Review 規則。

    action: "verified" | "edit" | "ambiguous" | "reject"
    """
    try:
        rule = review_rule(rule_id, action)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if rule is None:
        raise HTTPException(status_code=404, detail="規則不存在")

    return {
        "rule_id": rule.id,
        "verification_status": rule.verification_status.value,
    }
