"""
import_service.py — PDF Import Service

負責完整的 PDF 匯入管線：
Upload PDF → Extract Text → Detect Pages → Detect Sections → Create Chunks → AI Extraction → Create RuleNodes → Validation → Review Queue

遵循 Implementation Plan 的 PDF Import Pipeline。
"""
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.database import (
    create_document,
    complete_rulebook_import,
    get_rule_nodes_by_game,
    create_rule_nodes_bulk,
    get_document,
    get_rule_stats,
    update_document_status,
)
from src.gemini_pdf import extract_pdf_rules, PendingExtraction
from src.models import (
    ImportResult,
    ProcessingStatus,
    RuleDocument,
)
from src.pdf_parser import get_pdf_page_count, parse_pdf
from src.storage import store_rulebook
from src.validator import validate_rules
from src.models import ReviewItem, ReviewAction, VerificationStatus

logger = logging.getLogger(__name__)


def import_rulebook(
    game_id: str,
    pdf_path: str,
    filename: str,
    version: Optional[str] = None,
    language: str = "en",
    stored_path: Optional[str] = None,
    document_id: Optional[str] = None,
    max_pages: Optional[int] = None,
) -> ImportResult:
    """
    匯入規則書 PDF。

    Args:
        game_id: 遊戲 ID
        pdf_path: PDF 檔案路徑
        filename: 原始檔名
        version: 版本
        language: 語言

    Returns:
        ImportResult: 匯入結果統計
    """
    # 1. 建立 RuleDocument
    doc_id = document_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    existing = get_document(doc_id) if document_id else None
    if document_id:
        if not existing or existing.game_id != game_id or existing.file_path != stored_path:
            raise ValueError("Invalid document resume reference")
        if existing.processing_status == ProcessingStatus.PROCESSED:
            rules = [r for r in get_rule_nodes_by_game(game_id, limit=100000) if r.document_id == doc_id]
            return summarize_import(doc_id, rules)
    page_count = get_pdf_page_count(pdf_path)

    # Store the original PDF durably. Parsing still uses the request's temporary
    # local file, which is available in both local development and Cloud Run.
    stored_path = stored_path or store_rulebook(pdf_path, game_id, doc_id, filename)

    doc = RuleDocument(
        id=doc_id,
        game_id=game_id,
        filename=filename,
        version=version,
        language=language,
        file_path=stored_path,
        page_count=page_count,
        processing_status=ProcessingStatus.PROCESSING,
    )
    if not existing:
        create_document(doc)

    try:
        rules = extract_pdf_rules(pdf_path, doc_id, game_id, max_pages=max_pages)
        results = validate_rules(rules)
        if any(not result.is_valid for result in results):
            raise ValueError("規則來源或結構檢查失敗，未發布資料")
        reviews = []
        for result in results:
            if result.needs_review:
                result.rule.verification_status = VerificationStatus.AMBIGUOUS
                reviews.append(ReviewItem(id=f"REVIEW-{uuid.uuid4().hex[:8].upper()}",
                    rule_id=result.rule.id, game_id=game_id,
                    reason="; ".join(result.review_reasons), status=ReviewAction.AMBIGUOUS))
        complete_rulebook_import(doc_id, rules, reviews)
        return summarize_import(doc_id, rules)
    except PendingExtraction as pending:
        return {"document_id": doc_id, "processing_status": "processing",
                "completed_pages": pending.completed, "page_count": pending.total}

    except Exception as e:
        logger.error(f"❌ 匯入失敗: {e}")
        update_document_status(doc_id, ProcessingStatus.FAILED.value)
        raise


def get_import_status(doc_id: str) -> Optional[dict]:
    """取得匯入狀態。"""
    doc = get_document(doc_id)
    if doc is None:
        return None
    return {
        "document_id": doc.id,
        "game_id": doc.game_id,
        "filename": doc.filename,
        "page_count": doc.page_count,
        "processing_status": doc.processing_status.value,
        "created_at": doc.created_at.isoformat(),
    }


def summarize_import(doc_id, rules):
    return ImportResult(document_id=doc_id, rules_extracted=len(rules),
        explicit=sum(r.confidence.value == "explicit" for r in rules),
        derived=sum(r.confidence.value == "derived" for r in rules),
        ambiguous=sum(r.confidence.value == "ambiguous" for r in rules),
        conflict=sum(r.confidence.value == "conflict" for r in rules),
        need_review=sum(r.verification_status.value == "ambiguous" for r in rules))
