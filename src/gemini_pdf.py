"""Reusable, page-aware bilingual PDF extraction using Gemini native PDF input.

Only complete page sets are returned to the importer. Checkpoints contain drafts,
never published rules; the original PDF remains the source of truth.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
import threading
import urllib.error
import urllib.request
import uuid

import fitz
from pydantic import BaseModel, Field

from src.gemini_usage import generate, setting
from src.models import RuleCategory, RuleType, RuleNode, SourceReference, Confidence, VerificationStatus

logger = logging.getLogger(__name__)
PIPELINE_VERSION = 'bilingual-pdf-v2'
PDF_LOCK = threading.Lock()


class PendingExtraction(Exception):
    def __init__(self, completed: int, total: int):
        self.completed, self.total = completed, total
        super().__init__(f'{completed}/{total} pages completed')


class PDFRule(BaseModel):
    anchor: str
    title: str
    category: RuleCategory
    type: RuleType
    rule_text: str
    simplified_text: str
    section: str
    page_end: int
    uncertain: bool


class PDFPage(BaseModel):
    page: int
    rules: list[PDFRule]
    excluded_content: str
    warnings: list[str]


def check_page(payload: dict, page: int, total: int) -> PDFPage:
    result = PDFPage.model_validate(payload)
    if result.page != page:
        raise ValueError('Incorrect PDF page reference')
    for rule in result.rules:
        if not page <= rule.page_end <= min(page + 1, total):
            raise ValueError('Invalid source page range')
        if len(rule.rule_text.strip()) < 10:
            raise ValueError('Empty or fragmentary original text')
        for value in (rule.title, rule.simplified_text, rule.section):
            if not re.search(r'[\u3400-\u9fff]', value):
                raise ValueError('Missing Chinese content')
    if not result.rules and not result.excluded_content.strip():
        raise ValueError('Empty page result without explanation')
    return result


def normalize_anchor(value: str) -> str:
    value = re.sub(r'(?<=\w)-\s+(?=\w)', '', value)
    return re.sub(r'[^\w]+', ' ', value.casefold()).strip()


def request_page(pdf_bytes: bytes, page: int, first: int, last: int, total: int, context_bytes: bytes, page_text: str, game_id: str = '') -> PDFPage:
    raise RuntimeError('Gemini has been removed from Boardgame; cloud inference is disabled.')


def extract_pdf_rules(pdf_path: str, document_id: str, game_id: str, *, cache_dir: str | None = None, pages: list[int] | None = None, max_pages: int | None = None) -> list[RuleNode]:
    """Process all file pages; pages is for offline QA only, never partial publishing."""
    from src.config import DATA_DIR, get_cloud_settings
    model = setting('RULE_EXTRACTION_MODEL', 'gemini-3.8-flash')
    digest = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    cache_key = hashlib.sha256(f'{digest}:{model}:{PIPELINE_VERSION}'.encode()).hexdigest()
    directory = Path(cache_dir or DATA_DIR / 'extraction-cache') / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    bucket = None
    settings = get_cloud_settings()
    if settings.uses_cloud_storage and not cache_dir:
        from google.cloud import storage
        bucket = storage.Client().bucket(settings.storage_bucket)
    with fitz.open(pdf_path) as source:
        total = len(source)
    targets = list(range(1, total + 1)) if pages is None else pages
    if any(p < 1 or p > total for p in targets):
        raise ValueError('Invalid target page')
    if max_pages is not None:
        cached, missing = [], []
        remote_names = {b.name for b in bucket.list_blobs(prefix=f'extraction-cache/{cache_key}/')} if bucket else set()
        for page in targets:
            if (directory / f'{page:04d}.json').exists() or f'extraction-cache/{cache_key}/{page:04d}.json' in remote_names:
                cached.append(page)
            else:
                missing.append(page)
        targets = cached + missing[:max(1, max_pages)]

    def process(page: int) -> PDFPage:
        path = directory / f'{page:04d}.json'
        blob = bucket.blob(f'extraction-cache/{cache_key}/{page:04d}.json') if bucket else None
        if not path.exists() and blob and blob.exists():
            blob.download_to_filename(path)
        if path.exists():
            return check_page(json.loads(path.read_text(encoding='utf-8')), page, total)
        first, last = max(1, page - 1), min(total, page + 1)
        # Each worker opens its own document; no shared mutable PyMuPDF objects.
        with PDF_LOCK, fitz.open(pdf_path) as source, fitz.open() as excerpt, fitz.open() as primary:
            excerpt.insert_pdf(source, from_page=first - 1, to_page=last - 1)
            context_bytes = excerpt.tobytes(garbage=4, deflate=True)
            primary.insert_pdf(source, from_page=page-1, to_page=page-1)
            pdf_bytes = primary.tobytes(garbage=4, deflate=True)
            page_text = source[page-1].get_text()
        result = request_page(pdf_bytes, page, first, last, total, context_bytes, page_text, game_id=game_id)
        encoded = result.model_dump_json(indent=2)
        temp = path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
        temp.write_text(encoded, encoding='utf-8')
        temp.replace(path)
        if blob:
            blob.upload_from_string(encoded, content_type='application/json')
        return result

    results = {}
    workers = max(1, min(int(os.environ.get('RULE_EXTRACTION_WORKERS', '3')), 4))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(process, page): page for page in targets}
        for future in as_completed(pending):
            page = pending[future]
            results[page] = future.result()
            logger.info('PDF page %s/%s complete: %s rules', page, total, len(results[page].rules))
    if pages is None and len(results) < total:
        raise PendingExtraction(len(results), total)
    rules = []
    seen = set()
    for page in sorted(results):
        for item in results[page].rules:
            normalized = re.sub(r'\s+', ' ', item.rule_text).strip().casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            uncertain = item.uncertain
            rules.append(RuleNode(
                id=f'RULE-{uuid.uuid4().hex[:8].upper()}', game_id=game_id, document_id=document_id,
                title=item.title, category=item.category, type=item.type,
                rule_text=item.rule_text, simplified_text=item.simplified_text,
                source=SourceReference(document_id=document_id, page=page, section=item.section,
                    paragraph=f'PDF {page}–{item.page_end}' if item.page_end != page else None),
                confidence=Confidence.AMBIGUOUS if uncertain else Confidence.EXPLICIT,
                verification_status=VerificationStatus.AMBIGUOUS if uncertain else VerificationStatus.UNVERIFIED,
            ))
    if not rules:
        raise ValueError('PDF 中未找到完整可用規則；未發布資料')
    return rules
