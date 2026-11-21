"""
pdf_parser.py — PDF 解析與 Chunking

負責：
1. 從 PDF 提取文字（每頁保留）
2. 偵測章節（Section Detection）
3. 建立 Chunk（保留 document_id, page_number, section, raw_text）
4. 供 AI Extraction 使用

遵循 Implementation Plan 的 PDF Parsing Strategy：
PDF → Page → Section → Chunk → Rule Extraction
"""
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ── 資料結構 ────────────────────────────────────────────────────────

@dataclass
class PageText:
    """單頁文字。"""
    page_number: int
    text: str


@dataclass
class Section:
    """章節。"""
    title: str
    page_number: int
    text: str


@dataclass
class Chunk:
    """文字 Chunk — 保留來源資訊。"""
    document_id: str
    page_number: int
    section: str
    raw_text: str
    chunk_index: int = 0


# ── 章節標題偵測 ────────────────────────────────────────────────────

# 常見章節標題模式
SECTION_PATTERNS = [
    # 英文：全大寫或標題大小寫
    re.compile(r"^\s*([A-Z][A-Z\s&'\-]{3,})\s*$"),
    re.compile(r"^\s*(\d+\.?\s+[A-Z][A-Za-z\s&'\-]{3,})\s*$"),
    re.compile(r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,})\s*$"),
    # 中文章節
    re.compile(r"^\s*([一二三四五六七八九十]+[、.．]\s*.{2,20})\s*$"),
    re.compile(r"^\s*(\d+[、.．]\s*.{2,20})\s*$"),
    # 常見規則書章節關鍵字
    re.compile(
        r"^\s*(?:Game\s+Setup|Setup|Objective|Components|Game\s+Flow|Turn\s+Structure|Actions?|Scoring|End\s+of\s+Game|FAQ|Rules?)\s*$",
        re.IGNORECASE,
    ),
]

# 已知的章節標題（常見於規則書）
KNOWN_SECTIONS = [
    "game setup", "setup", "objective", "components", "game flow",
    "turn structure", "actions", "action", "scoring", "end of game",
    "end game", "faq", "rules", "rulebook", "introduction", "overview",
    "game overview", "player turn", "combat", "cards", "tiles",
    "tokens", "resources", "victory", "winning", "game end",
]

# 雜訊行模式（頁碼、頁首頁尾）
NOISE_PATTERNS = [
    re.compile(r"^\s*(?:page|pg|p)?\.?\s*\d+\s*(?:/\s*\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$"),
    re.compile(r"^\s*\[\s*\d+\s*\]\s*$"),
    re.compile(r"^\s*\{\s*\d+\s*\}\s*$"),
    re.compile(r"^\s*\(\s*\d+\s*\)\s*$"),
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*(copyright|©|all\s+rights?\s+reserved)", re.IGNORECASE),
]


def _is_noise_line(line: str) -> bool:
    """判斷是否為雜訊行。"""
    stripped = line.strip()
    if not stripped or len(stripped) <= 2:
        return True
    for pattern in NOISE_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def _is_section_title(line: str) -> Optional[str]:
    """判斷一行是否為章節標題，若是則回傳標題文字。"""
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None

    # 檢查已知章節
    lower = stripped.lower()
    for known in KNOWN_SECTIONS:
        if lower == known:
            return stripped

    # 檢查模式
    for pattern in SECTION_PATTERNS:
        match = pattern.match(stripped)
        if match:
            return stripped

    return None


# ── PDF 解析 ────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[PageText]:
    """從 PDF 提取每頁文字。"""
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]
        text_blocks.sort(key=lambda b: (b[1], b[0]))
        page_text = "\n".join(b[4].strip() for b in text_blocks if b[4].strip())

        # 清理雜訊行
        lines = page_text.split("\n")
        cleaned_lines = [line for line in lines if not _is_noise_line(line)]
        cleaned_text = "\n".join(cleaned_lines)

        if cleaned_text.strip():
            pages.append(PageText(page_number=page_num + 1, text=cleaned_text))

    doc.close()
    logger.info(f"📄 提取 {len(pages)} 頁文字")
    return pages


def detect_sections(pages: list[PageText]) -> list[Section]:
    """Keep each page's text exactly once and never borrow another page's citation."""
    sections: list[Section] = []
    title = "Unclassified"
    for page in pages:
        buffer: list[str] = []
        for line in page.text.split("\n"):
            heading = _is_section_title(line)
            if heading:
                if buffer:
                    sections.append(Section(title, page.page_number, "\n".join(buffer)))
                    buffer = []
                title = heading
            else:
                buffer.append(line)
        if any(line.strip() for line in buffer):
            sections.append(Section(title, page.page_number, "\n".join(buffer)))
    return sections


def create_chunks(
    sections: list[Section], document_id: str,
    chunk_size: int = 800, chunk_overlap: int = 100,
) -> list[Chunk]:
    """Use complete paragraphs; size is a soft target, never a reason to cut text.

    chunk_overlap is retained for caller compatibility. We do not duplicate
    partial sentences as overlap: source paragraphs are emitted exactly once.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    chunks: list[Chunk] = []
    for section in sections:
        paragraphs = re.split(r"\n\s*\n", section.text.strip())
        buffer = ""
        for paragraph in paragraphs:
            paragraph = re.sub(r"(?<=\w)-\n(?=\w)", "", paragraph)
            paragraph = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph).strip()
            if not paragraph:
                continue
            if buffer and len(buffer) + len(paragraph) > chunk_size:
                chunks.append(Chunk(document_id, section.page_number, section.title, buffer, len(chunks)))
                buffer = ""
            buffer = buffer + "\n\n" + paragraph if buffer else paragraph
        if buffer:
            chunks.append(Chunk(document_id, section.page_number, section.title, buffer, len(chunks)))
    return chunks


def parse_pdf(pdf_path: str, document_id: str) -> tuple[list[PageText], list[Section], list[Chunk]]:
    """完整 PDF 解析管線：提取 → 章節偵測 → Chunk。"""
    pages = extract_pages(pdf_path)
    sections = detect_sections(pages)
    chunks = create_chunks(sections, document_id)
    return pages, sections, chunks


def get_pdf_page_count(pdf_path: str) -> int:
    """取得 PDF 頁數。"""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count