"""
ingest.py — Board Game Rules PDF Ingestion Pipeline (v3 - Ultimate)

負責：
1. 從 /data 目錄讀取 PDF 規則書
2. 使用 PyMuPDF (fitz) 進行進階 PDF 解析
   - 自動處理雙欄／多欄排版
   - 清除頁碼、頁首頁尾等雜訊
3. 建立 LlamaIndex Document + Metadata（檔名作為 game_name）
4. 使用 bge-m3 高品質 Embedding 寫入 ChromaDB 向量資料庫（本地持久化）
5. 同時儲存文字語料庫至 corpus.json，供查詢階段 BM25 Hybrid Search 使用
"""

import json
import os
import re
import sys
import logging
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

# 確保專案根目錄在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── 設定 ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 集中設定
from src.config import (
    BASE_DIR, DATA_DIR, DB_DIR, CORPUS_FILE,
    COLLECTION_NAME, EMBED_MODEL_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

# ── 頁碼/雜訊過濾正則 ──────────────────────────────────────────────
# 常見頁碼模式：Page 1/20, 1 / 20, - 1 -, [1], {1}, (1), 1/20, p.1, p1
PAGE_NUMBER_PATTERNS = [
    re.compile(r'^\s*(?:page|pg|p)?\.?\s*\d+\s*(?:/\s*\d+)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*[-–—]\s*\d+\s*[-–—]\s*$'),
    re.compile(r'^\s*\[\s*\d+\s*\]\s*$'),
    re.compile(r'^\s*\{\s*\d+\s*\}\s*$'),
    re.compile(r'^\s*\(\s*\d+\s*\)\s*$'),
    re.compile(r'^\s*\d+\s*/\s*\d+\s*$'),
    re.compile(r'^\s*[-–—]\s*\d+\s*[-–—]\s*$'),
]

# 常見頁首/頁尾模式（版權、標題等）
HEADER_FOOTER_PATTERNS = [
    re.compile(r'^\s*(copyright|©|all\s+rights?\s+reserved)', re.IGNORECASE),
    re.compile(r'^\s*hegemony\s*(lead\s+(your\s+)?class\s+to\s+victory)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*lead\s+your\s+class\s+to\s+victory\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d+\s*$'),  # standalone numbers
]

# 連續重複行偵測（同一個短語出現太多次）
REPEATED_LINE_THRESHOLD = 5  # 同一行出現超過 5 次視為模板


def _is_noise_line(line: str) -> bool:
    """判斷一行文字是否為雜訊（頁碼、頁首頁尾等）。"""
    stripped = line.strip()
    if not stripped or len(stripped) <= 2:
        return True

    # 檢查頁碼模式
    for pattern in PAGE_NUMBER_PATTERNS:
        if pattern.match(stripped):
            return True

    # 檢查頁首頁尾模式
    for pattern in HEADER_FOOTER_PATTERNS:
        if pattern.match(stripped):
            return True

    return False


def _clean_text(text: str) -> str:
    """清理提取的文字：移除雜訊行。"""
    lines = text.split('\n')
    cleaned = [line for line in lines if not _is_noise_line(line)]
    return '\n'.join(cleaned)


def _detect_and_remove_repeated_lines(pages_text: list[str]) -> list[str]:
    """偵測並移除出現在每一頁的重複行（頁首/頁尾）。

    如果某行文字在超過一定比例的頁面中出現，視為模板文字並移除。
    """
    if len(pages_text) <= 2:
        return pages_text

    # 統計每行出現的頁面數
    line_page_count = {}
    for page_text in pages_text:
        seen_in_this_page = set()
        for line in page_text.split('\n'):
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                if stripped not in seen_in_this_page:
                    line_page_count[stripped] = line_page_count.get(stripped, 0) + 1
                    seen_in_this_page.add(stripped)

    # 找出出現在超過 60% 頁面的行（模板）
    threshold = max(2, len(pages_text) * 0.6)
    repeated_lines = {line for line, count in line_page_count.items() if count >= threshold}

    if repeated_lines:
        logger.info(f"   🔄 偵測到 {len(repeated_lines)} 個重複行（頁首/頁尾），將移除")

    # 從每頁中移除這些行
    cleaned_pages = []
    for page_text in pages_text:
        lines = page_text.split('\n')
        cleaned_lines = [l for l in lines if l.strip() not in repeated_lines]
        cleaned_pages.append('\n'.join(cleaned_lines))

    return cleaned_pages


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    使用 PyMuPDF 提取 PDF 文字。
    自動處理雙欄 / 多欄排版 + 清除頁碼/頁首/頁尾雜訊。
    """
    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]
        text_blocks.sort(key=lambda b: (b[1], b[0]))
        page_text = "\n".join(b[4].strip() for b in text_blocks if b[4].strip())
        page_text = _clean_text(page_text)
        if page_text:
            pages_text.append(page_text)

    doc.close()

    # 移除跨頁重複的頁首/頁尾
    pages_text = _detect_and_remove_repeated_lines(pages_text)

    return "\n\n".join(pages_text)


def create_documents_from_pdfs(data_dir: Path) -> list[Document]:
    """掃描 data_dir 下所有 PDF，產生 LlamaIndex Document（含 Metadata）。"""
    preprocessed_dir = data_dir / "preprocessed"
    md_files = list(preprocessed_dir.glob("*.md")) if preprocessed_dir.exists() else []

    if md_files:
        logger.info(f"📝 找到 {len(md_files)} 個預處理 Markdown 檔案，將優先使用")
        documents = []
        for md_path in tqdm(md_files, desc="📖 讀取 Markdown"):
            game_name = md_path.stem
            logger.info(f"  正在處理: {md_path.name} (game_name={game_name})")
            text = md_path.read_text(encoding="utf-8")
            if not text.strip():
                logger.warning(f"  ⚠️ {md_path.name} 內容為空，跳過")
                continue
            doc = Document(
                text=text,
                metadata={
                    "game_name": game_name,
                    "file_name": f"{game_name}.md",
                    "file_path": str(md_path),
                    "source": "preprocessed",
                },
            )
            documents.append(doc)
        logger.info(f"✅ 共建立 {len(documents)} 個 Document（來自預處理 Markdown）")
        return documents

    # 備援：使用原始 PDF（改善版解析）
    logger.info("📄 未找到預處理 Markdown，回退到原始 PDF 解析（已啟用雜訊過濾）")
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("❌ /data 目錄下沒有找到任何 PDF 檔案！")
        sys.exit(1)

    documents = []
    for pdf_path in tqdm(pdf_files, desc="📄 解析 PDF"):
        game_name = pdf_path.stem
        logger.info(f"  正在處理: {pdf_path.name} (game_name={game_name})")
        try:
            text = extract_text_from_pdf(str(pdf_path))
            if not text.strip():
                logger.warning(f"  ⚠️ {pdf_path.name} 提取內容為空，跳過")
                continue
            doc = Document(
                text=text,
                metadata={
                    "game_name": game_name,
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    "source": "raw_pdf",
                },
            )
            documents.append(doc)
        except Exception as e:
            logger.error(f"  ❌ 解析 {pdf_path.name} 失敗: {e}")
            continue

    logger.info(f"✅ 共建立 {len(documents)} 個 Document")
    return documents


def save_corpus_for_bm25(nodes: list) -> None:
    """將所有 chunk 文字儲存為 JSON，供查詢階段 BM25 使用。"""
    os.makedirs(DB_DIR, exist_ok=True)
    corpus = []
    for node in nodes:
        corpus.append({
            "doc_id": node.node_id,
            "text": node.get_content(),
            "metadata": node.metadata,
        })
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    logger.info(f"📦 BM25 語料庫已儲存: {CORPUS_FILE} ({len(corpus)} 個 chunk)")


def build_vector_index(documents: list[Document]) -> VectorStoreIndex:
    """建立向量索引並寫入 ChromaDB（使用 bge-m3 高品質 Embedding）。"""
    db_dir = str(DB_DIR)
    os.makedirs(db_dir, exist_ok=True)
    db = chromadb.PersistentClient(path=db_dir)
    
    # 檢查現有 collection 的 embedding 維度是否匹配
    try:
        existing_collection = db.get_collection(COLLECTION_NAME)
        # 嘗試用一個測試 embedding 來檢查維度
        test_embedding = [0.0] * 384  # multilingual-e5-small 的維度
        existing_collection.add(ids=["__dimension_test__"], embeddings=[test_embedding], documents=[""])
        existing_collection.delete(ids=["__dimension_test__"])
        logger.info(f"   ✅ 現有 collection 維度匹配，繼續使用")
        chroma_collection = existing_collection
    except Exception as e:
        if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
            logger.warning(f"   ⚠️ 現有 collection embedding 維度不符，將重建 collection")
            db.delete_collection(COLLECTION_NAME)
            chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
        else:
            raise
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info(f"🧠 載入 Embedding 模型: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    logger.info("✂️ 正在切分 documents 為 chunks...")
    all_nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
    logger.info(f"   ✅ 共 {len(all_nodes)} 個 chunks")

    if all_nodes:
        save_corpus_for_bm25(all_nodes)

    index = VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    logger.info(f"✅ 索引建立完成！已寫入 {db_dir}")
    logger.info(f"   Embedding 模型: {EMBED_MODEL_NAME}")
    logger.info(f"   Chunk 大小: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    return index


def main():
    logger.info("🚀 開始 Board Game Rules Ingestion Pipeline (v3)")
    logger.info(f"   Embedding 模型: {EMBED_MODEL_NAME}")
    logger.info(f"   Chunk 大小: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    logger.info(f"   雜訊過濾: 已啟用（頁碼、頁首頁尾、重複行）")
    documents = create_documents_from_pdfs(DATA_DIR)
    if not documents:
        logger.error("❌ 沒有任何有效的 Document 可供索引，結束。")
        sys.exit(1)
    build_vector_index(documents)
    logger.info("🎉 Ingestion 完成！")


if __name__ == "__main__":
    main()