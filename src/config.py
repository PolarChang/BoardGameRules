"""
config.py — Board Game RAG Engine 集中設定檔

所有共用設定集中在這裡，避免 app.py / query.py / ingest.py 重複定義。
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
CORPUS_FILE = DB_DIR / "corpus.json"

# ── 向量資料庫 ──
COLLECTION_NAME = "rulebooks"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-small"

# ── Chunking（ingest.py） ──
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

# ── 檢索參數（query.py / app.py） ──
SIMILARITY_TOP_K = 15
HYBRID_WEIGHT_VECTOR = 0.5
HYBRID_WEIGHT_BM25 = 0.5
RRF_K = 60

# ── MMR 多樣性 ──
MMR_LAMBDA = 0.7       # 0.7 = 偏相關性，0.3 = 偏多樣性
MMR_TOP_K = 10

# ── 自適應 Hybrid Search 權重 ──
SHORT_QUERY_THRESHOLD = 5
SHORT_QUERY_BM25_WEIGHT = 0.7
LONG_QUERY_VECTOR_WEIGHT = 0.7

# ── 結果過濾 ──
FILTER_NOISE_CHUNKS = True
NOISE_PATTERNS = [
    r'copyright\s*©',
    r'all\s+rights?\s+reserved',
    r'^\s*\d+\s*$',
    r'^\s*page\s+\d+',
    r'^\s*hegemony\s*(lead|rule)',
    r'lead\s+(your\s+)?class\s+to\s+victory',
    r'^\s*components?\s*$',
    r'(hegemony|rulebook|rules)\s*v?\d+\.?\d*',
    # 元件列表（多行連續的 "數量 項目名稱" 模式）
    r'^\d+\s+(GAME\s+)?BOARD',
    r'^\d+\s+PLAYER\s+BOARDS',
    r'^\d+\s+REGULAR\s+CARDS',
    r'^\d+\s+ACTION\s+CARDS?',
    r'^\d+\s+(CAPITALIST|WORKING|MIDDLE|STATE)\s',
    r'^\d+\s+COMPANY',
    r'^\d+\s+PLAYER\s+AIDS',
    r'^\d+\s+RULE\s+AIDS',
    r'^\d+\s+(COOPERATIVE|EXPORT|EVENT|POLITICAL)',
]

# 最小 chunk 長度（少於此字數視為無效內容）
MIN_CHUNK_CHARS = 100

# ── Cross-encoder Re-ranker ──
USE_RE_RANKER = False  # 關閉 Re-ranker 以節省記憶體
RE_RANKER_MODEL = "BAAI/bge-reranker-v2-m3"
