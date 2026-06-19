"""
Board Game RAG Engine - Web API Server (v3 - Ultimate)

以 FastAPI 提供 Web 介面，讓使用者可以在瀏覽器中查詢桌遊規則。
v3 改進：
  - Hybrid Search（向量 + BM25 + RRF 融合）
  - Query Expansion（桌遊同義詞擴展）
  - Cross-encoder Re-ranking
  - MMR 多樣性排序
  - Adaptive Hybrid Weights（短查詢偏 BM25，長查詢偏向量）
  - Noise Chunk Filtering（過濾版權頁、頁碼等雜訊）
  - 改善 LLM Prompt

本檔案為 Web 層，核心邏輯委託給 src/query.py，避免重複程式碼。
"""
import json
import logging
import os
import sys
from pathlib import Path

# 確保專案根目錄在 sys.path 中
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# 集中設定
from src.config import DB_DIR, COLLECTION_NAME, EMBED_MODEL_NAME, MMR_TOP_K

# ── 匯入共用核心函式（避免重複程式碼） ──
from src.query import (
    load_index,
    get_bm25_engine,
    get_reranker,
    expand_query,
    hybrid_search,
    reciprocal_rank_fusion,
    _get_adaptive_weights,
    _is_noise_chunk,
    mmr_diversity_ranking,
    rerank_with_cross_encoder,
    CHINESE_QA_PROMPT,
    get_available_games,
)
from src.config import (
    SIMILARITY_TOP_K,
    HYBRID_WEIGHT_VECTOR,
    HYBRID_WEIGHT_BM25,
    RRF_K,
    MMR_LAMBDA,
    FILTER_NOISE_CHUNKS,
    USE_RE_RANKER,
)

# 翻譯模組
from src.translator import translate_to_traditional_chinese

# ── 設定 ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CORPUS_FILE = DB_DIR / "corpus.json"

# ── 全域 ────────────────────────────────────────────────────────────
index = None

# ── Pydantic 模型 ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str
    game: str | None = None
    top_k: int = MMR_TOP_K

class QueryResult(BaseModel):
    rank: int
    score: float
    content: str

class QueryResponse(BaseModel):
    results: list[QueryResult]

class GameListResponse(BaseModel):
    games: list[str]


# ── FastAPI ─────────────────────────────────────────────────────────
app = FastAPI(title="Board Game Rules RAG Engine v3")


@app.on_event("startup")
async def startup():
    global index
    try:
        index = load_index()
        # 預先載入 BM25 與 Re-ranker，確保 Web 伺服器啟動後回應快速
        get_bm25_engine()
        get_reranker()
        logger.info("✅ 啟動完成，向量索引已載入")
    except SystemExit:
        logger.warning("⚠️ 啟動時索引載入失敗：找不到 db/ 目錄，請先執行 ingest 建立索引")
        index = None
    except RuntimeError as e:
        logger.warning(f"⚠️ 啟動時索引載入失敗: {e}")
        index = None
    except Exception as e:
        logger.warning(f"⚠️ 啟動時發生未預期錯誤: {e}")
        index = None


@app.get("/api/games", response_model=GameListResponse)
async def api_games():
    return GameListResponse(games=get_available_games())


@app.post("/api/query", response_model=QueryResponse)
async def api_query(req: QueryRequest):
    if index is None:
        raise HTTPException(status_code=503, detail="索引尚未載入")

    use_llm = bool(os.environ.get("OPENAI_API_KEY"))
    filters = None
    if req.game:
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        filters = MetadataFilters(filters=[ExactMatchFilter(key="game_name", value=req.game)])

    # ── Step 1: Query Expansion ──
    expanded_queries = expand_query(req.query)
    logger.info(f"🔍 查詢擴展: {req.query} → {len(expanded_queries)} 個變體")

    # ── Step 2: Hybrid Search（含自適應權重 + 所有擴展查詢變體） ──
    retriever = index.as_retriever(similarity_top_k=SIMILARITY_TOP_K, filters=filters)
    bm25 = get_bm25_engine()

    all_fused = []
    seen_ids = set()
    for eq in expanded_queries:
        fused = hybrid_search(retriever, eq, bm25, SIMILARITY_TOP_K, req.game)
        for item in fused:
            doc_id = item.get("node", None) and item["node"].node_id or item.get("bm25_text", "")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_fused.append(item)

    # 依 RRF 分數排序
    all_fused.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    logger.info(f"   📊 共 {len(all_fused)} 個不重複候選")

    # ── Step 3: 過濾雜訊 chunk ──
    if FILTER_NOISE_CHUNKS:
        before = len(all_fused)
        all_fused = [
            item for item in all_fused
            if not _is_noise_chunk(
                item.get("node", None) and item["node"].get_content() or item.get("bm25_text", "")
            )
        ]
        after = len(all_fused)
        if before > after:
            logger.info(f"   🗑️ 過濾 {before - after} 個雜訊 chunk")

    # ── Step 4: Cross-encoder Re-ranking ──
    if USE_RE_RANKER:
        logger.info("🏆 執行 Cross-encoder Re-ranking...")
        reranked = rerank_with_cross_encoder(all_fused, req.query, top_k=SIMILARITY_TOP_K)
    else:
        reranked = all_fused

    # ── Step 5: MMR Diversity ──
    diversified = mmr_diversity_ranking(reranked, None, lambda_param=MMR_LAMBDA, top_k=req.top_k)

    if use_llm:
        # ── LLM 模式 ──
        logger.info(f"💬 查詢: \"{req.query}\" (LLM=開啟)")
        context_parts = []
        for item in diversified:
            if item.get("node"):
                context_parts.append(item["node"].get_content())
            else:
                context_parts.append(item.get("bm25_text", ""))
        context_str = "\n\n---\n\n".join(context_parts)

        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        prompt = CHINESE_QA_PROMPT.format(context_str=context_str, query_str=req.query)
        response = llm.complete(prompt)
        results = [QueryResult(rank=1, score=1.0, content=str(response))]
    else:
        # ── 純檢索模式 ──
        logger.info(f"💬 查詢: \"{req.query}\" (LLM=關閉)")
        results = []
        rank = 1
        for item in diversified:
            if item.get("node"):
                content = item["node"].get_content()[:600]
                score = item.get("rerank_score", item.get("rrf_score", item.get("score", 0)))
            else:
                content = item.get("bm25_text", "")[:600]
                score = item.get("rerank_score", item.get("bm25_score", 0))
            translated = translate_to_traditional_chinese(content)
            results.append(QueryResult(rank=rank, score=round(float(score), 4), content=translated))
            rank += 1

    return QueryResponse(results=results)


# ── 靜態檔案與前端 ─────────────────────────────────────────────────
templates_dir = BASE_DIR / "templates"
templates_dir.mkdir(exist_ok=True)


@app.get("/health")
async def health_check():
    return {"status": "ok", "index_loaded": index is not None}


@app.get("/", response_class=HTMLResponse)
async def index_page():
    html_file = templates_dir / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>Board Game RAG Engine v3</h1><p>前端尚未建立。</p>")
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


def main():
    print("🎲 Board Game RAG Engine v3 - Ultimate")
    print(f"   Embedding: {EMBED_MODEL_NAME}")
    print(f"   Hybrid Search + Query Expansion + Re-ranker + MMR + Adaptive Weights + Noise Filter")
    print(f"   http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()