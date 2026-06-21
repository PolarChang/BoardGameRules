"""
query.py — Board Game Rules Query Engine (v3 - Ultimate)

負責：
1. 從 CLI 接收查詢文字與可選的 game_name 過濾條件
2. 載入 ChromaDB 向量索引 + BM25 語料庫
3. 執行 Hybrid Search（向量相似度 + BM25 關鍵字匹配）
4. 使用 Reciprocal Rank Fusion (RRF) 合併結果
5. Cross-encoder Re-ranker 重新排序
6. MMR 多樣性排序
7. Query Expansion 桌遊同義詞擴展
8. 支援 Metadata Filter（依 game_name 指定遊戲）
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

# 確保專案根目錄在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import chromadb
import jieba
import numpy as np
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core import PromptTemplate
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.chroma import ChromaVectorStore
from rank_bm25 import BM25Okapi

# 集中設定
from src.config import (
    BASE_DIR, DB_DIR, CORPUS_FILE,
    COLLECTION_NAME, EMBED_MODEL_NAME, HF_EMBED_MODEL_NAME, EMBED_TYPE,
    SIMILARITY_TOP_K, HYBRID_WEIGHT_VECTOR, HYBRID_WEIGHT_BM25, RRF_K,
    MMR_LAMBDA, MMR_TOP_K,
    SHORT_QUERY_THRESHOLD, SHORT_QUERY_BM25_WEIGHT, LONG_QUERY_VECTOR_WEIGHT,
    FILTER_NOISE_CHUNKS, NOISE_PATTERNS, MIN_CHUNK_CHARS,
    USE_RE_RANKER, RE_RANKER_MODEL,
)

# 翻譯模組
from src.translator import translate_to_traditional_chinese

# ── 設定 ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 全域快取 ──────────────────────────────────────────────────────
_reranker_model = None
_bm25_engine_cache = None


# ═══════════════════════════════════════════════════════════════════
# 1. 桌遊同義詞詞典（Query Expansion）
# ═══════════════════════════════════════════════════════════════════

BOARD_GAME_SYNONYMS = {
    # 勝利/得分
    "勝利": ["勝利", "獲勝", "贏", "win", "victory", "triumph", "VP", "分數", "得分", "計分"],
    "勝利條件": ["勝利條件", "勝利", "獲勝方式", "贏得遊戲", "game end", "win condition", "如何獲勝", "勝出"],
    "計分": ["計分", "得分", "分數", "VP", "victory point", "勝利點數", "scoring"],
    "VP": ["VP", "victory point", "勝利點數", "勝利分數", "分數", "得分"],

    # 遊戲流程
    "回合": ["回合", "輪", "turn", "round", "phase", "階段"],
    "行動": ["行動", "動作", "action", "move", "操作", "執行"],
    "設置": ["設置", "setup", "準備", "起始設置", "遊戲設置"],

    # 資源
    "資源": ["資源", "resource", "材料", "物資", "goods", "貨物"],
    "金錢": ["金錢", "錢", "money", "coin", "金幣", "資金", "貨幣", "財富"],
    "工人": ["工人", "worker", "劳工", "人力", "勞動力"],

    # 規則
    "規則": ["規則", "rule", "規範", "規定", "條例", "說明", "玩法"],
    "限制": ["限制", "limit", "constraint", "上限", "不得", "禁止"],

    # 遊戲類型
    "卡牌": ["卡牌", "卡片", "卡", "card", "牌"],
    "版圖": ["版圖", "board", "地圖", "圖板", "遊戲板"],
    "骰子": ["骰子", "dice", "骰"],

    # 特殊
    "能力": ["能力", "skill", "power", "ability", "特殊能力", "技能"],
    "效果": ["效果", "effect", "作用", "影響"],
    "階段": ["階段", "phase", "時期", "step", "步驟", "程序"],
    "政策": ["政策", "policy", "法案", "法律", "政見"],

    # 英文常用詞
    "how to": ["how to", "怎麼", "如何", "方式", "方法", "玩法"],
    "what is": ["what is", "什麼是", "說明", "解釋", "定義"],
}


# ═══════════════════════════════════════════════════════════════════
# 2. Cross-encoder Re-ranker（lazy loading + thread-safe）
# ═══════════════════════════════════════════════════════════════════

def get_reranker():
    """取得 Cross-encoder Re-ranker 模型（lazy loading，singleton）。"""
    global _reranker_model
    if _reranker_model is None and USE_RE_RANKER:
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"🧠 載入 Re-ranker 模型: {RE_RANKER_MODEL}")
            _reranker_model = CrossEncoder(RE_RANKER_MODEL)
            logger.info("✅ Re-ranker 模型載入完成")
        except Exception as e:
            logger.warning(f"⚠️ 載入 Re-ranker 失敗: {e}")
            logger.warning("   將跳過 Re-ranking 步驟")
    return _reranker_model


# ═══════════════════════════════════════════════════════════════════
# 3. Query Expansion
# ═══════════════════════════════════════════════════════════════════

def expand_query(query: str) -> list[str]:
    """查詢擴展：根據桌遊同義詞詞典產生多個查詢變體。

    例如 "勝利條件" → ["勝利條件", "win condition", "如何獲勝", "VP"]

    Args:
        query: 原始查詢

    Returns:
        擴展後的查詢列表（包含原始查詢）
    """
    expanded = [query]
    query_lower = query.lower()

    for key, synonyms in BOARD_GAME_SYNONYMS.items():
        key_lower = key.lower()
        if key_lower in query_lower:
            for syn in synonyms:
                syn_lower = syn.lower()
                if syn_lower not in query_lower:
                    # 修復：使用 flags=re.IGNORECASE 處理大小寫
                    expanded_query = re.sub(
                        re.escape(key), syn, query, flags=re.IGNORECASE
                    )
                    if expanded_query != query:
                        expanded.append(expanded_query)

    # 去重且保持順序
    unique_queries = list(dict.fromkeys(expanded))

    if len(unique_queries) > 1:
        logger.info(f"🔍 查詢擴展: {query} → {len(unique_queries)} 個變體")

    return unique_queries


# ═══════════════════════════════════════════════════════════════════
# 4. BM25 Engine（singleton）
# ═══════════════════════════════════════════════════════════════════

class BM25Engine:
    """BM25 關鍵字檢索引擎。"""

    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.corpus = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._load_corpus()

    def _load_corpus(self):
        if not self.corpus_path.exists():
            logger.warning(f"⚠️ BM25 語料庫不存在: {self.corpus_path}")
            return

        try:
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"❌ 讀取語料庫失敗: {e}")
            self.corpus = []

        if not self.corpus:
            logger.warning("⚠️ BM25 語料庫為空")
            return

        logger.info(f"🔤 建立 BM25 索引（{len(self.corpus)} 個 chunk）")
        self.tokenized_corpus = [
            list(jieba.cut(item["text"])) for item in self.corpus
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info("✅ BM25 索引建立完成")

    def search(self, query: str, top_k: int = SIMILARITY_TOP_K, game_name: str | None = None) -> list[dict]:
        if self.bm25 is None:
            return []

        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)

        results = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            item = self.corpus[i]
            meta = item.get("metadata", {})
            if game_name and meta.get("game_name") != game_name:
                continue
            results.append({
                "doc_id": item["doc_id"],
                "text": item["text"],
                "metadata": meta,
                "bm25_score": float(score),
            })

        results.sort(key=lambda x: x["bm25_score"], reverse=True)
        return results[:top_k]

    @property
    def is_ready(self) -> bool:
        return self.bm25 is not None


def get_bm25_engine() -> BM25Engine:
    """取得 BM25 引擎（singleton，避免每次查詢重新載入）。"""
    global _bm25_engine_cache
    if _bm25_engine_cache is None:
        _bm25_engine_cache = BM25Engine(CORPUS_FILE)
    return _bm25_engine_cache


# ═══════════════════════════════════════════════════════════════════
# 5. Hybrid Search + RRF
# ═══════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    vector_results: list,
    bm25_results: list,
    k: int = RRF_K,
    vector_weight: float = HYBRID_WEIGHT_VECTOR,
    bm25_weight: float = HYBRID_WEIGHT_BM25,
) -> list[dict]:
    """使用 Reciprocal Rank Fusion (RRF) 融合向量與 BM25 結果。"""
    rrf_scores = {}

    for rank, node_with_score in enumerate(vector_results, 1):
        doc_id = node_with_score.node.node_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + vector_weight * (1 / (k + rank))

    for rank, item in enumerate(bm25_results, 1):
        doc_id = item["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight * (1 / (k + rank))

    result_map = {}
    for node_with_score in vector_results:
        doc_id = node_with_score.node.node_id
        result_map[doc_id] = {
            "node": node_with_score.node,
            "score": node_with_score.score,
            "rrf_score": rrf_scores.get(doc_id, 0),
        }

    for item in bm25_results:
        doc_id = item["doc_id"]
        if doc_id not in result_map:
            result_map[doc_id] = {
                "node": None,
                "score": 0,
                "rrf_score": rrf_scores.get(doc_id, 0),
                "bm25_text": item["text"],
                "bm25_metadata": item["metadata"],
            }

    return sorted(result_map.values(), key=lambda x: x["rrf_score"], reverse=True)


def _get_adaptive_weights(query: str) -> tuple[float, float]:
    """根據查詢特性自動調整 Hybrid Search 權重。

    - 短查詢（<=5 字）、純英文查詢 → 偏 BM25（關鍵字匹配更有效）
    - 長查詢、含中文查詢 → 偏向量（語義理解更有效）

    Returns:
        (vector_weight, bm25_weight)
    """
    query_len = len(query)
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in query)

    if query_len <= SHORT_QUERY_THRESHOLD:
        logger.debug(f"   ⚙️ 自適應權重: 短查詢({query_len}字) → BM25優先")
        return (1 - SHORT_QUERY_BM25_WEIGHT, SHORT_QUERY_BM25_WEIGHT)
    elif not has_chinese and query_len < 20:
        logger.debug(f"   ⚙️ 自適應權重: 英文短查詢 → BM25優先")
        return (1 - SHORT_QUERY_BM25_WEIGHT, SHORT_QUERY_BM25_WEIGHT)
    else:
        logger.debug(f"   ⚙️ 自適應權重: {'長查詢' if query_len > 5 else '中英混合'} → 向量優先")
        return (LONG_QUERY_VECTOR_WEIGHT, 1 - LONG_QUERY_VECTOR_WEIGHT)


def _is_noise_chunk(text: str) -> bool:
    """判斷 chunk 是否為無意義的雜訊（版權、頁碼、元件列表等）。
    
    檢查項目：
    1. 第一行是否符合雜訊模式
    2. 整體長度是否過短
    3. 純元件列表（多行連續符合元件規則）
    """
    if not FILTER_NOISE_CHUNKS:
        return False

    stripped = text.strip()
    
    # 過濾過短的 chunk
    if len(stripped) < MIN_CHUNK_CHARS:
        return True

    first_line = stripped.split('\n')[0].strip()
    
    # 檢查第一行是否符合雜訊模式
    for pattern in NOISE_PATTERNS:
        if re.search(pattern, first_line, re.IGNORECASE):
            return True

    # 檢查整段文字是否超過 40% 的行是元件列表
    lines = stripped.split('\n')
    component_lines = 0
    for line in lines:
        trimmed = line.strip()
        # 檢查是否符合元件列表模式："數量 大寫名稱" 或 "數量 普通名詞"
        if re.match(r'^\d+\s+[A-Z]', trimmed):
            component_lines += 1
        # 檢查是否符合 "X COMPONENT_TYPE" 模式
        for pat in NOISE_PATTERNS:
            if re.search(pat, trimmed, re.IGNORECASE):
                component_lines += 1
                break

    if len(lines) > 3 and component_lines / len(lines) > 0.4:
        return True

    return False


def hybrid_search(
    retriever,
    query: str,
    bm25_engine: BM25Engine | None,
    top_k: int,
    game_name: str | None,
) -> list[dict]:
    """執行 Hybrid Search（向量 + BM25 + RRF），含自適應權重。"""
    vector_weight, bm25_weight = _get_adaptive_weights(query)

    vector_results = retriever.retrieve(query)

    bm25_results = bm25_engine.search(query, top_k=top_k, game_name=game_name) if bm25_engine else []

    if bm25_results:
        fused_results = reciprocal_rank_fusion(
            vector_results, bm25_results,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )
    else:
        fused_results = [
            {"node": r.node, "score": r.score, "rrf_score": 1.0}
            for r in vector_results
        ]

    return fused_results


# ═══════════════════════════════════════════════════════════════════
# 6. MMR 多樣性排序
# ═══════════════════════════════════════════════════════════════════

def mmr_diversity_ranking(
    results: list[dict],
    query_embedding: list[float] | None,
    lambda_param: float = MMR_LAMBDA,
    top_k: int = MMR_TOP_K,
) -> list[dict]:
    """使用 Maximal Marginal Relevance 對結果進行多樣性排序。

    MMR = λ * rel_score - (1-λ) * max_similarity_to_selected

    Args:
        results: 候選結果列表
        query_embedding: 查詢的 embedding（若無則使用 RRF score 代替 rel_score）
        lambda_param: 相關性 vs 多樣性權重（越高越偏相關性）
        top_k: 保留的結果數

    Returns:
        多樣性排序後的結果
    """
    if len(results) <= 3:
        return results[:top_k]

    selected = []
    candidate_scores = []
    for r in results:
        if r.get("node"):
            text = r["node"].get_content()[:200]
        else:
            text = r.get("bm25_text", "")[:200]
        candidate_scores.append({
            "text": text,
            "score": r.get("rerank_score", r.get("rrf_score", r.get("score", 0))),
            "data": r,
        })

    if not candidate_scores:
        return results[:top_k]

    candidate_scores.sort(key=lambda x: x["score"], reverse=True)
    selected.append(candidate_scores[0])
    remaining = candidate_scores[1:]

    while len(selected) < top_k and remaining:
        mmr_scores = []
        for cand in remaining:
            rel_score = cand["score"]
            cand_words = set(jieba.cut(cand["text"]))
            max_sim = 0
            for sel in selected:
                sel_words = set(jieba.cut(sel["text"]))
                if cand_words and sel_words:
                    jaccard = len(cand_words & sel_words) / len(cand_words | sel_words)
                    max_sim = max(max_sim, jaccard)

            mmr = lambda_param * rel_score - (1 - lambda_param) * max_sim
            mmr_scores.append(mmr)

        best_idx = int(np.argmax(mmr_scores))
        selected.append(remaining[best_idx])
        remaining.pop(best_idx)

    return [s["data"] for s in selected]


# ═══════════════════════════════════════════════════════════════════
# 7. Cross-encoder Re-ranking
# ═══════════════════════════════════════════════════════════════════

def rerank_with_cross_encoder(
    results: list[dict],
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """使用 Cross-encoder 模型對結果重新排序。

    Cross-encoder 會對 (query, paragraph) 逐對計算相關性分數，
    比向量相似度更精準。

    Args:
        results: 候選結果列表
        query: 原始查詢
        top_k: 保留的結果數

    Returns:
        重新排序後的結果
    """
    model = get_reranker()
    if model is None:
        return results[:top_k]

    pairs = []
    for item in results:
        if item.get("node"):
            text = item["node"].get_content()[:512]
        else:
            text = item.get("bm25_text", "")[:512]
        pairs.append((query, text))

    if not pairs:
        return results[:top_k]

    try:
        rerank_scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.warning(f"⚠️ Re-ranking 失敗: {e}")
        return results[:top_k]

    for i, score in enumerate(rerank_scores):
        results[i]["rerank_score"] = float(score)

    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    logger.info(f"   🏆 Re-ranking 完成，Top-1 分數: {results[0].get('rerank_score', 0):.4f}")

    return results[:top_k]


# ═══════════════════════════════════════════════════════════════════
# 8. LLM Prompt
# ═══════════════════════════════════════════════════════════════════

CHINESE_QA_PROMPT = PromptTemplate(
    "你是一位專業的桌遊裁判與規則專家。請根據以下提供的官方規則片段，"
    "嚴謹且有條理地回答玩家的問題。\n\n"
    "回答原則：\n"
    "1. 只根據提供的規則片段回答，不要憑空猜測\n"
    "2. 如果規則片段沒有明確提到，請說「規則中沒有明確說明」\n"
    "3. 盡量引用規則原文來支持你的回答\n"
    "4. 以繁體中文回答\n"
    "5. 用 **粗體** 標示關鍵術語\n"
    "6. 條列式整理（使用 - 或 1. 2. 3.）讓答案更易讀\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "玩家問題: {query_str}\n"
    "請用繁體中文回答："
)


# ═══════════════════════════════════════════════════════════════════
# 9. Index Loading
# ═══════════════════════════════════════════════════════════════════

def load_index() -> VectorStoreIndex:
    """載入 ChromaDB 中的向量索引。"""
    db_dir = str(DB_DIR)
    if not Path(db_dir).exists():
        raise FileNotFoundError(
            f"ChromaDB 目錄不存在: {db_dir}，請先執行 python main.py ingest 建立索引"
        )

    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info(f"🧠 載入 Embedding 模型: {EMBED_MODEL_NAME} (type={EMBED_TYPE})")
    if EMBED_TYPE == "openai" and os.environ.get("OPENAI_API_KEY"):
        from llama_index.embeddings.openai import OpenAIEmbedding
        embed_model = OpenAIEmbedding(model=EMBED_MODEL_NAME)
        logger.info("   ✅ 使用 OpenAI Embedding API（低記憶體模式）")
    else:
        if EMBED_TYPE == "openai" and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("   ⚠️ OPENAI_API_KEY 未設定，自動降級為本機 HuggingFace Embedding")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name=HF_EMBED_MODEL_NAME)
        logger.info(f"   ✅ 使用本機 HuggingFace Embedding: {HF_EMBED_MODEL_NAME}")
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    logger.info(f"✅ 索引載入完成 (collection: {COLLECTION_NAME}, embed_type={EMBED_TYPE})")
    return index


def build_filters(game_name: str | None) -> MetadataFilters | None:
    if not game_name:
        return None
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="game_name", value=game_name)]
    )
    logger.info(f"🔍 啟用 Metadata 過濾: game_name = {game_name}")
    return filters


def get_available_games() -> list[str]:
    """從 ChromaDB 取得所有遊戲名稱。"""
    db_dir = str(DB_DIR)
    if not Path(db_dir).exists():
        return []
    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection(COLLECTION_NAME)
    data = chroma_collection.get(include=["metadatas"])
    games = set()
    for meta in (data.get("metadatas") or []):
        if meta and "game_name" in meta:
            games.add(meta["game_name"])
    return sorted(games)


# ═══════════════════════════════════════════════════════════════════
# 10. Main Query Function（供 CLI 與 Web 共用）
# ═══════════════════════════════════════════════════════════════════

def query_rules(
    index: VectorStoreIndex,
    query_str: str,
    game_name: str | None = None,
    top_k: int = MMR_TOP_K,
) -> str:
    """執行規則查詢（含 Query Expansion, Hybrid Search, Re-ranking, MMR）。"""
    filters = build_filters(game_name)
    use_llm = bool(os.environ.get("OPENAI_API_KEY"))

    # ── Step 1: Query Expansion ──
    expanded_queries = expand_query(query_str)
    logger.info(f"🔍 查詢擴展: {len(expanded_queries)} 個變體 → 合併檢索")

    # ── Step 2: Hybrid Search ──
    logger.info(f"🔎 Hybrid 檢索 (top_k={SIMILARITY_TOP_K})")
    retriever = index.as_retriever(
        similarity_top_k=SIMILARITY_TOP_K,
        filters=filters,
    )
    bm25_engine = get_bm25_engine()

    all_fused_results = []
    seen_doc_ids = set()

    for eq in expanded_queries:
        fused = hybrid_search(retriever, eq, bm25_engine, SIMILARITY_TOP_K, game_name)
        for item in fused:
            doc_id = item.get("node", None) and item["node"].node_id or item.get("bm25_text", "")
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                all_fused_results.append(item)

    all_fused_results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    logger.info(f"   📊 共 {len(all_fused_results)} 個不重複候選段落")

    # ── Step 3: 過濾雜訊 chunk ──
    if FILTER_NOISE_CHUNKS:
        before = len(all_fused_results)
        all_fused_results = [
            item for item in all_fused_results
            if not _is_noise_chunk(
                item.get("node", None) and item["node"].get_content() or item.get("bm25_text", "")
            )
        ]
        after = len(all_fused_results)
        if before > after:
            logger.info(f"   🗑️ 過濾 {before - after} 個雜訊 chunk（版權/頁碼/頁首）")

    # ── Step 4: Cross-encoder Re-ranking ──
    if USE_RE_RANKER:
        logger.info("🏆 執行 Cross-encoder Re-ranking...")
        reranked = rerank_with_cross_encoder(all_fused_results, query_str, top_k=SIMILARITY_TOP_K)
    else:
        reranked = all_fused_results

    # ── Step 5: MMR 多樣性排序 ──
    logger.info(f"🎯 MMR 多樣性排序 (λ={MMR_LAMBDA}, top_k={top_k})")
    diversified = mmr_diversity_ranking(reranked, None, lambda_param=MMR_LAMBDA, top_k=top_k)

    if use_llm:
        # ── LLM 模式 ──
        logger.info("🤖 使用 LLM 生成回應")
        context_parts = []
        for item in diversified:
            if item.get("node"):
                context_parts.append(item["node"].get_content())
            else:
                context_parts.append(item.get("bm25_text", ""))
        context_str = "\n\n---\n\n".join(context_parts)

        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            import requests
            resp = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "openai/gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "你是一位專業的桌遊裁判與規則專家。請根據提供的規則片段回答問題，以繁體中文回答，用**粗體**標示關鍵術語，條列式整理。"},
                        {"role": "user", "content": f"規則片段：\n{context_str}\n\n問題：{query_str}\n\n請用繁體中文回答："},
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.1,
                }),
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        from llama_index.llms.openai import OpenAI
        llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        prompt = CHINESE_QA_PROMPT.format(context_str=context_str, query_str=query_str)
        response = llm.complete(prompt)
        return str(response)

    else:
        # ── 純檢索模式 ──
        logger.info("📄 使用純檢索模式（Hybrid + Re-rank + MMR + 翻譯）")
        lines = []
        rank = 1
        for item in diversified:
            if item.get("node"):
                raw_content = item["node"].get_content()[:600]
                score = item.get("score", 0)
                rerank_score = item.get("rerank_score", 0)
            else:
                raw_content = item.get("bm25_text", "")[:600]
                score = item.get("bm25_score", 0)
                rerank_score = item.get("rerank_score", 0)

            rrf_score = item.get("rrf_score", 0)
            lines.append(f"=== 結果 {rank} (向量: {score:.4f}, RRF: {rrf_score:.4f}, Re-rank: {rerank_score:.4f}) ===")
            translated = translate_to_traditional_chinese(raw_content)
            lines.append(translated)
            lines.append("")
            rank += 1

        return "\n".join(lines) if lines else "❌ 未找到相關規則段落。"


def interactive_mode(index: VectorStoreIndex):
    """互動式查詢模式。"""
    print("\n" + "=" * 70)
    print("🎲 Board Game Rules Query Engine v3（終極版）")
    print("=" * 70)
    print("功能:")
    print("  ✅ Hybrid Search (向量 + BM25)")
    print("  ✅ Query Expansion（桌遊同義詞擴展）")
    print("  ✅ Cross-encoder Re-ranking")
    print("  ✅ MMR 多樣性排序")
    print("=" * 70)
    print("指令:")
    print("  /game <名稱>  設定遊戲過濾")
    print("  /clear        清除遊戲過濾")
    print("  /top <數字>   設定檢索數量")
    print("  /help         顯示說明")
    print("  /exit         離開")
    print("=" * 70 + "\n")

    game_name = None
    top_k = MMR_TOP_K

    while True:
        try:
            user_input = input(f"{'[' + game_name + '] ' if game_name else ''}你: ").strip()

            if not user_input:
                continue

            if user_input == "/exit":
                print("👋 再見！")
                break
            elif user_input == "/help":
                print("  /game <名稱>  設定遊戲過濾")
                print("  /clear        清除遊戲過濾")
                print("  /top <數字>   設定檢索數量")
                print("  /help         顯示說明")
                print("  /exit         離開")
                continue
            elif user_input.startswith("/game "):
                game_name = user_input[6:].strip()
                print(f"✅ 遊戲過濾設定為: {game_name}")
                continue
            elif user_input == "/clear":
                game_name = None
                print("✅ 遊戲過濾已清除")
                continue
            elif user_input.startswith("/top "):
                try:
                    top_k = int(user_input[5:].strip())
                    print(f"✅ 檢索數量設定為: {top_k}")
                except ValueError:
                    print("❌ 請輸入有效數字")
                continue

            response = query_rules(index, user_input, game_name=game_name, top_k=top_k)
            print(f"\n🤖 回應:\n{response}\n")

        except KeyboardInterrupt:
            print("\n👋 再見！")
            break
        except Exception as e:
            logger.error(f"❌ 查詢失敗: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="🎲 Board Game Rules Query Engine (v3 - Ultimate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python src/query.py --query "Hegemony 的勝利條件是什麼？"
  python src/query.py --query "資源交換機制" --game "On Mars"
  python src/query.py --interactive
        """,
    )
    parser.add_argument("--query", "-q", type=str, help="查詢文字")
    parser.add_argument("--game", "-g", type=str, default=None, help="指定遊戲名稱（過濾）")
    parser.add_argument("--top-k", type=int, default=MMR_TOP_K, help="最終檢索數量")
    parser.add_argument("--interactive", "-i", action="store_true", help="互動模式")

    args = parser.parse_args()

    if not args.query and not args.interactive:
        parser.print_help()
        print("\n💡 提示: 使用 --interactive / -i 可進入互動模式")
        print("   或使用 --query / -q 直接查詢")
        sys.exit(0)

    index = load_index()

    if args.interactive:
        interactive_mode(index)
    else:
        response = query_rules(
            index,
            args.query,
            game_name=args.game,
            top_k=args.top_k,
        )
        print(f"\n🤖 回應:\n{response}\n")


if __name__ == "__main__":
    main()