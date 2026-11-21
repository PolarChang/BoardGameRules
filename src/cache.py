"""
cache.py — FAQ Cache / Token Optimization

遵循 Implementation Plan 的 Token Optimization：
Rule 4: FAQ Cache
Question → Normalize → Cache Search → Hit → Return / Miss → LLM

同時提供：
- 規則檢索結果快取（避免重複檢索）
- 問答快取（避免重複呼叫 LLM）
"""
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from src.config import DB_DIR

logger = logging.getLogger(__name__)

CACHE_FILE = DB_DIR / "qa_cache.json"
CACHE_TTL = 60 * 60 * 24 * 7  # 7 天


# ── 快取儲存 ────────────────────────────────────────────────────────

class QACache:
    """問答快取（持久化到 JSON 檔案）。"""

    def __init__(self, cache_file: Path = CACHE_FILE):
        self.cache_file = cache_file
        self._cache: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """從檔案載入快取。"""
        if self.cache_file.exists():
            try:
                self._cache = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.warning("⚠️ 快取檔案損壞，重新建立")
                self._cache = {}

    def _save(self) -> None:
        """儲存快取到檔案。"""
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.cache_file.write_text(
                json.dumps(self._cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"❌ 儲存快取失敗: {e}")

    def get(self, key: str) -> Optional[dict]:
        """取得快取項目（檢查 TTL）。"""
        item = self._cache.get(key)
        if item is None:
            return None

        # 檢查 TTL
        if time.time() - item.get("timestamp", 0) > CACHE_TTL:
            del self._cache[key]
            return None

        return item.get("data")

    def set(self, key: str, data: dict) -> None:
        """設定快取項目。"""
        self._cache[key] = {
            "timestamp": time.time(),
            "data": data,
        }
        self._save()

    def clear(self) -> None:
        """清除快取。"""
        self._cache = {}
        self._save()

    def size(self) -> int:
        """快取項目數量。"""
        return len(self._cache)


# ── 全域快取實例 ────────────────────────────────────────────────────

_qa_cache: Optional[QACache] = None


def get_qa_cache() -> QACache:
    """取得全域 QA 快取實例。"""
    global _qa_cache
    if _qa_cache is None:
        _qa_cache = QACache()
    return _qa_cache


# ── 問題正規化 ──────────────────────────────────────────────────────

def normalize_question(question: str) -> str:
    """
    正規化問題：
    1. 去除多餘空白
    2. 去除標點符號
    3. 轉為小寫
    4. 去除常見停用詞
    """
    # 去除多餘空白
    question = re.sub(r"\s+", " ", question.strip())

    # 去除標點符號
    question = re.sub(r"[，。！？、；：,.!?;:()\[\]{}「」『』\"'\-]", "", question)

    # 轉為小寫
    question = question.lower()

    # 去除常見停用詞
    stopwords = [
        "請問", "我想知道", "可以告訴我", "幫我", "一下",
        "please", "tell me", "can you", "i want to know",
    ]
    for word in stopwords:
        question = question.replace(word, "")

    return question.strip()


def make_cache_key(game_id: str, question: str) -> str:
    """建立快取鍵。"""
    normalized = normalize_question(question)
    raw = f"{game_id}:{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ── 快取操作 ────────────────────────────────────────────────────────

def get_cached_answer(game_id: str, question: str) -> Optional[dict]:
    """取得快取的問答結果。"""
    cache = get_qa_cache()
    key = make_cache_key(game_id, question)
    return cache.get(key)


def cache_answer(game_id: str, question: str, answer: dict) -> None:
    """快取問答結果。"""
    cache = get_qa_cache()
    key = make_cache_key(game_id, question)
    cache.set(key, answer)


# ── 規則檢索快取 ────────────────────────────────────────────────────

_retrieval_cache: dict[str, tuple[float, list]] = {}
RETRIEVAL_CACHE_TTL = 60 * 60  # 1 小時


def get_cached_retrieval(game_id: str, query: str) -> Optional[list]:
    """取得快取的檢索結果。"""
    key = make_cache_key(game_id, query)
    item = _retrieval_cache.get(key)
    if item is None:
        return None
    timestamp, data = item
    if time.time() - timestamp > RETRIEVAL_CACHE_TTL:
        del _retrieval_cache[key]
        return None
    return data


def cache_retrieval(game_id: str, query: str, results: list) -> None:
    """快取檢索結果。"""
    key = make_cache_key(game_id, query)
    _retrieval_cache[key] = (time.time(), results)