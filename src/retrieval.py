"""
retrieval.py — Rule Retrieval Service

負責從 SQLite 資料庫中檢索 RuleNode，支援：
- 關鍵字檢索（BM25-like scoring）
- Category 篩選
- Type 篩選
- Related Rules 展開
- Exceptions 檢索
- Source 相關性

Retrieval 優先考慮：
1. 同一遊戲
2. 同一 Category
3. Semantic similarity（透過關鍵字重疊）
4. Related Rules
5. Exceptions
6. Source relevance
"""
import re
from collections import Counter
from typing import Optional

from src.database import get_rule_node, get_rule_nodes_by_game
from src.models import RuleNode, RuleSearchResult


# ── 關鍵字權重 ──────────────────────────────────────────────────────

# 常見規則關鍵字，用於提升檢索相關性
CATEGORY_KEYWORDS = {
    "setup": ["setup", "準備", "設置", "放置", "初始", "開始前"],
    "objective": ["objective", "目標", "目的", "勝利條件", "win"],
    "components": ["component", "元件", "配件", "token", "board", "卡牌"],
    "game_flow": ["flow", "流程", "回合", "輪", "phase", "階段"],
    "turn_structure": ["turn", "回合", "輪到", "player turn", "行動順序"],
    "actions": ["action", "行動", "動作", "執行", "操作"],
    "resources": ["resource", "資源", "金錢", "money", "wood", "stone"],
    "mechanics": ["mechanic", "機制", "規則", "系統"],
    "cards": ["card", "卡牌", "卡", "deck", "手牌"],
    "tiles": ["tile", "板塊", "瓷磚"],
    "tokens": ["token", "標記", "指示物"],
    "combat": ["combat", "戰鬥", "攻擊", "防禦", "傷害"],
    "scoring": ["scoring", "計分", "分數", "得分", "point"],
    "end_game": ["end", "結束", "遊戲結束", "終局", "final"],
    "exceptions": ["exception", "例外", "特殊", "特別"],
    "faq": ["faq", "常見問題", "問答"],
}

TYPE_KEYWORDS = {
    "condition": ["if", "如果", "當", "when", "條件"],
    "action": ["action", "行動", "可以", "may", "執行"],
    "cost": ["cost", "花費", "支付", "消耗"],
    "effect": ["effect", "效果", "影響", "獲得"],
    "restriction": ["restriction", "限制", "不能", "cannot", "禁止"],
    "timing": ["timing", "時機", "時間點", "when", "during"],
    "exception": ["exception", "例外", "特殊", "特別"],
    "scoring": ["scoring", "計分", "分數", "得分"],
    "end_game": ["end", "結束", "遊戲結束"],
    "setup": ["setup", "準備", "設置"],
    "component": ["component", "元件", "配件"],
    "interaction": ["interaction", "互動", "交互"],
}


def _tokenize(text: str) -> set[str]:
    """將文字切分為 token 集合（支援中英文）。"""
    # 英文單字
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    # 中文詞（以 2-gram 方式處理）
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    for i in range(len(chinese_chars) - 1):
        words.add(chinese_chars[i] + chinese_chars[i + 1])
    return words


def _keyword_overlap(query_tokens: set[str], rule_text: str) -> float:
    """計算查詢與規則文字的關鍵字重疊分數。"""
    rule_tokens = _tokenize(rule_text)
    if not query_tokens or not rule_tokens:
        return 0.0
    overlap = query_tokens & rule_tokens
    return len(overlap) / max(len(query_tokens), 1)


def _category_match(query: str, category: str) -> float:
    """檢查查詢是否包含該 Category 的關鍵字。"""
    keywords = CATEGORY_KEYWORDS.get(category, [])
    query_lower = query.lower()
    matches = sum(1 for kw in keywords if kw.lower() in query_lower)
    if matches == 0:
        return 0.0
    return min(0.3, matches * 0.1)


def _type_match(query: str, rule_type: str) -> float:
    """檢查查詢是否包含該 Type 的關鍵字。"""
    keywords = TYPE_KEYWORDS.get(rule_type, [])
    query_lower = query.lower()
    matches = sum(1 for kw in keywords if kw.lower() in query_lower)
    if matches == 0:
        return 0.0
    return min(0.2, matches * 0.07)


def _source_relevance(rule: RuleNode, query: str) -> float:
    """計算來源相關性（章節名稱與查詢的匹配）。"""
    section = rule.source.section or ""
    if not section:
        return 0.0
    section_tokens = _tokenize(section)
    query_tokens = _tokenize(query)
    if not section_tokens or not query_tokens:
        return 0.0
    overlap = section_tokens & query_tokens
    return min(0.15, len(overlap) * 0.05)


def _exception_boost(rule: RuleNode, query: str) -> float:
    """例外規則的相關性提升（當查詢包含例外/特殊關鍵字時）。"""
    if rule.type.value != "exception":
        return 0.0
    exception_keywords = ["例外", "特殊", "exception", "special", "特別", "突破"]
    if any(kw in query.lower() for kw in exception_keywords):
        return 0.2
    return 0.0


def _verified_boost(rule: RuleNode) -> float:
    """已驗證規則的相關性提升。"""
    if rule.verification_status.value == "verified":
        return 0.1
    return 0.0


def _score_rule(rule: RuleNode, query: str) -> float:
    """計算單一規則的相關度分數。"""
    score = 0.0

    # 1. 關鍵字重疊（主要分數來源）
    query_tokens = _tokenize(query)
    score += _keyword_overlap(query_tokens, rule.rule_text) * 0.5
    score += _keyword_overlap(query_tokens, rule.title) * 0.3

    # 2. Category 匹配
    score += _category_match(query, rule.category.value)

    # 3. Type 匹配
    score += _type_match(query, rule.type.value)

    # 4. Source 相關性
    score += _source_relevance(rule, query)

    # 5. Exception 提升
    score += _exception_boost(rule, query)

    # 6. Verified 提升
    score += _verified_boost(rule)

    return score


def search_rules(
    game_id: str,
    query: str,
    limit: int = 10,
    category: Optional[str] = None,
    rule_type: Optional[str] = None,
    include_unverified: bool = False,
) -> list[RuleSearchResult]:
    """
    檢索遊戲的規則。

    Args:
        game_id: 遊戲 ID
        query: 查詢文字
        limit: 結果數量上限
        category: 篩選 Category
        rule_type: 篩選 Type
        include_unverified: 是否包含未驗證規則

    Returns:
        list[RuleSearchResult]: 排序後的規則檢索結果
    """
    # 取得遊戲的所有規則
    rules = get_rule_nodes_by_game(game_id, limit=5000)

    # 篩選
    if category:
        rules = [r for r in rules if r.category.value == category]
    if rule_type:
        rules = [r for r in rules if r.type.value == rule_type]
    if not include_unverified:
        rules = [
            r for r in rules
            if r.verification_status.value in ("verified", "unverified")
        ]

    # 評分
    scored = [(rule, _score_rule(rule, query)) for rule in rules]

    # 排序（分數高 → 低）
    scored.sort(key=lambda x: x[1], reverse=True)

    # 取前 limit 個
    results = []
    for rule, score in scored[:limit]:
        if score > 0:
            results.append(RuleSearchResult(rule_id=rule.id, score=round(score, 4)))

    return results


def get_related_rules(rule_id: str, limit: int = 5) -> list[RuleNode]:
    """取得與指定規則相關的規則（透過 related_rule_ids）。"""
    rule = get_rule_node(rule_id)
    if rule is None:
        return []

    related = []
    for related_id in rule.related_rule_ids[:limit]:
        related_rule = get_rule_node(related_id)
        if related_rule:
            related.append(related_rule)
    return related


def get_exceptions_for_rule(rule_id: str) -> list[RuleNode]:
    """取得與指定規則相關的例外規則。"""
    rule = get_rule_node(rule_id)
    if rule is None:
        return []

    # 1. 透過 related_rule_ids 找例外
    exceptions = []
    for related_id in rule.related_rule_ids:
        related_rule = get_rule_node(related_id)
        if related_rule and related_rule.type.value == "exception":
            exceptions.append(related_rule)

    # 2. 如果沒有，找同遊戲中 category 為 exceptions 的規則
    if not exceptions:
        game_rules = get_rule_nodes_by_game(rule.game_id, category="exceptions", limit=100)
        exceptions = [r for r in game_rules if r.verification_status.value == "verified"]

    return exceptions[:5]


def search_with_exceptions(
    game_id: str,
    query: str,
    limit: int = 10,
) -> dict:
    """
    檢索規則並同時取得相關例外。

    用於 AI Q&A 的 Exception Test：
    當查詢涉及「特殊能力是否可以突破基本限制？」時，
    必須同時檢索 base_rule + exception。

    Returns:
        {
            "rules": [...],
            "exceptions": [...]
        }
    """
    rules = search_rules(game_id, query, limit=limit)

    # 收集相關例外
    exceptions = []
    seen_ids = set()
    for result in rules:
        rule = get_rule_node(result.rule_id)
        if rule is None:
            continue
        for exc in get_exceptions_for_rule(rule.id):
            if exc.id not in seen_ids:
                seen_ids.add(exc.id)
                exceptions.append(RuleSearchResult(rule_id=exc.id, score=0.8))

    return {
        "rules": rules,
        "exceptions": exceptions,
    }