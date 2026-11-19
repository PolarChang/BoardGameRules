"""
qa_service.py — AI Q&A Service

負責 AI 規則問答：
1. Retrieve rules
2. Check source
3. Check exceptions
4. Determine confidence
5. Generate answer

遵循 Implementation Plan：
- 禁止每次問答重新傳整份 PDF
- 只使用 Retrieval 的 RuleNodes
- 必須提供 Rule IDs 和 source pages
- 規則書沒有足夠資訊時回答「規則書目前沒有足夠資訊判定。」
- 多條規則衝突時報告衝突，不猜測
"""
import logging
import os
from typing import Optional

from src.cache import cache_answer, get_cached_answer
from src.database import get_rule_node
from src.models import Confidence, QAResponse
from src.retrieval import search_with_exceptions

logger = logging.getLogger(__name__)


# ── System Prompt ───────────────────────────────────────────────────

QA_SYSTEM_PROMPT = """You are a board game rules assistant.

You must only answer based on retrieved verified rule data.

Never invent a rule.

Never rely on your pretrained knowledge when it conflicts with
the provided rule data.

If the rules do not contain enough information, answer:
「規則書目前沒有足夠資訊判定。」

If multiple rules conflict, report the conflict instead of guessing.

Every rules-related answer must provide Rule IDs and source pages.
"""


# ── 無 LLM 模式（規則式回答） ───────────────────────────────────────

def _format_rule_context(rules: list, exceptions: list) -> str:
    """格式化規則上下文供 LLM 使用。"""
    parts = []

    if rules:
        parts.append("=== 相關規則 ===")
        for item in rules:
            rule = get_rule_node(item.rule_id)
            if rule:
                parts.append(
                    f"[{rule.id}] ({rule.category.value}/{rule.type.value}) "
                    f"p.{rule.source.page}"
                )
                parts.append(f"  標題: {rule.title}")
                parts.append(f"  內容: {rule.rule_text}")
                if rule.exceptions:
                    parts.append(f"  例外: {'; '.join(rule.exceptions)}")
                parts.append("")

    if exceptions:
        parts.append("=== 相關例外 ===")
        for item in exceptions:
            rule = get_rule_node(item.rule_id)
            if rule:
                parts.append(
                    f"[{rule.id}] ({rule.category.value}/{rule.type.value}) "
                    f"p.{rule.source.page}"
                )
                parts.append(f"  標題: {rule.title}")
                parts.append(f"  內容: {rule.rule_text}")
                parts.append("")

    return "\n".join(parts)


def _answer_rule_based(question: str, rules: list, exceptions: list) -> QAResponse:
    """使用規則式模式回答（無 LLM 時）。"""
    if not rules and not exceptions:
        return QAResponse(
            answer="規則書目前沒有足夠資訊判定。",
            confidence=Confidence.AMBIGUOUS,
            rule_ids=[],
            sources=[],
        )

    # 收集規則與來源
    rule_ids = [item.rule_id for item in rules]
    sources = []
    for item in rules:
        rule = get_rule_node(item.rule_id)
        if rule:
            sources.append({
                "document_id": rule.source.document_id,
                "page": rule.source.page,
                "section": rule.source.section,
            })

    # 檢查是否有例外
    has_exceptions = len(exceptions) > 0

    # 建立回答
    if has_exceptions:
        answer = "根據規則書，此問題涉及例外情況。\n\n"
    else:
        answer = "根據規則書：\n\n"

    for item in rules[:3]:
        rule = get_rule_node(item.rule_id)
        if rule:
            answer += f"• {rule.rule_text}\n"
            answer += f"  （來源：{rule.source.section or '規則書'} p.{rule.source.page}）\n\n"

    if has_exceptions:
        answer += "⚠️ 注意例外：\n"
        for item in exceptions[:2]:
            rule = get_rule_node(item.rule_id)
            if rule:
                answer += f"• {rule.rule_text}\n"
                answer += f"  （來源：{rule.source.section or '規則書'} p.{rule.source.page}）\n\n"

    answer += f"相關規則：{', '.join(rule_ids[:5])}"

    return QAResponse(
        answer=answer,
        confidence=Confidence.EXPLICIT,
        rule_ids=rule_ids,
        sources=sources,
    )


# ── LLM 模式 ────────────────────────────────────────────────────────

def _answer_llm(question: str, rules: list, exceptions: list) -> QAResponse:
    """使用 LLM 回答。"""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _answer_rule_based(question, rules, exceptions)

    client = OpenAI(api_key=api_key)
    context = _format_rule_context(rules, exceptions)

    if not context.strip():
        return QAResponse(
            answer="規則書目前沒有足夠資訊判定。",
            confidence=Confidence.AMBIGUOUS,
            rule_ids=[],
            sources=[],
        )

    prompt = f"""問題：{question}

以下是從規則書檢索到的相關規則：

{context}

請根據以上規則回答問題。如果規則不足，請回答「規則書目前沒有足夠資訊判定。」"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        answer = response.choices[0].message.content

        # 收集規則與來源
        rule_ids = [item.rule_id for item in rules]
        sources = []
        for item in rules:
            rule = get_rule_node(item.rule_id)
            if rule:
                sources.append({
                    "document_id": rule.source.document_id,
                    "page": rule.source.page,
                    "section": rule.source.section,
                })

        # 判斷信心程度
        confidence = Confidence.EXPLICIT
        if "沒有足夠資訊" in answer or "無法判定" in answer:
            confidence = Confidence.AMBIGUOUS
        elif "衝突" in answer or "矛盾" in answer:
            confidence = Confidence.CONFLICT

        return QAResponse(
            answer=answer,
            confidence=confidence,
            rule_ids=rule_ids,
            sources=sources,
        )

    except Exception as e:
        logger.error(f"❌ LLM 回答失敗: {e}")
        return _answer_rule_based(question, rules, exceptions)


# ── 統一入口 ────────────────────────────────────────────────────────

def answer_question(game_id: str, question: str) -> QAResponse:
    """
    AI 問答統一入口。

    Token Optimization（Rule 4 - FAQ Cache）：
    Question → Normalize → Cache Search → Hit → Return / Miss → LLM

    Args:
        game_id: 遊戲 ID
        question: 問題

    Returns:
        QAResponse: 回答結果
    """
    # 1. 檢查快取
    cached = get_cached_answer(game_id, question)
    if cached is not None:
        logger.info("⚡ 快取命中，直接回傳")
        return QAResponse(**cached)

    # 2. Retrieve rules（含 exceptions）
    result = search_with_exceptions(game_id, question, limit=10)
    rules = result["rules"]
    exceptions = result["exceptions"]

    # 3. 如果沒有找到任何規則 → 無法判定
    if not rules and not exceptions:
        response = QAResponse(
            answer="規則書目前沒有足夠資訊判定。",
            confidence=Confidence.AMBIGUOUS,
            rule_ids=[],
            sources=[],
        )
        cache_answer(game_id, question, response.model_dump())
        return response

    # 3.5 相關性門檻：如果 top 規則分數過低，視為無足夠資訊
    MIN_RELEVANCE_SCORE = 0.15
    if rules and rules[0].score < MIN_RELEVANCE_SCORE:
        response = QAResponse(
            answer="規則書目前沒有足夠資訊判定。",
            confidence=Confidence.AMBIGUOUS,
            rule_ids=[],
            sources=[],
        )
        cache_answer(game_id, question, response.model_dump())
        return response

    # 4. 使用 LLM 或規則式回答
    if os.environ.get("OPENAI_API_KEY"):
        response = _answer_llm(question, rules, exceptions)
    else:
        response = _answer_rule_based(question, rules, exceptions)

    # 5. 快取結果
    cache_answer(game_id, question, response.model_dump())
    return response
