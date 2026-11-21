"""
validator.py — Rule Validation Service

負責驗證 RuleNode 的完整性與正確性：
1. Source 驗證（每個 Rule 必須有 document_id + page）
2. Category 驗證（必須是固定 Enum）
3. Type 驗證（必須是固定 Enum）
4. Confidence 驗證（必須是固定 Enum）
5. 建立 Review Queue（conflict, ambiguous, missing_source, low_confidence）

遵循 Implementation Plan：
- 所有 RuleNode 必須有 Source
- 無法取得來源時 verification_status = "missing_source"
- 該 Rule 不得進入 Verified
"""
import logging
import uuid
from typing import Optional

from src.database import (
    create_review_items_bulk,
    get_rule_node,
    update_rule_node,
    update_rule_verification,
)
from src.models import (
    Confidence,
    ReviewAction,
    ReviewItem,
    RuleCategory,
    RuleNode,
    RuleType,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


# ── 驗證結果 ────────────────────────────────────────────────────────

class ValidationResult:
    """單一規則的驗證結果。"""

    def __init__(self, rule: RuleNode):
        self.rule = rule
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.review_reasons: list[str] = []

    @property
    def is_valid(self) -> bool:
        """是否通過驗證（無錯誤）。"""
        return len(self.errors) == 0

    @property
    def needs_review(self) -> bool:
        """是否需要人工 Review。"""
        return len(self.review_reasons) > 0


# ── 驗證函式 ────────────────────────────────────────────────────────

def validate_source(rule: RuleNode, result: ValidationResult) -> None:
    """驗證 Source 完整性。"""
    source = rule.source

    # 必須有 document_id
    if not source.document_id:
        result.errors.append("缺少 document_id")
        result.review_reasons.append("missing_source")

    # 必須有 page
    if source.page is None or source.page <= 0:
        result.errors.append("缺少 page")
        result.review_reasons.append("missing_source")


def validate_category(rule: RuleNode, result: ValidationResult) -> None:
    """驗證 Category 是否為固定 Enum。"""
    try:
        RuleCategory(rule.category.value)
    except ValueError:
        result.errors.append(f"無效的 Category: {rule.category}")
        result.review_reasons.append("invalid_category")


def validate_type(rule: RuleNode, result: ValidationResult) -> None:
    """驗證 Type 是否為固定 Enum。"""
    try:
        RuleType(rule.type.value)
    except ValueError:
        result.errors.append(f"無效的 Type: {rule.type}")
        result.review_reasons.append("invalid_type")


def validate_confidence(rule: RuleNode, result: ValidationResult) -> None:
    """驗證 Confidence 是否為固定 Enum。"""
    try:
        Confidence(rule.confidence.value)
    except ValueError:
        result.errors.append(f"無效的 Confidence: {rule.confidence}")
        result.review_reasons.append("invalid_confidence")


def validate_rule_text(rule: RuleNode, result: ValidationResult) -> None:
    """驗證規則文字完整性。"""
    if not rule.rule_text or len(rule.rule_text.strip()) < 5:
        result.errors.append("規則文字太短或為空")
        result.review_reasons.append("empty_rule_text")

    if not rule.title or len(rule.title.strip()) < 2:
        result.warnings.append("標題太短或為空")


def validate_confidence_level(rule: RuleNode, result: ValidationResult) -> None:
    """驗證信心程度，低信心規則需要 Review。"""
    if rule.confidence == Confidence.AMBIGUOUS:
        result.review_reasons.append("ambiguous")
    elif rule.confidence == Confidence.CONFLICT:
        result.review_reasons.append("conflict")


def validate_verification_status(rule: RuleNode, result: ValidationResult) -> None:
    """驗證驗證狀態。"""
    if rule.verification_status == VerificationStatus.MISSING_SOURCE:
        result.review_reasons.append("missing_source")


def validate_rule(rule: RuleNode) -> ValidationResult:
    """驗證單一規則。"""
    result = ValidationResult(rule)

    validate_source(rule, result)
    validate_category(rule, result)
    validate_type(rule, result)
    validate_confidence(rule, result)
    validate_rule_text(rule, result)
    validate_confidence_level(rule, result)
    validate_verification_status(rule, result)

    return result


# ── 批次驗證 ────────────────────────────────────────────────────────

def validate_rules(rules: list[RuleNode]) -> list[ValidationResult]:
    """批次驗證規則。"""
    return [validate_rule(rule) for rule in rules]


def apply_validation(rules: list[RuleNode]) -> dict:
    """
    驗證規則並套用結果：
    1. 更新 verification_status
    2. 建立 Review Queue

    Returns:
        {
            "valid": int,
            "invalid": int,
            "needs_review": int,
            "review_items": int
        }
    """
    results = validate_rules(rules)
    review_items: list[ReviewItem] = []

    for result in results:
        rule = result.rule

        # 套用驗證狀態
        if not result.is_valid:
            # 有錯誤 → 標記為 missing_source 或 rejected
            if "missing_source" in result.review_reasons:
                rule.verification_status = VerificationStatus.MISSING_SOURCE
            else:
                rule.verification_status = VerificationStatus.REJECTED
        elif result.needs_review:
            # 需要 Review → 標記為 ambiguous
            rule.verification_status = VerificationStatus.AMBIGUOUS
        else:
            # 通過驗證 → unverified（等待人工確認）
            rule.verification_status = VerificationStatus.UNVERIFIED

        # 持久化更新後的 verification_status
        update_rule_node(rule)

        # 建立 Review Item
        if result.needs_review or not result.is_valid:
            reason = "; ".join(result.review_reasons) or "needs_review"
            review_items.append(ReviewItem(
                id=f"REVIEW-{uuid.uuid4().hex[:8].upper()}",
                rule_id=rule.id,
                game_id=rule.game_id,
                reason=reason,
                status=ReviewAction.AMBIGUOUS,
            ))

    # 批次建立 Review Items
    if review_items:
        create_review_items_bulk(review_items)

    return {
        "valid": sum(1 for r in results if r.is_valid),
        "invalid": sum(1 for r in results if not r.is_valid),
        "needs_review": sum(1 for r in results if r.needs_review),
        "review_items": len(review_items),
    }


# ── Review 操作 ─────────────────────────────────────────────────────

def review_rule(rule_id: str, action: str) -> Optional[RuleNode]:
    """
    人工 Review 規則。

    Args:
        rule_id: 規則 ID
        action: "verified" | "edit" | "ambiguous" | "reject"

    Returns:
        更新後的 RuleNode，或 None（找不到規則）
    """
    rule = get_rule_node(rule_id)
    if rule is None:
        return None

    if action == "verified":
        # 確認規則有 Source 才能 Verified
        if rule.source.document_id and rule.source.page > 0:
            rule.verification_status = VerificationStatus.VERIFIED
        else:
            rule.verification_status = VerificationStatus.MISSING_SOURCE
    elif action == "edit":
        rule.verification_status = VerificationStatus.UNVERIFIED
    elif action == "ambiguous":
        rule.verification_status = VerificationStatus.AMBIGUOUS
    elif action == "reject":
        rule.verification_status = VerificationStatus.REJECTED
    else:
        raise ValueError(f"無效的 Review 動作: {action}")

    update_rule_verification(rule.id, rule.verification_status.value)
    return rule


def get_review_priority(game_id: str) -> list[dict]:
    """
    取得 Review Queue，依優先度排序。

    優先顯示：
    1. conflict
    2. ambiguous
    3. missing_source
    4. low_confidence
    """
    from src.database import get_review_items

    items = get_review_items(game_id=game_id, limit=200)

    priority_order = {
        "conflict": 0,
        "ambiguous": 1,
        "missing_source": 2,
        "low_confidence": 3,
    }

    # 依優先度排序
    items.sort(key=lambda item: priority_order.get(item.reason, 99))

    result = []
    for item in items:
        rule = get_rule_node(item.rule_id)
        if rule:
            result.append({
                "review_id": item.id,
                "rule_id": item.rule_id,
                "reason": item.reason,
                "title": rule.title,
                "rule_text": rule.rule_text,
                "category": rule.category.value,
                "type": rule.type.value,
                "confidence": rule.confidence.value,
                "verification_status": rule.verification_status.value,
                "source": {
                    "document_id": rule.source.document_id,
                    "page": rule.source.page,
                    "section": rule.source.section,
                },
            })

    return result