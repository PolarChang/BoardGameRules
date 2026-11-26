"""
teaching_service.py — AI Teaching Service

負責 AI 教學：
Verified RuleNodes → Teaching Planner → Teaching Script → Interactive Teaching

教學順序預設：
1. Theme
2. Objective
3. Core Concept
4. Setup
5. Turn Structure
6. Basic Actions
7. Important Restrictions
8. Scoring
9. Common Mistakes

遵循 Implementation Plan：
- 教學不直接讀 PDF
- 使用 Verified RuleNodes
- 禁止修改規則、自創範例規則、忽略 Exception、把推測當官方規則
- 使用例子時必須標示「以下是示例，不是規則書中的實際情境。」
"""
import logging
import uuid
from typing import Optional

from src.database import get_rule_nodes_by_game
from src.models import RuleNode

logger = logging.getLogger(__name__)


# ── 教學順序 ────────────────────────────────────────────────────────

TEACHING_ORDER = [
    ("theme", "主題"),
    ("objective", "遊戲目標"),
    ("basic_info", "基本資訊"),
    ("setup", "遊戲設置"),
    ("turn_structure", "回合結構"),
    ("actions", "基本行動"),
    ("mechanics", "核心機制"),
    ("resources", "資源"),
    ("cards", "卡牌"),
    ("tiles", "板塊"),
    ("tokens", "標記"),
    ("combat", "戰鬥"),
    ("exceptions", "重要例外"),
    ("scoring", "計分"),
    ("end_game", "遊戲結束"),
]


# ── 教學腳本產生 ────────────────────────────────────────────────────

def _get_rules_for_category(
    rules: list[RuleNode],
    category: str,
    limit: int = 5,
) -> list[RuleNode]:
    """取得指定 Category 的已驗證規則。"""
    return [
        r for r in rules
        if r.category.value == category
        and r.verification_status.value == "verified"
    ][:limit]


def _get_common_mistakes(rules: list[RuleNode]) -> list[RuleNode]:
    """取得常見錯誤相關規則（restriction 類型）。"""
    return [
        r for r in rules
        if r.type.value == "restriction"
        and r.verification_status.value == "verified"
    ][:5]


def generate_teaching_script(game_id: str) -> list[dict]:
    """
    產生教學腳本。

    Args:
        game_id: 遊戲 ID

    Returns:
        list[dict]: 教學腳本（依教學順序排列）
    """
    # 取得遊戲的所有規則
    rules = get_rule_nodes_by_game(game_id, limit=5000)

    script = []

    for category, label in TEACHING_ORDER:
        category_rules = _get_rules_for_category(rules, category)

        if not category_rules:
            continue

        section = {
            "category": category,
            "title": label,
            "rules": [
                {
                    "rule_id": rule.id,
                    "title": rule.title,
                    "rule_text": rule.rule_text,
                    "simplified_text": rule.simplified_text,
                    "source": {
                        "page": rule.source.page,
                        "section": rule.source.section,
                    },
                }
                for rule in category_rules
            ],
        }
        script.append(section)

    # 加入常見錯誤
    mistakes = _get_common_mistakes(rules)
    if mistakes:
        script.append({
            "category": "common_mistakes",
            "title": "常見錯誤",
            "rules": [
                {
                    "rule_id": rule.id,
                    "title": rule.title,
                    "rule_text": rule.rule_text,
                    "simplified_text": rule.simplified_text,
                    "source": {
                        "page": rule.source.page,
                        "section": rule.source.section,
                    },
                }
                for rule in mistakes
            ],
        })

    return script


def create_teaching_session(game_id: str) -> dict:
    """
    建立教學 Session。

    Returns:
        {
            "session_id": str,
            "game_id": str,
            "script": list[dict]
        }
    """
    session_id = f"TEACH-{uuid.uuid4().hex[:8].upper()}"
    script = generate_teaching_script(game_id)

    return {
        "session_id": session_id,
        "game_id": game_id,
        "script": script,
    }


# ── 教學內容格式化 ──────────────────────────────────────────────────

def format_teaching_script(script: list[dict]) -> str:
    """將教學腳本格式化為可讀文字。"""
    lines = []
    lines.append("🎓 遊戲教學\n")
    lines.append("=" * 40)

    for section in script:
        lines.append(f"\n📌 {section['title']}")
        lines.append("-" * 30)

        for rule in section["rules"]:
            lines.append(f"\n• {rule['title']}")
            if rule.get("simplified_text"):
                lines.append(f"  {rule['simplified_text']}")
            else:
                lines.append(f"  {rule['rule_text']}")
            lines.append(f"  （來源：p.{rule['source']['page']}）")

    return "\n".join(lines)