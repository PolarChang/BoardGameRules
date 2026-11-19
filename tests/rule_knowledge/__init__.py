"""
test_rule_knowledge.py — Rule Knowledge Base Acceptance Tests

遵循 Implementation Plan 的 Acceptance Test：
1. Import Test — PDF → RuleNode
2. Source Test — 確認每個 Rule 有 page
3. Retrieval Test — 「怎麼結束遊戲？」必須找到 end_game
4. Exception Test — 「特殊能力是否可以突破基本限制？」必須同時 Retrieval base_rule + exception
5. Hallucination Test — Rulebook 完全沒有提到的事情 → Cannot determine
"""
import os
import sys
import tempfile
import uuid
from pathlib import Path

# 確保專案根目錄在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from src.database import (
    create_game,
    create_rule_node,
    get_rule_node,
    get_rule_nodes_by_game,
    init_db,
    update_rule_node,
)
from src.extractor import extract_rules_rule_based
from src.models import (
    Confidence,
    Game,
    GameStatus,
    RuleCategory,
    RuleNode,
    RuleType,
    SourceReference,
    VerificationStatus,
)
from src.pdf_parser import Chunk, parse_pdf
from src.qa_service import answer_question
from src.retrieval import search_rules, search_with_exceptions
from src.validator import apply_validation, review_rule


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db():
    """每個測試前初始化資料庫。"""
    init_db()


@pytest.fixture
def test_game():
    """建立測試遊戲。"""
    game_id = f"GAME-{uuid.uuid4().hex[:8].upper()}"
    game = Game(
        id=game_id,
        name="Test Game",
        publisher="Test Publisher",
        designer=["Test Designer"],
        status=GameStatus.DRAFT,
    )
    create_game(game)
    return game_id


@pytest.fixture
def test_document_id():
    """建立測試文件 ID。"""
    return f"DOC-{uuid.uuid4().hex[:8].upper()}"


def create_test_rule(
    game_id: str,
    document_id: str,
    category: RuleCategory,
    rule_type: RuleType,
    title: str,
    rule_text: str,
    page: int = 1,
    confidence: Confidence = Confidence.EXPLICIT,
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
) -> RuleNode:
    """建立測試規則。"""
    rule = RuleNode(
        id=f"RULE-{uuid.uuid4().hex[:8].upper()}",
        game_id=game_id,
        document_id=document_id,
        category=category,
        type=rule_type,
        title=title,
        rule_text=rule_text,
        source=SourceReference(
            document_id=document_id,
            page=page,
            section="Test Section",
        ),
        confidence=confidence,
        verification_status=verification_status,
    )
    create_rule_node(rule)
    return rule


# ── Import Test ────────────────────────────────────────────────────

class TestImport:
    """Import Test — PDF → RuleNode"""

    def test_parse_pdf_creates_chunks(self, test_document_id):
        """測試 PDF 解析能建立 Chunk。"""
        import fitz

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Game Setup\nPlayers place their tokens on the board.\n")
            page.insert_text((72, 144), "Scoring\nPlayers score points at the end of the game.\n")
            doc.save(tmp.name)
            doc.close()
            pdf_path = tmp.name

        try:
            pages, sections, chunks = parse_pdf(pdf_path, test_document_id)
            assert len(pages) > 0
            assert len(sections) > 0
            assert len(chunks) > 0
            # 每個 Chunk 必須有 document_id, page_number, section, raw_text
            for chunk in chunks:
                assert chunk.document_id == test_document_id
                assert chunk.page_number > 0
                assert chunk.section
                assert chunk.raw_text
        finally:
            Path(pdf_path).unlink(missing_ok=True)

    def test_extract_rules_from_chunks(self, test_game, test_document_id):
        """測試從 Chunk 提取 RuleNode。"""
        chunks = [
            Chunk(
                document_id=test_document_id,
                page_number=1,
                section="Game Setup",
                raw_text="Players place their tokens on the board. Each player draws 5 cards.",
                chunk_index=0,
            ),
            Chunk(
                document_id=test_document_id,
                page_number=2,
                section="Scoring",
                raw_text="Players score points at the end of the game. The player with the most points wins.",
                chunk_index=1,
            ),
        ]

        rules = extract_rules_rule_based(chunks, test_game)
        assert len(rules) > 0
        # 每個 Rule 必須有 Source
        for rule in rules:
            assert rule.source.document_id == test_document_id
            assert rule.source.page > 0


# ── Source Test ────────────────────────────────────────────────────

class TestSource:
    """Source Test — 確認每個 Rule 有 page"""

    def test_rule_has_source(self, test_game, test_document_id):
        """測試每個 Rule 有 page。"""
        rule = create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.SETUP,
            RuleType.SETUP,
            "Setup Rule",
            "Players place their tokens on the board.",
            page=5,
        )
        assert rule.source.page == 5
        assert rule.source.document_id == test_document_id

    def test_validation_marks_missing_source(self, test_game, test_document_id):
        """測試缺少 Source 的規則被標記為 missing_source。"""
        rule = RuleNode(
            id=f"RULE-{uuid.uuid4().hex[:8].upper()}",
            game_id=test_game,
            document_id=test_document_id,
            category=RuleCategory.SETUP,
            type=RuleType.SETUP,
            title="No Source Rule",
            rule_text="Players place their tokens on the board.",
            source=SourceReference(
                document_id="",
                page=0,
            ),
            confidence=Confidence.EXPLICIT,
            verification_status=VerificationStatus.UNVERIFIED,
        )
        create_rule_node(rule)

        validation = apply_validation([rule])
        assert validation["invalid"] > 0

        updated = get_rule_node(rule.id)
        assert updated.verification_status == VerificationStatus.MISSING_SOURCE


# ── Retrieval Test ─────────────────────────────────────────────────

class TestRetrieval:
    """Retrieval Test — 「怎麼結束遊戲？」必須找到 end_game"""

    def test_search_finds_end_game(self, test_game, test_document_id):
        """測試檢索能找到 end_game 規則。"""
        # 建立 end_game 規則
        create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.END_GAME,
            RuleType.END_GAME,
            "Game End",
            "The game ends when all players have taken their turns.",
            page=10,
            verification_status=VerificationStatus.VERIFIED,
        )
        # 建立其他規則
        create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.SETUP,
            RuleType.SETUP,
            "Setup",
            "Players place their tokens on the board.",
            page=1,
            verification_status=VerificationStatus.VERIFIED,
        )

        results = search_rules(test_game, "怎麼結束遊戲？", limit=10)
        assert len(results) > 0
        # 至少一個結果是 end_game
        end_game_found = False
        for result in results:
            rule = get_rule_node(result.rule_id)
            if rule and rule.category == RuleCategory.END_GAME:
                end_game_found = True
                break
        assert end_game_found


# ── Exception Test ─────────────────────────────────────────────────

class TestException:
    """Exception Test — 「特殊能力是否可以突破基本限制？」必須同時 Retrieval base_rule + exception"""

    def test_search_with_exceptions(self, test_game, test_document_id):
        """測試檢索同時找到 base_rule 和 exception。"""
        # 建立基本限制規則
        base_rule = create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.ACTIONS,
            RuleType.RESTRICTION,
            "Action Limit",
            "Players cannot take more than one action per turn.",
            page=3,
            verification_status=VerificationStatus.VERIFIED,
        )
        # 建立例外規則
        exception_rule = create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.EXCEPTIONS,
            RuleType.EXCEPTION,
            "Special Ability Exception",
            "Special abilities allow players to take additional actions.",
            page=7,
            verification_status=VerificationStatus.VERIFIED,
        )
        # 建立關聯
        base_rule.related_rule_ids.append(exception_rule.id)
        update_rule_node(base_rule)

        result = search_with_exceptions(test_game, "特殊能力是否可以突破基本限制？", limit=10)
        assert len(result["rules"]) > 0
        assert len(result["exceptions"]) > 0

        # 確認同時有 base_rule 和 exception
        rule_ids = [r.rule_id for r in result["rules"]]
        exception_ids = [r.rule_id for r in result["exceptions"]]
        assert base_rule.id in rule_ids or base_rule.id in exception_ids
        assert exception_rule.id in exception_ids or exception_rule.id in rule_ids


# ── Hallucination Test ─────────────────────────────────────────────

class TestHallucination:
    """Hallucination Test — Rulebook 完全沒有提到的事情 → Cannot determine"""

    def test_unknown_question_returns_cannot_determine(self, test_game, test_document_id):
        """測試規則書沒有提到的問題回傳 Cannot determine。"""
        # 建立一些規則
        create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.SETUP,
            RuleType.SETUP,
            "Setup",
            "Players place their tokens on the board.",
            page=1,
            verification_status=VerificationStatus.VERIFIED,
        )

        # 問一個規則書完全沒有提到的問題
        response = answer_question(test_game, "這個遊戲有擴充包嗎？")
        assert "沒有足夠資訊" in response.answer or "無法判定" in response.answer
        assert response.confidence == Confidence.AMBIGUOUS


# ── Review Test ────────────────────────────────────────────────────

class TestReview:
    """Review Test — Review Queue 操作"""

    def test_review_rule_verified(self, test_game, test_document_id):
        """測試 Review 規則為 Verified。"""
        rule = create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.SETUP,
            RuleType.SETUP,
            "Setup Rule",
            "Players place their tokens on the board.",
            page=5,
        )

        reviewed = review_rule(rule.id, "verified")
        assert reviewed.verification_status == VerificationStatus.VERIFIED

    def test_review_rule_rejected(self, test_game, test_document_id):
        """測試 Review 規則為 Rejected。"""
        rule = create_test_rule(
            test_game,
            test_document_id,
            RuleCategory.SETUP,
            RuleType.SETUP,
            "Setup Rule",
            "Players place their tokens on the board.",
            page=5,
        )

        reviewed = review_rule(rule.id, "reject")
        assert reviewed.verification_status == VerificationStatus.REJECTED