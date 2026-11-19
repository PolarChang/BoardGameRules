"""
models.py — Board Game Rules Knowledge Base 資料模型定義

定義所有結構化資料的 Pydantic 模型，用於 SQLite 資料庫儲存與 API 序列化。
"""
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enum 定義 ──────────────────────────────────────────────────────

class GameStatus(str, Enum):
    DRAFT = "draft"
    REVIEW = "review"
    PUBLISHED = "published"


class ProcessingStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class RuleCategory(str, Enum):
    BASIC_INFO = "basic_info"
    THEME = "theme"
    OBJECTIVE = "objective"
    COMPONENTS = "components"
    SETUP = "setup"
    GAME_FLOW = "game_flow"
    TURN_STRUCTURE = "turn_structure"
    ACTIONS = "actions"
    RESOURCES = "resources"
    MECHANICS = "mechanics"
    CARDS = "cards"
    TILES = "tiles"
    TOKENS = "tokens"
    COMBAT = "combat"
    SCORING = "scoring"
    END_GAME = "end_game"
    EXCEPTIONS = "exceptions"
    FAQ = "faq"
    UNCLASSIFIED = "unclassified"


class RuleType(str, Enum):
    FACT = "fact"
    DEFINITION = "definition"
    CONDITION = "condition"
    ACTION = "action"
    COST = "cost"
    EFFECT = "effect"
    RESTRICTION = "restriction"
    TIMING = "timing"
    EXCEPTION = "exception"
    SCORING = "scoring"
    END_GAME = "end_game"
    SETUP = "setup"
    COMPONENT = "component"
    INTERACTION = "interaction"


class Confidence(str, Enum):
    EXPLICIT = "explicit"
    DERIVED = "derived"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    MISSING_SOURCE = "missing_source"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"


class ReviewAction(str, Enum):
    VERIFIED = "verified"
    EDIT = "edit"
    AMBIGUOUS = "ambiguous"
    REJECT = "reject"


# ── 資料模型 ────────────────────────────────────────────────────────

class Game(BaseModel):
    """桌遊基本資訊。"""
    id: str = Field(..., description="遊戲 ID")
    name: str = Field(..., description="遊戲名稱")
    publisher: Optional[str] = Field(None, description="出版商")
    designer: list[str] = Field(default_factory=list, description="設計師列表")
    player_count: dict[str, int] = Field(
        default_factory=lambda: {"min": 1, "max": 4},
        description="玩家人數範圍",
    )
    play_time: dict[str, int] = Field(
        default_factory=lambda: {"min": 60, "max": 90},
        description="遊玩時間範圍（分鐘）",
    )
    complexity: Optional[str] = Field(None, description="複雜度")
    status: GameStatus = Field(default=GameStatus.DRAFT, description="狀態")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新時間")


class RuleDocument(BaseModel):
    """規則書文件。"""
    id: str = Field(..., description="文件 ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    filename: str = Field(..., description="原始檔名")
    version: Optional[str] = Field(None, description="版本")
    language: str = Field(default="en", description="語言")
    file_path: str = Field(..., description="檔案路徑")
    page_count: int = Field(default=0, description="頁數")
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.UPLOADED, description="處理狀態"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間")


class SourceReference(BaseModel):
    """規則來源引用。"""
    document_id: str = Field(..., description="文件 ID")
    page: int = Field(..., description="頁碼")
    section: Optional[str] = Field(None, description="章節名稱")
    paragraph: Optional[str] = Field(None, description="段落標識")


class RuleNode(BaseModel):
    """規則節點 — 整個系統最重要的 Entity。"""
    id: str = Field(..., description="規則 ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    document_id: str = Field(..., description="來源文件 ID")

    category: RuleCategory = Field(..., description="規則類別")
    type: RuleType = Field(..., description="規則類型")

    title: str = Field(..., description="規則標題")
    rule_text: str = Field(..., description="原始規則文字")
    simplified_text: Optional[str] = Field(None, description="簡化規則文字")

    conditions: list[str] = Field(default_factory=list, description="條件")
    effects: list[str] = Field(default_factory=list, description="效果")
    costs: list[str] = Field(default_factory=list, description="成本")
    restrictions: list[str] = Field(default_factory=list, description="限制")
    exceptions: list[str] = Field(default_factory=list, description="例外")
    related_rule_ids: list[str] = Field(default_factory=list, description="相關規則 ID")

    source: SourceReference = Field(..., description="來源引用")
    confidence: Confidence = Field(default=Confidence.EXPLICIT, description="信心程度")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED, description="驗證狀態"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="更新時間")


class Component(BaseModel):
    """遊戲元件。"""
    id: str = Field(..., description="元件 ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    name: str = Field(..., description="元件名稱")
    quantity: Optional[int] = Field(None, description="數量")
    description: Optional[str] = Field(None, description="描述")
    source: SourceReference = Field(..., description="來源引用")


class Card(BaseModel):
    """遊戲卡牌。"""
    id: str = Field(..., description="卡牌 ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    name: str = Field(..., description="卡牌名稱")
    card_type: Optional[str] = Field(None, description="卡牌類型")
    description: Optional[str] = Field(None, description="描述")
    effects: list[str] = Field(default_factory=list, description="效果")
    source: SourceReference = Field(..., description="來源引用")


class Action(BaseModel):
    """遊戲行動。"""
    id: str = Field(..., description="行動 ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    name: str = Field(..., description="行動名稱")
    description: Optional[str] = Field(None, description="描述")
    costs: list[str] = Field(default_factory=list, description="成本")
    effects: list[str] = Field(default_factory=list, description="效果")
    restrictions: list[str] = Field(default_factory=list, description="限制")
    source: SourceReference = Field(..., description="來源引用")


class ScoringRule(BaseModel):
    """計分規則。"""
    id: str = Field(..., description="計分規則 ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    name: str = Field(..., description="計分項目名稱")
    description: Optional[str] = Field(None, description="描述")
    points: Optional[str] = Field(None, description="分數計算方式")
    source: SourceReference = Field(..., description="來源引用")


class FAQ(BaseModel):
    """常見問題。"""
    id: str = Field(..., description="FAQ ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    question: str = Field(..., description="問題")
    answer: str = Field(..., description="答案")
    related_rule_ids: list[str] = Field(default_factory=list, description="相關規則 ID")
    source: SourceReference = Field(..., description="來源引用")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間")


class ReviewItem(BaseModel):
    """Review Queue 項目。"""
    id: str = Field(..., description="Review 項目 ID")
    rule_id: str = Field(..., description="規則 ID")
    game_id: str = Field(..., description="所屬遊戲 ID")
    reason: str = Field(..., description="需要 Review 的原因")
    status: ReviewAction = Field(default=ReviewAction.AMBIGUOUS, description="Review 狀態")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="建立時間")
    reviewed_at: Optional[datetime] = Field(None, description="Review 時間")


# ── API 請求/回應模型 ──────────────────────────────────────────────

class RuleSearchRequest(BaseModel):
    """規則檢索請求。"""
    query: str = Field(..., description="查詢文字")
    limit: int = Field(default=10, ge=1, le=50, description="結果數量上限")


class RuleSearchResult(BaseModel):
    """規則檢索結果。"""
    rule_id: str = Field(..., description="規則 ID")
    score: float = Field(..., description="相關度分數")


class RuleSearchResponse(BaseModel):
    """規則檢索回應。"""
    rules: list[RuleSearchResult] = Field(..., description="規則列表")


class QARequest(BaseModel):
    """AI 問答請求。"""
    question: str = Field(..., description="問題")


class QAResponse(BaseModel):
    """AI 問答回應。"""
    answer: str = Field(..., description="答案")
    confidence: Confidence = Field(..., description="信心程度")
    rule_ids: list[str] = Field(default_factory=list, description="相關規則 ID")
    sources: list[dict] = Field(default_factory=list, description="來源列表")


class TeachingSessionRequest(BaseModel):
    """AI 教學請求。"""
    game_id: str = Field(..., description="遊戲 ID")


class TeachingSessionResponse(BaseModel):
    """AI 教學回應。"""
    session_id: str = Field(..., description="教學 Session ID")
    script: list[dict] = Field(..., description="教學腳本")


class ImportResult(BaseModel):
    """PDF 匯入結果。"""
    document_id: str = Field(..., description="文件 ID")
    rules_extracted: int = Field(..., description="提取規則數量")
    explicit: int = Field(..., description="Explicit 規則數量")
    derived: int = Field(..., description="Derived 規則數量")
    ambiguous: int = Field(..., description="Ambiguous 規則數量")
    conflict: int = Field(..., description="Conflict 規則數量")
    need_review: int = Field(..., description="需要 Review 的規則數量")