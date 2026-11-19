"""
database.py — SQLite / PostgreSQL 資料庫層

負責所有結構化資料的持久化儲存：
- Game
- RuleDocument
- RuleNode
- Component / Card / Action / ScoringRule / FAQ
- ReviewItem

本機預設使用 SQLite；設定 Cloud SQL 環境變數後使用 PostgreSQL。
"""
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.config import DB_DIR, get_cloud_settings
from src.models import (
    Action,
    Card,
    Component,
    FAQ,
    Game,
    ReviewAction,
    ReviewItem,
    RuleDocument,
    RuleNode,
    ScoringRule,
    SourceReference,
)

DB_PATH = DB_DIR / "rules.db"
_sqlite3 = sqlite3


class _PostgresCursor:
    """Adapt this module's existing SQLite ``?`` placeholders for psycopg."""

    def __init__(self, cursor):
        self._cursor = cursor

    @staticmethod
    def _sql(query: str) -> str:
        return query.replace("?", "%s")

    def execute(self, query: str, params=None):
        return self._cursor.execute(self._sql(query), params)

    def executemany(self, query: str, params_seq):
        return self._cursor.executemany(self._sql(query), params_seq)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _PostgresConnection:
    """Small compatibility wrapper so existing CRUD code works in both modes."""

    def __init__(self, connection):
        self._connection = connection
        self.row_factory = None

    def cursor(self):
        return _PostgresCursor(self._connection.cursor())

    def commit(self):
        self._connection.commit()

    def close(self):
        self._connection.close()


def _connect():
    settings = get_cloud_settings()
    if settings.uses_postgres:
        from psycopg import connect
        from psycopg.rows import dict_row

        return _PostgresConnection(connect(settings.database_url, row_factory=dict_row))

    conn = _sqlite3.connect(str(DB_PATH))
    conn.row_factory = _sqlite3.Row
    return conn


class _DatabaseDriver:
    """Expose the small sqlite3 surface used below with Cloud SQL routing."""

    Row = _sqlite3.Row

    @staticmethod
    def connect(*_args, **_kwargs):
        return _connect()


# Existing CRUD functions intentionally keep their SQLite-shaped SQL.  This
# proxy routes their connections to PostgreSQL when Cloud Run supplies DB_HOST
# and DB_PASSWORD, while retaining SQLite as the zero-configuration default.
sqlite3 = _DatabaseDriver()


# ── 工具函式 ────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.utcnow().isoformat()


def _to_json(obj: Any) -> str:
    # Pydantic models → dict
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    # dict with nested Pydantic models → convert nested
    elif isinstance(obj, dict):
        obj = {
            k: v.model_dump() if hasattr(v, "model_dump") else v
            for k, v in obj.items()
        }
    return json.dumps(obj, ensure_ascii=False, default=str)


def _from_json(text: str, default: Any = None) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def _row_to_game(row: sqlite3.Row) -> Game:
    return Game(
        id=row["id"],
        name=row["name"],
        publisher=row["publisher"],
        designer=_from_json(row["designer"], []),
        player_count=_from_json(row["player_count"], {"min": 1, "max": 4}),
        play_time=_from_json(row["play_time"], {"min": 60, "max": 90}),
        complexity=row["complexity"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_document(row: sqlite3.Row) -> RuleDocument:
    return RuleDocument(
        id=row["id"],
        game_id=row["game_id"],
        filename=row["filename"],
        version=row["version"],
        language=row["language"],
        file_path=row["file_path"],
        page_count=row["page_count"],
        processing_status=row["processing_status"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


def _row_to_rule_node(row: sqlite3.Row) -> RuleNode:
    source_data = _from_json(row["source"], {})
    # 確保 source 是 dict（來自 JSON），轉為 SourceReference
    if isinstance(source_data, dict):
        source_ref = SourceReference(**source_data)
    else:
        source_ref = source_data
    return RuleNode(
        id=row["id"],
        game_id=row["game_id"],
        document_id=row["document_id"],
        category=row["category"],
        type=row["type"],
        title=row["title"],
        rule_text=row["rule_text"],
        simplified_text=row["simplified_text"],
        conditions=_from_json(row["conditions"], []),
        effects=_from_json(row["effects"], []),
        costs=_from_json(row["costs"], []),
        restrictions=_from_json(row["restrictions"], []),
        exceptions=_from_json(row["exceptions"], []),
        related_rule_ids=_from_json(row["related_rule_ids"], []),
        source=source_ref,
        confidence=row["confidence"],
        verification_status=row["verification_status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_review_item(row: sqlite3.Row) -> ReviewItem:
    return ReviewItem(
        id=row["id"],
        rule_id=row["rule_id"],
        game_id=row["game_id"],
        reason=row["reason"],
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        reviewed_at=(
            datetime.fromisoformat(row["reviewed_at"]) if row["reviewed_at"] else None
        ),
    )


# ── 資料庫初始化 ────────────────────────────────────────────────────

def init_db() -> None:
    """初始化資料庫 schema。"""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Game
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            publisher TEXT,
            designer TEXT DEFAULT '[]',
            player_count TEXT DEFAULT '{"min": 1, "max": 4}',
            play_time TEXT DEFAULT '{"min": 60, "max": 90}',
            complexity TEXT,
            status TEXT DEFAULT 'draft',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # RuleDocument
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rule_documents (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            version TEXT,
            language TEXT DEFAULT 'en',
            file_path TEXT NOT NULL,
            page_count INTEGER DEFAULT 0,
            processing_status TEXT DEFAULT 'uploaded',
            created_at TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    """)

    # RuleNode
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rule_nodes (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            document_id TEXT NOT NULL,
            category TEXT NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            rule_text TEXT NOT NULL,
            simplified_text TEXT,
            conditions TEXT DEFAULT '[]',
            effects TEXT DEFAULT '[]',
            costs TEXT DEFAULT '[]',
            restrictions TEXT DEFAULT '[]',
            exceptions TEXT DEFAULT '[]',
            related_rule_ids TEXT DEFAULT '[]',
            source TEXT NOT NULL,
            confidence TEXT DEFAULT 'explicit',
            verification_status TEXT DEFAULT 'unverified',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games (id),
            FOREIGN KEY (document_id) REFERENCES rule_documents (id)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_nodes_game ON rule_nodes (game_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_nodes_doc ON rule_nodes (document_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_nodes_category ON rule_nodes (category)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_rule_nodes_status ON rule_nodes (verification_status)"
    )

    # Component
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS components (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            name TEXT NOT NULL,
            quantity INTEGER,
            description TEXT,
            source TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    """)

    # Card
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cards (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            name TEXT NOT NULL,
            card_type TEXT,
            description TEXT,
            effects TEXT DEFAULT '[]',
            source TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    """)

    # Action
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            costs TEXT DEFAULT '[]',
            effects TEXT DEFAULT '[]',
            restrictions TEXT DEFAULT '[]',
            source TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    """)

    # ScoringRule
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scoring_rules (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            points TEXT,
            source TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    """)

    # FAQ
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS faqs (
            id TEXT PRIMARY KEY,
            game_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            related_rule_ids TEXT DEFAULT '[]',
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    """)

    # ReviewItem
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS review_items (
            id TEXT PRIMARY KEY,
            rule_id TEXT NOT NULL,
            game_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT DEFAULT 'ambiguous',
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            FOREIGN KEY (rule_id) REFERENCES rule_nodes (id),
            FOREIGN KEY (game_id) REFERENCES games (id)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_review_game ON review_items (game_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_review_status ON review_items (status)"
    )

    conn.commit()
    conn.close()


# ── Game CRUD ───────────────────────────────────────────────────────

def create_game(game: Game) -> Game:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    now = _now()
    cursor.execute(
        """
        INSERT INTO games (id, name, publisher, designer, player_count, play_time,
                           complexity, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game.id,
            game.name,
            game.publisher,
            _to_json(game.designer),
            _to_json(game.player_count),
            _to_json(game.play_time),
            game.complexity,
            game.status.value,
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return game


def get_game(game_id: str) -> Optional[Game]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM games WHERE id = ?", (game_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_game(row)


def get_all_games() -> list[Game]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM games ORDER BY name")
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_game(row) for row in rows]


def update_game(game: Game) -> Game:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE games
        SET name = ?, publisher = ?, designer = ?, player_count = ?, play_time = ?,
            complexity = ?, status = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            game.name,
            game.publisher,
            _to_json(game.designer),
            _to_json(game.player_count),
            _to_json(game.play_time),
            game.complexity,
            game.status.value,
            _now(),
            game.id,
        ),
    )
    conn.commit()
    conn.close()
    return game


def delete_game(game_id: str) -> bool:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM games WHERE id = ?", (game_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


# ── RuleDocument CRUD ───────────────────────────────────────────────

def create_document(doc: RuleDocument) -> RuleDocument:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO rule_documents (id, game_id, filename, version, language,
                                    file_path, page_count, processing_status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc.id,
            doc.game_id,
            doc.filename,
            doc.version,
            doc.language,
            doc.file_path,
            doc.page_count,
            doc.processing_status.value,
            doc.created_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return doc


def get_document(doc_id: str) -> Optional[RuleDocument]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rule_documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_document(row)


def get_documents_by_game(game_id: str) -> list[RuleDocument]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM rule_documents WHERE game_id = ? ORDER BY created_at",
        (game_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_document(row) for row in rows]


def update_document_status(doc_id: str, status: str) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE rule_documents SET processing_status = ? WHERE id = ?",
        (status, doc_id),
    )
    conn.commit()
    conn.close()


def update_document(doc: RuleDocument) -> RuleDocument:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE rule_documents
        SET filename = ?, version = ?, language = ?, file_path = ?,
            page_count = ?, processing_status = ?
        WHERE id = ?
        """,
        (
            doc.filename,
            doc.version,
            doc.language,
            doc.file_path,
            doc.page_count,
            doc.processing_status.value,
            doc.id,
        ),
    )
    conn.commit()
    conn.close()
    return doc


# ── RuleNode CRUD ───────────────────────────────────────────────────

def create_rule_node(rule: RuleNode) -> RuleNode:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    now = _now()
    cursor.execute(
        """
        INSERT INTO rule_nodes (id, game_id, document_id, category, type, title,
                                rule_text, simplified_text, conditions, effects, costs,
                                restrictions, exceptions, related_rule_ids, source,
                                confidence, verification_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rule.id,
            rule.game_id,
            rule.document_id,
            rule.category.value,
            rule.type.value,
            rule.title,
            rule.rule_text,
            rule.simplified_text,
            _to_json(rule.conditions),
            _to_json(rule.effects),
            _to_json(rule.costs),
            _to_json(rule.restrictions),
            _to_json(rule.exceptions),
            _to_json(rule.related_rule_ids),
            _to_json(rule.source),
            rule.confidence.value,
            rule.verification_status.value,
            now,
            now,
        ),
    )
    conn.commit()
    conn.close()
    return rule


def create_rule_nodes_bulk(rules: list[RuleNode], _connection=None) -> int:
    """批次建立 RuleNode，提升匯入效能。"""
    if not rules:
        return 0
    conn = _connection or sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    now = _now()
    rows = []
    for rule in rules:
        rows.append((
            rule.id,
            rule.game_id,
            rule.document_id,
            rule.category.value,
            rule.type.value,
            rule.title,
            rule.rule_text,
            rule.simplified_text,
            _to_json(rule.conditions),
            _to_json(rule.effects),
            _to_json(rule.costs),
            _to_json(rule.restrictions),
            _to_json(rule.exceptions),
            _to_json(rule.related_rule_ids),
            _to_json(rule.source),
            rule.confidence.value,
            rule.verification_status.value,
            now,
            now,
        ))
    cursor.executemany(
        """
        INSERT INTO rule_nodes (id, game_id, document_id, category, type, title,
                                rule_text, simplified_text, conditions, effects, costs,
                                restrictions, exceptions, related_rule_ids, source,
                                confidence, verification_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if _connection is None:
        conn.commit()
        conn.close()
    return len(rules)


def get_rule_node(rule_id: str) -> Optional[RuleNode]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM rule_nodes WHERE id = ?", (rule_id,))
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return _row_to_rule_node(row)


def get_rule_nodes_by_game(
    game_id: str,
    category: Optional[str] = None,
    verification_status: Optional[str] = None,
    limit: int = 1000,
) -> list[RuleNode]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    query = "SELECT * FROM rule_nodes WHERE game_id = ?"
    params: list[Any] = [game_id]
    if category:
        query += " AND category = ?"
        params.append(category)
    if verification_status:
        query += " AND verification_status = ?"
        params.append(verification_status)
    query += " ORDER BY created_at LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_rule_node(row) for row in rows]


def get_rule_nodes_by_document(doc_id: str) -> list[RuleNode]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM rule_nodes WHERE document_id = ? ORDER BY created_at",
        (doc_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_rule_node(row) for row in rows]


def update_rule_node(rule: RuleNode) -> RuleNode:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE rule_nodes
        SET category = ?, type = ?, title = ?, rule_text = ?, simplified_text = ?,
            conditions = ?, effects = ?, costs = ?, restrictions = ?, exceptions = ?,
            related_rule_ids = ?, source = ?, confidence = ?, verification_status = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            rule.category.value,
            rule.type.value,
            rule.title,
            rule.rule_text,
            rule.simplified_text,
            _to_json(rule.conditions),
            _to_json(rule.effects),
            _to_json(rule.costs),
            _to_json(rule.restrictions),
            _to_json(rule.exceptions),
            _to_json(rule.related_rule_ids),
            _to_json(rule.source),
            rule.confidence.value,
            rule.verification_status.value,
            _now(),
            rule.id,
        ),
    )
    conn.commit()
    conn.close()
    return rule


def update_rule_verification(rule_id: str, status: str) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE rule_nodes SET verification_status = ?, updated_at = ? WHERE id = ?",
        (status, _now(), rule_id),
    )
    conn.commit()
    conn.close()


def delete_rule_node(rule_id: str) -> bool:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("DELETE FROM rule_nodes WHERE id = ?", (rule_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def count_rule_nodes(game_id: Optional[str] = None) -> int:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    if game_id:
        cursor.execute("SELECT COUNT(*) FROM rule_nodes WHERE game_id = ?", (game_id,))
    else:
        cursor.execute("SELECT COUNT(*) FROM rule_nodes")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_rule_stats(game_id: str) -> dict:
    """取得遊戲的規則統計（用於 Import Result 顯示）。"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT confidence, COUNT(*) as cnt FROM rule_nodes WHERE game_id = ? GROUP BY confidence",
        (game_id,),
    )
    confidence_rows = cursor.fetchall()
    cursor.execute(
        "SELECT COUNT(*) as cnt FROM rule_nodes WHERE game_id = ? AND verification_status != 'verified'",
        (game_id,),
    )
    need_review = cursor.fetchone()["cnt"]
    conn.close()

    stats = {
        "explicit": 0,
        "derived": 0,
        "ambiguous": 0,
        "conflict": 0,
        "need_review": need_review,
    }
    for row in confidence_rows:
        key = row["confidence"]
        if key in stats:
            stats[key] = row["cnt"]
    return stats


# ── ReviewItem CRUD ─────────────────────────────────────────────────

def create_review_item(item: ReviewItem) -> ReviewItem:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO review_items (id, rule_id, game_id, reason, status, created_at, reviewed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.id,
            item.rule_id,
            item.game_id,
            item.reason,
            item.status.value,
            item.created_at.isoformat(),
            item.reviewed_at.isoformat() if item.reviewed_at else None,
        ),
    )
    conn.commit()
    conn.close()
    return item


def create_review_items_bulk(items: list[ReviewItem], _connection=None) -> int:
    if not items:
        return 0
    conn = _connection or sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    rows = [
        (
            item.id,
            item.rule_id,
            item.game_id,
            item.reason,
            item.status.value,
            item.created_at.isoformat(),
            item.reviewed_at.isoformat() if item.reviewed_at else None,
        )
        for item in items
    ]
    cursor.executemany(
        """
        INSERT INTO review_items (id, rule_id, game_id, reason, status, created_at, reviewed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if _connection is None:
        conn.commit()
        conn.close()
    return len(items)


def get_review_items(
    game_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> list[ReviewItem]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    query = "SELECT * FROM review_items WHERE 1=1"
    params: list[Any] = []
    if game_id:
        query += " AND game_id = ?"
        params.append(game_id)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY created_at LIMIT ?"
    params.append(limit)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [_row_to_review_item(row) for row in rows]


def update_review_item_status(item_id: str, status: str) -> None:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE review_items SET status = ?, reviewed_at = ? WHERE id = ?",
        (status, _now(), item_id),
    )
    conn.commit()
    conn.close()


# ── 其他 Entity CRUD（簡化版） ─────────────────────────────────────

def create_component(component: Component) -> Component:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO components (id, game_id, name, quantity, description, source)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            component.id,
            component.game_id,
            component.name,
            component.quantity,
            component.description,
            _to_json(component.source),
        ),
    )
    conn.commit()
    conn.close()
    return component


def create_card(card: Card) -> Card:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO cards (id, game_id, name, card_type, description, effects, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            card.id,
            card.game_id,
            card.name,
            card.card_type,
            card.description,
            _to_json(card.effects),
            _to_json(card.source),
        ),
    )
    conn.commit()
    conn.close()
    return card


def create_action(action: Action) -> Action:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO actions (id, game_id, name, description, costs, effects, restrictions, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            action.id,
            action.game_id,
            action.name,
            action.description,
            _to_json(action.costs),
            _to_json(action.effects),
            _to_json(action.restrictions),
            _to_json(action.source),
        ),
    )
    conn.commit()
    conn.close()
    return action


def create_scoring_rule(scoring: ScoringRule) -> ScoringRule:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO scoring_rules (id, game_id, name, description, points, source)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            scoring.id,
            scoring.game_id,
            scoring.name,
            scoring.description,
            scoring.points,
            _to_json(scoring.source),
        ),
    )
    conn.commit()
    conn.close()
    return scoring


def create_faq(faq: FAQ) -> FAQ:
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO faqs (id, game_id, question, answer, related_rule_ids, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            faq.id,
            faq.game_id,
            faq.question,
            faq.answer,
            _to_json(faq.related_rule_ids),
            _to_json(faq.source),
            faq.created_at.isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return faq


def get_faqs_by_game(game_id: str) -> list[FAQ]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM faqs WHERE game_id = ? ORDER BY created_at",
        (game_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        FAQ(
            id=row["id"],
            game_id=row["game_id"],
            question=row["question"],
            answer=row["answer"],
            related_rule_ids=_from_json(row["related_rule_ids"], []),
            source=_from_json(row["source"], {}),
            created_at=datetime.fromisoformat(row["created_at"]),
        )
        for row in rows
    ]


# ── 初始化 ──────────────────────────────────────────────────────────

def get_db_path() -> Path:
    return DB_PATH


def complete_rulebook_import(document_id, rules, review_items):
    """Commit rules, review queue and document completion together, once."""
    conn = sqlite3.connect(str(DB_PATH))
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE rule_documents SET processing_status = 'processed' WHERE id = ? AND processing_status IN ('processing', 'failed')", (document_id,))
        if cursor.rowcount != 1:
            raise ValueError("Document is missing or already completed")
        create_rule_nodes_bulk(rules, _connection=conn)
        create_review_items_bulk(review_items, _connection=conn)
        conn.commit()
    finally:
        # Closing an uncommitted transaction rolls back every change.
        conn.close()
