from fastapi.testclient import TestClient
from src import database


def test_local_catalog_supports_selection_creation_and_detail(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "catalog.db")
    for key in ("DB_HOST", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EDITORIAL_QA_PROVIDER", "local")
    monkeypatch.setenv("KNOWLEDGE_ADMIN_KEY", "test-catalog-admin")
    from src.local_knowledge_server import app
    from src.editorial_api import init_editorial_db
    from src.editorial_sessions import create as create_session
    database.init_db()
    init_editorial_db()
    client = TestClient(app)
    assert client.get("/api/knowledge/games").json() == {"games": []}
    body = {"id": "catalog-test", "name": "Catalog test", "designer": ["Designer"]}
    # Unauthenticated or wrong admin credential must be rejected with 401
    assert client.post("/api/knowledge/games", json=body).status_code == 401
    assert client.post("/api/knowledge/games", json=body, headers={"X-Knowledge-Admin": "wrong"}).status_code == 401
    session_token = create_session("test-catalog-admin")
    assert client.post("/api/knowledge/games", json=body, headers={"X-Knowledge-Admin": session_token}).status_code == 200
    games = client.get("/api/knowledge/games").json()["games"]
    assert len(games) == 1 and games[0]["id"] == body["id"]
    assert games[0]["designer"] == ["Designer"]
    assert client.get("/api/knowledge/games/catalog-test").json()["name"] == body["name"]
    assert client.get("/api/knowledge/games/missing").status_code == 404
    assert client.post("/api/knowledge/games", json=body, headers={"X-Knowledge-Admin": "test-catalog-admin"}).status_code == 400
    assert client.post("/api/knowledge/games", json={"name": "Missing ID"}, headers={"X-Knowledge-Admin": "test-catalog-admin"}).status_code == 422


def test_legacy_knowledge_write_routes_enforce_admin_credentials_and_session_revocation(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "legacy_auth.db")
    for key in ("DB_HOST", "DATABASE_URL", "RULEBOOK_STORAGE_BUCKET", "RULES_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KNOWLEDGE_DISABLE_LOCAL_ENV", "true")
    monkeypatch.setenv("EDITORIAL_DAILY_AI_LIMIT", "0")
    monkeypatch.setenv("ENABLE_LEGACY_PDF_EXTRACTION", "true")
    monkeypatch.setenv("KNOWLEDGE_ADMIN_KEY", "secret-admin-key")
    from fastapi import FastAPI
    from src.api import router as knowledge_router
    from src.editorial_api import init_editorial_db, router as editorial_router
    from src.editorial_sessions import create as create_session, revoke as revoke_session

    database.init_db()
    init_editorial_db()
    test_app = FastAPI()
    test_app.include_router(knowledge_router)
    test_app.include_router(editorial_router)
    client = TestClient(test_app)

    game_body = {"id": "legacy-auth-game", "name": "Legacy Auth Game"}
    # 1. POST /api/knowledge/games rejects missing or invalid credentials
    assert client.post("/api/knowledge/games", json=game_body).status_code == 401
    assert client.post("/api/knowledge/games", json=game_body, headers={"X-Knowledge-Admin": "wrong-key"}).status_code == 401

    # Valid session token creates game
    session_token = create_session("secret-admin-key")
    assert client.post("/api/knowledge/games", json=game_body, headers={"X-Knowledge-Admin": session_token}).status_code == 200

    # 2. PUT & DELETE /api/knowledge/games/{game_id} reject missing/wrong credentials
    assert client.put("/api/knowledge/games/legacy-auth-game", json={**game_body, "name": "Updated"}).status_code == 401
    assert client.put("/api/knowledge/games/legacy-auth-game", json={**game_body, "name": "Updated"}, headers={"X-Knowledge-Admin": "wrong"}).status_code == 401
    assert client.put("/api/knowledge/games/legacy-auth-game", json={**game_body, "name": "Updated"}, headers={"X-Knowledge-Admin": session_token}).status_code == 200

    # 3. Rulebook upload-url, import, and multipart upload reject missing, wrong, and revoked session tokens
    upload_url_path = "/api/knowledge/games/legacy-auth-game/rulebooks/upload-url"
    import_path = "/api/knowledge/games/legacy-auth-game/rulebooks/import"
    rulebooks_path = "/api/knowledge/games/legacy-auth-game/rulebooks"

    for path, kwargs in (
        (upload_url_path, {"json": {"filename": "rules.pdf"}}),
        (import_path, {"json": {"object_name": "obj.pdf", "filename": "rules.pdf"}}),
        (rulebooks_path, {"files": {"file": ("rules.txt", b"not-pdf", "text/plain")}}),
    ):
        assert client.post(path, **kwargs).status_code == 401
        assert client.post(path, headers={"X-Knowledge-Admin": "wrong-key"}, **kwargs).status_code == 401

    # With valid session token, requests pass auth and reach domain validation (503 for unconfigured storage bucket, 400 for non-pdf)
    assert client.post(upload_url_path, json={"filename": "rules.txt"}, headers={"X-Knowledge-Admin": session_token}).status_code == 400
    assert client.post(import_path, json={"object_name": "obj", "filename": "rules.txt"}, headers={"X-Knowledge-Admin": session_token}).status_code == 400
    assert client.post(rulebooks_path, files={"file": ("rules.txt", b"not-pdf", "text/plain")}, headers={"X-Knowledge-Admin": session_token}).status_code == 400

    # Revoke session token -> all legacy write endpoints immediately reject with 401
    revoke_session(session_token)
    assert client.post(upload_url_path, json={"filename": "rules.txt"}, headers={"X-Knowledge-Admin": session_token}).status_code == 401
    assert client.post(import_path, json={"object_name": "obj", "filename": "rules.txt"}, headers={"X-Knowledge-Admin": session_token}).status_code == 401
    assert client.post(rulebooks_path, files={"file": ("rules.txt", b"not-pdf", "text/plain")}, headers={"X-Knowledge-Admin": session_token}).status_code == 401
    assert client.delete("/api/knowledge/games/legacy-auth-game", headers={"X-Knowledge-Admin": session_token}).status_code == 401
    assert client.delete("/api/knowledge/games/legacy-auth-game", headers={"X-Knowledge-Admin": "secret-admin-key"}).status_code == 200


def test_split_deployment_rulebook_writes_accept_proxy_signed_private_session_across_separate_sqlite_dbs(tmp_path, monkeypatch):
    import hashlib
    import hmac
    import time
    from fastapi import FastAPI
    from src.api import router as knowledge_router
    from src.editorial_api import init_editorial_db, router as editorial_router
    from src.editorial_sessions import create as create_session, valid as valid_session

    for key in ("DB_HOST", "DATABASE_URL", "RULEBOOK_STORAGE_BUCKET", "RULES_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KNOWLEDGE_DISABLE_LOCAL_ENV", "true")
    monkeypatch.setenv("ENABLE_LEGACY_PDF_EXTRACTION", "true")
    monkeypatch.setenv("KNOWLEDGE_ADMIN_KEY", "shared-admin-password")
    monkeypatch.setenv("COMMUNITY_PROXY_SECRET", "split-proxy-signing-secret")

    # 1. Private DB issues a ksession.* token
    private_db = tmp_path / "private_ editorial.db"
    public_db = tmp_path / "public_cloud.db"
    monkeypatch.setattr(database, "DB_PATH", private_db)
    database.init_db()
    init_editorial_db()
    private_token = create_session("shared-admin-password")
    assert valid_session(private_token, "shared-admin-password") is True

    # 2. Switch to Public DB (separate SQLite database that does NOT have private_token in editorial_sessions!)
    monkeypatch.setattr(database, "DB_PATH", public_db)
    database.init_db()
    init_editorial_db()
    assert valid_session(private_token, "shared-admin-password") is False

    public_app = FastAPI()
    public_app.include_router(knowledge_router)
    public_app.include_router(editorial_router)
    client = TestClient(public_app)
    assert client.post(
        "/api/knowledge/games",
        json={"id": "split-game", "name": "Split Game"},
        headers={"X-Knowledge-Admin": "shared-admin-password"},
    ).status_code == 200

    upload_url_path = "/api/knowledge/games/split-game/rulebooks/upload-url"
    # Raw private_token sent directly to public service fails with 401 because public_db did not issue it
    assert client.post(upload_url_path, json={"filename": "rules.txt"}, headers={"X-Knowledge-Admin": private_token}).status_code == 401

    # Proxy-signed proof after verifying private_token against private service succeeds on public service (reaches 400 validation)
    now = str(int(time.time()))
    def sign(action: str, ts: str = now, game: str = "split-game") -> str:
        msg = f"rulebook-admin\n{ts}\nPOST\n{game}\n{action}"
        return hmac.new(b"split-proxy-signing-secret", msg.encode(), hashlib.sha256).hexdigest()

    assert client.post(
        upload_url_path,
        json={"filename": "rules.txt"},
        headers={
            "X-Knowledge-Admin": "",
            "X-Knowledge-Admin-Timestamp": now,
            "X-Knowledge-Admin-Signature": sign("rulebooks/upload-url"),
        },
    ).status_code == 400

    # Forged signature for wrong action, wrong game, or expired timestamp is rejected with 401
    for bad_ts, bad_sig in (
        (now, sign("rulebooks/import")),
        (now, sign("rulebooks/upload-url", game="other-game")),
        (str(int(now) - 120), sign("rulebooks/upload-url", ts=str(int(now) - 120))),
    ):
        assert client.post(
            upload_url_path,
            json={"filename": "rules.txt"},
            headers={
                "X-Knowledge-Admin": "",
                "X-Knowledge-Admin-Timestamp": bad_ts,
                "X-Knowledge-Admin-Signature": bad_sig,
            },
        ).status_code == 401
