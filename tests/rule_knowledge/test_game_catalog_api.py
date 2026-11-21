from fastapi.testclient import TestClient
from src import database


def test_local_catalog_supports_selection_creation_and_detail(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "catalog.db")
    for key in ("DB_HOST", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EDITORIAL_QA_PROVIDER", "local")
    from src.local_knowledge_server import app
    database.init_db()
    client = TestClient(app)
    assert client.get("/api/knowledge/games").json() == {"games": []}
    body = {"id": "catalog-test", "name": "Catalog test", "designer": ["Designer"]}
    assert client.post("/api/knowledge/games", json=body).status_code == 200
    games = client.get("/api/knowledge/games").json()["games"]
    assert len(games) == 1 and games[0]["id"] == body["id"]
    assert games[0]["designer"] == ["Designer"]
    assert client.get("/api/knowledge/games/catalog-test").json()["name"] == body["name"]
    assert client.get("/api/knowledge/games/missing").status_code == 404
    assert client.post("/api/knowledge/games", json=body).status_code == 400
    assert client.post("/api/knowledge/games", json={"name": "Missing ID"}).status_code == 422

