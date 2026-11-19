"""The exported CI fixture itself must stay isolated and model-free."""
import importlib.util
from pathlib import Path
from fastapi.testclient import TestClient


def test_fixture_boots_with_fresh_sqlite_and_private_original_pdf():
    path = Path(__file__).with_name('integration_server.py')
    spec = importlib.util.spec_from_file_location('isolated_ci_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with TestClient(module.create_app()) as client:
        health = client.get('/health').json()
        assert health['mode'] == 'isolated-integration'
        assert health['model_calls_enabled'] is False
        assert 'knowledge-integration-' in health['database']
        fixture = client.get('/integration-fixture').json()
        source = f"/api/knowledge/games/{fixture['game_id']}/editorial/imports/{fixture['import_id']}/source"
        assert client.get(source).status_code == 401
        original = client.get(source, headers={'X-Knowledge-Admin':'integration-only'})
        assert original.status_code == 200
        assert original.headers['content-type'] == 'application/pdf'
        assert original.content.startswith(b'%PDF-')
