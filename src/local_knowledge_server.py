"""Lightweight local knowledge API; avoids loading the legacy RAG/index stack."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header
from src.database import init_db
from src.editorial_api import router, init_editorial_db
from src.gemini_usage import setting
import os
import asyncio

# Local desktop server never implicitly falls back to a cloud provider.
os.environ['EDITORIAL_QA_PROVIDER'] = 'local'

# Only local administrative configuration is loaded; cloud model keys stay private.
if setting('KNOWLEDGE_ADMIN_KEY'):
    os.environ.setdefault('KNOWLEDGE_ADMIN_KEY', setting('KNOWLEDGE_ADMIN_KEY'))


@asynccontextmanager
async def lifespan(app):
    init_db()
    init_editorial_db()
    from src.local_source_worker import run_batch
    async def extract_sources():
        while True:
            try:
                await asyncio.to_thread(run_batch)
            except Exception:
                import logging
                logging.getLogger(__name__).exception('Source worker scan failed')
            await asyncio.sleep(10)
    task = asyncio.create_task(extract_sources())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)
app.include_router(router)
from src.game_catalog_api import router as game_catalog_router
app.include_router(game_catalog_router, prefix="/api/knowledge")


@app.get('/health')
def health():
    return {'status': 'ok', 'mode': 'local-knowledge'}


@app.post('/api/local/imports/claim')
def claim_any_game(x_knowledge_admin: str = Header(default='')):
    from src.editorial_api import admin
    from src.import_jobs import init_jobs, claim_next
    admin(x_knowledge_admin)
    init_jobs()
    return claim_next()


@app.get('/api/local/readiness')
def local_readiness(x_knowledge_admin: str = Header(default='')):
    from src.editorial_api import admin
    from src.local_runtime import readiness
    admin(x_knowledge_admin)
    return readiness(setting('LOCAL_QA_MODEL', 'qwen3:8b'))
