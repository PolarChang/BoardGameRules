"""Loopback-only integration fixture; never starts against an existing database."""
import asyncio
import io
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def create_app():
    # Clear cloud/model settings before importing any database or storage code.
    for key in ("DATABASE_URL", "DB_HOST", "DB_PASSWORD", "RULEBOOK_STORAGE_BUCKET",
                "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(key, None)
    os.environ.update(KNOWLEDGE_DISABLE_LOCAL_ENV="true", EDITORIAL_DAILY_AI_LIMIT="0", ENABLE_COMMUNITY="false",
                      ENABLE_LEGACY_PDF_EXTRACTION="false", KNOWLEDGE_ADMIN_KEY="integration-only")
    from fastapi import FastAPI, UploadFile
    from fastapi.responses import Response
    from src import database as db, editorial_api as editorial, editorial_visuals as visuals
    from src.models import Game
    import fitz

    root = Path(tempfile.mkdtemp(prefix="knowledge-integration-"))
    db.DB_PATH = root / "rules.db"
    visuals.DATA_DIR = root
    db.init_db()
    editorial.init_editorial_db()
    game_id = "knowledge-integration"
    db.create_game(Game(id=game_id, name="規則整合測試"))
    db.create_game(Game(id="knowledge-integration-other", name="來源隔離測試"))
    manuscript = {"blocks": [
        {"kind": "文件", "id": "META", "fields": {"遊戲": "規則整合測試", "版本": "測試版 1"}},
        {"kind": "來源", "id": "S1", "fields": {"名稱": "基本規則", "版本": "1", "PDF頁數": "3"}},
        {"kind": "來源", "id": "S2", "fields": {"名稱": "補充規則", "版本": "1", "PDF頁數": "1"}},
        {"kind": "術語", "id": "T1", "fields": {"中文": "資源", "原文": "Resource"}},
        {"kind": "規則", "id": "R1", "fields": {"大項": "進行遊戲", "標題": "支付資源", "內容": "支付 2 資源，完成一次行動。",
            "角色": "全部", "人數": "全部", "擴充": "全部", "條件": "回合中", "成本": "2 資源", "例外": "無",
            "來源": "S1", "PDF頁碼": "1-2", "原文": "Pay two resources to perform an action.", "狀態": "已確認"}},
        {"kind": "規則", "id": "R2", "fields": {"大項": "計分與結束", "標題": "補充計分", "內容": "每枚代幣得到 1 分。",
            "角色": "全部", "人數": "全部", "擴充": "全部", "條件": "遊戲結束", "例外": "無",
            "來源": "S2", "PDF頁碼": "1", "原文": "Each token scores one point.", "狀態": "已確認"}},
        {"kind": "FAQ", "id": "F1", "fields": {"問題": "支付多少資源？", "答案": "支付 2 資源。", "引用": "R1",
            "角色": "全部", "人數": "全部", "擴充": "全部", "狀態": "已確認"}},
        {"kind": "涵蓋", "id": "C1", "fields": {"章節": "行動", "來源": "S1", "PDF頁碼": "1-2", "規則": "R1", "狀態": "已整理"}},
        {"kind": "涵蓋", "id": "C2", "fields": {"章節": "封底", "來源": "S1", "PDF頁碼": "3", "狀態": "不適用", "說明": "出版資訊，沒有規則條文"}},
        {"kind": "涵蓋", "id": "C3", "fields": {"章節": "計分", "來源": "S2", "PDF頁碼": "1", "規則": "R2", "狀態": "已整理"}},
    ]}
    pdf = fitz.open()
    for i in range(3):
        page = pdf.new_page(width=300, height=400)
        page.insert_text((30, 40), f"Integration fixture - PDF page {i + 1}")
        page.draw_circle((150, 180), 45, color=(0.8, 0.1, 0.1), fill=(0.9, 0.2, 0.1))
    pdf_bytes = pdf.tobytes()
    pdf.close()
    from src import import_jobs
    import hashlib
    imported = import_jobs.enqueue(game_id, 'source-crop-fixture.pdf', pdf_bytes)
    lease = import_jobs.claim(game_id, 'integration-only')
    with fitz.open(stream=pdf_bytes, filetype='pdf') as source_pdf:
        crop = source_pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(90, 120, 210, 240)).tobytes('png')
    import_jobs.complete_page(game_id, imported['id'], import_jobs.PageResult(token=lease['token'], page=1,
        rules=[{'title': '圖片範例', 'translation': '紅色圓形', 'original': '待確認圖片', 'category': '卡牌與配件'}], warnings=[],
        locations=[{'block': 1, 'kind': 'image', 'bbox': [90, 120, 210, 240], 'scale': 2, 'image_sha256': hashlib.sha256(crop).hexdigest()}]), 'integration-only')
    import_jobs.pause(game_id, imported['id'], import_jobs.PauseRequest(token=lease['token']), 'integration-only')
    from src.import_quality import save_answer
    feedback_snapshot = import_jobs.build_draft(game_id, imported['id'])
    feedback_answer_id = save_answer(game_id, imported['id'], feedback_snapshot['version'], '回饋測試問題',
        {'answer': '待查證的測試答案', 'citations': [], 'model': 'fixture-no-inference'}, feedback_snapshot['document'])
    from src import import_question_jobs
    import_question_jobs.submit(game_id, imported['id'], import_question_jobs.QuestionJob(
        question='回饋測試問題', version=feedback_snapshot['version'], request_id='feedback-ui-fixture-001'), 'integration-only')
    feedback_lease = import_question_jobs.claim_next()
    import_question_jobs.finish(feedback_lease['id'], feedback_lease['token'], result={
        'answer': '待查證的測試答案（介面測試資料，未呼叫模型）', 'citations': [], 'kind': 'partial',
        'answer_id': feedback_answer_id, 'model': 'fixture-no-inference'})
    block = asyncio.run(visuals.crop(game_id, UploadFile(filename="fixture.pdf", file=io.BytesIO(pdf_bytes)),
        "S1", 1, "[0.3,0.3,0.7,0.6]", True, "integration-only"))["block"]
    block["fields"].update({"名稱": "行動代幣", "替代文字": "紅色圓形行動代幣", "相關規則": "R1", "狀態": "已確認"})
    manuscript["blocks"].append(block)
    manuscript["blocks"].extend([
        {"kind": "流程", "id": "FLOW-DEMO", "fields": {"名稱": "示範遊戲總覽", "範圍": "總覽", "起點": "NODE-ACTION", "說明": "自製測試規則，僅驗證介面及資料流程。", "狀態": "已確認"}},
        {"kind": "流程節點", "id": "NODE-ACTION", "fields": {"流程": "FLOW-DEMO", "標題": "執行行動", "順序": "1", "規則": "R1", "角色": "全部", "人數": "全部", "擴充": "全部", "結束": "否", "狀態": "已確認"}},
        {"kind": "流程節點", "id": "NODE-END", "fields": {"流程": "FLOW-DEMO", "標題": "最後計分", "順序": "2", "規則": "R2", "角色": "全部", "人數": "全部", "擴充": "全部", "結束": "是", "狀態": "已確認"}},
        {"kind": "流程連線", "id": "EDGE-END", "fields": {"流程": "FLOW-DEMO", "起點": "NODE-ACTION", "終點": "NODE-END", "條件": "遊戲結束", "順序": "1", "循環": "否", "狀態": "已確認"}},
        {"kind": "流程連線", "id": "EDGE-LOOP", "fields": {"流程": "FLOW-DEMO", "起點": "NODE-ACTION", "終點": "NODE-ACTION", "條件": "繼續遊戲", "順序": "2", "循環": "是", "狀態": "已確認"}},
    ])
    draft = editorial.save(game_id, editorial.Draft(document=manuscript), "integration-only")
    editorial.publish(game_id, editorial.Publish(version=draft["version"], confirmed=True), "integration-only")
    app = FastAPI(title="Isolated knowledge integration")
    app.include_router(editorial.router)
    from src.game_catalog_api import router as catalogue_router
    app.include_router(catalogue_router, prefix="/api/knowledge")

    @app.get("/health")
    def health():
        return {"mode": "isolated-integration", "game_id": game_id, "database": str(db.DB_PATH), "model_calls_enabled": False}

    @app.get("/integration-fixture")
    def fixture():
        text = "# 規則整理稿 v2\n\n" + "\n\n".join(
            f"## {b['kind']} {b['id']}\n" + "\n".join(f"{key}：{value}" for key, value in b["fields"].items())
            for b in manuscript["blocks"])
        return {"document": manuscript, "text": text, "version": draft["version"], "game_id": game_id, "import_id": imported['id'], "feedback_answer_id": feedback_answer_id}

    @app.get("/fixture.pdf")
    def fixture_pdf():
        return Response(pdf_bytes, media_type="application/pdf")

    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.getenv("KNOWLEDGE_INTEGRATION_BACKEND_PORT", "8013")))
