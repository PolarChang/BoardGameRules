"""
preprocess.py — PDF 預處理腳本：利用 LLM Vision 將 PDF 轉為繁體中文 Markdown

流程：
1. 使用 PyMuPDF 將 PDF 每一頁渲染為 PNG 圖片
2. 將圖片送給 Vision LLM（GPT-4o / Claude 3.5 Sonnet）
3. 將 LLM 回傳的繁體中文 Markdown 寫入 data/preprocessed/{game_name}.md
4. ingest.py 會優先讀取這些 Markdown 檔案而非原始 PDF
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from tqdm import tqdm

# ── 設定 ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"

# Vision LLM 支援的 providers
PROVIDER_OPENAI = "openai"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_GEMINI = "gemini"
PROVIDER_OPENROUTER = "openrouter"

SYSTEM_PROMPT = (
    "你是一位專業的桌遊規則書翻譯與排版專家。\n"
    "我會給你一張桌遊規則書的 PDF 頁面截圖。\n"
    "請你：\n"
    "1. 仔細閱讀頁面上的所有文字（包括雙欄、表格、圖示旁的文字）\n"
    "2. 將內容翻譯成 **精準的繁體中文**（台灣用語）\n"
    "3. 保留原始排版結構（標題層級、列表、表格等）\n"
    "4. 桌遊專有名詞請使用台灣桌遊圈通用譯名（例如：Action → 行動，Victory Point → 勝利點數，等等）\n"
    "5. 以 Markdown 格式輸出\n"
    "6. 如果頁面有圖片或圖示，請用 [圖表：簡短描述] 標註其位置與內容\n\n"
    "請直接輸出 Markdown 內容，不要加入任何前言或結語。"
)


# ── OpenRouter Vision Call ──────────────────────────────────────
def call_openrouter(image_base64: str, page_num: int) -> str:
    import requests

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("請設定 OPENROUTER_API_KEY 環境變數")

    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": "openai/gpt-4o",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"請將此 PDF 第 {page_num} 頁的內容翻譯為繁體中文 Markdown："},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
        }),
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


# ── Gemini (Google) Vision Call ─────────────────────────────────
def call_gemini(image_base64: str, page_num: int) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-001")
    response = model.generate_content(
        [
            SYSTEM_PROMPT + f"\n\n請將此 PDF 第 {page_num} 頁的內容翻譯為繁體中文 Markdown：",
            {"mime_type": "image/png", "data": base64.b64decode(image_base64)},
        ],
        generation_config={"temperature": 0.1, "max_output_tokens": 4096},
    )
    return response.text or ""


# ── OpenAI GPT-4o Call ──────────────────────────────────────────
def call_gpt4o(image_base64: str, page_num: int) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"請將此 PDF 第 {page_num} 頁的內容翻譯為繁體中文 Markdown："},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        max_tokens=4096,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


# ── Anthropic Claude 3.5 Sonnet Call ────────────────────────────
def call_claude(image_base64: str, page_num: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"請將此 PDF 第 {page_num} 頁的內容翻譯為繁體中文 Markdown：",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                ],
            }
        ],
    )
    return response.content[0].text or ""


# ── 主處理邏輯 ──────────────────────────────────────────────────
def preprocess_pdf(
    pdf_path: Path,
    provider: str = PROVIDER_OPENAI,
    dpi: int = 200,
    delay: float = 1.0,
) -> str:
    """將單一 PDF 轉換為繁體中文 Markdown。"""
    game_name = pdf_path.stem
    logger.info(f"📄 處理: {pdf_path.name}")

    # 1. 將 PDF 轉為圖片（使用 PyMuPDF，無需 poppler）
    logger.info(f"   🖼️ 轉換 PDF → 圖片 (DPI={dpi})...")
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    logger.info(f"   ✅ 共 {total_pages} 頁")

    # 2. 逐頁送給 LLM Vision
    markdown_pages = []
    for page_num in tqdm(range(total_pages), desc=f"   🤖 {provider} 處理中"):
        page = doc[page_num]
        # 將頁面渲染為高解析度 PNG 圖片
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # 重試機制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if provider == PROVIDER_OPENAI:
                    md_content = call_gpt4o(img_b64, page_num + 1)
                elif provider == PROVIDER_ANTHROPIC:
                    md_content = call_claude(img_b64, page_num + 1)
                elif provider == PROVIDER_GEMINI:
                    md_content = call_gemini(img_b64, page_num + 1)
                elif provider == PROVIDER_OPENROUTER:
                    md_content = call_openrouter(img_b64, page_num + 1)
                else:
                    raise ValueError(f"不支援的 provider: {provider}")

                markdown_pages.append(f"## 第 {page_num + 1} 頁\n\n{md_content.strip()}")
                break
            except Exception as e:
                logger.warning(f"   ⚠️ 第 {page_num + 1} 頁失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    logger.error(f"   ❌ 第 {page_num + 1} 頁放棄處理")
                    markdown_pages.append(f"## 第 {page_num + 1} 頁\n\n*[此頁處理失敗]*\n")

        # 避免 API rate limit
        if delay > 0:
            time.sleep(delay)

    # 3. 組合成完整 Markdown
    full_md = f"# {game_name} — 繁體中文規則書\n\n" + "\n\n---\n\n".join(markdown_pages)
    return full_md


def preprocess_all_pdfs(
    provider: str = PROVIDER_OPENAI,
    dpi: int = 200,
    delay: float = 1.0,
    force: bool = False,
):
    """掃描 data/ 下所有 PDF，逐一轉換為繁體中文 Markdown。"""
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.warning("❌ /data 目錄下沒有找到任何 PDF 檔案！")
        sys.exit(1)

    for pdf_path in pdf_files:
        game_name = pdf_path.stem
        output_path = PREPROCESSED_DIR / f"{game_name}.md"

        if output_path.exists() and not force:
            logger.info(f"⏭️  跳過 {pdf_path.name}（已存在預處理檔案，使用 --force 強制重新處理）")
            continue

        md_content = preprocess_pdf(pdf_path, provider=provider, dpi=dpi, delay=delay)
        output_path.write_text(md_content, encoding="utf-8")
        logger.info(f"✅ 已寫入: {output_path}")

    logger.info(f"🎉 所有 PDF 預處理完成！結果存放於: {PREPROCESSED_DIR}")


# ── CLI ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="🎲 PDF 預處理：利用 LLM Vision 將桌遊規則書轉為繁體中文 Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", choices=[PROVIDER_OPENAI, PROVIDER_ANTHROPIC, PROVIDER_GEMINI, PROVIDER_OPENROUTER],
                        default=PROVIDER_OPENAI,
                        help=f"Vision LLM 提供商（預設: {PROVIDER_OPENAI}）")
    parser.add_argument("--dpi", type=int, default=200,
                        help="PDF 轉圖片的 DPI（預設: 200，越高越清晰但耗費更多 tokens）")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="每頁之間的延遲秒數，避免 API rate limit（預設: 1.0）")
    parser.add_argument("--force", action="store_true",
                        help="強制重新處理已存在的檔案")
    parser.add_argument("--file", type=str, default=None,
                        help="只處理特定檔案（放在 data/ 下的 PDF 檔名）")

    args = parser.parse_args()

    # 檢查 API Key
    if args.provider == PROVIDER_OPENAI and not os.environ.get("OPENAI_API_KEY"):
        logger.error("❌ 請設定環境變數 OPENAI_API_KEY")
        logger.error("   或使用 --provider gemini (GEMINI_API_KEY) 或 --provider anthropic (ANTHROPIC_API_KEY)")
        sys.exit(1)
    if args.provider == PROVIDER_ANTHROPIC and not os.environ.get("ANTHROPIC_API_KEY"):
        logger.error("❌ 請設定環境變數 ANTHROPIC_API_KEY")
        sys.exit(1)
    if args.provider == PROVIDER_GEMINI and not os.environ.get("GEMINI_API_KEY"):
        logger.error("❌ 請設定環境變數 GEMINI_API_KEY")
        sys.exit(1)
    if args.provider == PROVIDER_OPENROUTER and not os.environ.get("OPENROUTER_API_KEY"):
        logger.error("❌ 請設定環境變數 OPENROUTER_API_KEY")
        sys.exit(1)

    if args.file:
        # 只處理單一檔案
        pdf_path = DATA_DIR / args.file
        if not pdf_path.exists():
            logger.error(f"❌ 找不到檔案: {pdf_path}")
            sys.exit(1)
        game_name = pdf_path.stem
        PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = PREPROCESSED_DIR / f"{game_name}.md"
        if output_path.exists() and not args.force:
            logger.info(f"⏭️  跳過 {pdf_path.name}（已存在）")
            return
        md_content = preprocess_pdf(pdf_path, provider=args.provider, dpi=args.dpi, delay=args.delay)
        output_path.write_text(md_content, encoding="utf-8")
        logger.info(f"✅ 已寫入: {output_path}")
    else:
        preprocess_all_pdfs(provider=args.provider, dpi=args.dpi, delay=args.delay, force=args.force)


if __name__ == "__main__":
    main()