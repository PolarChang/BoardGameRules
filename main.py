"""
Board Game RAG Engine (Local) - Entry Point

用法:
  python main.py              → 查詢模式（等同於 python src/query.py --interactive）
  python main.py ingest       → 建立索引（等同於 python src/ingest.py）
  python main.py query ...    → 查詢（等同於 python src/query.py ...）
  python main.py web          → 啟動 Web 伺服器（等同於 python app.py）
"""

import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main():
    if len(sys.argv) < 2:
        # 無參數 → 進入互動查詢模式
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "query.py"), "--interactive"])
        return

    command = sys.argv[1]
    rest_args = sys.argv[2:]

    if command == "ingest":
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "ingest.py"), *rest_args])
    elif command == "query":
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "query.py"), *rest_args])
    elif command == "web":
        subprocess.run([sys.executable, str(BASE_DIR / "app.py"), *rest_args])
    elif command == "preprocess":
        subprocess.run([sys.executable, str(BASE_DIR / "src" / "preprocess.py"), *rest_args])
    else:
        print(f"❌ 未知指令: {command}")
        print("可用指令: ingest, query, web, preprocess")
        print("  preprocess  → 利用 LLM Vision 將 PDF 轉為繁體中文 Markdown")
        print("  ingest      → 建立向量索引（優先使用 preprocessed Markdown）")
        print("  query       → 查詢模式")
        print("  web         → 啟動 Web 伺服器")
        print("或直接執行 main.py 進入互動查詢模式")
        sys.exit(1)


if __name__ == "__main__":
    main()
