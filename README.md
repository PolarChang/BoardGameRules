# 🎲 Board Game RAG Engine（本地端）

> 一個專為桌遊規則查詢開發的檢索增強生成（RAG）原型系統。  
> 專注於本地規則書的索引與問答，不涉及外部聯網或自動化部署。

---

## 專案結構

```
.
├── data/               # 存放 PDF 規則書（放入 .pdf 檔案）
├── db/                 # ChromaDB 向量資料庫儲存路徑（自動產生）
├── templates/          # Web 前端模板
│   └── index.html      # 網頁版查詢介面
├── src/
│   ├── ingest.py       # PDF 解析與資料寫入索引
│   └── query.py        # 查詢引擎與生成回應（CLI）
├── app.py              # Web API 伺服器（FastAPI）
├── main.py             # 統一入口點
├── requirements.txt    # 相依套件
└── README.md
```

---

## 快速開始

### 1️⃣ 環境設定

```bash
# 建立虛擬環境
python -m venv venv

# Windows：
venv\Scripts\activate

# macOS / Linux：
source venv/bin/activate
```

### 2️⃣ 安裝相依套件

```bash
pip install -r requirements.txt
```

> ⚠️ **注意**：本系統使用 **HuggingFace 本地 Embedding 模型**（`BAAI/bge-small-zh-v1.5`），無需任何 API Key 即可執行。  
> 首次執行時會自動下載模型（約 33MB），請確保網路連線暢通。

### 3️⃣ 放入 PDF 規則書

將你的桌遊規則書 **PDF 檔案** 放入 `data/` 目錄。  
支援中英文雙欄／多欄排版的規則書。

```
data/
├── On_Mars.pdf
├── Lisboa.pdf
└── ...
```

> **命名建議**：使用遊戲英文名稱作為檔名（如 `On_Mars.pdf`），系統會自動將檔名作為 `game_name` Metadata，方便後續依遊戲名稱篩選查詢。

### 4️⃣ 建立索引

```bash
python main.py ingest
# 或
python src/ingest.py
```

執行後會：
- 解析 `data/` 下所有 PDF
- 使用 PyMuPDF 進行進階文字提取（支援雙欄排版）
- 將文字分割為 Chunk（chunk_size=512, overlap=64）
- 使用 `bge-small-zh-v1.5` 產生向量嵌入
- 寫入 ChromaDB（儲存在 `db/` 目錄）

### 5️⃣ 開始查詢

**單次查詢（CLI）：**
```bash
python src/query.py --query "Lisboa 的影響力計分規則是什麼？"

# 指定遊戲篩選：
python src/query.py --query "資源交換機制" --game "On Mars"
```

**互動模式（CLI）：**
```bash
python main.py
# 或
python src/query.py --interactive
```

互動模式支援指令：
| 指令 | 說明 |
|------|------|
| `/game <名稱>` | 設定遊戲篩選 |
| `/clear` | 清除遊戲篩選 |
| `/top <數字>` | 設定檢索數量 |
| `/help` | 顯示說明 |
| `/exit` | 離開 |

**Web 介面：**
```bash
python main.py web
# 或
python app.py
```

開啟瀏覽器前往 `http://localhost:8000` 即可使用圖形化查詢介面。

---

## 功能特色

### ✅ 已完成

- [x] 基礎 Python 環境與 LlamaIndex 設定
- [x] ChromaDB 本地持久化儲存
- [x] 進階 PDF 解析（PyMuPDF，支援雙欄／多欄排版）
- [x] Metadata 篩選（支援指定 `game_name` 搜尋）
- [x] 互動式 CLI 查詢模式
- [x] Web 圖形化查詢介面（FastAPI）
- [x] 本地 Embedding 模型（無需 API / 外部服務）
- [x] 所有回答以繁體中文顯示

### 🔜 待開發

- [ ] 規則書的分層索引優化（目前是簡單的 chunk 切分）
- [ ] Multi-modal RAG：OCR 處理含圖片的規則
- [ ] 結果引用顯示（顯示資料來源與頁碼）
- [ ] 支援 LLM 摘要生成（需設定 API Key）

---

## 專案憲法（開發規範）

> 以下規範為本專案開發時必須遵守的核心原則。

### 命令執行規範

1. **禁止使用 `&&` 串聯命令**：在執行確認與截止（execution confirmation）時，不得使用 `&&` 將多個命令串聯在同一行執行。原因如下：
   - `&&` 會隱藏中間步驟的失敗細節，導致除錯困難
   - 無法在每一步之間確認執行結果與正確性
   - 違反「逐步確認、逐一執行」的原則
   
   ✅ 正確做法：分步執行，每一步獨立確認。
   ```bash
   cd project-dir
   python main.py ingest
   ```
   
   ❌ 錯誤做法：使用 `&&` 串聯。
   ```bash
   cd project-dir && python main.py ingest
   ```

2. **所有命令執行必須可獨立觀察結果**：每個命令應能單獨執行並觀察其輸出，確保每一步的結果都可被驗證。

---

## 技術架構

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF 規則書  │ ──→ │  PyMuPDF 解析  │ ──→ │  文字 Chunk  │
│   (data/)   │     │  (支援雙欄)   │     │  (512 tokens)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   查詢結果    │ ←── │  LlamaIndex   │ ←── │  ChromaDB   │
│   (回應)     │     │  Query Engine │     │  向量資料庫   │
└─────────────┘     └──────────────┘     └─────────────┘
```

**核心元件：**
- **LlamaIndex**：RAG 框架，管理文件索引、檢索與查詢
- **ChromaDB**：本地向量資料庫，持久化儲存嵌入向量
- **PyMuPDF（fitz）**：PDF 文字提取，支援複雜排版
- **BAAI/bge-small-zh-v1.5**：輕量中英雙語 Embedding 模型（336 維度）
- **FastAPI**：Web API 伺服器，提供 RESTful 查詢介面

---

## 開發指南

### 如何新增 PDF 解析功能

目前 `ingest.py` 使用 PyMuPDF 的區塊（block）提取模式。  
若要進一步優化（例如處理掃描件 PDF），可以整合 OCR 引擎：

```python
# ingest.py 中的 extract_text_from_pdf 可擴充為：
if not text.strip():
    # 回退到 OCR 模式
    import pytesseract
    text = pytesseract.image_to_string(page_image)
```

### 如何加入 LLM 生成回答

目前系統預設使用純檢索模式（Retrieval-Only），直接回傳規則書段落。  
若要啟用 LLM 生成摘要回答，設定環境變數即可：

```bash
# 使用 OpenAI
export OPENAI_API_KEY="sk-..."

# 或使用本地 LLM（via Ollama）
# 參考：https://docs.llamaindex.ai/en/stable/examples/llm/ollama/
```

### 如何變更輸出語言

所有使用者介面文字（包含 CLI 與 Web）已設定為繁體中文。  
若要變更語言，請修改：
- `src/query.py` 中的互動模式提示文字
- `templates/index.html` 中的 HTML 內容與 JavaScript 字串

---

## 需求

- Python 3.10+
- 4GB RAM（建議）
- 網路連線（首次執行需下載 Embedding 模型）

---

## License

MIT