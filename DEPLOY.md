# 部署規則引擎到線上

## 選項一：Railway（推薦，最簡單）

1. 安裝 Railway CLI：
   ```bash
   npm i -g @railway/cli
   ```

2. 登入並初始化：
   ```bash
   cd D:\BoardGameRules
   railway login
   railway init
   ```

3. 部署：
   ```bash
   railway up
   ```

4. 部署後會得到類似 `https://boardgame-rules.up.railway.app` 的 URL。

5. 在 `D:\Boardgame\.env.local` 中更新：
   ```
   BOARDGAME_RULES_API_URL=https://你的railway網址.up.railway.app
   ```

## 選項二：Render

1. 在 Render 建立新的 Web Service
2. 連接你的 Git repo
3. 設定：
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - 上傳 `data/` 資料夾（包含 PDF 和索引）

## 選項三：Fly.io

```bash
cd D:\BoardGameRules
fly launch
fly deploy
```

## 重要注意事項

### 1. 資料持久化
部署平台通常會丟棄檔案系統，所以需要：
- **Railway**：使用 Railway Volumes 掛載 `data/` 和 `db/`
- **Render**：使用 Render Disks
- **Fly.io**：使用 Fly Volumes

### 2. 環境變數
部署時需要設定：
```
OPENAI_API_KEY=你的key（可選，用於LLM模式）
```

### 3. 首次部署步驟
```bash
# 1. 本地建立索引
cd D:\BoardGameRules
python main.py ingest

# 2. 確認 db/ 和 data/ 已建立
# 3. 部署到線上平台
# 4. 確認線上平台的 data/ 和 db/ 有內容
```

### 4. 更新 Next.js 設定
部署完成後，在 `D:\Boardgame\.env.local` 中加入：
```
BOARDGAME_RULES_API_URL=https://你的部署網址
```

## 檔案說明

- `railway.toml` — Railway 部署設定
- `Procfile` — Heroku/Render 部署設定
- `requirements.txt` — Python 依賴（已加入 fastapi, uvicorn）
- `.gitignore` — Git 忽略清單（已排除 db/, data/preprocessed/ 等）