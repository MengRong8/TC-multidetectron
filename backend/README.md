# Backend API Server

提供假新聞檢測的 REST API 服務。

## 架構

```
輸入層:
  ├─ 方法1: 手動輸入 (POST /api/detect)
  └─ 方法2: URL 爬取 (POST /api/detect-from-url)

處理層:
  ├─ Text AIGC Model (RoBERTa) - 檢測文本是否為 AI 生成
  ├─ Text Intent Model (CB-LLM) - 檢測文本是否有誘導性
  └─ Image AIGC Model (CBM) - 檢測圖片是否為 AI 生成

決策層:
  └─ Linear Fusion Layer - 融合三個模型的分數

輸出層:
  ├─ 三個模型的獨立分數
  ├─ 最終融合分數 (0-1)
  └─ 判定結果 + 分析報告
```

## 模組説明

### 1. `crawler.py` - 新聞爬蟲模組
- 支援 7 個新聞網站: ETtoday, LTN, Chinatimes, TVBS, UDN, Yahoo, NowNews
- 自動識別新聞來源
- 提取標題、內容、圖片、作者、發布時間

### 2. `models.py` - 模型推理模組
- **TextAIGCModel**: RoBERTa 模型，檢測 AI 生成文本
- **TextIntentModel**: CB-LLM 模型，檢測誘導性文本
- **ImageAIGCModel**: CBM 模型，檢測 AI 生成圖片
- **ModelInference**: 統一推理介面

### 3. `fusion.py` - 決策層模組
- **DecisionFusionLayer**: 線性融合層
- **FusionStrategy**: 多種融合策略 (linear, max, min, avg, voting)
- **DecisionMaker**: 最終決策器，生成分析報告

### 4. `app.py` - Flask API 伺服器
- RESTful API 端點
- 錯誤處理
- CORS 支援

### 5. `config.py` - 配置檔案
- 模型路徑配置
- 融合權重配置
- API 伺服器配置

## API 端點

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "models_loaded": true,
  "decision_maker_loaded": true,
  "crawler_loaded": true
}
```

### 2. 爬取新聞
```bash
POST /api/crawl
Content-Type: application/json

{
  "url": "https://www.ettoday.net/news/..."
}
```

**Response:**
```json
{
  "url": "https://...",
  "title": "新聞標題",
  "content": "新聞內容...",
  "imageUrls": ["https://..."],
  "publishedAt": "2024-01-01T12:00:00",
  "author": "記者名",
  "siteName": "ETtoday"
}
```

### 3. 檢測假新聞（手動輸入）
```bash
POST /api/detect
Content-Type: application/json

{
  "title": "新聞標題",
  "content": "新聞內容...",
  "imageBase64": "data:image/jpeg;base64,...",  // 可選
  "imageUrl": "https://...",  // 可選
  "sourceUrl": "https://..."  // 可選
}
```

**Response:**
```json
{
  "id": "uuid-string",
  "title": "新聞標題",
  "sourceUrl": "https://...",
  "textAigcScore": {
    "modelName": "RoBERTa",
    "modelType": "Text AIGC Detection",
    "score": 0.85,
    "confidence": 0.90,
    "label": "AIGC",
    "details": ""
  },
  "textIntentScore": {
    "modelName": "CB-LLM (Qwen-3)",
    "modelType": "Text Intent Detection",
    "score": 0.60,
    "confidence": 0.70,
    "label": "Manipulative",
    "details": ""
  },
  "imageAigcScore": {
    "modelName": "CBM (Concept Bottleneck)",
    "modelType": "Image AIGC Detection",
    "score": 0.92,
    "confidence": 0.85,
    "label": "AI-generated",
    "details": ""
  },
  "finalScore": 0.79,
  "finalLabel": "Highly Likely Fake",
  "finalConfidence": 0.58,
  "report": "**Final Verdict: Highly Likely Fake**\n...",
  "analyzedAt": "2024-01-01T12:00:00"
}
```

### 4. 從 URL 檢測假新聞
```bash
POST /api/detect-from-url
Content-Type: application/json

{
  "url": "https://www.ettoday.net/news/..."
}
```

**Response:** 同上 `/api/detect`

## 使用方法

### 1. 安裝依賴
```bash
cd backend
pip install -r requirements.txt
```

### 2. 啟動伺服器

**方法 1: 使用腳本**
```bash
chmod +x start.sh
./start.sh
```

**方法 2: 直接運行**
```bash
python app.py --host 0.0.0.0 --port 5000
```

**方法 3: 開發模式**
```bash
python app.py --host 0.0.0.0 --port 5000 --debug
```

### 3. 測試 API

**測試健康檢查:**
```bash
curl http://localhost:5000/health
```

**測試爬蟲:**
```bash
curl -X POST http://localhost:5000/api/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.ettoday.net/news/20240101/2645678.htm"}'
```

**測試檢測:**
```bash
curl -X POST http://localhost:5000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "title": "測試新聞標題",
    "content": "這是測試新聞的內容..."
  }'
```

## 配置

編輯 `config.py` 來修改配置:

```python
# 模型權重
FUSION_CONFIG = {
    'weights': {
        'text_aigc': 0.4,    # RoBERTa 權重
        'text_intent': 0.3,  # CB-LLM 權重
        'image_aigc': 0.3    # CBM 權重
    }
}

# 服務器配置
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False
}
```

## 注意事項

1. **模型路徑**: 確保模型文件存在於配置的路徑
   - Text AIGC: `text_aigc/model/roberta-fake-news-detection-final/`
   - Image AIGC: `checkpoints/image/cbm-encoder.pt` & `cbm-predictor.pt`
   - Text Intent: 目前為佔位符實現

2. **GPU/CPU**: 自動檢測並使用可用的 GPU，否則使用 CPU

3. **內存**: 三個模型同時加載需要較大內存（建議 8GB+）

4. **併發**: 使用 Flask 的 threaded 模式支持併發請求

## 日誌

日誌文件保存在 `backend/logs/api.log`

查看實時日誌:
```bash
tail -f logs/api.log
```

## 故障排查

### 問題: 模型載入失敗
- 檢查模型檔案是否存在
- 檢查 PyTorch 版本是否相容
- 查看日誌取得詳細錯誤資訊

### 問題: 爬蟲失敗
- 檢查網路連線
- 查看目標網站是否可存取
- 確認 User-Agent 是否被封鎖

### 問題: CORS 錯誤
- 確認 `flask-cors` 已安裝
- 檢查 `CORS(app)` 是否正確配置

## 開發

### 新增新的融合策略
在 `fusion.py` 的 `FusionStrategy` 類中新增新方法:

```python
@staticmethod
def custom_fusion(scores: Dict[str, Optional[float]]) -> float:
    # 自訂融合邏輯
    pass
```

### 新增新的爬蟲支援
在 `crawler.py` 中新增新的解析方法:

```python
def _crawl_newsite(self, url: str) -> Dict:
    # 新網站的爬蟲邏輯
    pass
```

## License

MIT
