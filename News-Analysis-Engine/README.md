# News-Analysis-Engine 前端啟動說明

## 介紹
本資料夾為 MirageNews 假新聞偵測前端，負責 UI 顯示、API 呼叫、Top Concepts 物件框渲染。

## 目錄結構
- `src/components/ResultsDisplay.tsx`：主結果顯示元件，支援 Top Concepts 物件框可視化
- 其他元件、樣式、工具請見對應目錄

## 啟動方式

### 1. 安裝依賴
```bash
cd News-Analysis-Engine
npm install
# 或
pnpm install
```

### 2. 啟動開發伺服器
```bash
npm run dev
# 或
pnpm dev
```

預設會在 http://localhost:3000 啟動

### 3. 設定 API 端點
請於 `.env` 或 `src/config.ts` 設定後端 API 位置，例如：
```
VITE_API_BASE_URL=http://localhost:5000
```

### 4. 功能說明
- 支援顯示 CBM/OWLv2 Top Concepts 物件框（需後端回傳 box 座標）
- 支援多模型分數、融合分數、詳細報告顯示
- 支援圖片上疊加物件框（canvas/svg/CSS）

### 5. 測試
- 輸入標題、內容、圖片網址，送出後可看到分析結果與物件框
- 若無物件框資料，僅顯示分數與報告

## 注意事項
- 請確保後端 API 可連線，且 CORS 設定正確
- 若需自訂 port、API 路徑，請於 .env 設定

## 聯絡方式
如有問題請聯絡專案負責人或於 GitHub issue 提問。
