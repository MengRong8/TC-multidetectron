# TC-MultiDetectron：具高度可解釋性之繁體中文多模態假新聞檢測系統

> 國立中央大學資工系 大學專題　｜　指導教授：楊鎮華 教授
> 本人負責：**影像端 AI 生成痕跡偵測模組**（OWLv2 物件擷取＋概念瓶頸模型），並參與設計**跨模態決策融合層**
> 原始團隊 repo：[Howardisme/Multimodal_news_detector](https://github.com/Howardisme/Multimodal_news_detector)

生成式 AI 讓「有圖有真相」不再成立：文字可以由 LLM 改寫成帶有誘導性的敘事，配圖也能由擴散模型憑空生成。現有偵測器多半只給出一個「真／假」分數，使用者無從得知判斷依據。本專題針對繁體中文新聞，分別從**文字**與**圖片**兩條路徑判斷 AI 介入程度與語意操弄，並以概念瓶頸模型（Concept Bottleneck Model, CBM）讓每一個判斷都能對應到人可理解的概念。

## 方法：雙流解耦架構＋決策層融合

```
                    ┌─ Text AIGC   (RoBERTa)          → P(文字為 AI 生成)
 新聞文字 ──────────┤
                    └─ Text Intent (CB-LLM, 15 概念)   → P(各類誘導手法)        ┐
                                                                                ├─▶ 決策融合層 ─▶ 最終分數＋可解釋報告
 新聞圖片 ── OWLv2 物件擷取 ─┬─ 整張圖 Linear 分類     → P(圖片為 AI 生成)      │
                             └─ 物件區域 CBM (Top-k)   → P(物件層級異常)        ┘
```

| 串流 | 模組 | 輸出 |
|---|---|---|
| AI 介入程度 | **Text AIGC**：RoBERTa 判斷文字是否由 AI 生成 | 文字 AIGC 機率 |
| | **Image AIGC**：整張圖的線性分類，加上物件層級 CBM，取異常機率前 k 高的物件 | 圖片 AIGC 機率＋可疑物件 |
| 語意操弄程度 | **Text Intent**：CB-LLM 對 15 種誘導概念（如斷章取義、虛假引用、權威訴求、假兩難）輸出機率 | 15 維概念機率 |
| 決策 | **Fusion**：以各模組機率為特徵訓練線性融合層，自動學習權重 | 最終假新聞分數 |

## 影像端模組（本人負責）

1. **物件擷取**：以 OWLv2（`owlv2-base-patch16-ensemble`）做開放詞彙物件偵測，從新聞圖片中裁切出人物、建築、旗幟、麥克風等候選區域。
2. **概念編碼**：CBM Encoder 將影像特徵映射到約 300 個物件概念，使中間表示可被人解讀。
3. **AIGC 判斷**：CBM Predictor 依概念向量判斷真偽，並對各物件區域取 Max-Pooling／Top-k 異常分數，結合整張圖的線性分類結果，得到圖片最終的 AIGC 機率。
4. **可解釋輸出**：回報「哪個物件、哪個概念」最可能帶有生成痕跡，而不只是一個分數。

程式碼位於 [`img_cbm/`](img_cbm/)，模組細節見 [`img_cbm/README_IMG.md`](img_cbm/README_IMG.md)。

## 融合層的資料設計

為了讓融合層學到「AI 痕跡不等於假新聞」，最終訓練資料刻意組合不同情境：

| 組合 | 標籤 | 目的 |
|---|---|---|
| 真實圖片＋真實文字 | 真 | 基準 |
| 真實圖片＋無誘導的 AI 文字 | 真 | 有 AI 痕跡但無惡意語意，不應直接判假 |
| 真實圖片＋誘導性 AI 文字 | 假 | 學習 AI 痕跡與誘導概念同時出現的情形 |
| 真實文字＋AI 生成圖片 | 假 | 學習信任影像端的警訊 |
| AI 文字＋AI 圖片 | 假 | 全面造假的上界 |

## 系統實作

| 目錄 | 內容 |
|---|---|
| `data_preparation(crawler&cleaner)/` | 自由時報、聯合、TVBS、Yahoo、NOWnews 等新聞爬蟲與清洗流程 |
| `text_aigc/` | RoBERTa AIGC 訓練與 AI 文字生成 |
| `text_intent/` | CB-LLM 誘導概念訓練、評估與資料生成 |
| `text_nli/` | 事實一致性（NLI）實驗 |
| `img_cbm/` | 影像 CBM 訓練、測試與權重 |
| `backend/` | Flask API：網址爬取 → 三個模型推論 → 融合 → 分析報告；各模型以獨立 Docker 服務部署 |
| `News-Analysis-Engine/` | 前端網站（React＋Vite）與 API server |

### 執行

```bash
cp .env.example .env            # 視需要填入 HF_TOKEN、GOOGLE_API_KEY 等
docker compose -f docker-compose.dev.yml up --build
```

各模組也可單獨執行，例如影像端：

```bash
cd img_cbm
pip install -r requirements.txt
python test_img.py --mode image --model_class cbm-predictor
```

> 資料集與部分模型權重（RoBERTa、CB-LLM）因檔案大小與授權未收錄於本 repo。

## 致謝

本專題為團隊合作成果，文字端模組由組員負責開發。感謝楊鎮華教授的指導。
