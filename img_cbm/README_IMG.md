# Image CBM Model - 圖片概念瓶頸模型

這是 **Multimodal News Detector** 專案中的圖片 CBM (Concept Bottleneck Model) 模組。

## 模組結構

```
img_cbm/
├── models/
│   ├── __init__.py
│   └── mirage_img.py          # CBM 模型架構定義
├── checkpoints/
│   ├── best-cbm-encoder.pt    # CBM Encoder 權重 (1.9MB)
│   └── best-cbm-predictor.pt  # CBM Predictor 權重 (2.4KB)
├── configs/
│   ├── cbm-encoder.yaml       # Encoder 訓練配置
│   ├── cbm-predictor.yaml     # Predictor 訓練配置
│   └── *-train-noise.yaml     # 對抗訓練配置
├── data/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset 類別
│   ├── encode_image.py        # 圖片特徵編碼
│   └── class_names.txt        
├── utils/
│   ├── __init__.py
│   └── metrics.py             # 評估指標
├── train_img.py               # 訓練主程式
└── test_img.py                # 測試評估程式
```

## 模型功能

### CBM Encoder
- 輸入: 圖片特徵 `[N, 1408]` (來自 DINOv2 或 CLIP)
- 輸出: 300 個物件類別的概念向量 `[N, 300]`
- 作用: 將圖片編碼為可解釋的概念表示

### CBM Predictor
- 輸入: CBM 概念向量 `[N, 300]`
- 輸出: 真假新聞分類 `{0: Real, 1: Fake}`
- 作用: 基於概念進行最終決策

### 12 個核心概念
詳見 [data/class_names.txt](data/class_names.txt)，包含：
- bottle (瓶子), car (汽車), chair (椅子), cup (杯子)
- dining table (餐桌), person (人物), potted plant (盆栽)
- ... 等 12 個日常物件

## 快速開始

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 訓練模型
```bash
# 1. 訓練 CBM Encoder
python train_img.py --mode image --model_class cbm-encoder

# 2. 訓練 CBM Predictor
python train_img.py --mode image --model_class cbm-predictor
```

### 測試模型
```bash
python test_img.py --mode image --model_class cbm-predictor
```

## 模型性能

| 模型 | F1 Score | Accuracy | 檔案大小 |
|------|----------|----------|----------|
| CBM Encoder | - | - | 1.9 MB |
| CBM Predictor | ~0.85 | ~85% | 2.4 KB |


### configs/cbm-predictor.yaml
```yaml
train_dataset:
  data_path: encodings/predictions/image/best-cbm-encoder/train
eval_dataset:
  data_path: encodings/predictions/image/best-cbm-encoder/test
```
