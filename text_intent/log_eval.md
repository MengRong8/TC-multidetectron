正在載入 Qwen-3 模型...
正在掛載 LoRA Adapters...
=== 開始讀取所有資料集 ===
測試集準備完成: 3012 筆資料, 概念數: 15

=== 開始離線評估 ===

=== 載入模型: ./cbm_checkpoints_V3/cbm_epoch_1.pth ===
⚠️ checkpoint 非嚴格載入: missing=0, unexpected=1150

=== Epoch 1 Final 任務 ===
Accuracy: 53.65%
              precision    recall  f1-score   support

        Real     0.5153    1.0000    0.6801      1484
        Fake     1.0000    0.0864    0.1590      1528

    accuracy                         0.5365      3012
   macro avg     0.7576    0.5432    0.4196      3012
weighted avg     0.7612    0.5365    0.4158      3012

📊 Final 混淆矩陣已儲存: ./cbm_checkpoints_V3/eval_final_confusion_epoch_1.png

=== Epoch 1 Concept 任務 (Fake Only) ===
Top-1 Accuracy: 81.81%
              precision    recall  f1-score   support

         情緒化      0.575     0.359     0.442        64
   new_誇大與聳動      0.953     0.884     0.917        69
     new_煽動性      0.945     0.932     0.939        74
          不實      0.907     0.803     0.852        61
      new_主觀      1.000     0.985     0.992        65
          極端      0.335     0.958     0.496        71
         威脅性      0.868     0.930     0.898        71
        虛假引用      0.900     0.913     0.906        69
      new_偏頗      0.905     0.977     0.940        88
         標題黨      0.938     0.469     0.625        64
        譁眾取寵      1.000     0.984     0.992        62
        斷章取義      0.987     0.963     0.975        81
        情緒操控      1.000     0.595     0.746        74
        權威訴求      1.000     0.487     0.655        76
         假兩難      1.000     0.958     0.979        72

    accuracy                          0.818      1061
   macro avg      0.888     0.813     0.824      1061
weighted avg      0.890     0.818     0.827      1061

📊 Concept 混淆矩陣已儲存: ./cbm_checkpoints_V3/eval_concept_confusion_epoch_1.png
Top-3 Accuracy: 92.18%

=== 載入模型: ./cbm_checkpoints_V3/cbm_epoch_2.pth ===
⚠️ checkpoint 非嚴格載入: missing=0, unexpected=1150

=== Epoch 2 Final 任務 ===
Accuracy: 55.01%
              precision    recall  f1-score   support

        Real     0.5227    1.0000    0.6866      1484
        Fake     1.0000    0.1132    0.2034      1528

    accuracy                         0.5501      3012
   macro avg     0.7614    0.5566    0.4450      3012
weighted avg     0.7648    0.5501    0.4415      3012

📊 Final 混淆矩陣已儲存: ./cbm_checkpoints_V3/eval_final_confusion_epoch_2.png

=== Epoch 2 Concept 任務 (Fake Only) ===
Top-1 Accuracy: 82.00%
              precision    recall  f1-score   support

         情緒化      0.595     0.391     0.472        64
   new_誇大與聳動      0.969     0.913     0.940        69
     new_煽動性      0.971     0.892     0.930        74
          不實      0.938     0.738     0.826        61
      new_主觀      1.000     0.985     0.992        65
          極端      0.333     0.930     0.491        71
         威脅性      0.836     0.859     0.847        71
        虛假引用      0.889     0.928     0.908        69
      new_偏頗      0.907     1.000     0.951        88
         標題黨      0.941     0.500     0.653        64
        譁眾取寵      0.984     0.984     0.984        62
        斷章取義      0.987     0.963     0.975        81
        情緒操控      1.000     0.595     0.746        74
        權威訴求      0.957     0.592     0.732        76
         假兩難      1.000     0.944     0.971        72

    accuracy                          0.820      1061
   macro avg      0.887     0.814     0.828      1061
weighted avg      0.889     0.820     0.832      1061

📊 Concept 混淆矩陣已儲存: ./cbm_checkpoints_V3/eval_concept_confusion_epoch_2.png
Top-3 Accuracy: 92.27%

=== 載入模型: ./cbm_checkpoints_V3/cbm_epoch_3.pth ===
⚠️ checkpoint 非嚴格載入: missing=0, unexpected=1150

=== Epoch 3 Final 任務 ===
Accuracy: 52.16%
              precision    recall  f1-score   support

        Real     0.5074    1.0000    0.6732      1484
        Fake     1.0000    0.0569    0.1077      1528

    accuracy                         0.5216      3012
   macro avg     0.7537    0.5285    0.3905      3012
weighted avg     0.7573    0.5216    0.3863      3012

📊 Final 混淆矩陣已儲存: ./cbm_checkpoints_V3/eval_final_confusion_epoch_3.png

=== Epoch 3 Concept 任務 (Fake Only) ===
Top-1 Accuracy: 85.01%
              precision    recall  f1-score   support

         情緒化      0.618     0.328     0.429        64
   new_誇大與聳動      0.942     0.942     0.942        69
     new_煽動性      0.986     0.932     0.958        74
          不實      0.906     0.787     0.842        61
      new_主觀      1.000     0.985     0.992        65
          極端      0.418     0.859     0.562        71
         威脅性      0.896     0.845     0.870        71
        虛假引用      0.884     0.884     0.884        69
      new_偏頗      0.967     0.989     0.978        88
         標題黨      0.820     0.641     0.719        64
        譁眾取寵      1.000     0.984     0.992        62
        斷章取義      0.987     0.951     0.969        81
        情緒操控      0.889     0.757     0.818        74
        權威訴求      0.795     0.816     0.805        76
         假兩難      1.000     0.958     0.979        72

    accuracy                          0.850      1061
   macro avg      0.874     0.844     0.849      1061
weighted avg      0.876     0.850     0.854      1061

📊 Concept 混淆矩陣已儲存: ./cbm_checkpoints_V3/eval_concept_confusion_epoch_3.png
Top-3 Accuracy: 94.06%

✅ 評估完成
