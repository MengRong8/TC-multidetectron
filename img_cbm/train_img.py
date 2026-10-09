import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
import json
import os
import traceback

from models import get_model
from data import load_config, get_dataset, get_object_class, get_object_class_config

def create_balanced_sampler(dataset):
    """為每個 dataset 建立平衡抽樣器（防止類別不平衡）"""
    class_counts = torch.bincount(dataset.labels.long())
    if (class_counts == 0).any():
        print(f" Warning: Some classes have 0 samples → {class_counts}")
        return None
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[int(label)] for label in dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

def get_class_file(model_class):
    """根據模型類型返回對應的類別檔案路徑"""
    if model_class == "tbm-encoder":
        return "data/tbm_class_names.txt"
    return "data/class_names.txt"  # default for cbm-encoder

def collate_fn_api(batch):
    """
    Custom collate function for API mode - keep texts as list
    """
    if isinstance(batch[0], dict) and 'text' in batch[0]:
        return {
            'text': [item['text'] for item in batch],  # Keep as list of strings
            'label': torch.tensor([item['label'] for item in batch])
        }
    else:
        # Default collate for other cases
        return torch.utils.data.dataloader.default_collate(batch)

def main(mode, model_class, config_path=None):
    torch.manual_seed(42)
    
    print(f"\n Starting training script with mode='{mode}', model_class='{model_class}'")

    # -----------------------------
    # 1. 載入設定檔
    # -----------------------------
    if config_path:
        print(f"Using custom config: {config_path}")
    else:
        config_path = f"configs/{mode}/{model_class}.yaml"
        print(f"Using default config: {config_path}")
    
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    print(f" Loaded configuration keys: {list(config.keys())}")

    # -----------------------------
    # 2. 設定裝置與模型
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config['training']['batch_size']
    print(f" Using device: {device}, batch size = {batch_size}")

    model = get_model(config).to(device)
    print(f" Loaded model structure:\n{model}\n")

    # -----------------------------
    # 3️. per-class encoder 模式 (cbm-encoder / tbm-encoder)
    # -----------------------------
    if model_class in ("cbm-encoder", "tbm-encoder"):
        is_tbm = (model_class == "tbm-encoder")
        
        # CBM: 逐物件類別訓練（所有 classifier 在一個模型）
        class_file = get_class_file(model_class)
        classes = get_object_class(class_file)
        print(f" Using class file: {class_file}")
        print(f" Found {len(classes)} classes: {classes[:10]}{'...' if len(classes)>10 else ''}\n")
        
        # 從 config 讀取 crops 路徑和 split 設定
        crops_dir = config.get('crops_dir', 'encodings/crops')
        train_split = config.get('train_split', 'train')
        eval_split = config.get('eval_split', 'test')
        print(f" Using crops_dir: {crops_dir}")
        print(f" Using train_split: {train_split}, eval_split: {eval_split}\n")
        
        print(" Creating unified CBM model with 300 classifiers...")
        unified_model = model.to(device)
        print(f" Model structure: {unified_model}\n")

        for i, class_name in enumerate(tqdm(classes, desc="Training object classes")):
            print(f"\n==============================")
            print(f" Training encoder for class [{i+1}/{len(classes)}]: {class_name}")
            print(f"==============================")

            try:
                # 載入 dataset config，傳入自定義參數
                object_class_config = get_object_class_config(
                    class_name, 
                    encoder_type=model_class,
                    crops_dir=crops_dir,
                    train_split=train_split,
                    eval_split=eval_split
                )
                object_class_config["encoder_type"] = model_class
                object_class_config["training"] = config["training"]

                print(f" Train source: {object_class_config.get('train_dataset', {}).get('data_path', 'N/A')}")
                print(f" Eval  source: {object_class_config.get('eval_dataset', {}).get('data_path', 'N/A')}")

                # -----------------------------
                # 資料載入（CBM=圖片特徵；TBM=文字特徵）
                # -----------------------------
                train_dataset, _ = get_dataset(object_class_config, is_eval=False)
                eval_dataset, _ = get_dataset(object_class_config, is_eval=True)

                print(f" Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

                train_sampler = create_balanced_sampler(train_dataset)
                eval_sampler = create_balanced_sampler(eval_dataset)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
                eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler)

                print(f" Training model for class '{class_name}' (classifier_idx={i})...")

                # -----------------------------
                #  Training CBM (per-class binary classifier)
                # -----------------------------
                # CBM encoder trains one binary classifier per object class
                # 傳遞 classifier_idx 來指定訓練哪個 classifier
                train_model(unified_model, train_loader, eval_loader, object_class_config['training'], device, classifier_idx=i)

                # 儲存中間進度（每 10 個類別或最後一個）
                if (i + 1) % 10 == 0 or (i + 1) == len(classes):
                    save_path = config["training"]["save_path"]
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    print(f"Saving progress: {i+1}/{len(classes)} classes trained → {save_path}")

            except FileNotFoundError as e:
                print(f"Skipped class '{class_name}': Missing data file")
                print(f"   {e}")
                continue
            except Exception as e:
                print(f"Skipped class '{class_name}' due to error: {e}")
                traceback.print_exc()
                continue
        
        # -----------------------------
        # 5CBM-Encoder 訓練完成後，生成 predictions
        # -----------------------------
        print("\n" + "=" * 60)
        print("Generating predictions for all splits...")
        print("=" * 60)
        generate_cbm_predictions(config, classes, device, mode)
        print("\nCBM-Encoder training and prediction generation completed!")
        return


    print("\n Training completed successfully!")


def generate_cbm_predictions(config, classes, device, mode):
    """
    訓練完所有 CBM-Encoder 後，生成所有 split 的 predictions
    對每個圖片跑所有 300 個 object class encoder，生成 [N, 300] 的預測
    """
    from models import ObjectClassCBMEncoder
    
    # 從 config 讀取 split 名稱（支持自定義 split）
    train_split = config.get('train_split', 'train')
    eval_split = config.get('eval_split', 'test')
    splits = [train_split, eval_split]
    checkpoint_path = config['training']['save_path']
    
    # 從 checkpoint 路徑提取模型名稱（用於生成輸出目錄）
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
    
    # 載入統一的模型（包含所有 300 個 classifier）
    print("\nLoading unified CBM model with 300 classifiers...")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        unified_model = ObjectClassCBMEncoder().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        unified_model.load_state_dict(checkpoint['model_state_dict'])
        unified_model.eval()
        print(f"Loaded unified model from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 為每個 split 生成 predictions
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        for label_type in ['real', 'fake']:
            # 構建輸入特徵路徑（從 encodings/image/{split}/{label_type}.pt）
            input_path = f"encodings/image/{split}/{label_type}.pt"
            
            if not os.path.exists(input_path):
                print(f"Skipped {label_type}: {input_path} not found")
                continue
            
            # 載入圖片特徵 [N, 1408]
            print(f"  Loading {label_type} features from {input_path}")
            features = torch.load(input_path, weights_only=False)
            
            if isinstance(features, dict):
                features = features['features']
            
            features = features.to(device)
            num_samples = features.shape[0]
            print(f"    {num_samples} samples, shape: {features.shape}")
            
            # 初始化 predictions [N, 300]
            all_predictions = torch.zeros(num_samples, 300, device=device)
            
            # 對每個 object class 生成預測
            print(f"    Generating predictions for {len(classes)} classes...")
            with torch.no_grad():
                for idx in tqdm(range(len(classes)), desc=f"    {label_type}"):
                    # 使用 unified_model 和 classifier_index 生成預測
                    class_logits = unified_model(features, classifier_index=idx)  # [N, 1] logits
                    # 應用 sigmoid 轉換為概率
                    class_prob = torch.sigmoid(class_logits).squeeze()  # [N]
                    all_predictions[:, idx] = class_prob
            
            # 保存 predictions（使用checkpoint名稱作為目錄名）
            output_dir = f"encodings/predictions/image/{checkpoint_name}/{split}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/{label_type}.pt"
            
            # 移回 CPU 再保存
            all_predictions_cpu = all_predictions.cpu()
            torch.save(all_predictions_cpu, output_path)
            print(f"Saved to {output_path} with shape {all_predictions_cpu.shape}")


def add_adversarial_noise(features, epsilon=0.05, noise_type='gaussian'):
    """
    為特徵添加對抗性擾動，增強模型泛化能力
    Args:
        features: 輸入特徵 [batch_size, feature_dim]
        epsilon: 擾動強度
        noise_type: 'gaussian' (高斯噪聲) 或 'uniform' (均勻噪聲)
    Returns:
        添加擾動後的特徵
    """
    if noise_type == 'gaussian':
        # 高斯噪聲: N(0, epsilon)
        noise = torch.randn_like(features) * epsilon
    elif noise_type == 'uniform':
        # 均勻噪聲: U(-epsilon, epsilon)
        noise = (torch.rand_like(features) * 2 - 1) * epsilon
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type}")
    
    return features + noise


def train_model(model, train_loader, eval_loader, training_config, device, classifier_idx=None):
    """
    Train the model with the given data loaders and configuration.
    Args:
        classifier_idx: For CBM encoder, specifies which classifier to train (0-299)
    """
    # 如果是 CBM encoder 且有指定 classifier_idx，只優化該 classifier 的參數
    if classifier_idx is not None:
        # 只優化指定 classifier 的參數
        optimizer = torch.optim.Adam(
            model.classifiers[classifier_idx].parameters(), 
            lr=training_config.get('learning_rate', 0.001)
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=training_config.get('learning_rate', 0.001))
    
    # 對抗訓練參數
    use_adversarial = training_config.get('use_adversarial', False)
    adv_epsilon = training_config.get('adv_epsilon', 0.05)
    adv_lambda = training_config.get('adv_lambda', 0.3)
    noise_type = training_config.get('noise_type', 'gaussian')
    
    if use_adversarial:
        print(f"  Adversarial Training ENABLED")
        print(f"    - Epsilon (noise strength): {adv_epsilon}")
        print(f"    - Lambda (adversarial loss weight): {adv_lambda}")
        print(f"    - Noise type: {noise_type}")
    
    # 檢查模型類型來決定 loss function
    model_name = training_config.get('model_name', '')
    print(f" Training with model_name/class: '{model_name}'")
    
    is_api_model = 'api' in model_name.lower()
    is_encoder = 'encoder' in model_name.lower()
    
    if 'cbm' in model_name.lower():
        encoder_type = "cbm-encoder"
    else:
        encoder_type = None  # 對於 predictor 或其他模型
    
    if is_encoder:
        criterion = nn.BCEWithLogitsLoss()
        print(" Detected as ENCODER - will use multi-label training")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(" Detected as PREDICTOR/MIRAGE - will use single-label training")
    
    epochs = training_config.get('epochs', 100)
    save_path = training_config.get('save_path', 'checkpoints/model.pt')
    
    best_threshold = 0.5
    best_f1 = 0
    patience = training_config.get('patience', 10)
    no_improve_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad() # 清空梯度
            
            # Handle different batch formats
            if isinstance(batch, dict):
                # Check if using API (has 'text' key instead of 'input_ids')
                if 'text' in batch:
                    # API mode: pass raw texts as list
                    texts = batch['text']
                    if isinstance(texts, torch.Tensor):
                        raise TypeError("API model received tensor instead of text list")
                    labels = batch['label'].float().to(device)
                    outputs = model(texts)
                    
                    # 文本模式不支援對抗訓練
                    loss = criterion(outputs.squeeze(), labels)
                    
                elif 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].float().to(device)
                    outputs = model(input_ids, attention_mask)
                    
                    # Tokenized文本模式不支援對抗訓練
                    loss = criterion(outputs.squeeze(), labels)
                    
                elif 'features' in batch or 'data' in batch:
                    features = batch.get('features', batch.get('data')).to(device)
                    labels = batch['label'].float().to(device)
                    
                    # ========== 對抗訓練邏輯 ==========
                    if use_adversarial:
                        # 1. 乾淨樣本的損失
                        if classifier_idx is not None:
                            outputs_clean = model(features, classifier_index=classifier_idx)
                        else:
                            outputs_clean = model(features)
                        
                        if is_encoder and outputs_clean.dim() == 2 and outputs_clean.size(1) > 1:
                            labels_expanded = labels.unsqueeze(1).expand_as(outputs_clean)
                            loss_clean = criterion(outputs_clean, labels_expanded)
                        else:
                            loss_clean = criterion(outputs_clean.squeeze(), labels)
                        
                        # 2. 對抗樣本的損失
                        adv_features = add_adversarial_noise(features, epsilon=adv_epsilon, noise_type=noise_type)
                        
                        if classifier_idx is not None:
                            outputs_adv = model(adv_features, classifier_index=classifier_idx)
                        else:
                            outputs_adv = model(adv_features)
                        
                        if is_encoder and outputs_adv.dim() == 2 and outputs_adv.size(1) > 1:
                            loss_adv = criterion(outputs_adv, labels_expanded)
                        else:
                            loss_adv = criterion(outputs_adv.squeeze(), labels)
                        
                        # 3. 組合損失
                        loss = loss_clean + adv_lambda * loss_adv
                    else:
                        # 標準訓練（無對抗樣本）
                        if classifier_idx is not None:
                            outputs = model(features, classifier_index=classifier_idx)
                        else:
                            outputs = model(features)
                        
                        if is_encoder and outputs.dim() == 2 and outputs.size(1) > 1:
                            labels_expanded = labels.unsqueeze(1).expand_as(outputs)
                            loss = criterion(outputs, labels_expanded)
                        else:
                            loss = criterion(outputs.squeeze(), labels)
                else:
                    raise ValueError(f"Unrecognized batch format. Keys: {batch.keys()}")
            else:
                features, labels = batch
                features = features.to(device)
                labels = labels.float().to(device)
                
                # ========== 對抗訓練邏輯 ==========
                if use_adversarial:
                    # 1. 乾淨樣本的損失
                    if classifier_idx is not None:
                        outputs_clean = model(features, classifier_index=classifier_idx)
                    else:
                        outputs_clean = model(features)
                    
                    loss_clean = criterion(outputs_clean.squeeze(), labels)
                    
                    # 2. 對抗樣本的損失
                    adv_features = add_adversarial_noise(features, epsilon=adv_epsilon, noise_type=noise_type)
                    
                    if classifier_idx is not None:
                        outputs_adv = model(adv_features, classifier_index=classifier_idx)
                    else:
                        outputs_adv = model(adv_features)
                    
                    loss_adv = criterion(outputs_adv.squeeze(), labels)
                    
                    # 3. 組合損失
                    loss = loss_clean + adv_lambda * loss_adv
                else:
                    # 標準訓練（無對抗樣本）
                    if classifier_idx is not None:
                        outputs = model(features, classifier_index=classifier_idx)
                    else:
                        outputs = model(features)
                    
                    loss = criterion(outputs.squeeze(), labels)
            
            # 反向傳播（計算梯度）
            loss.backward() # 計算 ∂loss/∂W 和 ∂loss/∂bias
            # 更新權重（梯度下降)
            optimizer.step()    # W_new = W_old - lr × gradient
            
            epoch_loss += loss.item()
        
        # Evaluation
        model.eval()
        all_eval_preds = []
        all_eval_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                if isinstance(batch, dict):
                    if 'text' in batch:
                        texts = batch['text']
                        labels = batch['label']
                        outputs = model(texts)
                    elif 'input_ids' in batch:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label']
                        outputs = model(input_ids, attention_mask)
                    elif 'features' in batch or 'data' in batch:
                        features = batch.get('features', batch.get('data')).to(device)
                        labels = batch['label']
                        # 如果是 CBM encoder，傳遞 classifier_index
                        if classifier_idx is not None:
                            outputs = model(features, classifier_index=classifier_idx)
                        else:
                            outputs = model(features)   # 用當前 epoch 的 weights 預測
                    else:
                        raise ValueError(f"Unrecognized batch format. Keys: {batch.keys()}")
                else:
                    features, labels = batch
                    features = features.to(device)
                    # 如果是 CBM encoder，傳遞 classifier_index
                    if classifier_idx is not None:
                        outputs = model(features, classifier_index=classifier_idx)
                    else:
                        outputs = model(features)  # 用當前 epoch 的 weights 預測
                
                # Apply sigmoid since we used BCEWithLogitsLoss
                if is_encoder and outputs.dim() == 2 and outputs.size(1) > 1:
                    probs = torch.sigmoid(outputs).mean(dim=1).cpu().numpy()
                else:
                    probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                
                all_eval_preds.extend(probs)
                all_eval_labels.extend(labels.cpu().numpy())
        
        # Find best threshold
        thresholds = np.arange(0.1, 0.9, 0.05)  # [0.1, 0.15, 0.2, ..., 0.85]
        best_epoch_f1 = 0
        best_epoch_threshold = 0.5
        
        for threshold in thresholds:
            preds = (np.array(all_eval_preds) > threshold).astype(int)
            # 例如 threshold=0.4: [0.23>0.4, 0.78>0.4, ...] → [0, 1, ...]
            
            # 計算 F1
            f1 = f1_score(all_eval_labels, preds)
            # 更新最佳閾值
            if f1 > best_epoch_f1:
                best_epoch_f1 = f1
                best_epoch_threshold = threshold
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, F1: {best_epoch_f1:.4f}, Threshold: {best_epoch_threshold:.4f}")
        
        # Save best model
        if best_epoch_f1 > best_f1:     # 這個 epoch 的 F1 比之前更好
            best_f1 = best_epoch_f1
            best_threshold = best_epoch_threshold
            no_improve_count = 0
            
            # 儲存當前的 weights 和 threshold
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold
            }, save_path)
            print(f" Model saved with F1: {best_f1:.4f}, threshold: {best_threshold:.4f}")
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count}/{patience} epochs")
            
            if no_improve_count >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f" Training completed. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified configuration")
    parser.add_argument('--mode', type=str, choices=['image', 'text', 'multimodal'], required=True)
    parser.add_argument('--model_class', type=str, required=True)
    parser.add_argument('--config', type=str, help='Path to custom config file (optional)')
    args = parser.parse_args()
    main(args.mode, args.model_class, args.config)
