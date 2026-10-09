import torch
import numpy as np
import json
from tqdm import tqdm
from .metrics import calculate_metrics, find_best_threshold
from models import *

# === SHARED FUNCTIONS ===

def save_model_checkpoint(model, save_path, threshold):
    """
    Save the model state and threshold as a checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save.
        save_path (str): Path to save the checkpoint.
        threshold (float): The best threshold to save.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "best_threshold": threshold,
    }
    torch.save(checkpoint, save_path)


def load_model_checkpoint(model, save_path):
    """
    Load the model state and best threshold from a checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load the state into.
        save_path (str): Path to the checkpoint.
        
    Returns:
        model (torch.nn.Module): Model loaded with state_dict.
        best_threshold (float): Best threshold saved in the checkpoint.
    """
    checkpoint = torch.load(save_path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    best_threshold = checkpoint["best_threshold"] if "best_threshold" in checkpoint else None
    return model, best_threshold


# def evaluate_model(model, data_loader, criterion, device="cuda", threshold=0.5, cbm_encoder=None, concept_num=300):
#     """
#     Evaluate the model on a given dataset, calculating metrics using a specified threshold.
    
#     Args:
#         model (torch.nn.Module): The model to evaluate.
#         data_loader (DataLoader): DataLoader for the dataset.
#         criterion: Loss function.
#         device (str): Device for computation.
#         threshold (float): Threshold for binary classification.
#         cbm_encoder (torch.nn.Module, optional): CBM encoder model, if using CBM Predictor.
#         concept_num (int): Number of concepts in CBM Predictor.
        
#     Returns:
#         avg_loss (float): Average loss over the dataset.
#         metrics (dict): Calculated metrics using the specified threshold.
#     """
#     model.eval()
#     y_true = []
#     y_probs = []
#     total_loss = 0.0

#     with torch.no_grad():
#         for inputs, labels in data_loader:
#             inputs, labels = inputs.to(device), labels.to(device)

#             # Forward pass logic based on model type
#             if isinstance(model, ObjectClassCBMEncoder):
#                 classifier_idx = 0
#                 outputs = model(inputs.float(), classifier_idx)
#             # elif isinstance(model, ObjectClassCBMPredictor) and cbm_encoder:
#             #     concept_features = [cbm_encoder(inputs.float(), i) for i in range(concept_num)]
#             #     pred_scores = torch.cat(concept_features, dim=1)
#             #     outputs = model(pred_scores.to(device))
#             else:
#                 outputs = model(inputs.float())

#             # Calculate loss and store probabilities

#             loss = criterion(outputs.squeeze(-1), labels.float())
#             total_loss += loss.item()
#             y_true.extend(labels.cpu().numpy())
#             y_probs.extend(outputs.squeeze(-1).cpu().numpy())

#     avg_loss = total_loss / len(data_loader)
#     y_true = np.array(y_true)
#     y_probs = np.array(y_probs)
#     metrics = calculate_metrics(y_true, y_probs, threshold)

#     model.train()
#     return avg_loss, metrics, y_true, y_probs  # Return y_true and y_probs for threshold finding
    
def evaluate_model(model, dataloader, criterion, device, mode, threshold=0.5):
    """
    Evaluate model on the given dataloader.
    
    Args:
        threshold: Classification threshold (default: 0.5)
    """
    model.eval()
    total_loss = 0
    y_true = []
    y_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                # Handle dictionary batch format
                if 'input_ids' in batch:
                    # Text with BERT
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids, attention_mask)
                elif 'features' in batch or 'data' in batch:
                    # Pre-encoded features (for MiRAGe, Predictor, etc.)
                    inputs = batch.get('features', batch.get('data')).to(device)
                    labels = batch['label'].to(device)
                    outputs = model(inputs.float())
                else:
                    raise ValueError(f"Unrecognized batch format: {batch.keys()}")
            else:
                # Handle tuple batch format
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs.float())
            
            # Calculate loss
            if outputs.dim() > 1 and outputs.size(1) > 1:
                # Multi-output (encoder models)
                labels_expanded = labels.unsqueeze(1).expand_as(outputs)
                loss = criterion(outputs, labels_expanded.float())
            else:
                # Single output
                loss = criterion(outputs.squeeze(), labels.float())
            
            total_loss += loss.item()
            
            # Collect predictions
            if outputs.dim() > 1 and outputs.size(1) > 1:
                probs = torch.sigmoid(outputs).mean(dim=1)
            else:
                probs = torch.sigmoid(outputs.squeeze())
            
            y_probs.extend(probs.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # 轉換為 numpy array
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    # Calculate metrics with specified threshold
    avg_loss = total_loss / len(dataloader)
    metrics = calculate_metrics(y_true, y_probs, threshold=threshold)
    
    return avg_loss, metrics, y_true, y_probs


def save_metrics(metrics, save_path):
    """
    Save metrics as a JSON file.

    Args:
        metrics (dict): Dictionary of metrics to save.
        save_path (str): Path to the file where metrics will be saved.
    """
    with open(save_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")

# === TRAINING FUNCTION ===

def train_model(model, train_loader, eval_loader, training_config, device):
    """
    Train the model with the given data loaders and configuration.
    """
    from sklearn.metrics import f1_score
    
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config.get('learning_rate', 0.001))
    
    # 檢查模型類型來決定 loss function
    model_name = training_config.get('model_name', '')
    print(f"🔍 Training with model_name/class: '{model_name}'")
    
    is_api_model = 'api' in model_name.lower()
    is_encoder = 'encoder' in model_name.lower()
    
    if is_encoder:
        criterion = nn.BCEWithLogitsLoss()
        print("✅ Detected as ENCODER - will use multi-label training")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("✅ Detected as PREDICTOR - will use single-label training")
    
    epochs = training_config.get('epochs', 100)
    save_path = training_config.get('save_path', 'checkpoints/model.pt')
    
    best_threshold = 0.5
    best_f1 = 0
    patience = training_config.get('patience', 10)  # 連續 10 個 epoch 沒進步就停止
    no_improve_count = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            # Handle different batch formats
            if isinstance(batch, dict):
                # Check if using API (has 'text' key instead of 'input_ids')
                if 'text' in batch:
                    # API mode: pass raw texts as list
                    texts = batch['text']
                    if isinstance(texts, torch.Tensor):
                        raise TypeError("API model received tensor instead of text list")
                    labels = batch['label'].float().to(device)
                    outputs = model(texts)  # TBMEncoderAPI.forward(texts)
                # For text data with input_ids and attention_mask
                elif 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].float().to(device)
                    outputs = model(input_ids, attention_mask)
                elif 'features' in batch or 'data' in batch:
                    # For pre-encoded features (.pt files)
                    features = batch.get('features', batch.get('data')).to(device)
                    labels = batch['label'].float().to(device)
                    outputs = model(features)
                else:
                    raise ValueError(f"Unrecognized batch format. Keys: {batch.keys()}")
            else:
                # Tuple format (features, labels)
                features, labels = batch
                features = features.to(device)
                labels = labels.float().to(device)
                outputs = model(features)
            
            # For encoder models: expand labels to match multi-output shape
            if is_encoder and outputs.dim() == 2 and outputs.size(1) > 1:
                # outputs: [batch_size, num_classes]
                # labels: [batch_size] -> [batch_size, num_classes]
                labels_expanded = labels.unsqueeze(1).expand_as(outputs)
                loss = criterion(outputs, labels_expanded)
            else:
                # For single-output models
                loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 收集預測和標籤
        all_labels.extend(labels.cpu().numpy())  # 先移到 CPU
            
        # 根據不同的 encoder_type 處理
        if encoder_type == "cbm-encoder":
            predictions = torch.sigmoid(outputs)
            all_preds.extend(predictions.cpu().numpy())  # 先移到 CPU
        elif encoder_type == "tbm-encoder":
            _, predicted_classes = torch.max(outputs, 1)
            all_preds.extend(predicted_classes.cpu().numpy())  # 先移到 CPU
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Evaluation
        model.eval()
        all_eval_preds = []
        all_eval_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                if isinstance(batch, dict):
                    if 'text' in batch:
                        # API mode
                        texts = batch['text']
                        labels = batch['label']
                        outputs = model(texts)
                    elif 'input_ids' in batch:
                        # Local mode
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label']
                        outputs = model(input_ids, attention_mask)
                    elif 'features' in batch or 'data' in batch:
                        features = batch.get('features', batch.get('data')).to(device)
                        labels = batch['label']
                        outputs = model(features)
                    else:
                        raise ValueError(f"Unrecognized batch format. Keys: {batch.keys()}")
                else:
                    features, labels = batch
                    features = features.to(device)
                    outputs = model(features)
                
                # Apply sigmoid since we used BCEWithLogitsLoss
                if is_encoder and outputs.dim() == 2 and outputs.size(1) > 1:
                    # For encoder: average across all classifiers
                    probs = torch.sigmoid(outputs).mean(dim=1).cpu().numpy()
                else:
                    probs = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                
                all_eval_preds.extend(probs)
                all_eval_labels.extend(labels.numpy())
        
        # Find best threshold
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_epoch_f1 = 0
        best_epoch_threshold = 0.5
        
        for threshold in thresholds:
            preds = (np.array(all_eval_preds) > threshold).astype(int)
            f1 = f1_score(all_eval_labels, preds)
            if f1 > best_epoch_f1:
                best_epoch_f1 = f1
                best_epoch_threshold = threshold
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, F1: {best_epoch_f1:.4f}, Threshold: {best_epoch_threshold:.4f}")
        
        # Save best model
        if best_epoch_f1 > best_f1:
            best_f1 = best_epoch_f1
            best_threshold = best_epoch_threshold
            no_improve_count = 0  # 重置計數器
            torch.save({
                'model_state_dict': model.state_dict(),
                'threshold': best_threshold
            }, save_path)
            print(f"✅ Model saved with F1: {best_f1:.4f}, threshold: {best_threshold:.4f}")
        else:
            no_improve_count += 1
            print(f"⚠️  No improvement for {no_improve_count}/{patience} epochs")
            
            if no_improve_count >= patience:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"🎉 Training completed. Best F1: {best_f1:.4f}")



# === TESTING FUNCTION ===

# def test_model(model, test_loader, checkpoint_path, device="cuda", model_2=None):
#     """
#     Load the best model and threshold, then evaluate on the test set and save metrics.
    
#     Args:
#         model (torch.nn.Module): The trained model.
#         test_loader (DataLoader): DataLoader for the test set.
#         checkpoint_path (str): Path to the saved model checkpoint.
#         device (str): Device for computation (e.g., 'cuda' or 'cpu').

#     Returns:
#         dict: Test metrics.
#         list: List of predictions with true/pred labels.
#     """
#     criterion = torch.nn.BCELoss()
#     # Load model and best threshold
#     model, best_threshold = load_model_checkpoint(model, checkpoint_path)
#     # Evaluate on the test set with the loaded threshold
#     test_loss, test_metrics, y_true, y_probs = evaluate_model(model, test_loader, criterion, device, threshold=best_threshold)
#     # print("Test Metrics:", test_metrics)
#     #------------------------------
#     # === 新增：產生逐筆預測結果 ===
#     predictions = []
#     for i in range(len(y_true)):
#         pred_label = 1 if y_probs[i] >= best_threshold else 0
#         predictions.append({
#             "index": i,
#             "true": int(y_true[i]),
#             "prob": float(y_probs[i]),
#             "pred": pred_label,
#             "result": "fake" if pred_label == 1 else "real"
#         })

#     #------------------------------
#     return test_metrics,predictions
def test_model(model, test_loader, checkpoint_path, encoder_type="cbm-encoder", device="cuda"):
    if encoder_type == "cbm-encoder":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif encoder_type == "tbm-encoder":
        criterion = torch.nn.CrossEntropyLoss()
    else:
        # 對於 MiRAGe、Predictor 等其他模型，使用 BCEWithLogitsLoss
        criterion = torch.nn.BCEWithLogitsLoss()

    model, best_threshold = load_model_checkpoint(model, checkpoint_path, device)
    # 使用訓練時找到的最佳 threshold
    test_loss, test_metrics, y_true, y_probs = evaluate_model(
        model, test_loader, criterion, device, mode=encoder_type, threshold=best_threshold
    )

    predictions = []
    if encoder_type == "cbm-encoder":
        for i in range(len(y_true)):
            pred_label = 1 if y_probs[i] >= best_threshold else 0
            predictions.append({
                "index": i,
                "true": int(y_true[i]),
                "prob": float(y_probs[i]),
                "pred": pred_label,
                "result": "fake" if pred_label == 1 else "real"
            })
    elif encoder_type == "tbm-encoder":
        for i in range(len(y_true)):
            pred_label = int(np.argmax(y_probs[i]))
            predictions.append({
                "index": i,
                "true": int(y_true[i]),
                "pred": pred_label,
                "result": str(pred_label)
            })
    else:
        # 對於其他模型（MiRAGe, Linear, Predictor）
        for i in range(len(y_true)):
            pred_label = 1 if y_probs[i] >= best_threshold else 0
            predictions.append({
                "index": i,
                "true": int(y_true[i]),
                "prob": float(y_probs[i]),
                "pred": pred_label,
                "result": "fake" if pred_label == 1 else "real"
            })

    return test_metrics, predictions


def test_multimodal_model(image_model, text_model, test_loader, threshold=0.5, device='cuda'):
    image_model.eval()
    text_model.eval()
    criterion = torch.nn.BCELoss()
    y_true = []
    y_probs = []
    total_loss = 0.0

    with torch.no_grad():
        for image_inputs, text_inputs, labels in test_loader:
            image_inputs, text_inputs, labels = image_inputs.to(device), text_inputs.to(device), labels.to(device)

            # Forward pass logic based on model type
            if isinstance(image_model, ObjectClassCBMEncoder):
                classifier_idx = 0
                image_outputs = image_model(image_inputs.float(), classifier_idx)
            # elif isinstance(model, ObjectClassCBMPredictor) and cbm_encoder:
            #     concept_features = [cbm_encoder(inputs.float(), i) for i in range(concept_num)]
            #     pred_scores = torch.cat(concept_features, dim=1)
            #     outputs = model(pred_scores.to(device))
            else:
                image_outputs = image_model(image_inputs.float())

            text_outputs = text_model(text_inputs.float())
            # Calculate loss and store probabilities
            outputs = (image_outputs + text_outputs) / 2
            loss = criterion(outputs.squeeze(-1), labels.float())
            total_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_probs.extend(outputs.squeeze(-1).cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    metrics = calculate_metrics(y_true, y_probs, threshold)
    predictions = []
    for i in range(len(y_true)):
        pred_label = 1 if y_probs[i] >= threshold else 0
        predictions.append({
            "index": i,
            "true": int(y_true[i]),
            "prob": float(y_probs[i]),
            "pred": pred_label,
            "result": "fake" if pred_label == 1 else "real"
        })
    return metrics,predictions

def load_model_checkpoint(model, checkpoint_path, device):
    """
    Load model checkpoint and return model with best threshold
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 確保返回 tuple (model, threshold)
    best_threshold = checkpoint.get('threshold', 0.5)
    
    # 如果 threshold 是 numpy 類型，轉換為 Python float
    if hasattr(best_threshold, 'item'):
        best_threshold = best_threshold.item()
    
    model.eval()
    return model, best_threshold