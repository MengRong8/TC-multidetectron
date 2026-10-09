import re, os
import json
import torch
import random
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
# from lime.lime_text import LimeTextExplainer


MODEL_PATH = "model/roberta-fake-news-detection-final"
REAL_TEST_PATH = "ltn/ltn_cleaned2.json"
FAKE_TEST_PATH = "ltn/ltn_fake.json"

def load_test_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# 單筆測試
def predict_single(text, tokenizer, model, device): 
    # 將text token化
    inputs = tokenizer(
        text, 
        truncation=True, 
        padding=True, 
        max_length=512, 
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        
    pred = torch.argmax(probs, dim=-1).item()
    confidence = probs[0][pred].item()
    
    return pred, confidence

def clean_newlines(text: str) -> str:
    # 將多個換行（含前後空白）替換成一個空白
    text = re.sub(r"\s*\n\s*", " ", text)
    # 將多個空白壓成一個
    text = re.sub(r"\s{2,}", " ", text)
    
    return text.strip()

# 最終測試 (輸出f1...)
def testing(final_test_data, tokenizer, model, device, save_errors=True): 
    y_true = []
    y_pred = []
    y_conf = []
    error_cases = []
    
    for item in tqdm(final_test_data): 
        pred, conf = predict_single(
            item["content"], 
            tokenizer, 
            model, 
            device
        )
        y_true.append(item["label"])
        y_pred.append(pred)
        y_conf.append(conf)
        
        if pred != item["label"]:
            error_cases.append({
                "id": item["id"],
                "true_label": item["label"],
                "pred_label": pred,
                "confidence": conf,
                "content": item["content"]
            })
        
    print("=== OOD Test Result ===")
    print(classification_report(
        y_true,
        y_pred,
        target_names=["REAL", "FAKE"]
    ))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # if save_errors:
    #     with open("ood_error_cases_gptrewite_final.json", "w", encoding="utf-8") as f:
    #         json.dump(error_cases, f, ensure_ascii=False, indent=2)

    #     print(f"Saved {len(error_cases)} error cases")
       

if __name__ == "__main__": 
    # 載入 tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    real_test_data = load_test_data(REAL_TEST_PATH)
    fake_test_data = load_test_data(FAKE_TEST_PATH)
    
    real_dict = {item["id"]: item for item in real_test_data}
    fake_dict = {item["id"]: item for item in fake_test_data}

    common_ids = set(real_dict.keys()) & set(fake_dict.keys())

    print(f"Number of common articles: {len(common_ids)}")
    
    final_test_data = []
    for _id in common_ids:
        # REAL
        real_content = clean_newlines(real_dict[_id]["content"])
        fake_content = clean_newlines(fake_dict[_id]["content"])
        final_test_data.append({
            "id": _id,
            "content": real_content,
            "label": 0   # REAL
        })

        # FAKE
        final_test_data.append({
            "id": _id,
            "content": fake_content,
            "label": 1   # FAKE
        })
        
    print("Testing Dataset: ", final_test_data[:2]) 
    
    print("----------------------------------")
    print("TEST RESULT: ")
    
    testing(final_test_data, tokenizer, model, device)
    
    


