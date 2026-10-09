import re, os
import json
import torch
from datasets import Dataset, DatasetDict
import numpy as np
import random
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

REAL_PATHS = [
    "chinatime/chinatime_cleaned2.json",
    "ettoday/ettoday_cleaned.json",
    "ltn/ltn_cleaned2.json",
    "nownews/nownews_cleaned.json", 
    "yahoo/yahoo_cleaned2.json"
]

FAKE_PATHS = [
    "chinatime/chinatime_fake.json",
    "ettoday/ettoday_fake.json",
    "ltn/ltn_fake.json",
    "nownews/nownews_fake.json",
    "yahoo/yahoo_fake.json"
]

GLOBAL_SEED = 43
# 設定種子序號
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(GLOBAL_SEED)

def load_data(path): 
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def load_multiple_pairs(real_paths, fake_paths):
    final_data = []

    for real_path, fake_path in zip(real_paths, fake_paths):
        real_data = load_data(real_path)
        fake_data = load_data(fake_path)

        real_dict = {item["id"]: item for item in real_data}
        fake_dict = {item["id"]: item for item in fake_data}

        common_ids = sorted(set(real_dict) & set(fake_dict))
        print(f"{os.path.basename(real_path)} common ids: {len(common_ids)}")

        for _id in common_ids:
            final_data.append({
                "id": f"{os.path.basename(real_path)}_{_id}",
                "content": real_dict[_id]["content"],
                "label": 0
            })
            final_data.append({
                "id": f"{os.path.basename(fake_path)}_{_id}",
                "content": fake_dict[_id]["content"],
                "label": 1
            })

    return final_data
    
def clean_newlines(text: str) -> str:
    # 將多個換行（含前後空白）替換成一個空白
    text = re.sub(r"\s*\n\s*", " ", text)
    # 將多個空白壓成一個
    text = re.sub(r"\s{2,}", " ", text)
    
    return text.strip()

def prepare_dataset(final_data): 
    # 清理換行符號
    for item in final_data: 
        item["content"] = clean_newlines(item["content"])
    # 打亂順序  
    # random.shuffle(final_data)
    dataset = Dataset.from_list(final_data)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=GLOBAL_SEED)
    test_valid = split_dataset['test'].train_test_split(test_size=0.5, seed=GLOBAL_SEED)
    dataset_dict = DatasetDict({
        'train': split_dataset['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
    print(dataset_dict['train'][0], dataset_dict['validation'][0], dataset_dict['test'][0])
    return dataset_dict
    
# Tokenize
def tokenizer_function(examples): 
    return tokenizer(
        examples["content"], 
        truncation=True, 
        padding="max_length", 
        max_length=512 
    )

# Calculate evaluation metrics
def compute_metrics(eval_pred): 
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc, 
        "precision": precision, 
        "recall": recall, 
        "f1": f1
    }

def pretraining_test(tokenizer_dataset):
    sample = tokenizer_dataset["train"][0]
    print(sample.keys())
    print(len(sample["input_ids"]), len(sample["attention_mask"]))
    print(sample["label"])


if __name__ == "__main__":
    # 載入真實與假新聞資料
    final_data = load_multiple_pairs(REAL_PATHS, FAKE_PATHS)
    print(f"Total training samples: {len(final_data)}")
    
    # HuggingFace Dataset
    dataset_dict = prepare_dataset(final_data)
    
    #　Tokenizer loading，負責做文字轉換為模型可讀的格式
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_dataset = dataset_dict.map(tokenizer_function, batched=True)
    
    # Pretraining test
    pretraining_test(tokenizer_dataset)
    # Should see: 
    """
    dict_keys(['input_ids', 'attention_mask', 'label'])
    512 512
    0 or 1
    """
    
    EXPERIMENT_NAME = "roberta-fake-news-detection-overall"
    MODEL_SAVE_DIR = f"./model/{EXPERIMENT_NAME}"

    # Model loading，負責載入預訓練模型
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2  # 二分類 real vs. fake
    )
    # 設定訓練參數
    fp16 = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to="none"
    )
    
    # 建立 Trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenizer_dataset["train"],
        eval_dataset=tokenizer_dataset["validation"], 
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics
    )

    # 開始訓練
    trainer.train()

    results = trainer.evaluate(tokenizer_dataset["test"])
    print(results) # accuracy, f1, precision, recall
    
    trainer.save_model(MODEL_SAVE_DIR)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

