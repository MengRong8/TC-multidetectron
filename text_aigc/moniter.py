import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# -----------------------
# Model / Tokenizer
# -----------------------
MODEL_PATH = "./model/roberta-fake-news-detection-final"

device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -----------------------
# LIME explainer
# -----------------------
explainer = LimeTextExplainer(
    class_names=["REAL", "FAKE"],
    split_expression=r"\s+"  # 中文用空白切（夠用於風格分析）
)

# -----------------------
# Prediction function
# LIME 一定要回傳 numpy array (N, num_classes)
# -----------------------
def predict_proba(texts):
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    return probs.cpu().numpy()

# -----------------------
# Load error cases
# -----------------------
with open("ood_error_cases_pB.json", "r", encoding="utf-8") as f:
    error_cases = json.load(f)

for idx, sample in enumerate(error_cases[:3]):
    # sample = error_cases[idx]

    print("TRUE:", sample["true_label"])
    print("TEXT PREVIEW:", sample["content"][:20])

    # -----------------------
    # Explain
    # -----------------------
    exp = explainer.explain_instance(
        sample["content"],
        predict_proba,
        num_features=20,
        num_samples=3000,
        labels=[0, 1]
    )

    # -----------------------
    # Show result
    # -----------------------
    print("=== LIME Explanation (FAKE) ===")
    for token, weight in exp.as_list(label=1):
        print(f"{token:15s} {weight:+.3f}")
        print("----------------------------")
    print("================================")
