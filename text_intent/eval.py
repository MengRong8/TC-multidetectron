import glob
import logging
import os
import re

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import ConcatDataset, DataLoader, random_split
from tqdm import tqdm
from unsloth import FastLanguageModel

from dataloader import DirectPairedNewsDataset


class _SuppressAttentionMaskDeprecation(logging.Filter):
    """Avoid noisy logging formatting issues from transformers deprecation call."""

    TARGET_PREFIX = "The attention mask API under `transformers.modeling_attn_mask_utils`"

    def filter(self, record):
        return self.TARGET_PREFIX not in str(record.msg)


logging.getLogger("transformers.modeling_attn_mask_utils").addFilter(
    _SuppressAttentionMaskDeprecation()
)


# =========================
# 基本設定
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LOG_FILE = "log_eval.md"
CHECKPOINT_DIR = "./cbm_checkpoints_V3"
TOP_K = 3
FINAL_THRESHOLD = 0.5

model_name = "unsloth/Qwen3-8B"
max_seq_length = 1536
dtype = torch.bfloat16
load_in_4bit = True


def init_font():
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fallback_path = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Noto Sans CJK TC"
    elif os.path.exists(fallback_path):
        fm.fontManager.addfont(fallback_path)
        plt.rcParams["font.family"] = "Droid Sans Fallback"

    plt.rcParams["axes.unicode_minus"] = False


with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("")


def log_print(text):
    print(text)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(str(text) + "\n")


def find_epoch_from_path(path):
    match = re.search(r"cbm_epoch_(\d+)\.pth$", os.path.basename(path))
    if not match:
        return -1
    return int(match.group(1))


def load_model_state_only(path, model):
    """相容 train_third 的 checkpoint 格式與舊格式 state_dict。"""
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
        saved_epoch = int(payload.get("epoch", 0))
    else:
        state_dict = payload
        saved_epoch = None

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log_print(
            f"⚠️ checkpoint 非嚴格載入: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    return saved_epoch


class NewsCBM(torch.nn.Module):
    def __init__(self, backbone, num_concepts=15):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.concept_head = torch.nn.Linear(backbone.config.hidden_size, num_concepts)
        self.final_head = torch.nn.Linear(num_concepts, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        last_hidden_state = outputs.last_hidden_state[:, -1, :]

        concept_logits = self.concept_head(last_hidden_state)
        concept_logits = torch.clamp(concept_logits, min=-15, max=15)
        concept_probs = torch.sigmoid(concept_logits)
        final_logits = self.final_head(concept_probs)

        return concept_logits, final_logits


def build_test_loader(tokenizer):
    sources = [
        (
            "/home/howard/web_crawling/chinatime/chinatime_cleaned2.json",
            "/home/howard/cb_llm_induction/final_dataset/chinatime_final.json",
            "/home/howard/web_crawling/chinatime/chinatime_fake.json",
        ),
        (
            "/home/howard/web_crawling/ettoday/ettoday_cleaned.json",
            "/home/howard/cb_llm_induction/final_dataset/ettoday_final.json",
            "/home/howard/web_crawling/ettoday/ettoday_fake.json",
        ),
        (
            "/home/howard/web_crawling/ltn/ltn_cleaned2.json",
            "/home/howard/cb_llm_induction/final_dataset/ltn_final.json",
            "/home/howard/web_crawling/ltn/ltn_fake.json",
        ),
        (
            "/home/howard/web_crawling/nownews/nownews_cleaned.json",
            "/home/howard/cb_llm_induction/final_dataset/nownews_final.json",
            "/home/howard/web_crawling/nownews/nownews_fake.json",
        ),
        (
            "/home/howard/web_crawling/yahoo/yahoo_cleaned2.json",
            "/home/howard/cb_llm_induction/dataset/yahoo_rewrite_cleaned.json",
            "/home/howard/web_crawling/yahoo/yahoo_fake.json",
        ),
    ]

    all_datasets = []
    log_print("=== 開始讀取所有資料集 ===")

    for real_path, fake_path, filtered_path in sources:
        if os.path.exists(real_path) and os.path.exists(fake_path):
            ds = DirectPairedNewsDataset(
                real_json_path=real_path,
                fake_json_path=fake_path,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                filtered_json_path=filtered_path,
            )
            all_datasets.append(ds)
        else:
            log_print(f"Warning: 找不到檔案 {real_path} 或 {fake_path}")

    if not all_datasets:
        raise ValueError("❌ 錯誤：沒有讀取到任何資料。")

    final_dataset = ConcatDataset(all_datasets)
    total_count = len(final_dataset)
    test_size = int(total_count * 0.1)
    train_size = total_count - test_size

    _, test_set = random_split(
        final_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    concept_names = all_datasets[0].concept_map
    log_print(f"測試集準備完成: {len(test_set)} 筆資料, 概念數: {len(concept_names)}")
    return test_loader, concept_names


def evaluate_checkpoint(cbm_model, test_loader, concept_names, checkpoint_path):
    epoch_from_name = find_epoch_from_path(checkpoint_path)
    log_print(f"\n=== 載入模型: {checkpoint_path} ===")

    saved_epoch = load_model_state_only(checkpoint_path, cbm_model)
    display_epoch = saved_epoch if saved_epoch is not None else epoch_from_name

    cbm_model.eval()

    all_f_preds = []
    all_f_labels = []

    concept_true_labels = []
    concept_pred_labels = []
    concept_true_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {display_epoch} Eval"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            f_labels = batch["final_labels"].to(DEVICE).view(-1)
            c_labels_idx = batch["concept_idx"].to(DEVICE).view(-1)

            c_logits, f_logits = cbm_model(input_ids, attention_mask)

            f_logits = f_logits.squeeze(-1)
            f_probs = torch.sigmoid(f_logits.float())
            f_preds = (f_probs > FINAL_THRESHOLD).long()

            all_f_preds.extend(f_preds.cpu().tolist())
            all_f_labels.extend(f_labels.long().cpu().tolist())

            c_probs = torch.sigmoid(c_logits.float())

            fake_mask = f_labels > 0.5
            if fake_mask.any():
                fake_logits = c_logits[fake_mask]
                fake_probs = c_probs[fake_mask]
                fake_true_idx = c_labels_idx[fake_mask]
                fake_pred_idx = torch.argmax(fake_logits, dim=1)

                valid_mask = fake_true_idx >= 0
                if valid_mask.any():
                    concept_true_labels.extend(fake_true_idx[valid_mask].cpu().tolist())
                    concept_pred_labels.extend(fake_pred_idx[valid_mask].cpu().tolist())
                    concept_true_probs.extend(fake_probs[valid_mask].cpu().numpy().tolist())

    cm = confusion_matrix(all_f_labels, all_f_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real(0)", "Fake(1)"],
        yticklabels=["Real(0)", "Fake(1)"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Final Confusion Matrix - Epoch {display_epoch}")
    final_cm_path = os.path.join(
        CHECKPOINT_DIR, f"eval_final_confusion_epoch_{display_epoch}.png"
    )
    plt.tight_layout()
    plt.savefig(final_cm_path)
    plt.close()

    final_acc = accuracy_score(all_f_labels, all_f_preds)
    log_print(f"\n=== Epoch {display_epoch} Final 任務 ===")
    log_print(f"Accuracy: {final_acc:.2%}")
    log_print(
        classification_report(
            all_f_labels,
            all_f_preds,
            target_names=["Real", "Fake"],
            digits=4,
            zero_division=0,
        )
    )
    log_print(f"📊 Final 混淆矩陣已儲存: {final_cm_path}")

    if len(concept_true_labels) == 0:
        log_print("⚠️ 沒有可用的 Fake 概念標籤，跳過 Concept 評估。")
        return

    c_acc = accuracy_score(concept_true_labels, concept_pred_labels)
    log_print(f"\n=== Epoch {display_epoch} Concept 任務 (Fake Only) ===")
    log_print(f"Top-1 Accuracy: {c_acc:.2%}")

    unique_labels = sorted(set(concept_true_labels))
    label_names = [concept_names[i] for i in unique_labels]
    log_print(
        classification_report(
            concept_true_labels,
            concept_pred_labels,
            labels=unique_labels,
            target_names=label_names,
            digits=3,
            zero_division=0,
        )
    )

    cm_concept = confusion_matrix(
        concept_true_labels,
        concept_pred_labels,
        labels=list(range(len(concept_names))),
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_concept,
        annot=False,
        cmap="Reds",
        xticklabels=concept_names,
        yticklabels=concept_names,
    )
    plt.xlabel("Predicted Concept")
    plt.ylabel("True Concept")
    plt.title(f"Concept Confusion Matrix - Epoch {display_epoch}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    concept_cm_path = os.path.join(
        CHECKPOINT_DIR, f"eval_concept_confusion_epoch_{display_epoch}.png"
    )
    plt.savefig(concept_cm_path)
    plt.close()
    log_print(f"📊 Concept 混淆矩陣已儲存: {concept_cm_path}")

    probs_np = torch.tensor(concept_true_probs)
    true_np = torch.tensor(concept_true_labels)
    topk_idx = torch.topk(probs_np, k=min(TOP_K, probs_np.size(1)), dim=1).indices
    hits = (topk_idx == true_np.unsqueeze(1)).any(dim=1).float().mean().item()
    log_print(f"Top-{TOP_K} Accuracy: {hits:.2%}")


def main():
    init_font()

    log_print("正在載入 Qwen-3 模型...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_print("正在掛載 LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    test_loader, concept_names = build_test_loader(tokenizer)

    cbm_model = NewsCBM(model, num_concepts=len(concept_names)).to(DEVICE)
    if DEVICE == "cuda":
        cbm_model = cbm_model.to(torch.bfloat16)

    checkpoint_paths = sorted(
        glob.glob(os.path.join(CHECKPOINT_DIR, "cbm_epoch_*.pth")),
        key=find_epoch_from_path,
    )
    if not checkpoint_paths:
        best_path = os.path.join(CHECKPOINT_DIR, "best_cbm_model.pth")
        if os.path.exists(best_path):
            checkpoint_paths = [best_path]

    if not checkpoint_paths:
        raise FileNotFoundError(f"找不到 checkpoint: {CHECKPOINT_DIR}")

    log_print("\n=== 開始離線評估 ===")
    for ckpt in checkpoint_paths:
        evaluate_checkpoint(cbm_model, test_loader, concept_names, ckpt)

    log_print("\n✅ 評估完成")


if __name__ == "__main__":
    main()
