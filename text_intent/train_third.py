import torch
import glob
import os, re, gc
import logging
from bisect import bisect_right
from collections import Counter
import torch._inductor.config
import json
from transformers import AutoConfig
from unsloth import FastLanguageModel
from datasets import load_dataset
from torch.utils.data import ConcatDataset, random_split, DataLoader
import torch.nn as nn
from tqdm import tqdm
from dataloader import DirectPairedNewsDataset


class _SuppressAttentionMaskDeprecation(logging.Filter):
    """Avoid noisy logging formatting issues from transformers deprecation call."""

    TARGET_PREFIX = "The attention mask API under `transformers.modeling_attn_mask_utils`"

    def filter(self, record):
        return self.TARGET_PREFIX not in str(record.msg)


logging.getLogger("transformers.modeling_attn_mask_utils").addFilter(
    _SuppressAttentionMaskDeprecation()
)

# ==========================================
# 預訓練模型階段 (Model Loading)
# ==========================================

model_name = "unsloth/Qwen3-8B"

# 1. 設定參數
max_seq_length = 1536
dtype = torch.bfloat16
load_in_4bit = True # 4bit 量化，省顯存關鍵

# 2. 載入 Unsloth 優化模型
print("正在載入 Qwen-3 模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name, 
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# (Left Padding)
# 這是為了讓最後一個 Token 永遠是真實的句尾，而不是 Padding
tokenizer.padding_side = 'left' 

# 2. 確保 Pad Token 存在
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # 或是 tokenizer.pad_token_id = tokenizer.eos_token_id
    
print(f"Tokenizer Padding Side: {tokenizer.padding_side}") # 確認一下，應該要是 'left'
print(f"Pad Token: {tokenizer.pad_token}")

# 3. 掛載 LoRA Adapter
print("正在掛載 LoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # 建議 16, 32, 64
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_alpha = 16,
    lora_dropout = 0, # 設為 0 以獲得更快速度
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 節省顯存
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ==========================================
# 資料準備區塊 (Data Preparation)
# ==========================================

# 1. 定義來源 (Real, Fake, Filtered)
sources = [
    ("/home/howard/web_crawling/chinatime/chinatime_cleaned2.json", "/home/howard/cb_llm_induction/final_dataset/chinatime_final.json", "/home/howard/web_crawling/chinatime/chinatime_fake.json"),
    ("/home/howard/web_crawling/ettoday/ettoday_cleaned.json", "/home/howard/cb_llm_induction/final_dataset/ettoday_final.json", "/home/howard/web_crawling/ettoday/ettoday_fake.json"),
    ("/home/howard/web_crawling/ltn/ltn_cleaned2.json", "/home/howard/cb_llm_induction/final_dataset/ltn_final.json", "/home/howard/web_crawling/ltn/ltn_fake.json"),
    ("/home/howard/web_crawling/nownews/nownews_cleaned.json", "/home/howard/cb_llm_induction/final_dataset/nownews_final.json", "/home/howard/web_crawling/nownews/nownews_fake.json"),
    ("/home/howard/web_crawling/yahoo/yahoo_cleaned2.json", "/home/howard/cb_llm_induction/dataset/yahoo_rewrite_cleaned.json", "/home/howard/web_crawling/yahoo/yahoo_fake.json")
]

all_datasets = []


def _resolve_semantic_label(item, concept_map):
    if item.get("label") == 0:
        return "REAL"
    tbm_class = item.get("tbm_class")
    if tbm_class in concept_map:
        return tbm_class
    return "FAKE_UNKNOWN"


def count_subset_semantic_labels(subset, concept_map):
    """統計 random_split 後 subset 的語意標籤分佈（不經 tokenizer）。"""
    counts = Counter()
    concat_ds = subset.dataset
    cum_sizes = concat_ds.cumulative_sizes

    for global_idx in subset.indices:
        ds_idx = bisect_right(cum_sizes, global_idx)
        prev_cum = 0 if ds_idx == 0 else cum_sizes[ds_idx - 1]
        local_idx = global_idx - prev_cum

        raw_item = concat_ds.datasets[ds_idx].data[local_idx]
        label_name = _resolve_semantic_label(raw_item, concept_map)
        counts[label_name] += 1

    return counts


def print_semantic_counts(split_name, counts, concept_map):
    ordered_labels = ["REAL"] + concept_map + ["FAKE_UNKNOWN"]
    total = sum(counts.values())

    print(f"\n=== {split_name} 語意標籤統計 (總筆數: {total}) ===")
    for label_name in ordered_labels:
        print(f"{label_name}: {counts.get(label_name, 0)}")
    print("---------------------------")

print("=== 開始讀取所有資料集 ===") 
for real_path, fake_path, filtered_path in sources:
    if os.path.exists(real_path) and os.path.exists(fake_path):
        # 實例化你的 Class
        ds = DirectPairedNewsDataset(
            real_json_path=real_path,
            fake_json_path=fake_path,
            tokenizer=tokenizer, # 這裡用到了 Step 3 的 tokenizer
            max_length=max_seq_length,
            filtered_json_path=filtered_path # 加入廣告過濾清單
        )
        all_datasets.append(ds) # 將 dataloader 刑式存取至全資料集
    else:
        print(f"Warning: 找不到檔案 {real_path} 或 {fake_path}")

# 2. 合併與切分 (9:1)
if len(all_datasets) > 0:
    # A. 合併所有來源
    final_dataset = ConcatDataset(all_datasets)
    total_count = len(final_dataset)
    print(f"=== 合併完成，總資料筆數: {total_count} ===")

    # B. 計算切分數量 (90% Train, 10% Test)
    test_ratio = 0.1
    test_size = int(total_count * test_ratio) # 取 10%
    train_size = total_count - test_size      # 剩下的就是 90% (這樣最保險)

    print(f"預計切分 -> 訓練集: {train_size} 筆, 測試集: {test_size} 筆")

    # C. 執行隨機切分 (固定種子 seed=42 確保每次切的一樣)
    train_set, test_set = random_split(
        final_dataset, 
        [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # D. 統計 train/test 各語意標籤筆數（含 REAL 與 FAKE_UNKNOWN）
    concept_map = all_datasets[0].concept_map
    train_semantic_counts = count_subset_semantic_labels(train_set, concept_map)
    test_semantic_counts = count_subset_semantic_labels(test_set, concept_map)
    print_semantic_counts("Train", train_semantic_counts, concept_map)
    print_semantic_counts("Test", test_semantic_counts, concept_map)
    
    # E. 建立 DataLoader
    # num_workers=0 是為了避免在某些 Server/Windows 上多執行緒報錯，若很慢可改 2 或 4
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    
    print("✅ DataLoader 準備完成，隨時可以開始訓練！")

else:
    raise ValueError("❌ 錯誤：沒有讀取到任何資料，請檢查路徑與 sources 列表。")


# ==========================================
# CBM 架構
# ==========================================

class NewsCBM(torch.nn.Module):
    def __init__(self, backbone, num_concepts=15):
        super().__init__()
        self.backbone = backbone
        # 繼承 LLM 的設定 ex. hidden size
        self.config = backbone.config
        # 15 個概念分類頭 concept head
        self.concept_head = torch.nn.Linear(backbone.config.hidden_size, num_concepts)
        
        # 最終真假預測頭 final head
        self.final_head = torch.nn.Linear(num_concepts, 1)

    def forward(self, input_ids, attention_mask=None):
        # 直接呼叫 base model，取得 last_hidden_state（避免 CausalLMOutput 無此欄位）
        outputs = self.backbone.model.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            return_dict=True
        )
        # 取最後一個 token 的特徵 = 全文精華
        last_hidden_state = outputs.last_hidden_state[:, -1, :]
        
        # CBM Extraction
        concept_logits = self.concept_head(last_hidden_state)
        # 【新增】限速器 (Logit Clamping)
        concept_logits = torch.clamp(concept_logits, min=-15, max=15)
        concept_probs = torch.sigmoid(concept_logits)
        final_logits = self.final_head(concept_probs)
        
        return concept_logits, final_logits

# 初始化你的 CBM 模型（包裝 unsloth model）
cbm_model = NewsCBM(model).to("cuda")

print("🔧 正在將 CBM 模型權重轉換為 BFloat16...")
cbm_model = cbm_model.to(torch.bfloat16)

# ==========================================
# 訓練設定與迴圈 (Training Loop)
# ==========================================

# A. 設定超參數
LEARNING_RATE = 2e-4
NUM_EPOCHS = 5
ENABLE_EVAL = False
GRADIENT_ACCUMULATION_STEPS = 4 # 如果顯存不夠，把 batch_size 調小，把這個調大
SAVE_DIR = "./cbm_checkpoints_V3"
os.makedirs(SAVE_DIR, exist_ok=True)


def find_latest_epoch_checkpoint(save_dir):
    """回傳 (epoch_index_from_0, checkpoint_path)，若無則回傳 (0, None)。"""
    pattern = os.path.join(save_dir, "cbm_epoch_*.pth")
    candidates = glob.glob(pattern)
    if not candidates:
        return 0, None

    best_epoch = -1
    best_path = None
    for path in candidates:
        m = re.search(r"cbm_epoch_(\d+)\.pth$", os.path.basename(path))
        if not m:
            continue
        epoch_num = int(m.group(1))
        if epoch_num > best_epoch:
            best_epoch = epoch_num
            best_path = path

    if best_path is None:
        return 0, None

    # 檔名是 1-based（cbm_epoch_1），for-loop 的 epoch 是 0-based。
    return best_epoch, best_path


def load_resume_checkpoint(path, model, optimizer):
    """載入 checkpoint，兼容舊格式（僅 model state_dict）。"""

    def _load_model_state(state_dict):
        try:
            model.load_state_dict(state_dict, strict=True)
            print("✅ checkpoint 權重 strict=True 載入成功")
            return
        except RuntimeError as e:
            print(f"⚠️ strict=True 載入失敗，改用 strict=False 相容載入。原因: {e}")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(
                "ℹ️ 相容載入完成 "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    payload = torch.load(path, map_location="cpu")

    # 新格式: dict 包含 model_state / optimizer_state / 其他訓練狀態
    if isinstance(payload, dict) and "model_state" in payload:
        _load_model_state(payload["model_state"])

        if "optimizer_state" in payload and payload["optimizer_state"] is not None:
            try:
                optimizer.load_state_dict(payload["optimizer_state"])
                print("✅ optimizer state 載入成功（含動量）")
            except Exception as e:
                print(f"⚠️ optimizer state 載入失敗，將使用新 optimizer 狀態。原因: {e}")

        loaded_epoch = int(payload.get("epoch", 0))
        loaded_best = float(payload.get("best_val_c_loss", float("inf")))
        loaded_patience = int(payload.get("patience_counter", 0))
        return loaded_epoch, loaded_best, loaded_patience

    # 舊格式: 直接就是 model state_dict
    _load_model_state(payload)
    return None, None, None


def build_checkpoint_payload(model, optimizer, epoch, best_val_c_loss, patience_counter):
    """建立完整訓練狀態 checkpoint。"""
    return {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_c_loss": float(best_val_c_loss),
        "patience_counter": int(patience_counter),
    }

# Early Stopping 參數
PATIENCE = 3  # 無改善持續幾次 epoch 就停止訓練
BEST_VAL_LOSS = float('inf')  # 初始化最佳驗證損失
PATIENCE_COUNTER = 0
best_val_c_loss = BEST_VAL_LOSS
patience_counter = PATIENCE_COUNTER

# B. 定義優化器與損失函數
# 這裡建議把 Learning Rate 分組：Backbone (LoRA) 用小一點，Head 用大一點 (可選)
optimizer = torch.optim.AdamW(cbm_model.parameters(), lr=LEARNING_RATE)

# Concept Loss
# 使用 CrossEntropyLoss 以及為每個概念分類添加權重 weights
weights_list = [
    1.0, 1.0, 1.0, 1.0,  # 情緒化(難), 誇大, 煽動性(難), 聳動
    1.0, 1.0, 1.0, 1.0,  # 不實, 主觀(難), 極端(難), 威脅性(難)
    1.0, 1.0, 1.0, 1.0,  # 虛假引用, 偏頗(難), 標題黨, 譁眾取寵
    1.0, 1.0, 1.0   # 斷章取義(易), 情緒操控(中), 權威訴求, 假兩難
]

# pos_weight 專門用來處理正樣本不平衡，不需要 label_smoothing 參數
pos_weight = torch.tensor(weights_list).to("cuda").float()

criterion_concept = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Final Loss
criterion_final = nn.BCEWithLogitsLoss()

# D. 嘗試從最新 checkpoint 自動續跑
start_epoch, resume_path = find_latest_epoch_checkpoint(SAVE_DIR)
if resume_path is not None and start_epoch < NUM_EPOCHS:
    print(f"♻️ 偵測到 checkpoint，從下一個 epoch 續跑: {resume_path}")
    loaded_epoch, loaded_best, loaded_patience = load_resume_checkpoint(
        resume_path,
        cbm_model,
        optimizer,
    )
    if loaded_best is not None:
        best_val_c_loss = loaded_best
        patience_counter = loaded_patience
    if loaded_epoch is not None:
        start_epoch = loaded_epoch
    print(f"✅ 已載入權重，將從 Epoch {start_epoch + 1}/{NUM_EPOCHS} 開始")
elif resume_path is not None and start_epoch >= NUM_EPOCHS:
    print("ℹ️ 最新 checkpoint 已達或超過 NUM_EPOCHS，將不再進行訓練。")
else:
    print("🆕 未偵測到既有 checkpoint，從頭開始訓練。")

# C. 訓練主迴圈
cbm_model.train() # 切換到訓練模式

for epoch in range(start_epoch, NUM_EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
    
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training")
    
    for step, batch in enumerate(progress_bar):
        # 1. 搬移資料到 GPU
        input_ids = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        c_labels = batch['concept_labels'].to("cuda") # 18維
        f_labels = batch['final_labels'].to("cuda")   # 1維
        
        # 2. Forward (前向傳播)
        c_logits, f_logits = cbm_model(input_ids, attention_mask)
        
        # 3. Calculate Loss (雙重 Loss)
        # 你可以加權重： loss = 0.8 * loss_c + 0.2 * loss_f
        fake_mask = (f_labels.view(-1) > 0.5)
        if fake_mask.any():
            loss_c = criterion_concept(
                c_logits[fake_mask].float(),
                c_labels[fake_mask].float()
            )
        else:
            # 若該 batch 無 fake，concept loss 設為 0，僅更新 final head
            loss_c = torch.tensor(0.0, device=c_logits.device)
        loss_f = criterion_final(f_logits.float(), f_labels)
        # 權重先設1:1
        loss = loss_c + loss_f
        
        # [新增] NaN 偵測 (選用，方便除錯)
        if torch.isnan(loss):
            print(f"⚠️ Warning: NaN detected at step {step}")
            # 【關鍵】強制清空可能殘留的梯度，確保乾淨
            optimizer.zero_grad()
            continue # 跳過這一步，不要更新參數
        
        # 4. Backward (反向傳播)
        # 處理 Gradient Accumulation (模擬大 Batch Size)
        loss = loss / GRADIENT_ACCUMULATION_STEPS 
        loss.backward()
        
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(cbm_model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()
            optimizer.zero_grad()
        
        # 5. 更新進度條顯示
        total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({
            "Loss": f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}", 
            "C_Loss": f"{loss_c.item():.4f}",
            "F_Loss": f"{loss_f.item():.4f}"
        })

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

    # 每個 epoch 都先保存，可在被中斷後續跑
    current_epoch_path = os.path.join(SAVE_DIR, f"cbm_epoch_{epoch+1}.pth")
    torch.save(
        build_checkpoint_payload(
            cbm_model,
            optimizer,
            epoch + 1,
            best_val_c_loss,
            patience_counter,
        ),
        current_epoch_path,
    )
    print(f"💾 模型已備份: {current_epoch_path}")

    if not ENABLE_EVAL:
        print("⏭️ 本次設定為不做評估，直接進入下一個訓練 epoch。")
        cbm_model.train()
        continue

    # ==========================================
    # D. 每個 Epoch 結束後跑一次測試集 (Evaluation) + Early Stopping
    # ==========================================
    print("正在評估測試集...")

    optimizer.zero_grad()       # 清空梯度
    torch.cuda.empty_cache()    # 強制釋放 PyTorch 佔用的未用緩存 (關鍵!)
    gc.collect()                # 強制 Python 進行垃圾回收 (回收 RAM)

    cbm_model.eval() # 切換到評估模式 (關閉 Dropout)

    correct_f = 0
    total_samples = 0
    val_total_c_loss = 0 # 用來算平均 Concept Loss
    
    with torch.no_grad(): # 評估不需要算梯度，省記憶體
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to("cuda")
            attention_mask = batch['attention_mask'].to("cuda")
            c_labels = batch['concept_labels'].to("cuda") # 驗證也需要 Concept Labels
            f_labels = batch['final_labels'].to("cuda")
            
            # Forward
            c_logits, f_logits = cbm_model(input_ids, attention_mask)
            
            # [新增] 計算 Validation Concept Loss (這是 Early Stopping 的指標)
            fake_mask = (f_labels.view(-1) > 0.5)
            if fake_mask.any():
                val_loss_c = criterion_concept(
                    c_logits[fake_mask].float(),
                    c_labels[fake_mask].float()
                )
            else:
                val_loss_c = torch.tensor(0.0, device=c_logits.device)
            val_total_c_loss += val_loss_c.item()
            
            # CBM 最後一層通常輸出 [32, 1]，如果不壓扁，跟 [32] 的 labels 比較會出錯
            f_logits = f_logits.squeeze(-1)
            probs = torch.sigmoid(f_logits.float())
            # 計算準確率 (真假新聞辨識率)
            preds = (probs > 0.5).long()
            f_labels = f_labels.long()
            
            # correct_f += (preds == f_labels).sum().item()
            correct_f += (preds.view(-1) == f_labels.view(-1)).sum().item()
            total_samples += f_labels.view(-1).size(0)
        
    avg_val_c_loss = val_total_c_loss / len(test_loader)
    
        
    # 防止除以零 (雖然不太可能發生)
    if total_samples > 0:
        accuracy = correct_f / total_samples
        print(f"Epoch {epoch+1} Test Accuracy (真假判斷): {accuracy:.2%} | Val Concept Loss: {avg_val_c_loss:.4f}")
    else:
        print("⚠️ 測試集樣本數為 0，無法計算準確率。")
    
    # ---------------------------------------------------
    # 3. Early Stopping Logic
    # ---------------------------------------------------
    if avg_val_c_loss < best_val_c_loss:
        print(f"🔥 Loss Improved ({best_val_c_loss:.4f} -> {avg_val_c_loss:.4f}). Updating Best Model...")
        best_val_c_loss = avg_val_c_loss
        patience_counter = 0 # 重置忍耐值
        
        # 額外存一份 "best_cbm_model.pth"
        # 這樣你以後要推理時，不用去查 log 到底是第幾 epoch 最好，直接讀這個檔就好
        best_save_path = os.path.join(SAVE_DIR, "best_cbm_model.pth")
        torch.save(
            build_checkpoint_payload(
                cbm_model,
                optimizer,
                epoch + 1,
                best_val_c_loss,
                patience_counter,
            ),
            best_save_path,
        )
        
    else:
        patience_counter += 1
        print(f"⏳ No improvement. Patience: {patience_counter}/{PATIENCE}")
    
    # 觸發早停
    if patience_counter >= PATIENCE:
        print(f"🛑 Early Stopping Triggered! Training stopped at Epoch {epoch+1}.")
        print(f"   Best Validation Loss was: {best_val_c_loss:.4f}")
        break 

    cbm_model.train() # 切回訓練模式

print("訓練結束！最佳模型已存為 best_cbm_model.pth")