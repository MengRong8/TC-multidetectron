import json
import torch
from torch.utils.data import Dataset, DataLoader

# CB-LLM training dataloader
class DirectPairedNewsDataset(Dataset):
    def __init__(self, real_json_path, fake_json_path, tokenizer, max_length, filtered_json_path=None):
        # 1. 讀取資料
        print("正在讀取並進行 1:1 配對...")
        with open(real_json_path, 'r', encoding='utf-8') as f:
            real_data_raw = json.load(f)
        with open(fake_json_path, 'r', encoding='utf-8') as f:
            fake_data_raw = json.load(f)

        # 1.5 讀取廣告過濾清單 (若有提供)
        valid_titles = None
        if filtered_json_path:
            print(f"正在讀取廣告過濾清單: {filtered_json_path}...")
            try:
                with open(filtered_json_path, 'r', encoding='utf-8') as f:
                    filtered_data = json.load(f)
                    # 建立有效標題集合 (用於快速查找)
                    valid_titles = set(item['title'].strip() for item in filtered_data if 'title' in item)
                print(f"已載入 {len(valid_titles)} 筆有效標題 (過濾廣告用)。")
            except Exception as e:
                print(f"Warning: 無法讀取過濾清單 {filtered_json_path}: {e}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 定義 16 種誘導類別順序 (原18種，移除模糊, 誘導性, 聳動, 原_主觀, 原_煽動性，原_誇大, 原_偏頗)
        self.concept_map = [
            # "誘導性", "聳動", "煽動性"
            "情緒化", "new_誇大與聳動", "new_煽動性", 
            # "模糊", "主觀", "偏頗", 
            "不實", "new_主觀", "極端", "威脅性", "虛假引用", "new_偏頗", 
            "標題黨", "譁眾取寵", "斷章取義", "情緒操控", "權威訴求", "假兩難"
        ]

        # =================================================
        # [邏輯修正] 1:1 嚴格配對 (依賴 Title)
        # =================================================
        
        # A. 建立真實新聞的索引 (以 Title 為 Key)
        # strip() 去除首尾空白，避免因空白造成的對齊失敗
        real_map = {item['title'].strip(): item for item in real_data_raw}
        
        self.data = []
        matched_count = 0
        
        # B. 遍歷每一篇假新聞
        for fake_item in fake_data_raw:
            title_key = fake_item['title'].strip()
            
            # [新增] 廣告過濾檢查
            # 如果有提供過濾清單，且標題不在清單中，則視為廣告/無效資料，直接跳過
            if valid_titles is not None:
                if title_key not in valid_titles:
                    continue

            # 嘗試尋找對應的真實原稿
            if title_key in real_map:
                real_item = real_map[title_key]
                
                # C. 處理假新聞 (Fake)
                # 確保它有正確的 tbm_class，並且 label 設為 1
                fake_entry = fake_item.copy()
                fake_entry['label'] = 1  # 強制標記為假
                self.data.append(fake_entry)
                
                # D. 處理真新聞 (Real)
                # 確保 label 設為 0，且 tbm_class 設為 None (全 0 向量)
                real_entry = real_item.copy()
                real_entry['label'] = 0  # 強制標記為真
                real_entry['tbm_class'] = None 
                self.data.append(real_entry)
                
                matched_count += 1
            else:
                # 若標題對不起來，建議印出來檢查一下
                print(f"Warning: 找不到原稿 -> {title_key[:10]}...")
                pass
                
        print(f"此為 {real_json_path} 資料集，共配對 {matched_count} 組 (總資料筆數: {len(self.data)})")
        print("---------------------------")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 處理文本
        full_text = f"標題：{item['title']}\n內文：{item['content']}"
        
        inputs = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # --- [修改後] BCEWithLogitsLoss 用法 (One-Hot) ---
        concept_target = torch.zeros(len(self.concept_map), dtype=torch.float)

        concept_idx_int = -1 # 預設無效索引
        
        if item['label'] == 1: # 只有假新聞才有誘導類別
            if item.get('tbm_class') in self.concept_map:
                class_idx = self.concept_map.index(item['tbm_class'])
                
                # 設定 One-Hot (BCE Target)
                concept_target[class_idx] = 1.0 
                
                # 設定 Integer Index (Evaluation 用)
                concept_idx_int = class_idx
        
        # 3. 處理最終真假標籤
        final_label = torch.tensor([float(item['label'])])

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "concept_labels": concept_target, # 16 個概念標籤 (One-Hot)
            "concept_idx": torch.tensor(concept_idx_int, dtype=torch.long), # 評估用 (保持與 concept_labels 一致)
            "final_labels": final_label # 真新聞/假新聞
        }