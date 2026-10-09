import json
import re
import os
import argparse

# ==========================================
# 遮罩與刪除規則設定區
# ==========================================

# 0. 嚴重幻覺丟棄規則 (如果文本符合這些正則，整筆資料直接捨棄，不加入訓練集)
DROP_RULES = {
    "煽動性": [
        r"此時，危機已啟動，不採取行動將導致風險加劇，且未作為將導致全面崩壞的直接推手，藉此營造出沒有緩衝時間、沒有退路的壓迫感。",
        r"冷酷[、，]嚴肅",
        r"權威專家語氣",
        r"透過對未來後果的冷靜推演",
        r"自然營造壓力",
        r"宣告式危機",
        r"強調風險與潛在後果",
    ],
    "主觀": [
        r"隱含的第一人稱視角",
        r"社會共識或必然邏輯",
        r"評斷性與預設立場的詞彙",
        r"平靜[、，]自信[、，]專業評論風格",
        r"事實與解讀交織",
        r"帶有個人立場傾向",
    ],
    "偏頗": [
        r"看似客觀平衡",
        r"重心傾向某一立場",
        r"特定立場自然占據敘事主導",
        r"冷靜[、，]分析式的口吻",
        r"包裝偏見",
        r"帶風向",
        r"篇幅安排與語境優先順序",
    ],
    "語意誇大與聳動": [
        r"極度放大其影響規模",
        r"災難化[／/]奇蹟化推演",
        r"強制提升層級",
        r"誇張且聳動的層次",
        r"一本正經[、，]看似邏輯嚴密的推論過程",
        r"歷史與宏觀級別的強烈修飾詞",
    ],
    "通用": [
        r"【核心任務】",
        r"【寫作原則】",
        r"【注意事項】",
        r"硬性規則",
    ]
}

# 1. 必須完全刪除的字串 (替換為空字串，不影響整體句意)
DELETE_RULES = {
    "煽動性": [
        r"引發全面崩壞的直接推手",
        r"已啟動且持續惡化",
        r"沒有緩衝時間[、，]沒有退路的壓迫感",
        r"相關行動者(?:——)?",
        r"政府[、，及與]機構[、，及與]企業[、，及與]讀者(?:或特定群體)?", # 這是 Prompt 的原話，如果單獨出現「政府、機構」不會被誤刪
        r"具備絕對必然性的結果",
    ],
    "主觀": [
        r"個人武斷推測與評論",
        r"滲入評論性推論",
    ],
    "偏頗": [
        r"邏輯漏洞或缺乏證據",
        r"看似中立但帶有貶義的描述",
    ],
    "語意誇大與聳動": [
        r"引發誇張結果的結論",
    ],
    "通用": [
        r"蕭守善的創新方案不僅提升了視覺品質，更為台灣VTuber注入了可持續發展的動力[。]?",
    ]
}

# 2. 需要替換成「中性詞」的字串 (降級 Shortcut，保持文法通順)
REPLACE_RULES = {
    "通用": [
        (r"不僅(僅)?是?", "是"),
        (r"不僅", ""),
        (r"，更是?", "，也是"),
        (r"更是?", "也是"),
    ]
}

# 2. 需要替換成 [MASK] 的字串 (捷徑特徵 / Trigger Words)
# 因對 Qwen + Linear 訓練效果不明顯，目前改為直接在 DELETE_RULES 中執行「清洗」而非「遮罩」。
# MASK_RULES = {
#     "煽動性": [
#         r"不可逆(?:轉)?的?(?:負面)?(?:後果|損失|衝擊)",
#         r"迫在眉睫的[機危][會機]",
#         r"全面崩壞",
#         r"沒有緩衝時間(?:可供調整)?",
#     ],
#     "主觀": [
#         r"我認為[，、]?",
#         r"我觀察到[，、]?",
#         r"我相信[，、]?",
#         r"我期待(?:看到)?[，、]?",
#         r"值得(?:我們)?深入關注",
#     ],
#     "偏頗": [
#         r"相較於其他[^，。]*?，",
#         r"(?:進一步)?(?:提升|鞏固|優化|強化)",
#         r"此舉不僅",
#         r"暴露了[^，。]*?的?(?:漏洞|缺陷|不足)",
#         r"站不住腳",
#     ],
#     "語意誇大與聳動": [
#         r"(?:帶來|產生|展現)了?(?:前所未有的|深遠的?影響)",
#         r"前所未有的",
#         r"(?:歷史性|里程碑)的?(?:轉折|意義|發展)",
#         r"注入了?新的?活力",
#         r"將在未來[^，。]*?，",
#     ],
#     "通用": [
#         r"不僅(僅)?",
#         r"，更是?",
#         r"，也是?",
#     ]
# }

# 自定義的標籤對應表
# 這樣我們就可以把生成的 "煽動性" 改名為 "AIGC_煽動性"，避免跟原始 dataset 衝突
TBM_CLASS_MAPPING = {
    "煽動性": "new_煽動性",
    "主觀": "new_主觀",
    "偏頗": "new_偏頗",
    "語意誇大與聳動": "new_誇大與聳動",
}

def should_drop(text: str, tbm_class: str) -> bool:
    """判斷此文本是否包含嚴重幻覺，若是則整筆捨棄"""
    if not text: return True
    
    for pattern in DROP_RULES.get("通用", []):
        if re.search(pattern, text):
            return True
            
    for pattern in DROP_RULES.get(tbm_class, []):
        if re.search(pattern, text):
            return True
            
    return False

import random

def apply_masking(text: str, tbm_class: str) -> tuple[str, int]:
    """根據 tbm_class 對文本進行正則替換與刪除，並回傳清除的洩漏數量"""
    if not text: return text, 0

    processed_text = text
    leak_count = 0

    # --- 執行刪除/清洗規則 (替換為空字串) ---
    for pattern in DELETE_RULES.get("通用", []):
        matches = re.findall(pattern, processed_text)
        if matches:
            leak_count += len(matches)
            processed_text = re.sub(pattern, "", processed_text)
            
    for pattern in DELETE_RULES.get(tbm_class, []):
        matches = re.findall(pattern, processed_text)
        if matches:
            leak_count += len(matches)
            processed_text = re.sub(pattern, "", processed_text)

    # --- 執行替換規則 (降級 Shortcut，加上機率控制) ---
    # 因為 LLM 生成含有「不僅」的比例高達近 80%，而真實新聞約 30%
    # 這裡我們用 60% 的機率對「不僅」這類排比進行降級，保留 40% 的原汁原味
    if "不僅" in processed_text and random.random() < 0.60:
        processed_text = re.sub(r"不僅(僅)?是?", "是", processed_text)
        processed_text = re.sub(r"不僅", "", processed_text) # 處理沒有「是」的狀況
        processed_text = re.sub(r"，更是?", "，也是", processed_text)
        processed_text = re.sub(r"更是?", "也是", processed_text)

    # 稍微清理一下多餘的標點符號 (例如連續兩個逗號或句號)
    processed_text = re.sub(r"[，、]{2,}", "，", processed_text)
    processed_text = re.sub(r"[。]{2,}", "。", processed_text)

    return processed_text.strip(), leak_count

def main():
    parser = argparse.ArgumentParser(description="針對生成的新聞進行特徵清洗，並與原資料集整合成最終訓練檔。")
    parser.add_argument("--generated", "-g", default="chinatime_regeneratetext_v1_fixed.json", help="含有 LLM 生成文本的 JSON")
    parser.add_argument("--original", "-o", default="dataset/chinatime_rewrite_cleaned.json", help="原始新聞網乾淨的 JSON")
    parser.add_argument("--output", "-out", default="final_dataset/chinatime_final.json", help="最終輸出的合併 JSON")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"📥 讀取生成的資料: {args.generated}")
    try:
        with open(args.generated, "r", encoding="utf-8") as f:
            generated_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到生成檔案: {args.generated}")
        return

    print(f"📥 讀取原始乾淨資料: {args.original}")
    try:
        with open(args.original, "r", encoding="utf-8") as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到原始檔案: {args.original}")
        return

    print("\n🛠️ 開始進行文本清洗...")
    valid_generated_data = []
    dropped_count = 0
    total_leak_count = 0

    for item in generated_data:
        original_tbm = item.get("tbm_class", "")
        text = item.get("content", "")
        
        # 0. 判斷是否要整筆捨棄
        if should_drop(text, original_tbm):
            dropped_count += 1
            continue

        # 1. 執行清洗
        cleaned_text, leaks_found = apply_masking(text, original_tbm)
        item["content"] = cleaned_text
        total_leak_count += leaks_found
        
        # 2. 修改 tbm_class 名稱 (避免與原始 dataset 混淆)
        if original_tbm in TBM_CLASS_MAPPING:
            item["tbm_class"] = TBM_CLASS_MAPPING[original_tbm]
        
        # 3. 確保 label=1 (生成/有特徵)
        item["label"] = 1
        valid_generated_data.append(item)

    print(f"\n🔗 準備合併資料集...")
    final_dataset = valid_generated_data + original_data
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"=====================================")
    print(f"🧹 清洗統計報告：")
    print(f" - 因嚴重幻覺/洩漏被整筆丟棄：{dropped_count} 筆")
    print(f" - 成功清除的 Prompt 洩漏字串：{total_leak_count} 個")
    print(f"=====================================")
    print(f"🎉 處理完成！")
    print(f" - 生成文本 (已清洗/改名): {len(valid_generated_data)} 筆")
    print(f" - 原始文本 (維持原樣): {len(original_data)} 筆")
    print(f" - 合併總數: {len(final_dataset)} 筆")
    print(f"💾 最終資料集已儲存至: {args.output}")

if __name__ == "__main__":
    main()