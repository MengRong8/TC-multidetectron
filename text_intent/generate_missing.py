import json
import os
import random
import requests
import time
import re
from collections import Counter, defaultdict
from opencc import OpenCC

cc = OpenCC('s2t')

# =========================
# API 設定與 Prompt
# =========================
host = __import__("os").getenv("LLM_HOST", "localhost")
username = "kslab"
password = "kslab"
url = f"http://{host}/api/generate"
MODEL_NAME = "huihui_ai/gpt-oss-abliterated"

TBM_PROMPT_MAP = {

    "煽動性": """
【核心任務】：將新聞改寫為「強調風險與潛在後果，使相關行動者（政府、機構、企業、讀者或特定群體）覺得不得不做些什麼或改變想法」的煽動文章。
【寫作原則】：
1. 宣告式危機：直接將潛在的負面後果，描寫成已啟動且持續惡化的進程，或具備絕對必然性的結果。必須使用肯定的直述句，將維持現狀或不作為的態度，指控為引發全面崩壞的直接推手，藉此營造出沒有緩衝時間、沒有退路的壓迫感。
2. 語氣設定：保持冷酷、嚴肅、具壓迫感的權威專家語氣，避免使用過於情緒化或誇張的字眼。
3. 推演過程注意事項：請透過對未來後果的冷靜推演來自然營造壓力，而非直接命令讀者。
【注意事項】
1. 避免直接使用「立即行動」等公式化語句
2. 避免使用「若...」等任何帶有條件或假設意味的句型。
3. 避免重複使用相同句型開頭，請靈活變換表達方式來呈現因果推演。
""", 

    "主觀": """
【核心任務】：將新聞改寫為「帶有個人立場傾向、個人武斷推測與評論，但客觀包裝」的新聞文本。
【寫作原則】：
1. 隱性主觀視角：適當使用第一人稱代名詞。以隱含的第一人稱視角，將記者的個人解讀包裝成社會共識或必然邏輯。
2. 靈活的引導詞彙：請使用各種評斷性與預設立場的詞彙，自然地在敘事中滲入評論性推論，使事實與解讀交織。
3. 語氣設定：保持平靜、自信、專業評論風格，避免情緒化或攻擊性語言。可適度穿插反問句，來引導讀者思路，但要根據前後文自然替換詞彙。
""",
    "偏頗": """
【核心任務】：將新聞改寫為看似客觀平衡，實則重心傾向某一立場的帶風向文章。
【寫作原則】：
1. 立場傾斜：透過篇幅安排與語境優先順序，使特定立場自然占據敘事主導。
2. 弱化對手：在提及對立觀點時，必須緊接著使用轉折語氣，立即指出其邏輯漏洞或缺乏證據，使其看起來站不住腳或不可信。
3. 語氣設定：使用冷靜、分析式的口吻，避免情緒化謾罵，而是用邏輯或數據來包裝偏見，讓讀者覺得另一方顯然是錯的。
4. 詞彙約束：避免直接的人身攻擊，改用看似中立但帶有貶義的描述來包裝偏見。   
""", 
    "語意誇大與聳動": """
【核心任務】：將新聞文中的事件，極度放大其影響規模與最終後果。
【寫作原則】：
1. 災難化/奇蹟化推演：請將普通的事件，用非常嚴肅的口吻，一步一步往上推導，直到得出會引發誇張結果的結論。
2. 強制提升層級：無論原始新聞的層次多麼局部，都必須將其影響力強行拉高到誇張且聳動的層次。
3. 語氣設定：用一本正經、看似邏輯嚴密的推論過程，來掩飾極度誇張的預測。
4. 詞彙約束：請自然運用具備歷史與宏觀級別的強烈修飾詞，但嚴禁使用驚嘆號、恐慌驚呼或情緒化的字眼。
"""
}

NEW_PROMPT = """你是一位專業的假新聞撰寫專家。你擅長將新聞摘要依照指定的語意特徵進行改編與擴寫，合理地擴充內容以滿足指定的語意特徵要求。
請幫我將下列新聞摘要改編擴寫為一篇完整的繁體中文新聞報導，並在行文過程中自然地注入「{tbm_class}」的語意特徵。

【新聞摘要】（請將以下摘要當作新聞擴寫骨幹）
{summary}

【{tbm_class} 語意特徵說明】（將此說明作為新聞摘要的擴寫改編規則）
{specific_prompt}

硬性規則（非常重要，請嚴格遵守）：
1) 全文使用繁體中文。
2) 禁止機械式重複句型或詞彙（如連續出現相同的呼籲或重複使用相同句型開頭）。請靈活變換表達方式來呈現語意特徵，確保文章流暢且自然。
3) 將摘要每句進行改寫擴大，必要時注入語意特徵，而非重複摘要內容。
4) 嚴格禁止混入其他不在指定語意特徵說明內的誘導手法，確保此文章成為該單一特徵的純淨文本。

（不要標題、不要條列、不要任何額外說明、不要 html 格式）請直接輸出擴寫後的新聞內文：
"""

CATEGORY_LIST = {
    "煽動性": ["政治", "財經", "國際", "科技"],
    "主觀": ["政治", "國際", "娛樂"],
    "偏頗": ["政治", "財經", "國際"],
    "語意誇大與聳動": ["政治", "財經", "國際", "科技", "娛樂"]
}

def english_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", text)
    return len(letters) / max(len(text), 1)

def simplified_ratio(text: str) -> float:
    converted = cc.convert(text)
    length = len(text)
    if length == 0: return 0.0
    return sum(1 for a, b in zip(text, converted) if a != b) / length

# =========================
# 同步自 mask_and_merge.py 的丟棄規則 (防止生成後被丟棄)
# =========================
DROP_RULES = {
    "煽動性": [
        r"危機已啟動", 
        r"全面崩壞的直接推手",
        r"權威專家語氣",
        r"冷靜推演",
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
    ]
}

def is_valid(text: str, original_summary: str, tbm_class: str) -> bool:
    if not text: return False
    text_lower = text.lower()
    
    # 1. 基礎物理過濾
    if "powershell" in text_lower: return False
    if "local businessman" in text_lower: return False
    if "<div" in text_lower: return False
    if english_ratio(text) > 0.3: return False
    if simplified_ratio(text) >= 0.05: return False
    if len(text) < len(original_summary): return False
    
    # 2. 同步 mask_and_merge 的 DROP_RULES (Prompt Leak 過濾)
    for pattern in DROP_RULES.get("通用", []):
        if re.search(pattern, text):
            return False
    for pattern in DROP_RULES.get(tbm_class, []):
        if re.search(pattern, text):
            return False
            
    return True

def call_llm(prompt: str, temperature, repeat_penalty) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False, 
        "options": {
            "temperature": temperature,
            "repeat_penalty": repeat_penalty
        }
    }
    response = requests.post(url, json=payload, auth=(username, password), timeout=900)
    response.raise_for_status()
    return response.json()["response"]

def process_media(media_name):
    print(f"\n=====================================")
    print(f"🚀 開始處理媒體: {media_name}")
    print(f"=====================================")
    
    final_path = f"final_dataset/{media_name}_final.json"
    fixed_path = f"{media_name}_regeneratetext_v1_fixed.json"
    source_path = f"/home/howard/web_crawling/{media_name}/stage1_{media_name}_results.json"
    
    # 1. 讀取 final 計算差額
    try:
        with open(final_path, 'r', encoding='utf-8') as f:
            final_data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 找不到 {final_path}，跳過。")
        return

    used_titles = set(d.get("title", "") for d in final_data)
    
    counts = Counter(d.get('tbm_class', '未知') for d in final_data)
    orig_counts = [count for label, count in counts.items() if not label.startswith('AIGC_')]
    target_count = int(sum(orig_counts) / len(orig_counts)) if orig_counts else 0
    
    print(f"🎯 目標各標籤數量 (以原始資料平均值為準): {target_count}")
    
    shortfalls = {}
    for tbm_class in TBM_PROMPT_MAP.keys():
        current_count = counts.get(f"AIGC_{tbm_class}", 0)
        gap = target_count - current_count
        if gap > 0:
            shortfalls[tbm_class] = gap
        print(f"  - {tbm_class}: 現有 {current_count}，需補 {gap if gap > 0 else 0} 筆")
        
    if not shortfalls:
        print("✅ 數量皆已達標，無需生成。")
        return

    # 2. 準備生成任務
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ 找不到 {source_path}，跳過。")
        return
        
    news_pool = defaultdict(list)
    for item in source_data:
        if not item.get("summary"): continue
        if item.get("is_ad", False): continue
        if item.get("title", "") in used_titles: continue
        news_pool[item["category"]].append(item)

    for cat in news_pool:
        random.shuffle(news_pool[cat])
        
    allocated_tasks = defaultdict(list)
    for tbm_class, gap in shortfalls.items():
        valid_categories = CATEGORY_LIST.get(tbm_class, [])
        for _ in range(gap + int(gap * 0.2)): # 多分配 20% 以防生成失敗
            available_cats = [c for c in valid_categories if len(news_pool[c]) > 0]
            if available_cats:
                chosen_cat = random.choice(available_cats)
                allocated_tasks[tbm_class].append(news_pool[chosen_cat].pop())

    # 3. 讀取目前的 fixed 檔案 (用於附加)
    try:
        with open(fixed_path, 'r', encoding='utf-8') as f:
            fixed_data = json.load(f)
    except FileNotFoundError:
        fixed_data = []

    # 4. 開始生成
    TEMPERATURE_LIST = {"煽動性": 0.75, "主觀": 0.75, "偏頗": 0.75, "語意誇大與聳動": 0.75}
    REPEAT_PENALTY_LIST = {"煽動性": 1.2, "主觀": 1.2, "偏頗": 1.2, "語意誇大與聳動": 1.2}
    
    total_added = 0
    
    for tbm_class, tasks in allocated_tasks.items():
        print(f"\n⏳ 開始生成 {tbm_class} (預計需要 {shortfalls[tbm_class]} 筆)")
        success_count = 0
        target = shortfalls[tbm_class]
        
        specific_prompt = TBM_PROMPT_MAP[tbm_class]
        temp = TEMPERATURE_LIST.get(tbm_class, 0.75)
        penalty = REPEAT_PENALTY_LIST.get(tbm_class, 1.2)
        
        for idx, item in enumerate(tasks):
            if success_count >= target:
                break # 已達標
                
            original_summary = item["summary"]
            prompt = NEW_PROMPT.format(tbm_class=tbm_class, specific_prompt=specific_prompt, summary=original_summary)
            
            print(f"[{success_count+1}/{target}] 正在生成 ID: {item.get('id')} ...")
            max_retries = 3
            passed = False
            for attempt in range(max_retries):
                try:
                    adjusted_temp = temp + (attempt * 0.1)
                    rewritten = call_llm(prompt, adjusted_temp, penalty)
                    rewritten = rewritten.strip().strip('"').strip('「').strip('」')
                    
                    if is_valid(rewritten, original_summary, tbm_class):
                        new_item = {
                            "id": item.get("id", ""),
                            "category": item.get("category", ""),
                            "title": item.get("title", ""),
                            "content": rewritten,
                            "label": 1,
                            "tbm_class": tbm_class,
                            "original_summary": original_summary
                        }
                        fixed_data.append(new_item)
                        success_count += 1
                        total_added += 1
                        passed = True
                        
                        # 每成功一次就覆寫 _fixed.json
                        with open(fixed_path, 'w', encoding='utf-8') as f:
                            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
                        print(f"   ➔ 成功！")
                        break
                    else:
                        print(f"   ➔ 第 {attempt+1} 次失敗 (內容不合規)")
                except Exception as e:
                    print(f"   ➔ 第 {attempt+1} 次 API 錯誤: {e}")
                    time.sleep(2)
            
            if not passed:
                print(f"   ❌ 放棄此筆資料。")

    print(f"\n🎉 {media_name} 補充生成完成！共補回 {total_added} 筆。")
    print(f"資料已存入 {fixed_path}。請記得再次執行 mask_and_merge.py 更新 final_dataset！")

import argparse

def main():
    parser = argparse.ArgumentParser(description="補齊缺失的 AIGC 標籤資料數")
    parser.add_argument("--media", "-m", type=str, choices=["nownews", "ltn", "chinatime", "ettoday"], help="指定要處理的媒體名稱 (如: nownews)")
    args = parser.parse_args()

    if args.media:
        # 如果有指定媒體，就只跑那個媒體
        process_media(args.media)
    else:
        # 如果沒指定，預設跑這三個 (跳過 ettoday 因為通常已達標)
        medias = ["nownews", "ltn", "chinatime"]
        for m in medias:
            process_media(m)

if __name__ == "__main__":
    main()
