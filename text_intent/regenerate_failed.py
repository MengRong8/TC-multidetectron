import json
import re
import requests
import time
from opencc import OpenCC

cc = OpenCC('s2t')

# =========================
# API 設定與 Prompt (繼承自 generate.py)
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
3. 推演過程注意事項：請透過對未來後果的冷靜推演來自然營造壓力，而非直接命令行動者。
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
4) 若摘要同時提供中文與英文名稱，請使用「中文名（English Name）」格式；若摘要未提供英文名稱，不得自行補充。
5) 嚴格禁止混入其他不在指定語意特徵說明內的誘導手法，確保此文章成為該單一特徵的純淨文本。

（不要標題、不要條列、不要任何額外說明、不要 html 格式）請直接輸出擴寫後的新聞內文：
"""

def english_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", text)
    return len(letters) / max(len(text), 1)

def simplified_ratio(text: str) -> float:
    converted = cc.convert(text)
    length = len(text)
    if length == 0: return 0.0
    return sum(1 for a, b in zip(text, converted) if a != b) / length

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

def is_valid(text: str, original_summary: str) -> bool:
    if english_ratio(text) > 0.3: return False
    if simplified_ratio(text) >= 0.05: return False
    if len(text) < len(original_summary): return False
    return True

def main():
    file_path = "chinatime_regeneratetext_v1.json"
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    success = []
    fail = []
    
    # 1. 篩選出成功的與失敗的
    for item in data:
        text = item.get("content", "")
        original_summary = item.get("original_summary", "")
        if is_valid(text, original_summary):
            success.append(item)
        else:
            fail.append(item)
            
    print(f"✅ 原有合格資料：{len(success)} 筆")
    print(f"❌ 需要重新生成的資料：{len(fail)} 筆\n")
    
    # 2. 針對失敗的重新生成
    TEMPERATURE_LIST = {"煽動性": 0.75, "主觀": 0.75, "偏頗": 0.75, "語意誇大與聳動": 0.75}
    REPEAT_PENALTY_LIST = {"煽動性": 1.2, "主觀": 1.2, "偏頗": 1.2, "語意誇大與聳動": 1.2}
    
    max_retries = 3
    final_success = list(success)  # 複製一份成功的清單
    still_failed = []
    
    for idx, item in enumerate(fail, 1):
        tbm_class = item.get("tbm_class", "")
        original_summary = item.get("original_summary", "")
        
        print(f"[{idx}/{len(fail)}] 重新生成 ID: {item.get('id')}, 類別: {tbm_class}...")
        
        passed = False
        for attempt in range(max_retries):
            specific_prompt = TBM_PROMPT_MAP.get(tbm_class, "")
            prompt = NEW_PROMPT.format(tbm_class=tbm_class, specific_prompt=specific_prompt, summary=original_summary)
            temp = TEMPERATURE_LIST.get(tbm_class, 0.75)
            penalty = REPEAT_PENALTY_LIST.get(tbm_class, 1.2)
            
            try:
                # 稍微提高溫度，讓模型有機會跳出先前的短句死胡同
                adjusted_temp = temp + (attempt * 0.1) 
                rewritten = call_llm(prompt, adjusted_temp, penalty)
                rewritten = rewritten.strip().strip('"').strip('「').strip('」')
                
                if is_valid(rewritten, original_summary):
                    item["content"] = rewritten
                    final_success.append(item)
                    print(f"   ➔ 第 {attempt+1} 次重試成功！長度: {len(rewritten)}")
                    passed = True
                    break
                else:
                    print(f"   ➔ 第 {attempt+1} 次重試失敗 (長度 {len(rewritten)} vs 摘要 {len(original_summary)}, 簡中: {simplified_ratio(rewritten):.2f}, 英文: {english_ratio(rewritten):.2f})")
            except Exception as e:
                print(f"   ➔ API 呼叫錯誤: {e}")
                time.sleep(2)
        
        if not passed:
            print(f"   ❌ 放棄生成 ID: {item.get('id')}，已達最大重試次數。")
            still_failed.append(item)
            
    # 3. 寫回原本的檔案（只存最終成功的，或是你也可以決定保留還是丟棄最終失敗的）
    # 這裡選擇只把成功的存回去，確保 json 內全部都是乾淨的資料
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_success, f, ensure_ascii=False, indent=2)
        
    print(f"\n🎉 重新生成作業完成！")
    print(f"總筆數從 {len(data)} 變更為 {len(final_success)} 筆。已覆寫 {file_path}")
    if still_failed:
        print(f"⚠️ 仍有 {len(still_failed)} 筆無法成功生成，已將其排除。")

if __name__ == "__main__":
    main()
