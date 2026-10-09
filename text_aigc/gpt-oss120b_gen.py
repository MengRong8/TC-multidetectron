import os
import requests
import json
from clean_AIGCnews import simplified_ratio, english_ratio
from collections import Counter

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:11434/api/generate")
MODEL_NAME = "gpt-oss:20b"
# input_path = "chinatime/stage1_chinatime_results.json"
# output_path = "chinatime/chinatime_gptoss_fake.json"
# failed_path = "chinatime/chinatime_failed.json"

def rewrite_prompt(summary, target_len):
    origin_prompt = f"""
    你是一位台灣的新聞稿編輯。請根據【摘要】撰寫一篇完整的繁體中文新聞稿。

    目標：
    - 全文長度約 {target_len} 字（±10%）
    - 只輸出純新聞內文（不要標題、不要條列、不要任何額外說明）
    - 全文使用繁體中文

    硬性規則（非常重要，請嚴格遵守）：
    1) 不得照抄摘要句子，需改寫語序與表達，但不得改變事實。
    2) 不得新增摘要中沒有出現的「人名、公司/機構名、地名、日期、數字、法規名稱、產品名」。
    3) 不得捏造引述或說法（例如「某某表示」若摘要沒寫就不能加）。
    4) 允許補充「通用背景敘述」讓文章更順，但背景不得包含任何新的專有名詞與數字。
    5) 若摘要中出現中文譯名且同時有英文原名，請以「中文名（English Name）」格式呈現；如果摘要沒有英文原名，請不要自行補英文名。

    【摘要】
    {summary}
    """
    prompt_B = f"""
    你是一位台灣的即時新聞記者，需根據摘要快速撰寫一則新聞內文。
    要求：
    - 可使用常見新聞慣用句
    - 保持事實正確，不可捏造人名與數字
    - 行文自然，不必過度正式
    - 全文繁體中文
    
    輸出：
    約 {target_len} 字的新聞內文
    只輸出純新聞內文（不要標題、不要條列、不要任何額外說明）
    
    摘要：
    {summary}
    """
    
    return prompt_B

def safe_mkdir_for_file(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def load_json_if_exists(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def regenerate_fail_only(
    retrieved_path, 
    success_path,
    fail_path,
    output_success_path
):
    # 讀取已成功資料
    with open(success_path, "r", encoding="utf-8") as f:
        success_data = json.load(f)

    with open(retrieved_path, "r", encoding="utf-8") as f: 
        retrieved_data = json.load(f)
    
    success_by_id = {item["id"]: item for item in success_data}
    retrieved_by_id = {item["id"]: item for item in retrieved_data}

    # 讀取 fail
    with open(fail_path, "r", encoding="utf-8") as f:
        fail_data = json.load(f)

    fail_next = []
    generated_count = 0  # ✅ 成功生成計數器

    for idx, item in enumerate(fail_data):
        print(f"重生 fail 第 {idx+1} 筆 (id={item['id']})")

        summary = retrieved_by_id[item["id"]]["summary"].strip()
        # summary = item["summary"].strip()
        target_len = len(summary) * 4
        rewriteprompt = rewrite_prompt(summary=summary, target_len=target_len)

        payload = {
            "model": MODEL_NAME,
            "prompt": rewriteprompt,
            "stream": False
        }

        try:
            response = requests.post(LLM_API_URL, json=payload, timeout=900)
            response.raise_for_status()
            result = response.json()
            rewritten = result.get("response", "").strip()

            # 檢查生成內容是否符合要求
            if english_ratio(rewritten) > 0.3:
                raise ValueError("GENERATED_CONTENT_TOO_ENGLISH")
            if simplified_ratio(rewritten) >= 0.05:
                raise ValueError("GENERATED_CONTENT_TOO_SIMPLIFIED")
            
            success_by_id[item["id"]] = {
                "id": item["id"],
                "category": item["category"],
                "title": item["title"],
                "content": rewritten,
                "label": 1,
                "image_url": ""
            }
            generated_count += 1
            # ✅ 每 30 筆就寫回 fake.json
            if generated_count % 30 == 0:
                success_sorted = sorted(
                    success_by_id.values(),
                    key=lambda x: x["id"]
                )
                with open(output_success_path, "w", encoding="utf-8") as f:
                    json.dump(success_sorted, f, ensure_ascii=False, indent=2)

                print(f"  💾 已累積 {generated_count} 筆，暫存寫入 fake.json")

        except Exception as e:
            fail_next.append({
                "id": item["id"],
                "category": item["category"],
                "title": item["title"],
                # "summary": item["summary"],
                "error": str(e)
            })

    # 只輸出成功的
    success_sorted = sorted(success_by_id.values(), key=lambda x: x["id"])
    with open(output_success_path, "w", encoding="utf-8") as f:
        json.dump(success_sorted, f, ensure_ascii=False, indent=2)

    # 仍失敗的，留給下一輪
    with open(fail_path, "w", encoding="utf-8") as f:
        json.dump(fail_next, f, ensure_ascii=False, indent=2)

    print("✅ 成功與失敗已分流完成")


def generate_with_LLM(input_path, output_path, failed_path, sample_size):
    safe_mkdir_for_file(output_path)
    safe_mkdir_for_file(failed_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 允許斷點續跑：如果 output 已存在就先載入
    output = load_json_if_exists(output_path, default=[])
    failed = load_json_if_exists(failed_path, default=[])

    # 用 id 做去重，避免重複寫入
    done_ids = set(str(x.get("id")) for x in output if x.get("id") is not None)

    for idx, item in enumerate(data[:sample_size]):
        item_id = item.get("id")
        item_id_str = str(item_id) if item_id is not None else None

        print(f"正在處理第 {idx+1} 筆… id={item_id}")

        # ✅ is_ad==1 跳過
        if str(item.get("is_ad", 0)) == "1":
            print("  ↪ is_ad=1，跳過")
            continue

        # ✅ 已做過就跳過（避免重跑浪費）
        if item_id_str and item_id_str in done_ids:
            print("  ↪ 已完成過，跳過")
            continue

        content = item.get("summary", "").strip()
        title = item.get("title", "").strip()
        category = item.get("category", "")

        if not content:
            # 把 summary 空的也記錄起來，方便你回頭查
            failed.append({
                "id": item_id,
                "category": category,
                "title": title,
                "summary": content,
                "error": "EMPTY_SUMMARY"
            })
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            print("  ❌ summary 為空，已記錄到 failed")
            continue

        orig_len = len(content) * 4  # ✅ 你指定固定 *4
        prompt = rewrite_prompt(summary=content, target_len=orig_len)

        payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}

        try:
            response = requests.post(LLM_API_URL, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()

            rewritten = result.get("response", "")
            if not isinstance(rewritten, str):
                rewritten = str(rewritten)
            rewritten = rewritten.strip()

            # 有些模型可能回空字串：也算失敗，方便重跑
            if not rewritten:
                raise ValueError("EMPTY_GENERATION_RESPONSE")

            output.append({
                "id": item_id,
                "category": category,
                "title": title,
                "content": rewritten,
                "label": 1,
                "image_url": ""
            })
            if item_id_str:
                done_ids.add(item_id_str)

            # ✅ 暫存 output
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print("  ✔ 生成成功，已暫存")

        except Exception as e:
            failed.append({
                "id": item_id,
                "category": category,
                "title": title,
                "summary": content,
                "error": repr(e)
            })
            with open(failed_path, "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
            print(f"  ❌ 失敗：{e}（已記錄到 failed）")

    # 最後再寫一次（保險）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(failed_path, "w", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # with open(input_path, "r", encoding="utf-8") as f:
    #     data = json.load(f)
    # real_size = len(data)
    # print(f"資料總筆數：{real_size}")
    # generate_with_LLM(input_path, output_path, failed_path, sample_size=real_size)
    
    # generate_with_LLM(
    #     input_path="chinatime/stage1_chinatime_results.json",
    #     output_path="chinatime_promptB.json",
    #     failed_path="chinatime/chinatime_failed_B.json",
    #     sample_size=150
    # )

    regenerate_fail_only(
        retrieved_path="ettoday/stage1_ettoday_results.json", 
        success_path="ettoday/ettoday_fake.json",
        fail_path="ettoday/ettoday_failed.json",
        output_success_path="ettoday/ettoday_fake.json"
    )
    