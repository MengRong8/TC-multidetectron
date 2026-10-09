import re
from opencc import OpenCC
import json
from collections import Counter

cc = OpenCC('s2t')  # 只用來「判斷」，不做實際轉換

def english_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", text)
    return len(letters) / max(len(text), 1)

# def is_traditional_chinese(text: str) -> bool:
#     converted = cc.convert(text)
#     diff = sum(1 for a, b in zip(text, converted) if a != b)
#     return diff / max(len(text), 1) < 0.01

def simplified_ratio(text: str) -> float:
    # 轉成繁體
    converted = cc.convert(text)
    length = len(text)
    
    if length == 0:
        return 0.0

    # 計算有多少字跟原本不一樣（不一樣代表原本可能是簡體）
    diff_count = sum(1 for a, b in zip(text, converted) if a != b)

    # 分母改用「文章總長度」
    return diff_count / length


if __name__ == "__main__":
    # === 路徑設定 ===
    fake_path = "yahoo/yahoo_fake.json"
    fail_path = "yahoo/yahoo_failed.json"

    # === 讀取原始資料 ===
    with open(fake_path, encoding="utf-8") as f:
        data = json.load(f)

    success = []
    fail = []

    stats = Counter()

    # === 分類 ===
    for item in data:
        text = item["content"]

        if english_ratio(text) > 0.3:
            stats["english"] += 1
            fail.append(item)

        elif simplified_ratio(text) >= 0.05:
            stats["simplified"] += 1
            fail.append(item)

        else:
            success.append(item)

    # === 統計 ===
    stats["success"] = len(success)
    stats["fail"] = len(fail)
    stats["total"] = len(data)

    print("=== 統計結果 ===")
    for k, v in stats.items():
        print(f"{k}: {v}")


    # # === 覆寫 fake.json（只保留 success） ===
    # with open(fake_path, "w", encoding="utf-8") as f:
    #     json.dump(success, f, ensure_ascii=False, indent=2)

    # # === 寫入 fail.json（供重新生成） ===
    # with open(fail_path, "w", encoding="utf-8") as f:
    #     json.dump(fail, f, ensure_ascii=False, indent=2)

    print(f"\n已更新：")
    print(f"- fake.json：{len(success)} 筆（保留合格資料）")
    print(f"- fail.json：{len(fail)} 筆（待重新生成）")
    print(fail[:])

