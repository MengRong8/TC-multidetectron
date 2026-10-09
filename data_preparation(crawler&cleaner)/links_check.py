import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
import cloudscraper

LINK_FILE = "chinatime_links.json"

import cloudscraper
from bs4 import BeautifulSoup

scraper = cloudscraper.create_scraper()

def extract_source(url):
    resp = scraper.get(url)
    resp.encoding = "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. 正常桌機版 (2020–2024)
    node = soup.select_one("header.article-header div.source")
    if node:
        return node.text.strip()

    # 2. 手機版
    node = soup.select_one("div.article-hd div.source")
    if node:
        return node.text.strip()

    # 3. meta-info 舊版
    node = soup.select_one(".meta-info .source")
    if node:
        return node.text.strip()

    # 4. 舊報紙版 (2020–2021)
    node = soup.select_one("span.source-data")
    if node:
        return node.text.strip()

    # 5. 極舊版作者欄
    author = soup.select_one("div.author, span.author")
    if author:
        text = author.text
        if "工商" in text:
            return "工商時報"
        if "旺報" in text:
            return "旺報"
        if "中時" in text or "本報" in text:
            return "中國時報"

    # 6. regex：從「XXX／台北報導」推論
    import re
    m = re.search(r"／(.+?)報導", soup.text)
    if m:
        return "中國時報"  # 預設報導來源

    return None



def count_industrial_times():
    scraper = cloudscraper.create_scraper()

    # 讀取你的 links 文件
    with open(LINK_FILE, "r", encoding="utf-8") as f:
        links = json.load(f)

    total = len(links)
    industrial_count = 0

    for idx, (url, category) in enumerate(links):
        print(f"[{idx+1}/{total}] 抓取來源：{url}")

        source = extract_source(url)
        print(" → 來源：", source)

        if source == "工商時報":
            industrial_count += 1

    print("\n=============================")
    print(f"總共：{total} 篇中時新聞")
    print(f"其中《工商時報》來源：{industrial_count} 篇")
    print("=============================\n")


if __name__ == "__main__":
    count_industrial_times()

# print(f"\n總共有 {result} 篇新聞來自《工商時報》")


# def preview_contents(input_path, output_path="content_preview_china.txt", n=30):
#     with open(input_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     lines = []
#     for item in data:
#         cid = item.get("id")
#         title = item.get("title", "")
#         content = item.get("content", "")

#         if not content:
#             preview = "[EMPTY CONTENT]"
#         else:
#             head = content[:n].replace("\n", "\\n")
#             tail = content[-n:].replace("\n", "\\n")
#             preview = f"{head} ... {tail}"

#         block = (
#             f"ID: {cid}\n"
#             f"TITLE: {title}\n"
#             f"PREVIEW: {preview}\n"
#         )
#         lines.append(block)

#     with open(output_path, "w", encoding="utf-8") as f:
#         f.write("".join(lines))

#     print(f"Saved preview to {output_path}")

# if __name__ == "__main__":
#     preview_contents("chinatime_cleaned.json")

# with open("ltn_real.json", "r", encoding="utf-8") as f: 
#     data = json.load(f)

# # 記錄
# seen_url = {}            # url -> (id)
# seen_content = {}        # content_hash -> (id, url)
# dup_same_url = []        # (id1, id2, url)
# dup_same_content = []    # (id1, id2, url1, url2)

# for item in data:
#     _id = item["id"]
#     url = item["url"]
#     content = item["content"]

#     # ============
#     # 1️⃣ 同 URL duplicate
#     # ============
#     if url in seen_url:
#         dup_same_url.append((seen_url[url], _id, url))
#     else:
#         seen_url[url] = _id

#     # ============
#     # 2️⃣ 同內容 duplicate（不同 URL）
#     # ============
#     content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

#     if content_hash in seen_content:
#         prev_id, prev_url = seen_content[content_hash]

#         # 不管 URL 是否相同，你說都算，所以直接加入
#         dup_same_content.append((prev_id, _id, prev_url, url))
#     else:
#         seen_content[content_hash] = (item["id"], item["url"])


# # ============
# # 統計
# # ============
# print("=== Duplicate by URL (same URL) ===")
# print("Count:", len(dup_same_url), "\n")

# print("=== Duplicate by Content (same content) ===")
# print("Count:", len(dup_same_content)-len(dup_same_url), "\n")

# total = len(dup_same_url) + len(dup_same_content)
# print(total)
