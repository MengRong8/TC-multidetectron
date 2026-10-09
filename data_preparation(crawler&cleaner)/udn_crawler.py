import os
import re
import json
import time
import random
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Google Custom Search API 參數
API_KEY = ""
CX = ""

# 抓取時間範圍設定
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 1, 1)

month_ranges = []
current_date = start_date
while current_date < end_date:
    next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
    before = next_month.strftime("%Y-%m-%d")
    after = current_date.strftime("%Y-%m-%d")
    month_ranges.append((after, before))
    current_date = next_month

# ======== 你要改這裡 ========
categories = ["政治", "財經", "娛樂", "科技", "國際"]

HEADERS = {"User-Agent": "Mozilla/5.0"}
IMG_DIR = "./udn_news"
OUTPUT_FILE = "udn_real.json"
LINK_FILE = "udn_links.json"
MAX_RESULTS = 10  # 要取多少篇


# 建立資料夾
os.makedirs(IMG_DIR, exist_ok=True)

# ===== Method 1: Google Custom Search API =====
links = []

for category in categories:
    for after, before in month_ranges:
        query = f"site:https://udn.com/news {category} before:{before} after:{after}"
        print(f"[CSE] 搜尋 {category}: {after}~{before}")

        # 只用 start=1 → 最少 API、最多 10 筆，再切 MAX_RESULTS
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}&start=1"
        res = requests.get(url).json()

        items = res.get("items", [])[:MAX_RESULTS]
        for item in items:
            link = item["link"]
            if "udn.com" in link:
                links.append((link, category))

        time.sleep(1)

# 去重
unique = {}
for link, cat in links:
    if link not in unique:
        unique[link] = cat
links = list(unique.items())

print(f"共取得 {len(links)} 篇 UDN 連結")

with open(LINK_FILE, "w", encoding="utf-8") as f:
    json.dump(links, f, ensure_ascii=False, indent=4)


# ============== 工具函式 ==============
def clean_text(text: str) -> str:
    text = re.sub(r"（.*?圖／.*?）", "", text)
    text = re.sub(r"\(.*?圖／.*?\)", "", text)
    text = re.sub(r"〔記者.*?／.*?〕", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


# ============== 進入每篇新聞抓資料 ==============
dataset = []

for idx, (url, category) in enumerate(links):
    print("-------")
    print(f"id: {idx+1}, url: {url}")

    try:
        res = requests.get(url, headers=HEADERS, timeout=8)
        res.raise_for_status()
    except:
        print("❌ 無法開啟頁面")
        continue

    soup = BeautifulSoup(res.text, "html.parser")

    # ========= 抓標題 ==========
    title_tag = soup.select_one("h1")
    title = title_tag.get_text(strip=True) if title_tag else "無標題"
    print("標題:", title)

    # ========= 抓主體內容 ==========
    # UDN 正文：div.article-content__editor 或 div#story_body_content
    content_div = (
        soup.select_one("div.article-content__editor")
        or soup.select_one("div#story_body_content")
        or soup.select_one("section.article-content-wrapper")
    )

    if not content_div:
        print("❌ 找不到正文，跳過")
        continue

    # 移除廣告、腳本
    for bad in content_div.select("script, iframe, ins, .adv, .advertisement"):
        bad.decompose()

    # 取得所有 p
    paragraphs = content_div.find_all("p")

    filtered = []
    bad_words = ["延伸閱讀", "更多閱讀", "▲", "資料照"]

    for p in paragraphs:
        t = p.get_text(strip=True)
        if any(b in t for b in bad_words):
            continue
        t = clean_text(t)
        if t:
            filtered.append(t)

    content = "\n".join(filtered)
    if not content:
        print("❌ 無內文")
        continue

    print("內文前 20 字:", content[:20])

    # ========= 抓主圖 ==========
    img = (
        soup.select_one("figure img")
        or soup.select_one(".hero-image img")
        or soup.select_one("meta[property='og:image']")
    )

    img_url = None

    if img:
        if img.has_attr("src"):
            img_url = img["src"]
        elif img.has_attr("content"):  # og:image
            img_url = img["content"]

    if img_url and img_url.startswith("//"):
        img_url = "https:" + img_url

    # 下載圖片
    img_path = None
    if img_url:
        try:
            img_data = requests.get(img_url).content
            safe_name = re.sub(r"[\\/:*?\"<>|]", "_", title[:12])
            img_path = os.path.join(IMG_DIR, safe_name + ".jpg")

            with open(img_path, "wb") as f:
                f.write(img_data)

            print("圖片下載成功:", img_path)
        except:
            print("❌ 圖片下載失敗")

    # ========= 放入 dataset ==========
    dataset.append({
        "id": idx + 1,
        "category": category,
        "url": url,
        "title": title,
        "content": content,
        "label": 0,
        "image_path": img_path
    })

    if (idx + 1) % 50 == 0:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"💾 已暫存 {len(dataset)} 筆")

    time.sleep(random.uniform(0.5, 1.2))


# ============== 結尾：寫入 JSON ==============
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"✅ 爬蟲完成，共 {len(dataset)} 筆 UDN 新聞已存入 {OUTPUT_FILE}")
