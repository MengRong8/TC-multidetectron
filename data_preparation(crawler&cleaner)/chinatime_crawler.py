# 網路爬蟲策略
# 1: Request + BeautifulSoup for 靜態頁面 such as ETtoday

import os
import json
import time
import requests
import random
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

print("月份區間:", month_ranges)
print("len:", len(month_ranges))
# 分類設定
categories = ["政治", "財經", "娛樂", "科技", "國際"]
print("分類:", categories)

# 基本設定
HEADERS = {"User-Agent": "Mozilla/5.0"}
IMG_DIR = "./chinatime_news"
OUTPUT_FILE = "chinatime_real.json"
LINK_FILE = "chinatime_links.json"
MAX_RESULTS = 30  # 每月每類最多筆數
os.makedirs(IMG_DIR, exist_ok=True)

# =====Method 1: Google Custom Search API=====
links = []
for category in categories:  # 新聞種類
    for after, before in month_ranges:  # 時間區間
        query = f"site:https://www.chinatimes.com {category} before:{before} after:{after}"
        print(f"搜尋分類: {category}, 時間: {after} ~ {before}")
        for start in range(1, MAX_RESULTS + 1, 10):
            url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}&start={start}"
            res = requests.get(url).json()
            new_links = [item["link"] for item in res.get("items", []) if "chinatimes.com" in item["link"]]

            links.extend((link, category) for link in new_links)
            print(f"第 {(start-1)//10 + 1} 批，共 {len(new_links)} 則")

            time.sleep(1)  # 避免請求過快

# 去重
unique_links = {}
for link, category in links:
    if link not in unique_links:
        unique_links[link] = category
links = list(unique_links.items())
print(f"Method: Google CSE -> 找到 {len(links)} 篇 Chinatime 新聞連結")

# 儲存links到檔案以便後續使用
with open(LINK_FILE, "w", encoding="utf-8") as f:
    json.dump(links, f, ensure_ascii=False, indent=4)
print(f"🔖 已將新聞連結存成 {LINK_FILE}")


# =====進入每篇新聞頁面抓取內容===== 

# 確保輸出檔案存在
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("[]")

dataset = []
for idx, (news_url, category) in enumerate(links):
    print("-------")
    print(f"id: {idx+1}, url: {news_url}, category: {category}")
    try:
        try:
            if "?" in news_url:
                safe_url = news_url + "&chdtv"
            else:
                safe_url = news_url + "?chdtv"
            res = requests.get(safe_url, headers=HEADERS, timeout=60)
            res.raise_for_status()
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout: {safe_url}")
            time.sleep(3)
            continue
        except requests.exceptions.RequestException as e:
            print(f"❌ 請求錯誤: {e}")
            continue

        new_soup = BeautifulSoup(res.text, "html.parser")

        # ========= 1. 標題 =========
        title_tag = new_soup.select_one("h1.article-title")
        title = title_tag.get_text(strip=True) if title_tag else "無標題"
        print("抓到文章標題:", title)

        # ========= 2. 第一個 column-left & 內文 =========
        wrapper = new_soup.select_one("[class*='column-wrapper']")
        if not wrapper:
            print("❌ 找不到 column-wrapper，跳過")
            continue

        first_col_left = wrapper.select_one("div.column-left")
        if not first_col_left:
            print("❌ 找不到 column-left，跳過")
            continue

        article_div = first_col_left.select_one("div.article-body[itemprop='articleBody']")
        if not article_div:
            print("❌ 找不到 article-body，跳過")
            continue

        # ========= 3. 主圖（優先 main-figure，其次內文第一張） =========
        filename = None
        img_tag = first_col_left.select_one("div.main-figure img.photo") \
                  or first_col_left.select_one("div.main-figure img") \
                  or article_div.select_one("img")

        img_url = None
        if img_tag:
            img_url = (
                img_tag.get("data-src")
                or img_tag.get("data-original")
                or img_tag.get("src")
            )

        if img_url and img_url.startswith("//"):
            img_url = "https:" + img_url

        if img_url:
            try:
                img_data = requests.get(img_url, headers=HEADERS, timeout=10).content
                filename = os.path.join(
                    IMG_DIR,
                    title[:10].replace("/", "_") + ".jpg"
                )
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"📸 圖片已下載: {filename}")
            except Exception as e:
                print("圖片下載失敗:", e)
        else:
            print("⚠️ 沒有找到主圖")

        # ========= 4. 清掉不要的區塊 =========
        for bad in article_div.select(
            "figure, #donate-form-container, .donation, .donation-prompt, "
            ".promote-word, .article-hash-tag, .ad, .google-news-promote, "
            "script, iframe"
        ):
            bad.decompose()

        # ========= 5. 擷取純文字內文 =========
        paragraphs = []
        for p in article_div.find_all("p"):
            text = p.get_text(strip=True)
            if not text:
                continue
            paragraphs.append(text)

        # 段落文字過濾
        skip_contains = [
            "中時新聞網", "中時電子報",
            "本報系資料照", "本報系資料照）",
            "贊助本文章",
            "下載中時", "追蹤中時新聞網",
        ]

        filtered_paragraphs = []
        for t in paragraphs:
            if any(bad in t for bad in skip_contains):
                print(f"⚠️ 跳過段落: {t[:20]}")
                continue
            filtered_paragraphs.append(t)

        content = "\n".join(filtered_paragraphs).strip()
        if not content:
            print("❌ 無內文，跳過")
            continue

        print("抓到文章內容:", content[:20], "...")

        # ========= 6. 存進 dataset =========
        dataset.append({
            "id": idx + 1,
            "category": category,
            "url": safe_url,
            "title": title,
            "content": content,
            "label": 0,
            "image_path": filename,
        })

        if (idx + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            print(f"💾 已暫存 {len(dataset)} 筆資料")

    except Exception as e:
        print("抓取文章失敗:", e)

    time.sleep(random.uniform(0.8, 1.5))

# ====== 存成 JSON 檔案 =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"✅ 資料已存成 {len(dataset)} 筆 中時新聞資料到 {OUTPUT_FILE}")
