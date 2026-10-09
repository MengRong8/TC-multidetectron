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
IMG_DIR = "./ettoday_news"
OUTPUT_FILE = "ettoday_real.json"
LINK_FILE = "ettoday_links.json"
MAX_RESULTS = 30  # 每月每類最多筆數
os.makedirs(IMG_DIR, exist_ok=True)

# =====Method 1: Google Custom Search API=====
links = []
for category in categories:  # 新聞種類
    for after, before in month_ranges:  # 時間區間
        query = f"site:https://www.ettoday.net {category} before:{before} after:{after}"
        print(f"搜尋分類: {category}, 時間: {after} ~ {before}")
        for start in range(1, MAX_RESULTS + 1, 10):
            url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}&start={start}"
            res = requests.get(url).json()
            new_links = [item["link"] for item in res.get("items", []) if "ettoday.net/news" in item["link"]]

            links.extend((link, category) for link in new_links)
            print(f"第 {(start-1)//10 + 1} 批，共 {len(new_links)} 則")

            time.sleep(1)  # 避免請求過快

# 去重
unique_links = {}
for link, category in links:
    if link not in unique_links:
        unique_links[link] = category
links = list(unique_links.items())
print(f"Method: Google CSE -> 找到 {len(links)} 篇 ETtoday 新聞連結")

# 儲存links到檔案以便後續使用
with open(LINK_FILE, "w", encoding="utf-8") as f:
    json.dump(links, f, ensure_ascii=False, indent=4)
print(f"🔖 已將新聞連結存成 {LINK_FILE}")

# =====進入每篇新聞頁面抓取內容===== 

# 確保輸出檔案存在
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("[]")
# 進度保存
# def append_to_json(file_path, new_data):
#     """安全地 append 一筆資料到 JSON 陣列"""
#     with open(file_path, "r+", encoding="utf-8") as f:
#         data = json.load(f)
#         data.append(new_data)
#         f.seek(0)
#         json.dump(data, f, ensure_ascii=False, indent=4)
#         f.truncate()

dataset = []
for idx, (news_url, category) in enumerate(links):
    print("-------")
    print(f"id: {idx+1}, url: {news_url}, category: {category}")
    try:
        try:
            res = requests.get(news_url, headers=HEADERS, timeout=10)
            res.raise_for_status()  # 非200會拋錯
        except requests.exceptions.Timeout:
            print(f"⏰ Timeout: {news_url}")
            time.sleep(3)
            continue
        except requests.exceptions.RequestException as e:
            print(f"❌ 請求錯誤: {e}")
            continue

        new_soup = BeautifulSoup(res.text, "html.parser")

        # 抓標題
        title_tag = new_soup.select_one("h1.title")
        title = title_tag.get_text(strip=True) if title_tag else "無標題"
        print("抓到文章標題:", title)

        # 抓內文容器
        article_div = new_soup.select_one("div.story")
        if not article_div:
            article_div = new_soup.select_one("article.story")

        # 移除廣告
        for ad in article_div.select(".ad_in_news, .ad_readmore, .lazyload-ad, script, ins"):
            ad.decompose()

        # 抓段落
        paragraphs = article_div.find_all("p") if article_div else []
        filtered_paragraphs = []
        skip_keywords = ["記者", "報導", "中心／", "整理／", "新聞組／"] # 記者...報導 需len考慮
        skip_k = ["▲"] # 圖解
        skip_contains = ["ETtoday新聞雲", "ETtoday新聞雲報導", "更多新聞", "延伸閱讀", "（圖／", "(圖／", "圖／資料照片", "照片／", "／台北報導", "／綜合報導", "更多鏡週刊報導"]
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            if any(text.startswith(k) for k in skip_keywords) and len(text) < 15:
                print(f"⚠️ 跳過來源段落: {text}")
                continue
            
            if any(text.startswith(k) for k in skip_k):
                print(f"⚠️ 跳過段落: {text[:20]}")
                continue
            # 跳過包含 (ETtoday) 、(圖) 相關字眼的段落
            if any(k in text for k in skip_contains):
                print(f"⚠️ 跳過段落: {text[:20]}")
                continue
            filtered_paragraphs.append(p)

        content = "\n".join([p.get_text(strip=True) for p in filtered_paragraphs])
        content = "\n".join([line for line in content.split("\n") if line.strip() != ""])
        if not content:
            print("❌ 無內文，跳過")
            continue

        print("抓到文章內容:", content[:20], "...")

        # 抓圖片
        figure = new_soup.select_one("figure.story img, .story img")
        img_url = figure["src"] if figure else None
        if img_url.startswith("//"):
            img_url = "https:" + img_url

        filename = None

        if img_url:
            try:
                img_data = requests.get(img_url).content
                filename = os.path.join(IMG_DIR, title[:10].replace("/", "_") + ".jpg")
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"圖片已下載: {filename}")
            except Exception as e:
                print("圖片下載失敗:", e)

        # 加入 dataset
        dataset.append({
            "id": idx + 1,
            "category": category,
            "url": news_url,
            "title": title,
            "content": content,
            "label": 0,
            "image_path": filename
        })
        if (idx + 1) % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            print(f"💾 已暫存 {len(dataset)} 筆資料")

    except Exception as e:
        print("抓取文章失敗:", e)
        
    time.sleep(random.uniform(0.8, 1.5))

# ======存成 JSON 檔案=====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"✅ 資料已存成 {len(dataset)} 筆 ETtoday 新聞資料到 {OUTPUT_FILE}")
