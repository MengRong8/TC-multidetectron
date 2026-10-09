# 網路爬蟲策略
# 1: Request + BeautifulSoup for 靜態頁面such as Yahoo
import os
import json
import time
import feedparser
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

# Google Custom Search API 參數
API_KEY = ""
CX = ""

# Query Setting: query = "site:tw.news.yahoo.com {categories} before:{before} after:{after}"
# Time: 
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
# Category:
categories = ["政治", "財經", "娛樂", "科技", "國際"]
print("分類:", categories)


# 基本設定
HEADERS = {"User-Agent": "Mozilla/5.0"}
IMG_DIR = "./news_images"
OUTPUT_FILE = "yahoo_real.json"
MAX_RESULTS = 45  # 每個時間段(time)、每個分類(category)最多抓取的新聞數量
os.makedirs(IMG_DIR, exist_ok=True)

# =====Method 1: RSS(抓取近期新聞)=====
# rss = feedparser.parse("https://tw.news.yahoo.com/rss/politics") #ex. 可更改html最後的category參數抓不同分類
# links = [entry.link for entry in rss.entries]
# print(f"Method:RSS -> 找到 {len(links)} 篇新聞連結")

# =====Method 2: Google Custom Search API=====
links = []
for category in categories: # 新聞種類
    for after, before in month_ranges: # 時間區間
        query = f"site:tw.news.yahoo.com {category} before:{before} after:{after}"
        print(f"搜尋分類: {category}, 時間: {after} ~ {before}")
        for start in range(1, MAX_RESULTS + 1, 10): # , 10): # start=1, 11, 21, ...
            url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}&start={start}"
            res = requests.get(url).json()
            new_links = [item["link"] for item in res.get("items", []) if "tw.news.yahoo.com" in item["link"]]
            
            # 把(url, category)存成tuple，方便後續存檔
            links.extend((link, category) for link in new_links)
            print(f"第 {(start-1)//10 + 1} 批，共 {len(new_links)} 則")
            
            time.sleep(1)  # 避免請求過快

# 去重
unique_links = {}
for link, category in links:
    if link not in unique_links:
        unique_links[link] = category
links = list(unique_links.items())  # 轉回 list of tuples

print(f"Method:Google CSE -> 找到 {len(links)} 篇新聞連結")
# =====進入每篇新聞頁面抓內文與圖片=====
dataset = []

for idx, (news_url, category) in enumerate(links): # 抓links全部
    print("-------")
    print(f"id: {idx+1}, url: {news_url}, category: {category}")
    try:
        res = requests.get(news_url, headers=HEADERS) # 進入URL
        new_soup = BeautifulSoup(res.text, "html.parser")

        # 先找到裝內文的容器(article)
        article_div = new_soup.select_one("article .atoms")

        if article_div:
            # 抓標題
            title = new_soup.find("h1").text if new_soup.find("h1") else "無標題"
            print("抓到文章標題:", title)
            # 移除內文的廣告
            for ad in article_div.select(".recommendation-contents, .bg-card-bg-1"):
                ad.decompose()  
            # 抓內文
            paragraphs = article_div.select("p.mb-module-gap.break-words") # 文章段落
            # filer: 篩選掉段落來源
            filtered_paragraphs = []
            skip_keywords = ["報導", "中心／", "記者", "新聞網／", "新聞組／", "整理／", "綜合報導"]
            for p in paragraphs:
                text = p.get_text(strip=True)
                if any(text.startswith(keyword) for keyword in skip_keywords) and len(text)<11:
                    print(f"⚠️ 跳過來源段落: {text}")
                    continue
                filtered_paragraphs.append(p)

            content = "\n".join([p.get_text(strip=True) for p in filtered_paragraphs])
            content = "\n".join([line for line in content.split("\n") if line.strip() != ""]) # 移除空行
            if not content:
                content = "無內文, 跳過"
                continue
            print("抓到文章內容:", content[:20], "...")

            # 抓圖片
            figure = new_soup.select_one("figure img")
            img_url = figure.get("src") if figure else None

            if img_url:
                try:
                    img_data = requests.get(img_url).content
                    os.makedirs(IMG_DIR, exist_ok=True)
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
                "label": 0,  # 預設標籤為 0(real)
                "image_path": filename if img_url else None
            })
    except Exception as e:
        print("抓取文章失敗:", e)
    time.sleep(1) # 避免請求過快

# ======存成 JSON 檔案=====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)

print(f"資料已存成 {len(dataset)} 筆新聞資料到 {OUTPUT_FILE}")
