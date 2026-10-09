import os
import re
import time
import json
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

print("月份區間:", month_ranges)
print("len:", len(month_ranges))

# 分類設定
categories = ["政治", "財經", "娛樂", "科技", "國際"]
print("分類:", categories)

# 基本設定
HEADERS = {"User-Agent": "Mozilla/5.0"}
IMG_DIR = "./tvbs_news"
OUTPUT_FILE = "tvbs_real.json"
LINK_FILE = "tvbs_links.json"
MAX_RESULTS = 5  # 每月每類最多筆數
os.makedirs(IMG_DIR, exist_ok=True)

# =====Method 1: Google Custom Search API=====
links = []
for category in categories:
    for after, before in month_ranges:
        query = f"site:https://news.tvbs.com.tw {category} before:{before} after:{after}"
        print(f"搜尋分類: {category}, 時間: {after} ~ {before}")
        for start in range(1, MAX_RESULTS + 1, 10):
            url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CX}&key={API_KEY}&start={start}"
            res = requests.get(url).json()
            new_links = [item["link"] for item in res.get("items", []) if "news.tvbs.com.tw" in item["link"]]
            links.extend((link, category) for link in new_links)
            print(f"第 {(start-1)//10 + 1} 批，共 {len(new_links)} 則")
            time.sleep(1)

# 去重
unique_links = {}
for link, category in links:
    if link not in unique_links:
        unique_links[link] = category
links = list(unique_links.items())
print(f"Method: Google CSE -> 找到 {len(links)} 篇 自由時報 新聞連結")

# 儲存 links
with open(LINK_FILE, "w", encoding="utf-8") as f:
    json.dump(links, f, ensure_ascii=False, indent=4)
print(f"🔖 已將新聞連結存成 {LINK_FILE}")

# =====進入每篇新聞頁面抓取內容===== 
if not os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("[]")

def clean_text(text: str) -> str:
    """清理 TVBS 常見樣板"""
    text = re.sub(r"責任編輯：.*", "", text)
    text = re.sub(r"封面圖／.*", "", text)
    text = re.sub(r"（圖／.*?）", "", text)
    text = re.sub(r"\(圖／.*?\)", "", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

with open("tvbs_links.json", "r", encoding="utf-8") as f:
    links = json.load(f)

dataset = []
for idx, (news_url, category) in enumerate(links):
    print("-------")
    print(f"id: {idx+1}, url: {news_url}, category: {category}")
    try:
        res = requests.get(news_url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        new_soup = BeautifulSoup(res.text, "html.parser")

        # 標題
        title_tag = new_soup.select_one("h1.title")
        title = title_tag.get_text(strip=True) if title_tag else "無標題"
        print("抓到文章標題:", title)

        # 內文容器
        article_div = new_soup.select_one("[itemprop='articleBody']")
        if not article_div:
            print("❌ 找不到內文區塊，跳過")
            continue
        # ======================================================
        #  圖片抓取邏輯：先抓 articleBody 外的 img_box，否則抓內文第一張
        # ======================================================     
        filename = None
        img_url = None
        
        # 2) fallback：抓 articleBody 裡面的第一張圖片（內文圖片）
        inner_img = article_div.select_one("div.img.margin_b20 img, .img.margin_b20 img, .img img")
        if inner_img:
            img_url = inner_img.get("data-original")
            print("📸 使用 articleBody 內的第一張圖片")
        else:
            print("⚠️ 沒有找到任何圖片")
            img_url = None
            

        if img_url:
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            bad_ext = [".svg", ".gif", ".webp"]
            if any(ext in img_url.lower() for ext in bad_ext):
                print("⚠️ 發現非新聞圖片 (SVG/GIF/WEBP)，跳過:", img_url)
                img_url = None

            # TVBS 新聞圖片通常都在 upload/
            if img_url and "upload" not in img_url:
                print("⚠️ 非新聞主圖（缺少 upload/），跳過:", img_url)
                img_url = None

            try:
                img_data = requests.get(img_url).content
                filename = os.path.join(IMG_DIR, title[:10].replace("/", "_") + ".jpg")
                with open(filename, "wb") as f:
                    f.write(img_data)
                print(f"圖片已下載: {filename}")
            except Exception as e:
                print("圖片下載失敗:", e)
        

        remove_selectors = [
            # 廣告區塊
            ".ad_pc", ".ad_mo", ".adsbox",
            "[id^='news_pc_read_in']", "[id^='news_m_read_in']", ".center"

            # 圖片與圖說
            ".img", ".margin_b20", ".lazyimage", "img", "figure", "picture",
            ".font_color5", ".img_small",

            # 宣傳區塊
            ".widely_declared",

            # 技術性無用標籤
            "iframe", "script", "ins"
        ]

        for sel in remove_selectors:
            for ad in article_div.select(sel):
                ad.decompose()
        
        raw_text = article_div.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        content = "\n".join(lines)

        # 段落過濾
        skip_contains = ["作者：", "圖／", "圖／TVBS", "責任編輯", "活動辦法", "更多新聞", "延伸閱讀"]

        clean_lines = [
            ln for ln in content.split("\n")
            if not any(skip in ln for skip in skip_contains)
        ]

        content = "\n".join(clean_lines)
        if not content:
            print("❌ 無內文，跳過")
            continue
        print("抓到文章內容:", content[:20], "...")

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

print(f"✅ 資料已存成 {len(dataset)} 筆 TVBS 新聞資料到 {OUTPUT_FILE}")
