import json
import re
import os
import hashlib
from typing import List, Dict

class NewsCleaner:

    def __init__(self):
        self.simplified_chars = set("专产历压厉参双县后听团园实导层岁条样标权汉济现环联设证谈请购贸阳陈长")
        # 可擴充 skip list
        self.skip_patterns = {
            "yahoo": [
                r"更多.*新聞", r"延伸閱讀：", r"看更多", r"看影片", r"推薦閱讀", r"更多世界日報.*", r"更多.*?新聞.*?", r"更多.*?報導.*", r".*?（翻攝.*?）", r".*來源：.*", r"今日.*?推薦.*?影音.*?", r"▲.*?", 
                r"→.*?", r".*?相關新聞影音", r"\(文圖提供.*", r"\(?原始連結\)?", r"作者[為：].*", r"照片[為：].*", r"[●◎◤⊙].*", r"（更多新聞：.*", r".*?責任編輯.*?", r"(實習記者|核稿編輯).*?", 
                r"[iI]mage [sS]ource.*", r"延伸閱讀.*", r"今日熱門.*", r".*[（(][^）)]*提供[）)].*", r"中央社記者.*?", r".*?圖[：:].*", r".*36氪.*?", r"（民視新聞.*?）", r"^[▶].*?$", r"看更多.*", r"👉.*"
            ],
            "ltn": [
                r"〔記者.*?報導〕", r"自由時報", r"更多相關新聞", r"《.*?》", r".*?／特稿", r"武漢肺炎懶人包.*$", r"相關新聞連結：.*" ,r"相關新聞請見：[^\\n]*", r"相關新聞︰.*", r"(首次上稿|上稿時間)[^\\n]*",
                r"（?資料來源：.*", r"整理：.*", r"「武漢肺炎專區」.*"
            ],
            "ettoday": [
                r"記者.*?／.*?報導", r"★.*?★", r"▼.*?", r"延伸閱讀", r"^[►▶︎].*$", r".*?更多(熱門|相關)新聞.*", r".*?其他人也看了.*?", r".*?圖片授權來源：.*?", r"相關新聞：.*", r"舉凡餐券.*", r"相關新聞\.\.\."
            ],
            "chinatimes": [
                r"更多 CTWANT 報導", r"中時新聞網", r"（待續）", r"（圖.*?）"
            ],
            "nownews": [
                r"NOWnews 今日新聞", r"延伸閱讀", r"看更多", r"（圖.*?）", r"※.*",r"※【.*", r"NOWnews 今日新聞", r"】提醒.*", r"因應(新冠|武漢)肺炎疫情，疾管署.*", r"^1922", r"」專線，或「", r"0800-001922", r"」，並依指示配戴口罩儘速就醫，同時主動告知醫師旅遊史及接觸史.*", r"自殺不能解決問題.*", r"透過守門123步驟.*",
                r"少一份毒品.*"
            ],

            # 全站共通 pattern
            "common": [
                r"（圖.*?）", r"\(圖.*?\)", r".*?圖／.*?", r".*?／圖.*",          # 各式圖示
                r"※.*?", r"★.*?", r"^【.*?】",  
                r"^.*?[／/]\s*.*?報導$", r"^\(?待續\)?$", r"（.*?(示意圖|資料照|資料圖|翻攝自).*?）", r"^圖.*／.*", r"快訊／.*", r"獨／.*", r"亂！.*", r".*?美麗最大的祕密.*?", r"文／.*?", r"瘋言瘋語：", r"文 / .*?"
                r"^https?://\S+", r"記者.+綜合報導", r"記者.+台北報導", r"＊.*?"
            ]
        }
        self.sub_patterns = {
            "ltn": [
                (r"（(?:圖文|圖|文)[^）]*?）", r""), 
                (r"（[左右].*）", r""), 
                (r"[（(]?[左右上下中]\d+[）)]?", ""),
                (r"［記者[^］]+?報導］", ""),
                (r"（記者.*?）", ""),
                (r"〔.*?〕", ""),
                (r"〔.*?］", ""),
                (r"（編輯.*?）", ""),
                (r"（編譯.*?）", ""),
                (r"\\b", ""), 
                (r"https?://[A-Za-z0-9\-\._~:/?#\[\]@!$&\*+,;=%]+", "<LINK>"), 
            ], 
            "ettoday": [
                (r"（(?:圖文|圖|文)[^）]*?）", r""), 
                (r"（[左右].*）", r""), 
                (r"[（(]?[左右上下中]\d+[）)]?", ""),
                (r"［記者[^］]+?報導］", ""),
                (r"（記者.*?）", ""),
                (r"〔.*?〕", ""),
                (r"〔.*?］", ""),
                (r"（編輯[^）]*）[0-9]*?", ""),
                (r"（編譯：[^）]*）[0-9]*?", ""),
                (r"（譯者[^）]*）[0-9]*?", ""),
                (r"（編譯：[^）]*）[0-9]*?", ""),
                (r"\\b", ""), 
                (r"https?://[A-Za-z0-9\-\._~:/?#\[\]@!$&\*+,;=%]+", "<LINK>"), 
            ], 
            "chinatimes": [
                (r"（(?:圖文|圖|文)[^）]*?）", r""), 
                (r"（[左右].*）", r""), 
                (r"[（(]?[左右上下中]\d+[）)]?", ""),
                (r"［記者[^］]+?報導］", ""),
                (r"（記者.*?）", ""),
                (r"〔.*?〕", ""),
                (r"〔.*?］", ""),
                (r"（編輯[^）]*）[0-9]*?", ""),
                (r"（編譯：[^）]*）[0-9]*?", ""),
                (r"（譯者[^）]*）[0-9]*?", ""),
                (r"（編譯：[^）]*）[0-9]*?", ""),
                (r"\\b", ""), 
                (r"https?://[A-Za-z0-9\-\._~:/?#\[\]@!$&\*+,;=%]+", "<LINK>"), 
            ],
            "nownews": [
                (r"（(?:圖文|圖|文)[^）]*?）", r""), 
                (r"（[左右].*）", r""), 
                (r"[（(]?[左右上下中]\d+[）)]?", ""),
                (r"［記者[^］]+?報導］", ""),
                (r"（記者.*?）", ""),
                (r"〔.*?〕", ""),
                (r"〔.*?］", ""),
                (r"（編輯.*?）", ""),
                (r"（譯者[^）]*）[0-9]+", ""),
                (r"（編譯.*?）", ""),
                (r"「看看政治聽聽[^」]*」，?", ""),
                (r"（軍聞社[^）]*）", ""),
                (r"（中央社[^）]*）", ""),
                (r"\\b", ""), 
                (r"https?://[A-Za-z0-9\-\._~:/?#\[\]@!$&\*+,;=%]+", "<LINK>"), 
                (r"★[ ]?", ""), 
            ], 
            "yahoo": [
                (r"（(?:圖文|圖|文)[^）]*?）", r""), 
                (r"（[左右].*）", r""), 
                (r"[（(]?[左右上下中]\d+[）)]?", ""),
                (r"［記者[^］]+?報導］", ""),
                (r"（記者.*?）", ""),
                (r"〔.*?〕", ""),
                (r"〔.*?］", ""),
                (r"（編輯.*?）", ""),
                (r"（譯者[^）]*）[0-9]+", ""),
                (r"（編譯.*?）", ""),
                (r"「看看政治聽聽[^」]*」，?", ""),
                (r"（軍聞社[^）]*）", ""),
                (r"（中央社[^）]*）", ""),
                (r"https?://[A-Za-z0-9\-\._~:/?#\[\]@!$&\*+,;=%]+", "<LINK>"), 
                (r"★[ ]?", ""), 
                (r"\[(新頭殼newtalk|周刊王CTWANT)\][ ]?", ""), 
                (r"（[^）]*／綜合報導）", ""), 
                (r"（延伸閱讀：[^）]*）", ""),
            ]
        }
    
    # 簡體字檢測
    def has_simplified(self, text: str) -> bool: 
        return any(char in self.simplified_chars for char in text)
    
    # 中文存在檢測
    def has_chinese(self, text):
        return any('\u4e00' <= ch <= '\u9fff' for ch in text)
    
    # 「台灣彩券網站 」
    LOTTERY_KEYWORDS = [r"「台灣彩券網站 」", r"「台灣彩券網站」", r"台灣彩券各項開獎獎項及獎號"]
    def is_lottery_news(self, text: str) -> bool:
        return any(keyword in text for keyword in self.LOTTERY_KEYWORDS)
    
    # 清除社群媒體連結與誤抓內容
    def is_social_embed(self, line): 
        keys = [
            "twitter", "twitter.com", "t.co/", "x.com",
            "instagram", "instagram.com", "facebook.com", "fb.watch",
            "facebook", "youtube.com", "youtu.be", "tiktok.com"
        ]
        lower = line.lower()
        if not self.has_chinese(line) and any(k in lower for k in keys): 
            return True
        return False
    
    # 去重：同 URL、或不同 URL 但內容相同
    def remove_duplicate(self, data: List[Dict]) -> List[Dict]:
        seen_urls = set()
        seen_contents = {}
        new_data = []
        count = 0
        
        for item in data:
            url = item.get("url", "")
            content = item.get("content", "").strip()

            # 用 sha256 保證唯一（比 hash() 更穩）
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # case 1 — URL 重複（跳過）
            if url in seen_urls:
                count += 1
                continue

            # case 2 — 內容重複：保留第一篇，跳過後面
            if content_hash in seen_contents:
                count += 1
                continue

            # 記錄第一次遇到的
            seen_urls.add(url)
            seen_contents[content_hash] = item["id"]

            new_data.append(item)
        print(f"Removed {count} duplicate articles.")
        
        return new_data


    def remove_invalid_image(self, data: List[Dict]) -> List[Dict]:
        new_data = []
        for item in data:
            img = item.get("image_path", "")
            if img and os.path.exists(img):
                new_data.append(item)
        print(f"Removed {len(data) - len(new_data)} articles with invalid image paths.")
        return new_data

    def clean_text(self, text: str, source: str) -> str:
        if not text:
            return ""

        # 按原始格式拆行（完全保留換行）
        lines = text.split("\n")
        cleaned_lines = []


        # 取得該新聞網的 pattern 與共通 pattern
        patterns = self.skip_patterns.get(source.lower(), []) + self.skip_patterns["common"]
        sub_list = self.sub_patterns.get(source.lower(), [])

        skip_next = False # 解決 CTWANT 換行問題

        for ln in lines:
            original_ln = ln  # 保留原始行
            remove_line = False
            
            # 換行處理
            if skip_next:
                skip_next = False
                continue
            
            if self.is_social_embed(ln):
                remove_line = True
            if not self.has_chinese(ln):
                remove_line = True
            for p in patterns:
                try:
                    # 若整行符合噪音 → 刪掉整行
                    if re.fullmatch(p, ln.strip()):
                        remove_line = True
                        break

                except re.error as e:
                    print(f"[REGEX ERROR] pattern={p}, error={e}")
                    continue

            # rule D: 跨行污染觸發（下一行一起跳）
            if re.search(r"(更多 CTWANT 文章|《TVBS》提醒您：|x 三立新聞網提醒您：)", ln):
                remove_line = True
                skip_next = True   # 👈 下一行一併清除

            if remove_line:
                continue
            
            for sub in sub_list:
                try:
                    ln = re.sub(sub[0], sub[1], ln)
                except:
                    pass
                
            cleaned_lines.append(ln)


        # 保留所有 \n
        cleaned = "\n".join(cleaned_lines)

        return cleaned


    def clean_all(self, data: List[Dict]) -> List[Dict]:
        output = []
        for item in data:
            title = item.get("title", "")
            content = item.get("content", "")
            
            if self.is_lottery_news(content):
                continue
            
            # 簡體字判斷，如果是簡體就將該篇文章跳過
            if self.has_simplified(title) or self.has_simplified(content): 
                continue
            
            source = self.detect_source(item["url"])
            cleaned = self.clean_text(content, source)
            
            if len(cleaned.replace("\n", "").strip()) < 40:
                continue
            
            item["content"] = cleaned
            output.append(item)
        return output

    def detect_source(self, url: str) -> str:
        if "yahoo" in url:
            return "yahoo"
        if "ltn" in url or "liberty" in url:
            return "ltn"
        if "ettoday" in url:
            return "ettoday"
        if "chinatimes" in url:
            return "chinatimes"
        if "nownews" in url:
            return "nownews"
        return "common"

    # 重排id
    def reorder_ids(self, data: List[Dict]) -> List[Dict]:
        for idx, item in enumerate(data):
            item["id"] = idx + 1
        return data

def run_pipeline(input_path: str, output_path: str):
    cleaner = NewsCleaner()

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Step 1: remove duplicate URLs...")
    data = cleaner.remove_duplicate(data)

    print("Step 2: remove invalid image paths...")
    data = cleaner.remove_invalid_image(data)

    print("Step 3: clean text content...")
    data = cleaner.clean_all(data)

    print("Step 4: reorder IDs...")
    data = cleaner.reorder_ids(data)

    print("Saving output...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("Finished!")


if __name__ == "__main__":
    run_pipeline("yahoo_cleaned2.json", "yahoo_cleaned2.json")
