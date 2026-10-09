"""
新聞內容清洗模組
提供 crawler 必經的文本清洗流程。
"""

from typing import Dict, List
import re


__all__ = ["NewsCleaner"]


class NewsCleaner:
	def __init__(self):
		# 可擴充 skip list
		self.skip_patterns = {
			"yahoo": [
				r"更多.*新聞", r"延伸閱讀：", r"看更多", r"看影片", r"推薦閱讀", r"更多世界日報.*", r"更多.*?新聞.*?", r"更多.*?報導.*", r".*?（翻攝.*?）", r".*來源：.*", r"今日.*?推薦.*?影音.*?", r"▲.*?",
				r"→.*?", r".*?相關新聞影音", r"\(文圖提供.*", r"\(?原始連結\)?", r"作者[為：].*", r"照片[為：].*", r"[●◎◤⊙].*", r"（更多新聞：.*", r".*?責任編輯.*?", r"(實習記者|核稿編輯).*?",
				r"[iI]mage [sS]ource.*", r"延伸閱讀.*", r"今日熱門.*", r".*[（(][^）)]*提供[）)].*", r"中央社記者.*?", r".*?圖[：:].*", r".*36氪.*?", r"（民視新聞.*?）", r"^[▶].*?$", r"看更多.*", r"👉.*",
			],
			"ltn": [
				r"〔記者.*?報導〕", r"自由時報", r"更多相關新聞", r"《.*?》", r".*?／特稿", r"武漢肺炎懶人包.*$", r"相關新聞連結：.*", r"相關新聞請見：[^\\n]*", r"相關新聞︰.*", r"(首次上稿|上稿時間)[^\\n]*",
				r"（?資料來源：.*", r"整理：.*", r"「武漢肺炎專區」.*",
			],
			"ettoday": [
				r"記者.*?／.*?報導", r"★.*?★", r"▼.*?", r"延伸閱讀", r"^[►▶︎].*$", r".*?更多(熱門|相關)新聞.*", r".*?其他人也看了.*?", r".*?圖片授權來源：.*?", r"相關新聞：.*", r"舉凡餐券.*", r"相關新聞\.\.\.",
			],
			"chinatimes": [
				r"更多 CTWANT 報導", r"中時新聞網", r"（待續）", r"（圖.*?）",
			],
			"nownews": [
				r"NOWnews 今日新聞", r"延伸閱讀", r"看更多", r"（圖.*?）", r"※.*", r"※【.*", r"NOWnews 今日新聞", r"】提醒.*", r"因應(新冠|武漢)肺炎疫情，疾管署.*", r"^1922", r"」專線，或「", r"0800-001922", r"」，並依指示配戴口罩儘速就醫，同時主動告知醫師旅遊史及接觸史.*", r"自殺不能解決問題.*", r"透過守門123步驟.*",
				r"少一份毒品.*",
			],
			# 全站共通 pattern
			"common": [
				r"（圖.*?）", r"\(圖.*?\)", r".*?圖／.*?", r".*?／圖.*",
				r"※.*?", r"★.*?", r"^【.*?】",
				r"^.*?[／/]\s*.*?報導$", r"^\(?待續\)?$", r"（.*?(示意圖|資料照|資料圖|翻攝自).*?）", r"^圖.*／.*", r"快訊／.*", r"獨／.*", r"亂！.*", r".*?美麗最大的祕密.*?", r"文／.*?", r"瘋言瘋語：", r"文 / .*?",
				r"^https?://\S+", r"記者.+綜合報導", r"記者.+台北報導", r"＊.*?",
			],
		}

		self.sub_patterns = {
			"ltn": [
				(r"（(?:圖文|圖|文)[^）]*?）", ""),
				(r"（[左右].*）", ""),
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
				(r"（(?:圖文|圖|文)[^）]*?）", ""),
				(r"（[左右].*）", ""),
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
				(r"（(?:圖文|圖|文)[^）]*?）", ""),
				(r"（[左右].*）", ""),
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
				(r"（(?:圖文|圖|文)[^）]*?）", ""),
				(r"（[左右].*）", ""),
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
				(r"（(?:圖文|圖|文)[^）]*?）", ""),
				(r"（[左右].*）", ""),
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
			],
		}

		# 行內尾巴污染切斷關鍵字
		self.inline_cut_markers = [
			"更多中天新聞網報導",
			"更多 CTWANT 報導",
			"更多CTWANT報導",
			"延伸閱讀",
		]

	def has_chinese(self, text: str) -> bool:
		return any("\u4e00" <= ch <= "\u9fff" for ch in text)

	def is_social_embed(self, line: str) -> bool:
		keys = [
			"twitter", "twitter.com", "t.co/", "x.com",
			"instagram", "instagram.com", "facebook.com", "fb.watch",
			"facebook", "youtube.com", "youtu.be", "tiktok.com",
		]
		lower = line.lower()
		return (not self.has_chinese(line)) and any(k in lower for k in keys)

	def detect_source(self, url: str) -> str:
		lower = (url or "").lower()
		if "yahoo" in lower:
			return "yahoo"
		if "ltn" in lower or "liberty" in lower:
			return "ltn"
		if "ettoday" in lower:
			return "ettoday"
		if "chinatimes" in lower:
			return "chinatimes"
		if "nownews" in lower:
			return "nownews"
		return "common"

	def _truncate_inline_tail(self, line: str) -> str:
		cut_pos = None
		for marker in self.inline_cut_markers:
			pos = line.find(marker)
			if pos >= 0 and (cut_pos is None or pos < cut_pos):
				cut_pos = pos
		if cut_pos is None:
			return line
		return line[:cut_pos].strip()

	def clean_text(self, text: str, source: str) -> str:
		if not text:
			return ""

		lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
		cleaned_lines: List[str] = []

		patterns = self.skip_patterns.get((source or "").lower(), []) + self.skip_patterns["common"]
		sub_list = self.sub_patterns.get((source or "").lower(), [])

		skip_next = False

		for ln in lines:
			ln = ln.strip()
			if not ln:
				continue

			if skip_next:
				skip_next = False
				continue

			remove_line = False
			if self.is_social_embed(ln):
				remove_line = True
			if not self.has_chinese(ln):
				remove_line = True

			for p in patterns:
				try:
					if re.fullmatch(p, ln):
						remove_line = True
						break
				except re.error:
					continue

			if re.search(r"(更多 CTWANT 文章|《TVBS》提醒您：|x 三立新聞網提醒您：)", ln):
				remove_line = True
				skip_next = True

			if remove_line:
				continue

			for sub in sub_list:
				try:
					ln = re.sub(sub[0], sub[1], ln)
				except re.error:
					continue

			ln = self._truncate_inline_tail(ln)
			if not ln:
				continue

			cleaned_lines.append(ln)

		cleaned = "\n".join(cleaned_lines)
		cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
		return cleaned
