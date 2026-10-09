"""
通用新聞爬蟲模塊
整合多個新聞源（ETtoday, LTN, Chinatimes, TVBS, UDN, Yahoo, NowNews）
"""
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional
import re
from urllib.parse import urlparse
import logging

try:
    from backend.cleaner import NewsCleaner
except ModuleNotFoundError:
    from cleaner import NewsCleaner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


__all__ = ['NewsCrawler']

class NewsCrawler:
    """統一的新聞爬蟲接口"""
    
    SUPPORTED_DOMAINS = {
        'ettoday.net': 'ettoday',
        'ltn.com.tw': 'ltn',
        'chinatimes.com': 'chinatimes',
        'tvbs.com.tw': 'tvbs',
        'udn.com': 'udn',
        'yahoo.com': 'yahoo',
        'nownews.com': 'nownews'
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.cleaner = NewsCleaner()
    
    def identify_source(self, url: str) -> Optional[str]:
        """識別新聞來源"""
        domain = urlparse(url).netloc.lower()
        for key, source in self.SUPPORTED_DOMAINS.items():
            if key in domain:
                return source
        return None
    
    def crawl(self, url: str) -> Dict:
        """
        爬取新聞內容
        返回: {
            'url': str,
            'title': str,
            'content': str,
            'images': List[str],
            'author': Optional[str],
            'published_at': Optional[str],
            'site_name': str
        }
        """
        source = self.identify_source(url)
        if not source:
            logger.warning(f"Unsupported URL: {url}")
            return self._normalize_result(self._generic_crawl(url))
        
        logger.info(f"Crawling {source}: {url}")
        
        # 根據不同新聞源調用不同的解析器
        crawlers = {
            'ettoday': self._crawl_ettoday,
            'ltn': self._crawl_ltn,
            'chinatimes': self._crawl_chinatimes,
            'tvbs': self._crawl_tvbs,
            'udn': self._crawl_udn,
            'yahoo': self._crawl_yahoo,
            'nownews': self._crawl_nownews
        }
        
        crawler_func = crawlers.get(source, self._generic_crawl)
        result = crawler_func(url)
        if isinstance(result, dict):
            result.setdefault('source', source)
        return self._normalize_result(result)

    def _normalize_result(self, result: Dict) -> Dict:
        """統一清理輸出，避免沿用舊圖片。"""
        raw_content = result.get('content', '')
        source = result.get('source') or self.identify_source(result.get('url', '')) or 'common'
        content = self.cleaner.clean_text(raw_content, source)
        title = self._clean_text(result.get('title', ''))
        images = result.get('images', [])

        if not isinstance(images, list):
            images = []
        else:
            images = [img.strip() for img in images if isinstance(img, str) and img.strip()]

        # 若缺少內文或圖片，強制清空圖片，避免沿用舊圖
        if not content or not images:
            images = []

        result['title'] = title
        result['content'] = content
        result['images'] = images
        return result
    
    def _fetch_html(self, url: str) -> BeautifulSoup:
        """獲取網頁 HTML"""
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        if not text:
            return ""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in text.split('\n')]
        lines = [ln for ln in lines if ln]
        return '\n'.join(lines)
    
    def _crawl_ettoday(self, url: str) -> Dict:
        """ETtoday 爬蟲 - 基於原始 ettoday_crawler.py 實現"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'ETtoday'
        }
        
        try:
            # 標題
            title_elem = soup.select_one('h1.title')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 內容容器
            article_div = soup.select_one('div.story')
            if not article_div:
                article_div = soup.select_one('article.story')
            
            if article_div:
                # 移除廣告
                for ad in article_div.select('.ad_in_news, .ad_readmore, .lazyload-ad, script, ins'):
                    ad.decompose()
                
                # 抓段落並過濾
                paragraphs = article_div.find_all('p')
                filtered_paragraphs = []
                skip_keywords = ['記者', '報導', '中心／', '整理／', '新聞組／']
                skip_k = ['▲']
                skip_contains = ['ETtoday新聞雲', '更多新聞', '延伸閱讀', '（圖／', '(圖／', '照片／']
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    # 跳過記者來源段（短於15字）
                    if any(text.startswith(k) for k in skip_keywords) and len(text) < 15:
                        continue
                    # 跳過圖說
                    if any(text.startswith(k) for k in skip_k):
                        continue
                    # 跳過包含特定字眼
                    if any(k in text for k in skip_contains):
                        continue
                    filtered_paragraphs.append(text)
                
                result['content'] = self._clean_text('\n'.join(filtered_paragraphs))
            else:
                result['content'] = ''
            
            # 圖片
            images = []
            img_elems = soup.find_all('img', class_='main-image')
            for img in img_elems:
                src = img.get('src')
                if src:
                    images.append(src)
            result['images'] = images
            
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"ETtoday parsing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _crawl_ltn(self, url: str) -> Dict:
        """自由時報 爬蟲 - 基於原始 ltn_crawler.py 實現"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'Liberty Times'
        }
        
        try:
            # 標題
            title_elem = soup.select_one('h1')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 內容容器
            article_div = soup.select_one('div.text.boxTitle')
            if not article_div:
                article_div = soup.select_one('div.text.boxTitle.boxText')
            
            if article_div:
                # 移除廣告和繼續閱讀提示
                for ad in article_div.select(
                    "div[id^='ad-'], .ad_box, .ad_ph, .ad, .suggest_pc, "
                    ".subs_eDM, .appE1121, iframe, script, ins"
                ):
                    ad.decompose()
                
                # 移除「請繼續往下閱讀」
                for p in article_div.find_all('p'):
                    if '請繼續往下閱讀' in p.get_text(strip=True):
                        p.decompose()
                
                # 抓段落（不包括圖片區塊內的p）
                paragraphs = [
                    p for p in article_div.find_all('p')
                    if not p.find_parent(class_='photo')
                ]
                
                # 過濾特定內容
                skip_contains = ['自由時報', '延伸閱讀', '（圖／', '(圖／', '▲', '資料照', '點圖放大']
                filtered_paragraphs = []
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if any(k in text for k in skip_contains):
                        continue
                    if text:
                        filtered_paragraphs.append(text)
                
                result['content'] = self._clean_text('\n'.join(filtered_paragraphs))
            else:
                result['content'] = ''
            
            # 圖片
            images = []
            img_elems = soup.find_all('img', class_='lazy')
            for img in img_elems:
                src = img.get('data-src') or img.get('src')
                if src and not src.endswith('.svg'):
                    images.append(src)
            result['images'] = images
            
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"LTN parsing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _crawl_chinatimes(self, url: str) -> Dict:
        """中時電子報 爬蟲 - 基於原始 chinatime_crawler.py 實現"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'China Times'
        }
        
        try:
            # 標題
            title_elem = soup.select_one('h1.article-title')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 找到 column-wrapper 和 column-left
            wrapper = soup.select_one("[class*='column-wrapper']")
            if not wrapper:
                result['content'] = ''
                result['images'] = []
                return result
            
            first_col_left = wrapper.select_one('div.column-left')
            if not first_col_left:
                result['content'] = ''
                result['images'] = []
                return result
            
            # 內容
            article_div = first_col_left.select_one("div.article-body[itemprop='articleBody']")
            if article_div:
                # 移除廣告和腳本
                for bad in article_div.select('script, iframe, ins, .ad'):
                    bad.decompose()
                
                paragraphs = article_div.find_all('p')
                filtered_paragraphs = []
                skip_contains = ['（圖／', '(圖／', '▲', '延伸閱讀']
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if any(k in text for k in skip_contains):
                        continue
                    if text:
                        filtered_paragraphs.append(text)
                
                result['content'] = self._clean_text('\n'.join(filtered_paragraphs))
            else:
                result['content'] = ''
            
            # 圖片 - 優先 main-figure，其次內文第一張
            images = []
            img_tag = (first_col_left.select_one('div.main-figure img.photo') 
                      or first_col_left.select_one('div.main-figure img')
                      or (article_div.select_one('img') if article_div else None))
            
            if img_tag:
                img_url = (img_tag.get('data-src') 
                          or img_tag.get('data-original') 
                          or img_tag.get('src'))
                if img_url:
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    images.append(img_url)
            
            result['images'] = images
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"Chinatimes parsing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _crawl_tvbs(self, url: str) -> Dict:
        """TVBS 爬蟲 - 基於原始 tvbs_crawler.py 實現"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'TVBS'
        }
        
        try:
            # 標題
            title_elem = soup.select_one('h1.title')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 內容容器
            article_div = soup.select_one("[itemprop='articleBody']")
            if not article_div:
                result['content'] = ''
                result['images'] = []
                return result
            
            # 圖片 - 抓 articleBody 內的第一張
            images = []
            inner_img = article_div.select_one('div.img.margin_b20 img, .img.margin_b20 img, .img img')
            if inner_img:
                img_url = inner_img.get('data-original')
                if img_url:
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    # 只抓新聞圖片（包含 upload/）
                    if 'upload' in img_url and not any(ext in img_url.lower() for ext in ['.svg', '.gif', '.webp']):
                        images.append(img_url)
            
            # 移除廣告和圖片區塊
            remove_selectors = [
                '.ad_pc', '.ad_mo', '.adsbox',
                "[id^='news_pc_read_in']", "[id^='news_m_read_in']", '.center',
                '.img', '.margin_b20', '.lazyimage', 'img', 'figure', 'picture',
                'script', 'iframe', 'ins'
            ]
            for sel in remove_selectors:
                for tag in article_div.select(sel):
                    tag.decompose()
            
            # 抓段落
            paragraphs = article_div.find_all('p')
            filtered_paragraphs = []
            skip_contains = ['責任編輯：', '封面圖／', '（圖／', '(圖／']
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                if any(k in text for k in skip_contains):
                    continue
                if text:
                    filtered_paragraphs.append(text)
            
            result['content'] = self._clean_text('\n'.join(filtered_paragraphs))
            result['images'] = images
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"TVBS parsing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _crawl_udn(self, url: str) -> Dict:
        """聯合新聞網 爬蟲 - 基於原始 udn_crawler.py 實現"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'UDN'
        }
        
        try:
            # 標題
            title_elem = soup.select_one('h1')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 內容容器
            content_div = (soup.select_one('div.article-content__editor')
                          or soup.select_one('div#story_body_content')
                          or soup.select_one('section.article-content-wrapper'))
            
            if content_div:
                # 移除廣告和腳本
                for bad in content_div.select('script, iframe, ins, .adv, .advertisement'):
                    bad.decompose()
                
                # 抓段落
                paragraphs = content_div.find_all('p')
                filtered_paragraphs = []
                skip_contains = ['延伸閱讀', '更多閱讀', '▲', '資料照', '（圖／', '(圖／']
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if any(k in text for k in skip_contains):
                        continue
                    if text:
                        filtered_paragraphs.append(text)
                
                result['content'] = self._clean_text('\n'.join(filtered_paragraphs))
            else:
                result['content'] = ''
            
            # 圖片
            images = []
            img = (soup.select_one('figure img')
                  or soup.select_one('.hero-image img')
                  or soup.select_one("meta[property='og:image']"))
            
            if img:
                img_url = None
                if img.has_attr('src'):
                    img_url = img['src']
                elif img.has_attr('data-src'):
                    img_url = img['data-src']
                elif img.name == 'meta' and img.has_attr('content'):
                    img_url = img['content']
                
                if img_url:
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    images.append(img_url)
            
            result['images'] = images
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"UDN parsing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _crawl_yahoo(self, url: str) -> Dict:
        """Yahoo 新聞 爬蟲 - 基於原始 yahoo_crawler.py 實現"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'Yahoo News'
        }
        
        try:
            # 標題
            title_elem = soup.find('h1')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 內容容器
            article_div = soup.select_one('article .atoms')
            
            if article_div:
                # 移除廣告
                for ad in article_div.select('.recommendation-contents, .bg-card-bg-1'):
                    ad.decompose()
                
                # 抓段落
                paragraphs = article_div.select('p.mb-module-gap.break-words')
                filtered_paragraphs = []
                skip_keywords = ['報導', '中心／', '記者', '新聞網／', '新聞組／', '整理／', '綜合報導']
                
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    # 跳過來源段落（短於11字）
                    if any(text.startswith(k) for k in skip_keywords) and len(text) < 11:
                        continue
                    if text:
                        filtered_paragraphs.append(text)
                
                result['content'] = self._clean_text('\n'.join(filtered_paragraphs))
            else:
                result['content'] = ''
            
            # 圖片
            images = []
            figure = soup.select_one('figure img')
            if figure:
                img_url = figure.get('src')
                if img_url:
                    images.append(img_url)
            result['images'] = images
            
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"Yahoo parsing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _crawl_nownews(self, url: str) -> Dict:
        """今日新聞 爬蟲 - 基於原始 nownews_crawler.py 實現"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'NOW News'
        }
        
        try:
            # 標題
            title_elem = soup.select_one('h1')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 內容 - 使用原始選擇器：div#articleContent.article-content-edtor
            article_block = soup.select_one('div#articleContent.article-content-edtor')
            
            if article_block:
                # 移除廣告和非內容區塊（基於原始實現）
                remove_selectors = [
                    '.ad-blk', '.ad-blk1', '.aditem', '.hr-separator',
                    '.zoomIn02', '.zoomIn03', '.fade', 'script', 'iframe',
                    '.related-item', 'aside.article-aside'
                ]
                for sel in remove_selectors:
                    for tag in article_block.select(sel):
                        tag.decompose()
                
                # 獲取純文字內容
                raw_text = article_block.get_text(separator='\n', strip=True)
                lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
                content = '\n'.join(lines)
                
                # 段落過濾（移除圖說、作者等）
                skip_contains = ['作者：', '（圖/', '圖／', '（ 圖 /', '圖 /', 
                                '（圖', '(圖', '責任編輯', '▲', '更多新聞', '延伸閱讀']
                clean_lines = [
                    ln for ln in content.split('\n')
                    if not any(skip in ln for skip in skip_contains)
                ]
                result['content'] = self._clean_text('\n'.join(clean_lines))
            else:
                result['content'] = ''
            
            # 圖片 - 從 figure.image img 提取（基於原始實現）
            images = []
            figure = soup.select_one('figure.image img')
            if figure:
                img_url = figure.get('data-src') or figure.get('src')
                if img_url:
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    images.append(img_url)
            result['images'] = images
            
            # 作者和時間
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"NOW News parsing error: {e}")
            result['error'] = str(e)
        
        return result
    
    def _generic_crawl(self, url: str) -> Dict:
        """通用爬蟲（用於不支持的網站）"""
        soup = self._fetch_html(url)
        
        result = {
            'url': url,
            'site_name': 'Unknown'
        }
        
        try:
            # 嘗試提取標題
            title_elem = soup.find('h1') or soup.find('title')
            result['title'] = self._clean_text(title_elem.text if title_elem else '')
            
            # 嘗試提取所有段落作為內容
            paragraphs = soup.find_all('p')
            content = ' '.join([p.text for p in paragraphs if p.text.strip()])
            result['content'] = self._clean_text(content)
            
            # 提取圖片
            images = []
            img_elems = soup.find_all('img')
            for img in img_elems:
                if img.get('src'):
                    images.append(img['src'])
            result['images'] = images[:10]  # 限制10張
            
            result['author'] = None
            result['published_at'] = None
            
        except Exception as e:
            logger.error(f"Generic crawl error: {e}")
            result['error'] = str(e)
        
        return result


# 使用示例
if __name__ == "__main__":
    crawler = NewsCrawler()
    
    # 測試 ETtoday
    test_url = "https://www.ettoday.net/news/20240101/2645678.htm"
    result = crawler.crawl(test_url)
    
    print(f"Title: {result.get('title')}")
    print(f"Content preview: {result.get('content', '')[:200]}...")
    print(f"Images found: {len(result.get('images', []))}")
