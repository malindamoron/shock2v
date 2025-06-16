
import asyncio
import aiohttp
import time
import random
import json
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import feedparser
import newspaper
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from fake_useragent import UserAgent
import requests
from lxml import html
import sqlite3
from datetime import datetime, timedelta
import logging
import threading
from queue import Queue
import ssl
import certifi
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

class StealthOpenWebCrawler:
    def __init__(self, max_workers=50, delay_range=(1, 3)):
        self.max_workers = max_workers
        self.delay_range = delay_range
        self.ua = UserAgent()
        self.session = self._create_session()
        self.crawl_queue = Queue()
        self.processed_urls = set()
        self.results = []
        self.db_path = 'shock2/data/raw/crawler_cache.db'
        self._init_database()
        self.target_sources = self._load_news_sources()
        self.logger = self._setup_logger()
        
    def _create_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = certifi.where()
        return session
        
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS crawled_articles (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                title TEXT,
                content TEXT,
                published_date TEXT,
                source TEXT,
                hash TEXT,
                crawl_timestamp TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/crawler.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _load_news_sources(self):
        return [
            'https://rss.cnn.com/rss/edition.rss',
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://www.reuters.com/rssFeed/worldNews',
            'https://feeds.npr.org/1001/rss.xml',
            'https://feeds.washingtonpost.com/rss/world',
            'https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/section/world/rss.xml',
            'https://feeds.theguardian.com/theguardian/world/rss',
            'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
            'https://feeds.foxnews.com/foxnews/latest',
            'https://feeds.abcnews.com/abcnews/topstories',
            'https://feeds.nbcnews.com/nbcnews/public/news',
            'https://feeds.skynews.com/feeds/rss/world.xml',
            'https://feeds.content.dowjones.io/public/rss/mw_topstories',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://feeds.fortune.com/fortune/headlines',
            'https://feeds.techcrunch.com/TechCrunch/',
            'https://feeds.wired.com/wired/index',
            'https://feeds.arstechnica.com/arstechnica/index',
            'https://rss.slashdot.org/Slashdot/slashdotMain',
            'https://feeds.venturebeat.com/VentureBeat',
        ]
        
    async def _get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
    async def crawl_rss_feeds(self):
        """Crawl RSS feeds from major news sources"""
        articles = []
        
        for rss_url in self.target_sources:
            try:
                await asyncio.sleep(random.uniform(*self.delay_range))
                
                headers = await self._get_headers()
                response = self.session.get(rss_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    feed = feedparser.parse(response.content)
                    
                    for entry in feed.entries:
                        article_data = {
                            'url': entry.link,
                            'title': entry.title,
                            'summary': getattr(entry, 'summary', ''),
                            'published': getattr(entry, 'published', ''),
                            'source': urlparse(rss_url).netloc,
                            'crawl_timestamp': datetime.now().isoformat()
                        }
                        
                        # Get full article content
                        full_content = await self._extract_article_content(entry.link)
                        if full_content:
                            article_data['content'] = full_content
                            articles.append(article_data)
                            self._store_article(article_data)
                            
            except Exception as e:
                self.logger.error(f"Error crawling RSS {rss_url}: {str(e)}")
                
        return articles
        
    async def _extract_article_content(self, url):
        """Extract full article content using newspaper3k and selenium"""
        try:
            # First try with newspaper3k
            article = newspaper.Article(url)
            article.download()
            article.parse()
            
            if len(article.text) > 200:
                return {
                    'text': article.text,
                    'authors': article.authors,
                    'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                    'keywords': article.keywords,
                    'summary': article.summary
                }
                
        except Exception as e:
            self.logger.warning(f"Newspaper3k failed for {url}: {str(e)}")
            
        # Fallback to selenium for dynamic content
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument(f'--user-agent={self.ua.random}')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            
            # Wait for content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Extract content using multiple selectors
            content_selectors = [
                'article', '.article-content', '.post-content', 
                '.entry-content', '.story-body', '.article-body',
                '.content', '.main-content', 'main'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    content = elements[0].text
                    break
                    
            driver.quit()
            
            if len(content) > 200:
                return {'text': content, 'extraction_method': 'selenium'}
                
        except Exception as e:
            self.logger.error(f"Selenium extraction failed for {url}: {str(e)}")
            
        return None
        
    def _store_article(self, article_data):
        """Store article in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create hash for deduplication
            content_hash = hashlib.md5(
                (article_data['title'] + article_data.get('content', {}).get('text', '')).encode()
            ).hexdigest()
            
            cursor.execute('''
                INSERT OR IGNORE INTO crawled_articles 
                (url, title, content, published_date, source, hash, crawl_timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article_data['url'],
                article_data['title'],
                json.dumps(article_data.get('content', {})),
                article_data.get('published', ''),
                article_data['source'],
                content_hash,
                article_data['crawl_timestamp'],
                json.dumps(article_data)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing article: {str(e)}")
            
    async def discover_trending_topics(self):
        """Discover trending topics from social media and news aggregators"""
        trending_sources = [
            'https://trends.google.com/trends/trendingsearches/daily/rss',
            'https://www.reddit.com/r/worldnews/hot/.rss',
            'https://www.reddit.com/r/news/hot/.rss',
        ]
        
        trends = []
        for source in trending_sources:
            try:
                headers = await self._get_headers()
                response = self.session.get(source, headers=headers)
                
                if 'reddit' in source:
                    # Parse Reddit RSS
                    feed = feedparser.parse(response.content)
                    for entry in feed.entries:
                        trends.append({
                            'topic': entry.title,
                            'source': 'reddit',
                            'url': entry.link,
                            'score': 1.0
                        })
                else:
                    # Parse Google Trends
                    feed = feedparser.parse(response.content)
                    for entry in feed.entries:
                        trends.append({
                            'topic': entry.title,
                            'source': 'google_trends',
                            'searches': entry.get('ht:approx_traffic', 0),
                            'score': 2.0
                        })
                        
            except Exception as e:
                self.logger.error(f"Error discovering trends from {source}: {str(e)}")
                
        return trends
        
    async def targeted_keyword_crawl(self, keywords):
        """Crawl specific keywords across multiple search engines"""
        results = []
        search_engines = [
            'https://duckduckgo.com/html/?q=',
            'https://www.bing.com/search?q=',
        ]
        
        for keyword in keywords:
            for engine in search_engines:
                try:
                    search_url = engine + keyword.replace(' ', '+')
                    headers = await self._get_headers()
                    
                    response = self.session.get(search_url, headers=headers)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract search results
                    if 'duckduckgo' in engine:
                        links = soup.find_all('a', class_='result__a')
                    else:
                        links = soup.find_all('h2')
                        
                    for link in links[:10]:  # Top 10 results
                        if 'duckduckgo' in engine:
                            url = link.get('href')
                        else:
                            a_tag = link.find('a')
                            url = a_tag.get('href') if a_tag else None
                            
                        if url and url.startswith('http'):
                            content = await self._extract_article_content(url)
                            if content:
                                results.append({
                                    'keyword': keyword,
                                    'url': url,
                                    'content': content,
                                    'search_engine': engine
                                })
                                
                    await asyncio.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    self.logger.error(f"Error in keyword crawl for '{keyword}': {str(e)}")
                    
        return results
        
    async def run_continuous_crawl(self):
        """Run continuous crawling operation"""
        self.logger.info("Starting continuous crawl operation")
        
        while True:
            try:
                # Crawl RSS feeds
                rss_articles = await self.crawl_rss_feeds()
                self.logger.info(f"Crawled {len(rss_articles)} articles from RSS feeds")
                
                # Discover trending topics
                trends = await self.discover_trending_topics()
                self.logger.info(f"Discovered {len(trends)} trending topics")
                
                # Extract keywords from trends
                trending_keywords = [trend['topic'] for trend in trends[:5]]
                
                # Targeted keyword crawling
                keyword_results = await self.targeted_keyword_crawl(trending_keywords)
                self.logger.info(f"Crawled {len(keyword_results)} articles from keyword search")
                
                # Sleep before next cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in continuous crawl: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

if __name__ == "__main__":
    crawler = StealthOpenWebCrawler()
    asyncio.run(crawler.run_continuous_crawl())
