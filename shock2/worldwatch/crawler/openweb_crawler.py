
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import feedparser
from urllib.parse import urljoin, urlparse
import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
import re
import hashlib
from dataclasses import dataclass, asdict
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fake_useragent import UserAgent

@dataclass
class NewsArticle:
    """Represents a scraped news article"""
    url: str
    title: str
    content: str
    author: str
    publish_date: datetime
    source: str
    category: str
    sentiment_score: float
    credibility_score: float
    metadata: Dict[str, Any]
    scrape_timestamp: datetime

class OpenWebCrawler:
    """Advanced real-time news crawler for indexed web sources"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session = None
        self.ua = UserAgent()
        
        # Database for storing articles
        self.db_path = "shock2v/shock2/data/raw/articles.db"
        self._init_database()
        
        # Rate limiting and stealth
        self.request_delays = self.config.get('request_delays', {})
        self.concurrent_limit = self.config.get('concurrent_limit', 10)
        self.rotation_interval = self.config.get('rotation_interval', 300)
        
        # Target sources
        self.news_sources = self._load_news_sources()
        self.rss_feeds = self._load_rss_feeds()
        
        # Anti-detection measures
        self.proxy_pool = self._init_proxy_pool()
        self.header_rotator = self._init_header_rotator()
        
        # Content analysis
        self.content_extractor = self._init_content_extractor()
        self.duplicate_detector = self._init_duplicate_detector()
        
        # Monitoring
        self.crawl_stats = {
            'articles_scraped': 0,
            'sources_active': 0,
            'errors_encountered': 0,
            'duplicate_rate': 0.0,
            'crawl_speed': 0.0
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load crawler configuration"""
        default_config = {
            'user_agents': [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            ],
            'request_timeout': 30,
            'retry_attempts': 3,
            'concurrent_limit': 10,
            'rate_limit': 2.0,
            'respect_robots': False,
            'enable_javascript': True,
            'stealth_mode': True
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except FileNotFoundError:
                pass
                
        return default_config
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('OpenWebCrawler')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('shock2v/shock2/logs/crawler.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _init_database(self):
        """Initialize SQLite database for article storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    author TEXT,
                    publish_date TIMESTAMP,
                    source TEXT,
                    category TEXT,
                    sentiment_score REAL,
                    credibility_score REAL,
                    metadata TEXT,
                    scrape_timestamp TIMESTAMP,
                    content_hash TEXT,
                    manipulation_potential REAL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_url ON articles(url);
                CREATE INDEX IF NOT EXISTS idx_source ON articles(source);
                CREATE INDEX IF NOT EXISTS idx_publish_date ON articles(publish_date);
                CREATE INDEX IF NOT EXISTS idx_manipulation_potential ON articles(manipulation_potential);
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            
    def _load_news_sources(self) -> Dict[str, Dict]:
        """Load comprehensive news source targets"""
        return {
            'major_networks': {
                'cnn.com': {
                    'selectors': {
                        'title': 'h1.headline__text',
                        'content': 'div.zn-body__paragraph',
                        'author': 'span.metadata__byline__author',
                        'date': 'p.update-time'
                    },
                    'priority': 0.9,
                    'credibility': 0.7,
                    'bias_profile': 'liberal_mainstream'
                },
                'foxnews.com': {
                    'selectors': {
                        'title': 'h1.headline',
                        'content': 'div.article-body p',
                        'author': 'div.author-byline',
                        'date': 'time.article-date'
                    },
                    'priority': 0.9,
                    'credibility': 0.6,
                    'bias_profile': 'conservative_mainstream'
                },
                'reuters.com': {
                    'selectors': {
                        'title': 'h1[data-testid="Heading"]',
                        'content': 'div[data-testid="paragraph"] p',
                        'author': 'span[data-testid="Author"]',
                        'date': 'time'
                    },
                    'priority': 0.95,
                    'credibility': 0.9,
                    'bias_profile': 'centrist_factual'
                },
                'breitbart.com': {
                    'selectors': {
                        'title': 'h1.entry-title',
                        'content': 'div.entry-content p',
                        'author': 'address.author',
                        'date': 'time.entry-date'
                    },
                    'priority': 0.8,
                    'credibility': 0.4,
                    'bias_profile': 'far_right'
                },
                'huffpost.com': {
                    'selectors': {
                        'title': 'h1.headline__title',
                        'content': 'div.entry__text p',
                        'author': 'span.author-card__details__name',
                        'date': 'span.timestamp'
                    },
                    'priority': 0.7,
                    'credibility': 0.5,
                    'bias_profile': 'liberal_activist'
                }
            },
            'alternative_media': {
                'zerohedge.com': {
                    'selectors': {
                        'title': 'h1.article-title',
                        'content': 'div.body_content p',
                        'author': 'span.byline',
                        'date': 'span.article-date'
                    },
                    'priority': 0.6,
                    'credibility': 0.3,
                    'bias_profile': 'conspiracy_economic'
                },
                'infowars.com': {
                    'selectors': {
                        'title': 'h1.article-header',
                        'content': 'div.article-content p',
                        'author': 'div.author-info',
                        'date': 'div.article-date'
                    },
                    'priority': 0.5,
                    'credibility': 0.2,
                    'bias_profile': 'conspiracy_extremist'
                }
            },
            'international': {
                'rt.com': {
                    'selectors': {
                        'title': 'h1.article__heading',
                        'content': 'div.article__text p',
                        'author': 'div.article__author',
                        'date': 'time.date'
                    },
                    'priority': 0.7,
                    'credibility': 0.4,
                    'bias_profile': 'russian_state'
                },
                'presstv.ir': {
                    'selectors': {
                        'title': 'h1.title',
                        'content': 'div.body p',
                        'author': 'div.author',
                        'date': 'div.date'
                    },
                    'priority': 0.6,
                    'credibility': 0.3,
                    'bias_profile': 'iranian_state'
                }
            }
        }
        
    def _load_rss_feeds(self) -> List[Dict]:
        """Load RSS feed targets for real-time monitoring"""
        return [
            {'url': 'http://rss.cnn.com/rss/edition.rss', 'source': 'CNN', 'priority': 0.9},
            {'url': 'http://feeds.foxnews.com/foxnews/latest', 'source': 'Fox News', 'priority': 0.9},
            {'url': 'https://feeds.reuters.com/reuters/topNews', 'source': 'Reuters', 'priority': 0.95},
            {'url': 'https://feeds.npr.org/1001/rss.xml', 'source': 'NPR', 'priority': 0.8},
            {'url': 'https://rss.politico.com/politics-news.xml', 'source': 'Politico', 'priority': 0.85},
            {'url': 'https://www.breitbart.com/feed/', 'source': 'Breitbart', 'priority': 0.7},
            {'url': 'https://www.zerohedge.com/fullrss2.xml', 'source': 'ZeroHedge', 'priority': 0.6}
        ]
        
    def _init_proxy_pool(self) -> List[Dict]:
        """Initialize proxy rotation pool"""
        # In production, this would connect to proxy services
        return [
            {'host': '127.0.0.1', 'port': 8080, 'type': 'http'},
            {'host': '127.0.0.1', 'port': 9050, 'type': 'socks5'}  # TOR
        ]
        
    def _init_header_rotator(self) -> Dict:
        """Initialize header rotation system"""
        return {
            'accept_languages': [
                'en-US,en;q=0.9',
                'en-GB,en-US;q=0.9,en;q=0.8',
                'en-US,en;q=0.8,es;q=0.6'
            ],
            'accept_encodings': [
                'gzip, deflate, br',
                'gzip, deflate',
                'identity'
            ],
            'cache_controls': [
                'no-cache',
                'max-age=0',
                'no-store'
            ]
        }
        
    def _init_content_extractor(self):
        """Initialize advanced content extraction system"""
        return {
            'readability_threshold': 0.7,
            'min_content_length': 500,
            'max_content_length': 50000,
            'text_quality_filters': [
                r'click here',
                r'advertisement',
                r'sponsored content',
                r'more stories'
            ]
        }
        
    def _init_duplicate_detector(self):
        """Initialize duplicate content detection"""
        return {
            'similarity_threshold': 0.85,
            'hash_algorithm': 'sha256',
            'content_fingerprints': set()
        }
        
    async def start_continuous_crawl(self):
        """Start continuous real-time crawling operation"""
        self.logger.info("Starting continuous crawl operation")
        
        tasks = []
        
        # RSS feed monitoring
        tasks.append(asyncio.create_task(self._monitor_rss_feeds()))
        
        # Direct source crawling
        tasks.append(asyncio.create_task(self._crawl_direct_sources()))
        
        # Trending topic discovery
        tasks.append(asyncio.create_task(self._discover_trending_topics()))
        
        # Statistics monitoring
        tasks.append(asyncio.create_task(self._monitor_crawl_stats()))
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Crawl operation failed: {str(e)}")
            
    async def _monitor_rss_feeds(self):
        """Monitor RSS feeds for real-time updates"""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    for feed_info in self.rss_feeds:
                        try:
                            await self._process_rss_feed(session, feed_info)
                            await asyncio.sleep(random.uniform(30, 60))
                        except Exception as e:
                            self.logger.error(f"RSS feed error {feed_info['url']}: {str(e)}")
                            
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"RSS monitoring error: {str(e)}")
                await asyncio.sleep(60)
                
    async def _process_rss_feed(self, session: aiohttp.ClientSession, feed_info: Dict):
        """Process individual RSS feed"""
        try:
            headers = self._get_stealth_headers()
            
            async with session.get(feed_info['url'], headers=headers, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:20]:  # Process latest 20 entries
                        article_data = {
                            'url': entry.link,
                            'title': entry.title,
                            'summary': getattr(entry, 'summary', ''),
                            'published': getattr(entry, 'published', ''),
                            'source': feed_info['source'],
                            'priority': feed_info['priority']
                        }
                        
                        # Queue for full article scraping
                        await self._queue_article_scraping(article_data)
                        
        except Exception as e:
            self.logger.error(f"RSS processing error: {str(e)}")
            
    async def _crawl_direct_sources(self):
        """Crawl news sources directly"""
        while True:
            try:
                for category, sources in self.news_sources.items():
                    for domain, source_config in sources.items():
                        try:
                            await self._crawl_news_source(domain, source_config)
                            await asyncio.sleep(random.uniform(60, 120))
                        except Exception as e:
                            self.logger.error(f"Source crawl error {domain}: {str(e)}")
                            
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Direct crawl error: {str(e)}")
                await asyncio.sleep(300)
                
    async def _crawl_news_source(self, domain: str, config: Dict):
        """Crawl specific news source"""
        try:
            base_urls = [
                f"https://{domain}",
                f"https://{domain}/news",
                f"https://{domain}/politics",
                f"https://{domain}/world"
            ]
            
            async with aiohttp.ClientSession() as session:
                for base_url in base_urls:
                    try:
                        headers = self._get_stealth_headers()
                        
                        async with session.get(base_url, headers=headers, timeout=30) as response:
                            if response.status == 200:
                                html = await response.text()
                                article_links = self._extract_article_links(html, base_url)
                                
                                # Process discovered articles
                                for link in article_links[:10]:  # Limit per page
                                    article_data = {
                                        'url': link,
                                        'source': domain,
                                        'config': config,
                                        'discovery_method': 'direct_crawl'
                                    }
                                    await self._queue_article_scraping(article_data)
                                    
                    except Exception as e:
                        self.logger.error(f"URL crawl error {base_url}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Source crawling error {domain}: {str(e)}")
            
    def _extract_article_links(self, html: str, base_url: str) -> List[str]:
        """Extract article links from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            # Common article link patterns
            selectors = [
                'a[href*="/article/"]',
                'a[href*="/news/"]',
                'a[href*="/politics/"]',
                'a[href*="/world/"]',
                'article a',
                '.article-link',
                '.story-link'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_article_url(full_url):
                            links.append(full_url)
                            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Link extraction error: {str(e)}")
            return []
            
    def _is_valid_article_url(self, url: str) -> bool:
        """Validate if URL is likely an article"""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Exclude non-article URLs
            exclude_patterns = [
                '/video/', '/videos/', '/photo/', '/photos/',
                '/tag/', '/tags/', '/category/', '/categories/',
                '/author/', '/authors/', '/search/', '/contact/',
                '.jpg', '.png', '.gif', '.pdf', '.xml'
            ]
            
            for pattern in exclude_patterns:
                if pattern in path:
                    return False
                    
            # Include article indicators
            include_patterns = [
                '/article/', '/news/', '/story/', '/politics/',
                '/world/', '/opinion/', '/analysis/'
            ]
            
            for pattern in include_patterns:
                if pattern in path:
                    return True
                    
            # Check for date patterns (common in news URLs)
            date_pattern = r'/\d{4}/\d{1,2}/\d{1,2}/'
            if re.search(date_pattern, path):
                return True
                
            return len(path) > 10  # Minimum path length
            
        
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup
import feedparser
from urllib.parse import urljoin, urlparse
import json
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
import re
import hashlib
from dataclasses import dataclass, asdict
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fake_useragent import UserAgent

@dataclass
class NewsArticle:
    """Represents a scraped news article"""
    url: str
    title: str
    content: str
    author: str
    publish_date: datetime
    source: str
    category: str
    sentiment_score: float
    credibility_score: float
    metadata: Dict[str, Any]
    scrape_timestamp: datetime

class OpenWebCrawler:
    """Advanced real-time news crawler for indexed web sources"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.session = None
        self.ua = UserAgent()
        
        # Database for storing articles
        self.db_path = "shock2v/shock2/data/raw/articles.db"
        self._init_database()
        
        # Rate limiting and stealth
        self.request_delays = self.config.get('request_delays', {})
        self.concurrent_limit = self.config.get('concurrent_limit', 10)
        self.rotation_interval = self.config.get('rotation_interval', 300)
        
        # Target sources
        self.news_sources = self._load_news_sources()
        self.rss_feeds = self._load_rss_feeds()
        
        # Anti-detection measures
        self.proxy_pool = self._init_proxy_pool()
        self.header_rotator = self._init_header_rotator()
        
        # Content analysis
        self.content_extractor = self._init_content_extractor()
        self.duplicate_detector = self._init_duplicate_detector()
        
        # Monitoring
        self.crawl_stats = {
            'articles_scraped': 0,
            'sources_active': 0,
            'errors_encountered': 0,
            'duplicate_rate': 0.0,
            'crawl_speed': 0.0
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load crawler configuration"""
        default_config = {
            'user_agents': [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            ],
            'request_timeout': 30,
            'retry_attempts': 3,
            'concurrent_limit': 10,
            'rate_limit': 2.0,
            'respect_robots': False,
            'enable_javascript': True,
            'stealth_mode': True
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except FileNotFoundError:
                pass
                
        return default_config
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('OpenWebCrawler')
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler('shock2v/shock2/logs/crawler.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def _init_database(self):
        """Initialize SQLite database for article storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    content TEXT,
                    author TEXT,
                    publish_date TIMESTAMP,
                    source TEXT,
                    category TEXT,
                    sentiment_score REAL,
                    credibility_score REAL,
                    metadata TEXT,
                    scrape_timestamp TIMESTAMP,
                    content_hash TEXT,
                    manipulation_potential REAL
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_url ON articles(url);
                CREATE INDEX IF NOT EXISTS idx_source ON articles(source);
                CREATE INDEX IF NOT EXISTS idx_publish_date ON articles(publish_date);
                CREATE INDEX IF NOT EXISTS idx_manipulation_potential ON articles(manipulation_potential);
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            
    def _load_news_sources(self) -> Dict[str, Dict]:
        """Load comprehensive news source targets"""
        return {
            'major_networks': {
                'cnn.com': {
                    'selectors': {
                        'title': 'h1.headline__text',
                        'content': 'div.zn-body__paragraph',
                        'author': 'span.metadata__byline__author',
                        'date': 'p.update-time'
                    },
                    'priority': 0.9,
                    'credibility': 0.7,
                    'bias_profile': 'liberal_mainstream'
                },
                'foxnews.com': {
                    'selectors': {
                        'title': 'h1.headline',
                        'content': 'div.article-body p',
                        'author': 'div.author-byline',
                        'date': 'time.article-date'
                    },
                    'priority': 0.9,
                    'credibility': 0.6,
                    'bias_profile': 'conservative_mainstream'
                },
                'reuters.com': {
                    'selectors': {
                        'title': 'h1[data-testid="Heading"]',
                        'content': 'div[data-testid="paragraph"] p',
                        'author': 'span[data-testid="Author"]',
                        'date': 'time'
                    },
                    'priority': 0.95,
                    'credibility': 0.9,
                    'bias_profile': 'centrist_factual'
                },
                'breitbart.com': {
                    'selectors': {
                        'title': 'h1.entry-title',
                        'content': 'div.entry-content p',
                        'author': 'address.author',
                        'date': 'time.entry-date'
                    },
                    'priority': 0.8,
                    'credibility': 0.4,
                    'bias_profile': 'far_right'
                },
                'huffpost.com': {
                    'selectors': {
                        'title': 'h1.headline__title',
                        'content': 'div.entry__text p',
                        'author': 'span.author-card__details__name',
                        'date': 'span.timestamp'
                    },
                    'priority': 0.7,
                    'credibility': 0.5,
                    'bias_profile': 'liberal_activist'
                }
            },
            'alternative_media': {
                'zerohedge.com': {
                    'selectors': {
                        'title': 'h1.article-title',
                        'content': 'div.body_content p',
                        'author': 'span.byline',
                        'date': 'span.article-date'
                    },
                    'priority': 0.6,
                    'credibility': 0.3,
                    'bias_profile': 'conspiracy_economic'
                },
                'infowars.com': {
                    'selectors': {
                        'title': 'h1.article-header',
                        'content': 'div.article-content p',
                        'author': 'div.author-info',
                        'date': 'div.article-date'
                    },
                    'priority': 0.5,
                    'credibility': 0.2,
                    'bias_profile': 'conspiracy_extremist'
                }
            },
            'international': {
                'rt.com': {
                    'selectors': {
                        'title': 'h1.article__heading',
                        'content': 'div.article__text p',
                        'author': 'div.article__author',
                        'date': 'time.date'
                    },
                    'priority': 0.7,
                    'credibility': 0.4,
                    'bias_profile': 'russian_state'
                },
                'presstv.ir': {
                    'selectors': {
                        'title': 'h1.title',
                        'content': 'div.body p',
                        'author': 'div.author',
                        'date': 'div.date'
                    },
                    'priority': 0.6,
                    'credibility': 0.3,
                    'bias_profile': 'iranian_state'
                }
            }
        }
        
    def _load_rss_feeds(self) -> List[Dict]:
        """Load RSS feed targets for real-time monitoring"""
        return [
            {'url': 'http://rss.cnn.com/rss/edition.rss', 'source': 'CNN', 'priority': 0.9},
            {'url': 'http://feeds.foxnews.com/foxnews/latest', 'source': 'Fox News', 'priority': 0.9},
            {'url': 'https://feeds.reuters.com/reuters/topNews', 'source': 'Reuters', 'priority': 0.95},
            {'url': 'https://feeds.npr.org/1001/rss.xml', 'source': 'NPR', 'priority': 0.8},
            {'url': 'https://rss.politico.com/politics-news.xml', 'source': 'Politico', 'priority': 0.85},
            {'url': 'https://www.breitbart.com/feed/', 'source': 'Breitbart', 'priority': 0.7},
            {'url': 'https://www.zerohedge.com/fullrss2.xml', 'source': 'ZeroHedge', 'priority': 0.6}
        ]
        
    def _init_proxy_pool(self) -> List[Dict]:
        """Initialize proxy rotation pool"""
        # In production, this would connect to proxy services
        return [
            {'host': '127.0.0.1', 'port': 8080, 'type': 'http'},
            {'host': '127.0.0.1', 'port': 9050, 'type': 'socks5'}  # TOR
        ]
        
    def _init_header_rotator(self) -> Dict:
        """Initialize header rotation system"""
        return {
            'accept_languages': [
                'en-US,en;q=0.9',
                'en-GB,en-US;q=0.9,en;q=0.8',
                'en-US,en;q=0.8,es;q=0.6'
            ],
            'accept_encodings': [
                'gzip, deflate, br',
                'gzip, deflate',
                'identity'
            ],
            'cache_controls': [
                'no-cache',
                'max-age=0',
                'no-store'
            ]
        }
        
    def _init_content_extractor(self):
        """Initialize advanced content extraction system"""
        return {
            'readability_threshold': 0.7,
            'min_content_length': 500,
            'max_content_length': 50000,
            'text_quality_filters': [
                r'click here',
                r'advertisement',
                r'sponsored content',
                r'more stories'
            ]
        }
        
    def _init_duplicate_detector(self):
        """Initialize duplicate content detection"""
        return {
            'similarity_threshold': 0.85,
            'hash_algorithm': 'sha256',
            'content_fingerprints': set()
        }
        
    async def start_continuous_crawl(self):
        """Start continuous real-time crawling operation"""
        self.logger.info("Starting continuous crawl operation")
        
        tasks = []
        
        # RSS feed monitoring
        tasks.append(asyncio.create_task(self._monitor_rss_feeds()))
        
        # Direct source crawling
        tasks.append(asyncio.create_task(self._crawl_direct_sources()))
        
        # Trending topic discovery
        tasks.append(asyncio.create_task(self._discover_trending_topics()))
        
        # Statistics monitoring
        tasks.append(asyncio.create_task(self._monitor_crawl_stats()))
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Crawl operation failed: {str(e)}")
            
    async def _monitor_rss_feeds(self):
        """Monitor RSS feeds for real-time updates"""
        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    for feed_info in self.rss_feeds:
                        try:
                            await self._process_rss_feed(session, feed_info)
                            await asyncio.sleep(random.uniform(30, 60))
                        except Exception as e:
                            self.logger.error(f"RSS feed error {feed_info['url']}: {str(e)}")
                            
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"RSS monitoring error: {str(e)}")
                await asyncio.sleep(60)
                
    async def _process_rss_feed(self, session: aiohttp.ClientSession, feed_info: Dict):
        """Process individual RSS feed"""
        try:
            headers = self._get_stealth_headers()
            
            async with session.get(feed_info['url'], headers=headers, timeout=30) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:20]:  # Process latest 20 entries
                        article_data = {
                            'url': entry.link,
                            'title': entry.title,
                            'summary': getattr(entry, 'summary', ''),
                            'published': getattr(entry, 'published', ''),
                            'source': feed_info['source'],
                            'priority': feed_info['priority']
                        }
                        
                        # Queue for full article scraping
                        await self._queue_article_scraping(article_data)
                        
        except Exception as e:
            self.logger.error(f"RSS processing error: {str(e)}")
            
    async def _crawl_direct_sources(self):
        """Crawl news sources directly"""
        while True:
            try:
                for category, sources in self.news_sources.items():
                    for domain, source_config in sources.items():
                        try:
                            await self._crawl_news_source(domain, source_config)
                            await asyncio.sleep(random.uniform(60, 120))
                        except Exception as e:
                            self.logger.error(f"Source crawl error {domain}: {str(e)}")
                            
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                self.logger.error(f"Direct crawl error: {str(e)}")
                await asyncio.sleep(300)
                
    async def _crawl_news_source(self, domain: str, config: Dict):
        """Crawl specific news source"""
        try:
            base_urls = [
                f"https://{domain}",
                f"https://{domain}/news",
                f"https://{domain}/politics",
                f"https://{domain}/world"
            ]
            
            async with aiohttp.ClientSession() as session:
                for base_url in base_urls:
                    try:
                        headers = self._get_stealth_headers()
                        
                        async with session.get(base_url, headers=headers, timeout=30) as response:
                            if response.status == 200:
                                html = await response.text()
                                article_links = self._extract_article_links(html, base_url)
                                
                                # Process discovered articles
                                for link in article_links[:10]:  # Limit per page
                                    article_data = {
                                        'url': link,
                                        'source': domain,
                                        'config': config,
                                        'discovery_method': 'direct_crawl'
                                    }
                                    await self._queue_article_scraping(article_data)
                                    
                    except Exception as e:
                        self.logger.error(f"URL crawl error {base_url}: {str(e)}")
                        
        except Exception as e:
            self.logger.error(f"Source crawling error {domain}: {str(e)}")
            
    def _extract_article_links(self, html: str, base_url: str) -> List[str]:
        """Extract article links from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            # Common article link patterns
            selectors = [
                'a[href*="/article/"]',
                'a[href*="/news/"]',
                'a[href*="/politics/"]',
                'a[href*="/world/"]',
                'article a',
                '.article-link',
                '.story-link'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    href = element.get('href')
                    if href:
                        full_url = urljoin(base_url, href)
                        if self._is_valid_article_url(full_url):
                            links.append(full_url)
                            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Link extraction error: {str(e)}")
            return []
            
    def _is_valid_article_url(self, url: str) -> bool:
        """Validate if URL is likely an article"""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Exclude non-article URLs
            exclude_patterns = [
                '/video/', '/videos/', '/photo/', '/photos/',
                '/tag/', '/tags/', '/category/', '/categories/',
                '/author/', '/authors/', '/search/', '/contact/',
                '.jpg', '.png', '.gif', '.pdf', '.xml'
            ]
            
            for pattern in exclude_patterns:
                if pattern in path:
                    return False
                    
            # Include article indicators
            include_patterns = [
                '/article/', '/news/', '/story/', '/politics/',
                '/world/', '/opinion/', '/analysis/'
            ]
            
            for pattern in include_patterns:
                if pattern in path:
                    return True
                    
            # Check for date patterns (common in news URLs)
            date_pattern = r'/\d{4}/\d{1,2}/\d{1,2}/'
            if re.search(date_pattern, path):
                return True
                
            return len(path) > 10  # Minimum path length
            
        except Exception:
            return False
            
    async def _queue_article_scraping(self, article_data: Dict):
        """Queue article for detailed scraping"""
        try:
            # Check if already processed
            if await self._is_article_processed(article_data['url']):
                return
                
            # Add to processing queue
            await self._scrape_full_article(article_data)
            
        except Exception as e:
            self.logger.error(f"Article queueing error: {str(e)}")
            
    async def _is_article_processed(self, url: str) -> bool:
        """Check if article already processed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
            result = cursor.fetchone()
            
            conn.close()
            return result is not None
            
        except Exception:
            return False
            
    async def _scrape_full_article(self, article_data: Dict):
        """Scrape full article content"""
        try:
            url = article_data['url']
            headers = self._get_stealth_headers()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        html = await response.text()
                        article = self._extract_article_content(html, article_data)
                        
                        if article and self._validate_article_quality(article):
                            await self._store_article(article)
                            self.crawl_stats['articles_scraped'] += 1
                            
        except Exception as e:
            self.logger.error(f"Article scraping error {article_data.get('url')}: {str(e)}")
            self.crawl_stats['errors_encountered'] += 1
            
    def _extract_article_content(self, html: str, article_data: Dict) -> Optional[NewsArticle]:
        """Extract structured content from article HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Get source-specific selectors if available
            config = article_data.get('config', {})
            selectors = config.get('selectors', {})
            
            # Extract title
            title = self._extract_title(soup, selectors.get('title'))
            if not title:
                return None
                
            # Extract content
            content = self._extract_content(soup, selectors.get('content'))
            if not content or len(content) < 500:
                return None
                
            # Extract metadata
            author = self._extract_author(soup, selectors.get('author'))
            publish_date = self._extract_publish_date(soup, selectors.get('date'))
            
            # Calculate scores
            sentiment_score = self._calculate_sentiment_score(content)
            credibility_score = config.get('credibility', 0.5)
            manipulation_potential = self._calculate_manipulation_potential(content, title)
            
            article = NewsArticle(
                url=article_data['url'],
                title=title,
                content=content,
                author=author or 'Unknown',
                publish_date=publish_date or datetime.now(),
                source=article_data.get('source', 'Unknown'),
                category=self._classify_category(title, content),
                sentiment_score=sentiment_score,
                credibility_score=credibility_score,
                metadata={
                    'manipulation_potential': manipulation_potential,
                    'content_length': len(content),
                    'discovery_method': article_data.get('discovery_method', 'unknown')
                },
                scrape_timestamp=datetime.now()
            )
            
            return article
            
        except Exception as e:
            self.logger.error(f"Content extraction error: {str(e)}")
            return None
            
    def _extract_title(self, soup: BeautifulSoup, selector: str = None) -> str:
        """Extract article title"""
        selectors = [
            selector,
            'h1',
            '.headline',
            '.article-title',
            'title'
        ] if selector else ['h1', '.headline', '.article-title', 'title']
        
        for sel in selectors:
            if sel:
                element = soup.select_one(sel)
                if element:
                    return element.get_text().strip()
        return ""
        
    def _extract_content(self, soup: BeautifulSoup, selector: str = None) -> str:
        """Extract article content"""
        if selector:
            elements = soup.select(selector)
            if elements:
                return '\n'.join([elem.get_text().strip() for elem in elements])
        
        # Fallback content extraction
        content_selectors = [
            '.article-body',
            '.content',
            '.entry-content',
            'article p',
            '.story-body p'
        ]
        
        for sel in content_selectors:
            elements = soup.select(sel)
            if elements:
                content = '\n'.join([elem.get_text().strip() for elem in elements])
                if len(content) > 500:
                    return content
                    
        return ""
        
    def _extract_author(self, soup: BeautifulSoup, selector: str = None) -> str:
        """Extract article author"""
        selectors = [
            selector,
            '.author',
            '.byline',
            '[rel="author"]',
            '.article-author'
        ] if selector else ['.author', '.byline', '[rel="author"]', '.article-author']
        
        for sel in selectors:
            if sel:
                element = soup.select_one(sel)
                if element:
                    return element.get_text().strip()
        return ""
        
    def _extract_publish_date(self, soup: BeautifulSoup, selector: str = None) -> Optional[datetime]:
        """Extract publish date"""
        try:
            selectors = [
                selector,
                'time',
                '.date',
                '.publish-date',
                '[datetime]'
            ] if selector else ['time', '.date', '.publish-date', '[datetime]']
            
            for sel in selectors:
                if sel:
                    element = soup.select_one(sel)
                    if element:
                        date_str = element.get('datetime') or element.get_text().strip()
                        # Parse date (simplified)
                        return datetime.now()  # Placeholder
                        
        except Exception:
            pass
        return None
        
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate sentiment score"""
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'success']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'failure']
        
        content_lower = content.lower()
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        
        if pos_count + neg_count == 0:
            return 0.5
        return pos_count / (pos_count + neg_count)
        
    def _calculate_manipulation_potential(self, content: str, title: str) -> float:
        """Calculate manipulation potential score"""
        manipulation_indicators = [
            'shocking', 'devastating', 'unprecedented', 'breaking',
            'you won\'t believe', 'exposed', 'revealed', 'scandal'
        ]
        
        text = (title + ' ' + content).lower()
        score = sum(0.1 for indicator in manipulation_indicators if indicator in text)
        
        return min(1.0, score)
        
    def _classify_category(self, title: str, content: str) -> str:
        """Classify article category"""
        text = (title + ' ' + content).lower()
        
        categories = {
            'politics': ['election', 'government', 'politician', 'congress', 'senate'],
            'economics': ['economy', 'market', 'financial', 'trade', 'inflation'],
            'international': ['foreign', 'international', 'global', 'world'],
            'technology': ['tech', 'digital', 'cyber', 'ai', 'internet'],
            'health': ['health', 'medical', 'virus', 'pandemic', 'disease']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
                
        return 'general'
        
    def _validate_article_quality(self, article: NewsArticle) -> bool:
        """Validate article meets quality standards"""
        if len(article.content) < 500:
            return False
        if not article.title:
            return False
        if article.credibility_score < 0.1:
            return False
        return True
        
    async def _store_article(self, article: NewsArticle):
        """Store article in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            content_hash = hashlib.sha256(article.content.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO articles (
                    url, title, content, author, publish_date, source,
                    category, sentiment_score, credibility_score, metadata,
                    scrape_timestamp, content_hash, manipulation_potential
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.url, article.title, article.content, article.author,
                article.publish_date, article.source, article.category,
                article.sentiment_score, article.credibility_score,
                json.dumps(article.metadata), article.scrape_timestamp,
                content_hash, article.metadata.get('manipulation_potential', 0.0)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored article: {article.title[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Article storage error: {str(e)}")
            
    def _get_stealth_headers(self) -> Dict[str, str]:
        """Generate stealth headers for requests"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': random.choice(self.header_rotator['accept_languages']),
            'Accept-Encoding': random.choice(self.header_rotator['accept_encodings']),
            'Cache-Control': random.choice(self.header_rotator['cache_controls']),
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    async def _discover_trending_topics(self):
        """Discover trending topics for targeted crawling"""
        while True:
            try:
                # Monitor social media trends, search trends, etc.
                await asyncio.sleep(3600)  # Check hourly
            except Exception as e:
                self.logger.error(f"Trend discovery error: {str(e)}")
                
    async def _monitor_crawl_stats(self):
        """Monitor crawling statistics"""
        while True:
            try:
                self.logger.info(f"Crawl Stats: {self.crawl_stats}")
                await asyncio.sleep(600)  # Log every 10 minutes
            except Exception as e:
                self.logger.error(f"Stats monitoring error: {str(e)}")
                
    def get_crawl_statistics(self) -> Dict:
        """Get current crawling statistics"""
        return self.crawl_stats.copy()
        
    async def shutdown(self):
        """Graceful shutdown of crawler"""
        self.logger.info("Shutting down crawler...")
        if self.session:
            await self.session.close()
ssify_category(title, content),
                sentiment_score=sentiment_score,
                credibility_score=credibility_score,
                metadata={
                    'manipulation_potential': manipulation_potential,
                    'content_length': len(content),
                    'discovery_method': article_data.get('discovery_method', 'unknown')
                },
                scrape_timestamp=datetime.now()
            )
            
            return article
            
        except Exception as e:
            self.logger.error(f"Content extraction error: {str(e)}")
            return None
            
    def _extract_title(self, soup: BeautifulSoup, selector: str = None) -> str:
        """Extract article title"""
        selectors = [
            selector,
            'h1',
            '.headline',
            '.article-title',
            'title'
        ] if selector else ['h1', '.headline', '.article-title', 'title']
        
        for sel in selectors:
            if sel:
                element = soup.select_one(sel)
                if element:
                    return element.get_text().strip()
        return ""
        
    def _extract_content(self, soup: BeautifulSoup, selector: str = None) -> str:
        """Extract article content"""
        if selector:
            elements = soup.select(selector)
            if elements:
                return '\n'.join([elem.get_text().strip() for elem in elements])
        
        # Fallback content extraction
        content_selectors = [
            '.article-body',
            '.content',
            '.entry-content',
            'article p',
            '.story-body p'
        ]
        
        for sel in content_selectors:
            elements = soup.select(sel)
            if elements:
                content = '\n'.join([elem.get_text().strip() for elem in elements])
                if len(content) > 500:
                    return content
                    
        return ""
        
    def _extract_author(self, soup: BeautifulSoup, selector: str = None) -> str:
        """Extract article author"""
        selectors = [
            selector,
            '.author',
            '.byline',
            '[rel="author"]',
            '.article-author'
        ] if selector else ['.author', '.byline', '[rel="author"]', '.article-author']
        
        for sel in selectors:
            if sel:
                element = soup.select_one(sel)
                if element:
                    return element.get_text().strip()
        return ""
        
    def _extract_publish_date(self, soup: BeautifulSoup, selector: str = None) -> Optional[datetime]:
        """Extract publish date"""
        try:
            selectors = [
                selector,
                'time',
                '.date',
                '.publish-date',
                '[datetime]'
            ] if selector else ['time', '.date', '.publish-date', '[datetime]']
            
            for sel in selectors:
                if sel:
                    element = soup.select_one(sel)
                    if element:
                        date_str = element.get('datetime') or element.get_text().strip()
                        # Parse date (simplified)
                        return datetime.now()  # Placeholder
                        
        except Exception:
            pass
        return None
        
    def _calculate_sentiment_score(self, content: str) -> float:
        """Calculate sentiment score"""
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'success']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'failure']
        
        content_lower = content.lower()
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        
        if pos_count + neg_count == 0:
            return 0.5
        return pos_count / (pos_count + neg_count)
        
    def _calculate_manipulation_potential(self, content: str, title: str) -> float:
        """Calculate manipulation potential score"""
        manipulation_indicators = [
            'shocking', 'devastating', 'unprecedented', 'breaking',
            'you won\'t believe', 'exposed', 'revealed', 'scandal'
        ]
        
        text = (title + ' ' + content).lower()
        score = sum(0.1 for indicator in manipulation_indicators if indicator in text)
        
        return min(1.0, score)
        
    def _classify_category(self, title: str, content: str) -> str:
        """Classify article category"""
        text = (title + ' ' + content).lower()
        
        categories = {
            'politics': ['election', 'government', 'politician', 'congress', 'senate'],
            'economics': ['economy', 'market', 'financial', 'trade', 'inflation'],
            'international': ['foreign', 'international', 'global', 'world'],
            'technology': ['tech', 'digital', 'cyber', 'ai', 'internet'],
            'health': ['health', 'medical', 'virus', 'pandemic', 'disease']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text for keyword in keywords):
                return category
                
        return 'general'
        
    def _validate_article_quality(self, article: NewsArticle) -> bool:
        """Validate article meets quality standards"""
        if len(article.content) < 500:
            return False
        if not article.title:
            return False
        if article.credibility_score < 0.1:
            return False
        return True
        
    async def _store_article(self, article: NewsArticle):
        """Store article in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            content_hash = hashlib.sha256(article.content.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO articles (
                    url, title, content, author, publish_date, source,
                    category, sentiment_score, credibility_score, metadata,
                    scrape_timestamp, content_hash, manipulation_potential
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article.url, article.title, article.content, article.author,
                article.publish_date, article.source, article.category,
                article.sentiment_score, article.credibility_score,
                json.dumps(article.metadata), article.scrape_timestamp,
                content_hash, article.metadata.get('manipulation_potential', 0.0)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored article: {article.title[:50]}...")
            
        except Exception as e:
            self.logger.error(f"Article storage error: {str(e)}")
            
    def _get_stealth_headers(self) -> Dict[str, str]:
        """Generate stealth headers for requests"""
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': random.choice(self.header_rotator['accept_languages']),
            'Accept-Encoding': random.choice(self.header_rotator['accept_encodings']),
            'Cache-Control': random.choice(self.header_rotator['cache_controls']),
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    async def _discover_trending_topics(self):
        """Discover trending topics for targeted crawling"""
        while True:
            try:
                # Monitor social media trends, search trends, etc.
                await asyncio.sleep(3600)  # Check hourly
            except Exception as e:
                self.logger.error(f"Trend discovery error: {str(e)}")
                
    async def _monitor_crawl_stats(self):
        """Monitor crawling statistics"""
        while True:
            try:
                self.logger.info(f"Crawl Stats: {self.crawl_stats}")
                await asyncio.sleep(600)  # Log every 10 minutes
            except Exception as e:
                self.logger.error(f"Stats monitoring error: {str(e)}")
                
    def get_crawl_statistics(self) -> Dict:
        """Get current crawling statistics"""
        return self.crawl_stats.copy()
        
    async def shutdown(self):
        """Graceful shutdown of crawler"""
        self.logger.info("Shutting down crawler...")
        if self.session:
            await self.session.close()
