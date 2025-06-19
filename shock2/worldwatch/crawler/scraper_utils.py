
import asyncio
import aiohttp
import requests
from bs4 import BeautifulSoup, Comment
import lxml.html
from lxml import etree
import json
import re
import time
import random
import hashlib
from urllib.parse import urljoin, urlparse, parse_qs
from fake_useragent import UserAgent
import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import undetected_chromedriver as uc
import cloudscraper
import execjs
import js2py
from newspaper import Article
import trafilatura
import readability
from goose3 import Goose
import newspaper
import dateutil.parser
from datetime import datetime, timedelta
import logging
import sqlite3
import threading
from queue import Queue
import base64
from cryptography.fernet import Fernet
import pickle
import gzip
import zlib
from typing import Dict, List, Optional, Tuple, Union
import cssselect
import html2text
import nltk
from textstat import flesch_reading_ease
import langdetect
import chardet

class AdvancedHTMLParser:
    """Advanced HTML parsing with multiple extraction strategies"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.logger = self._setup_logger()
        self.goose = Goose()
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = True
        self.html2text.ignore_images = True
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/scraper_utils.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def extract_content_multiple_methods(self, html_content: str, url: str = "") -> Dict:
        """Extract content using multiple parsing methods and return best result"""
        results = {}
        
        try:
            # Method 1: Newspaper3k
            try:
                article = Article(url)
                article.set_html(html_content)
                article.parse()
                results['newspaper'] = {
                    'title': article.title,
                    'text': article.text,
                    'authors': article.authors,
                    'publish_date': article.publish_date,
                    'keywords': article.keywords,
                    'summary': article.summary,
                    'top_image': article.top_image,
                    'score': len(article.text) / 100  # Simple scoring
                }
            except Exception as e:
                self.logger.warning(f"Newspaper extraction failed: {str(e)}")
                results['newspaper'] = {'score': 0}
                
            # Method 2: Trafilatura
            try:
                extracted = trafilatura.extract(html_content, include_comments=False, 
                                              include_tables=True, include_formatting=True)
                if extracted:
                    results['trafilatura'] = {
                        'text': extracted,
                        'title': self._extract_title_from_html(html_content),
                        'score': len(extracted) / 100
                    }
                else:
                    results['trafilatura'] = {'score': 0}
            except Exception as e:
                self.logger.warning(f"Trafilatura extraction failed: {str(e)}")
                results['trafilatura'] = {'score': 0}
                
            # Method 3: Readability
            try:
                doc = readability.Document(html_content)
                results['readability'] = {
                    'title': doc.title(),
                    'text': self.html2text.handle(doc.summary()),
                    'score': len(doc.summary()) / 100
                }
            except Exception as e:
                self.logger.warning(f"Readability extraction failed: {str(e)}")
                results['readability'] = {'score': 0}
                
            # Method 4: Goose3
            try:
                article = self.goose.extract(raw_html=html_content, final_url=url)
                results['goose'] = {
                    'title': article.title,
                    'text': article.cleaned_text,
                    'authors': [article.authors] if article.authors else [],
                    'publish_date': article.publish_date,
                    'top_image': article.top_image.src if article.top_image else None,
                    'score': len(article.cleaned_text) / 100
                }
            except Exception as e:
                self.logger.warning(f"Goose extraction failed: {str(e)}")
                results['goose'] = {'score': 0}
                
            # Method 5: Custom BeautifulSoup parsing
            try:
                custom_result = self._custom_bs4_extraction(html_content)
                results['custom'] = custom_result
            except Exception as e:
                self.logger.warning(f"Custom extraction failed: {str(e)}")
                results['custom'] = {'score': 0}
                
            # Select best result
            best_method = max(results.keys(), key=lambda k: results[k].get('score', 0))
            best_result = results[best_method]
            best_result['extraction_method'] = best_method
            best_result['all_results'] = results
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Content extraction failed: {str(e)}")
            return {'error': 'extraction_failed', 'all_results': results}
            
    def _custom_bs4_extraction(self, html_content: str) -> Dict:
        """Custom BeautifulSoup-based content extraction"""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()
            
        # Extract title
        title = ""
        title_selectors = ['h1', 'title', '.title', '.headline', '[class*="title"]', '[class*="headline"]']
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                title = title_elem.get_text().strip()
                break
                
        # Extract main content
        content_selectors = [
            'article', '.article-content', '.post-content', '.entry-content',
            '.story-body', '.article-body', '.content', '.main-content',
            'main', '[role="main"]', '.text', '.body'
        ]
        
        content = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator=' ', strip=True)
                if len(content) > 200:  # Minimum content length
                    break
                    
        # If no content found, try paragraph aggregation
        if len(content) < 200:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])
            
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        return {
            'title': title,
            'text': content,
            'metadata': metadata,
            'score': len(content) / 100
        }
        
    def _extract_title_from_html(self, html_content: str) -> str:
        """Extract title from HTML"""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Try multiple title sources
        title_sources = [
            soup.find('title'),
            soup.find('h1'),
            soup.find(attrs={'property': 'og:title'}),
            soup.find(attrs={'name': 'twitter:title'}),
            soup.find(class_=re.compile(r'title|headline', re.I))
        ]
        
        for source in title_sources:
            if source:
                title = source.get('content', '') or source.get_text('')
                if title.strip():
                    return title.strip()
                    
        return "No Title Found"
        
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract comprehensive metadata from HTML"""
        metadata = {}
        
        # Open Graph metadata
        og_tags = soup.find_all(attrs={'property': re.compile(r'^og:')})
        for tag in og_tags:
            property_name = tag.get('property', '').replace('og:', '')
            content = tag.get('content', '')
            if property_name and content:
                metadata[f'og_{property_name}'] = content
                
        # Twitter Card metadata
        twitter_tags = soup.find_all(attrs={'name': re.compile(r'^twitter:')})
        for tag in twitter_tags:
            name = tag.get('name', '').replace('twitter:', '')
            content = tag.get('content', '')
            if name and content:
                metadata[f'twitter_{name}'] = content
                
        # Standard meta tags
        meta_tags = soup.find_all('meta')
        for tag in meta_tags:
            name = tag.get('name', '') or tag.get('property', '')
            content = tag.get('content', '')
            if name and content:
                metadata[name] = content
                
        # Extract publish date
        date_selectors = [
            '[datetime]', '[data-published]', '[data-date]',
            '.date', '.published', '.timestamp', 'time'
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem:
                date_text = (date_elem.get('datetime') or 
                           date_elem.get('data-published') or 
                           date_elem.get('data-date') or 
                           date_elem.get_text())
                try:
                    parsed_date = dateutil.parser.parse(date_text)
                    metadata['publish_date'] = parsed_date.isoformat()
                    break
                except:
                    continue
                    
        return metadata

class UAXExtractor:
    """User Agent eXtraction - Advanced user agent and fingerprint spoofing"""
    
    def __init__(self):
        self.ua = UserAgent()
        self.session_cache = {}
        self.fingerprint_cache = {}
        
    def get_stealth_headers(self, target_domain: str = "") -> Dict[str, str]:
        """Generate stealth headers optimized for target domain"""
        base_headers = {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Charset': 'utf-8, iso-8859-1;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        # Domain-specific header optimization
        if 'facebook' in target_domain:
            base_headers.update({
                'Sec-Ch-Ua': '"Google Chrome";v="119", "Chromium";v="119", "Not?A_Brand";v="24"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"Windows"'
            })
        elif 'twitter' in target_domain or 'x.com' in target_domain:
            base_headers.update({
                'X-Twitter-Active-User': 'yes',
                'X-Twitter-Client-Language': 'en'
            })
        elif any(news_site in target_domain for news_site in ['cnn', 'bbc', 'reuters', 'nytimes']):
            base_headers.update({
                'DNT': '1',
                'Sec-GPC': '1'
            })
            
        return base_headers
        
    def rotate_user_agent(self, platform: str = 'random') -> str:
        """Rotate user agent with specific platform targeting"""
        if platform == 'mobile':
            return self.ua.random_mobile
        elif platform == 'desktop':
            return self.ua.random_desktop
        elif platform == 'chrome':
            return self.ua.chrome
        elif platform == 'firefox':
            return self.ua.firefox
        else:
            return self.ua.random
            
    def generate_browser_fingerprint(self) -> Dict:
        """Generate realistic browser fingerprint"""
        fingerprints = [
            {
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'screen_resolution': '1920x1080',
                'timezone': 'America/New_York',
                'language': 'en-US',
                'platform': 'Win32',
                'webgl_vendor': 'Google Inc. (Intel)',
                'webgl_renderer': 'ANGLE (Intel, Intel(R) HD Graphics 630 Direct3D11 vs_5_0 ps_5_0, D3D11-27.20.100.8935)'
            },
            {
                'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
                'screen_resolution': '1440x900',
                'timezone': 'America/Los_Angeles',
                'language': 'en-US',
                'platform': 'MacIntel',
                'webgl_vendor': 'Apple Inc.',
                'webgl_renderer': 'Apple GPU'
            }
        ]
        
        return random.choice(fingerprints)

class JSBypassEngine:
    """JavaScript bypass and execution engine"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        self.js_context = execjs.get()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/js_bypass.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def bypass_cloudflare(self, url: str, headers: Dict = None) -> requests.Response:
        """Bypass Cloudflare protection using cloudscraper"""
        try:
            scraper = cloudscraper.create_scraper(
                browser={
                    'browser': 'chrome',
                    'platform': 'windows',
                    'desktop': True
                }
            )
            
            if headers:
                scraper.headers.update(headers)
                
            response = scraper.get(url, timeout=30)
            return response
            
        except Exception as e:
            self.logger.error(f"Cloudflare bypass failed for {url}: {str(e)}")
            return None
            
    def execute_javascript(self, js_code: str, context_vars: Dict = None) -> any:
        """Execute JavaScript code safely"""
        try:
            if context_vars:
                # Inject context variables
                for var_name, var_value in context_vars.items():
                    js_code = f"var {var_name} = {json.dumps(var_value)};\n" + js_code
                    
            result = self.js_context.eval(js_code)
            return result
            
        except Exception as e:
            self.logger.error(f"JavaScript execution failed: {str(e)}")
            return None
            
    def extract_dynamic_content(self, url: str, wait_for_element: str = None) -> str:
        """Extract content from JavaScript-heavy pages using undetected Chrome"""
        try:
            options = uc.ChromeOptions()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            
            driver = uc.Chrome(options=options, version_main=119)
            
            # Execute stealth script
            stealth_script = """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
            """
            
            driver.execute_script(stealth_script)
            driver.get(url)
            
            # Wait for specific element if provided
            if wait_for_element:
                WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                )
            else:
                time.sleep(5)  # General wait for page load
                
            # Extract page source
            page_source = driver.page_source
            driver.quit()
            
            return page_source
            
        except Exception as e:
            self.logger.error(f"Dynamic content extraction failed for {url}: {str(e)}")
            return ""
            
    def bypass_recaptcha(self, driver, wait_time: int = 30) -> bool:
        """Attempt to bypass reCAPTCHA (educational purposes only)"""
        try:
            # Look for reCAPTCHA iframe
            recaptcha_iframe = driver.find_element(By.CSS_SELECTOR, 'iframe[src*="recaptcha"]')
            
            if recaptcha_iframe:
                self.logger.info("reCAPTCHA detected, implementing bypass strategy")
                
                # Switch to reCAPTCHA iframe
                driver.switch_to.frame(recaptcha_iframe)
                
                # Find and click the checkbox
                checkbox = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CLASS_NAME, "recaptcha-checkbox-border"))
                )
                
                # Human-like clicking
                action = ActionChains(driver)
                action.move_to_element(checkbox)
                action.pause(random.uniform(0.5, 1.5))
                action.click()
                action.perform()
                
                # Switch back to main frame
                driver.switch_to.default_content()
                
                # Wait for potential challenge resolution
                time.sleep(wait_time)
                
                return True
                
        except Exception as e:
            self.logger.warning(f"reCAPTCHA bypass attempt failed: {str(e)}")
            return False
            
        return False

class ContentAnalyzer:
    """Advanced content analysis and quality assessment"""
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/content_analyzer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def analyze_content_quality(self, content: str) -> Dict:
        """Comprehensive content quality analysis"""
        analysis = {}
        
        try:
            # Basic metrics
            analysis['word_count'] = len(content.split())
            analysis['char_count'] = len(content)
            analysis['sentence_count'] = len(re.split(r'[.!?]+', content))
            analysis['paragraph_count'] = len(content.split('\n\n'))
            
            # Readability metrics
            try:
                analysis['flesch_reading_ease'] = flesch_reading_ease(content)
                analysis['readability_grade'] = self._calculate_readability_grade(content)
            except:
                analysis['flesch_reading_ease'] = 0
                analysis['readability_grade'] = 'unknown'
                
            # Language detection
            try:
                analysis['language'] = langdetect.detect(content)
                analysis['language_confidence'] = langdetect.detect_langs(content)[0].prob
            except:
                analysis['language'] = 'unknown'
                analysis['language_confidence'] = 0
                
            # Content structure analysis
            analysis['structure_score'] = self._analyze_structure(content)
            
            # Information density
            analysis['information_density'] = self._calculate_information_density(content)
            
            # Overall quality score
            analysis['quality_score'] = self._calculate_overall_quality(analysis)
            
        except Exception as e:
            self.logger.error(f"Content analysis failed: {str(e)}")
            analysis['error'] = 'analysis_failed'
            
        return analysis
        
    def _calculate_readability_grade(self, content: str) -> str:
        """Calculate readability grade level"""
        try:
            # Simple Flesch-Kincaid grade level approximation
            words = len(content.split())
            sentences = len(re.split(r'[.!?]+', content))
            syllables = self._count_syllables(content)
            
            if sentences == 0 or words == 0:
                return 'unknown'
                
            grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
            
            if grade < 6:
                return 'elementary'
            elif grade < 9:
                return 'middle_school'
            elif grade < 13:
                return 'high_school'
            else:
                return 'college'
                
        except:
            return 'unknown'
            
    def _count_syllables(self, content: str) -> int:
        """Approximate syllable count"""
        words = re.findall(r'\b\w+\b', content.lower())
        syllable_count = 0
        
        for word in words:
            # Simple syllable counting heuristic
            vowels = 'aeiouy'
            word = word.lower()
            count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    count += 1
                prev_was_vowel = is_vowel
                
            # Handle silent e
            if word.endswith('e'):
                count -= 1
                
            # Every word has at least one syllable
            if count == 0:
                count = 1
                
            syllable_count += count
            
        return syllable_count
        
    def _analyze_structure(self, content: str) -> float:
        """Analyze content structure quality"""
        score = 0.0
        
        # Check for headings/subheadings
        if re.search(r'^[A-Z][^.!?]*:?\s*$', content, re.MULTILINE):
            score += 0.2
            
        # Check for proper paragraphing
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 1:
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)
            if 50 <= avg_paragraph_length <= 150:  # Ideal paragraph length
                score += 0.3
                
        # Check for lists or bullet points
        if re.search(r'^\s*[•\-\*\d+\.]\s+', content, re.MULTILINE):
            score += 0.2
            
        # Check for quotes
        if '"' in content or ''' in content or '"' in content:
            score += 0.1
            
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', content)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
            if 15 <= avg_sentence_length <= 25:  # Ideal sentence length
                score += 0.2
                
        return min(score, 1.0)
        
    def _calculate_information_density(self, content: str) -> float:
        """Calculate information density score"""
        words = content.split()
        
        if not words:
            return 0.0
            
        # Count unique words
        unique_words = set(word.lower().strip('.,!?;:"()[]{}') for word in words)
        
        # Count common stop words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'])
        
        content_words = unique_words - stop_words
        
        # Calculate density
        density = len(content_words) / len(words) if words else 0
        
        return min(density * 2, 1.0)  # Scale to 0-1 range
        
    def _calculate_overall_quality(self, analysis: Dict) -> float:
        """Calculate overall content quality score"""
        try:
            scores = []
            
            # Word count score (optimal range: 300-2000 words)
            word_count = analysis.get('word_count', 0)
            if 300 <= word_count <= 2000:
                scores.append(1.0)
            elif word_count < 300:
                scores.append(word_count / 300)
            else:
                scores.append(max(0.5, 2000 / word_count))
                
            # Readability score
            flesch_score = analysis.get('flesch_reading_ease', 0)
            if 60 <= flesch_score <= 90:  # Optimal readability
                scores.append(1.0)
            else:
                scores.append(max(0.3, min(flesch_score / 90, 1.0)))
                
            # Structure score
            scores.append(analysis.get('structure_score', 0))
            
            # Information density score
            scores.append(analysis.get('information_density', 0))
            
            # Language confidence score
            scores.append(analysis.get('language_confidence', 0))
            
            return sum(scores) / len(scores) if scores else 0.0
            
        except:
            return 0.0

class ProxyManager:
    """Advanced proxy management and rotation"""
    
    def __init__(self):
        self.proxy_pools = {
            'residential': [],
            'datacenter': [],
            'mobile': [],
            'tor': []
        }
        self.active_proxies = {}
        self.failed_proxies = set()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/proxy_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def add_proxy_pool(self, proxy_type: str, proxies: List[str]):
        """Add proxies to specific pool"""
        if proxy_type in self.proxy_pools:
            self.proxy_pools[proxy_type].extend(proxies)
            self.logger.info(f"Added {len(proxies)} proxies to {proxy_type} pool")
            
    def get_proxy(self, proxy_type: str = 'random') -> Optional[str]:
        """Get a working proxy from specified pool"""
        if proxy_type == 'random':
            available_types = [t for t in self.proxy_pools.keys() if self.proxy_pools[t]]
            if not available_types:
                return None
            proxy_type = random.choice(available_types)
            
        available_proxies = [p for p in self.proxy_pools[proxy_type] if p not in self.failed_proxies]
        
        if not available_proxies:
            return None
            
        proxy = random.choice(available_proxies)
        return proxy
        
    def test_proxy(self, proxy: str, test_url: str = "http://httpbin.org/ip") -> bool:
        """Test if proxy is working"""
        try:
            proxy_dict = {
                'http': proxy,
                'https': proxy
            }
            
            response = requests.get(test_url, proxies=proxy_dict, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                self.failed_proxies.add(proxy)
                return False
                
        except Exception as e:
            self.failed_proxies.add(proxy)
            self.logger.warning(f"Proxy test failed for {proxy}: {str(e)}")
            return False
            
    def rotate_proxy(self, session: requests.Session, proxy_type: str = 'random') -> bool:
        """Rotate proxy for given session"""
        new_proxy = self.get_proxy(proxy_type)
        
        if new_proxy:
            session.proxies = {
                'http': new_proxy,
                'https': new_proxy
            }
            return True
            
        return False

class CacheManager:
    """Intelligent caching system for scraped content"""
    
    def __init__(self, cache_dir: str = 'shock2/data/cache'):
        self.cache_dir = cache_dir
        self.db_path = f"{cache_dir}/cache.db"
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self._init_cache_db()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/cache_manager.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _init_cache_db(self):
        """Initialize cache database"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                id INTEGER PRIMARY KEY,
                url_hash TEXT UNIQUE,
                url TEXT,
                content_hash TEXT,
                compressed_content BLOB,
                metadata TEXT,
                created_timestamp TEXT,
                accessed_timestamp TEXT,
                access_count INTEGER DEFAULT 1,
                expiry_timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def cache_content(self, url: str, content: str, metadata: Dict = None, ttl_hours: int = 24):
        """Cache content with compression and encryption"""
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Compress and encrypt content
            compressed_content = gzip.compress(content.encode())
            encrypted_content = self.cipher.encrypt(compressed_content)
            
            # Calculate expiry
            expiry_time = datetime.now() + timedelta(hours=ttl_hours)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO cache_entries 
                (url_hash, url, content_hash, compressed_content, metadata, 
                 created_timestamp, accessed_timestamp, expiry_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                url_hash, url, content_hash, encrypted_content,
                json.dumps(metadata or {}),
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                expiry_time.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cached content for URL: {url}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache content for {url}: {str(e)}")
            
    def get_cached_content(self, url: str) -> Optional[Tuple[str, Dict]]:
        """Retrieve cached content if valid and not expired"""
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT compressed_content, metadata, expiry_timestamp 
                FROM cache_entries 
                WHERE url_hash = ?
            ''', (url_hash,))
            
            result = cursor.fetchone()
            
            if result:
                encrypted_content, metadata_json, expiry_str = result
                
                # Check if expired
                expiry_time = datetime.fromisoformat(expiry_str)
                if datetime.now() > expiry_time:
                    self._delete_cache_entry(url_hash)
                    conn.close()
                    return None
                    
                # Decrypt and decompress content
                compressed_content = self.cipher.decrypt(encrypted_content)
                content = gzip.decompress(compressed_content).decode()
                metadata = json.loads(metadata_json)
                
                # Update access statistics
                cursor.execute('''
                    UPDATE cache_entries 
                    SET accessed_timestamp = ?, access_count = access_count + 1
                    WHERE url_hash = ?
                ''', (datetime.now().isoformat(), url_hash))
                
                conn.commit()
                conn.close()
                
                self.logger.info(f"Retrieved cached content for URL: {url}")
                return content, metadata
                
            conn.close()
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached content for {url}: {str(e)}")
            return None
            
    def _delete_cache_entry(self, url_hash: str):
        """Delete specific cache entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM cache_entries WHERE url_hash = ?', (url_hash,))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to delete cache entry: {str(e)}")
            
    def cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM cache_entries 
                WHERE expiry_timestamp < ?
            ''', (datetime.now().isoformat(),))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired cache: {str(e)}")

# Utility functions for easy access
def quick_extract_content(url: str, method: str = 'auto') -> Dict:
    """Quick content extraction with automatic method selection"""
    parser = AdvancedHTMLParser()
    
    # Get content
    if method == 'dynamic':
        js_engine = JSBypassEngine()
        html_content = js_engine.extract_dynamic_content(url)
    else:
        response = requests.get(url, headers=UAXExtractor().get_stealth_headers(url))
        html_content = response.text
        
    # Extract content
    result = parser.extract_content_multiple_methods(html_content, url)
    
    # Analyze quality
    if 'text' in result:
        analyzer = ContentAnalyzer()
        result['quality_analysis'] = analyzer.analyze_content_quality(result['text'])
        
    return result

def bypass_protection_and_extract(url: str) -> Dict:
    """Bypass protection and extract content"""
    js_engine = JSBypassEngine()
    
    # Try Cloudflare bypass first
    response = js_engine.bypass_cloudflare(url)
    
    if response and response.status_code == 200:
        parser = AdvancedHTMLParser()
        return parser.extract_content_multiple_methods(response.text, url)
    else:
        # Fall back to dynamic extraction
        html_content = js_engine.extract_dynamic_content(url)
        if html_content:
            parser = AdvancedHTMLParser()
            return parser.extract_content_multiple_methods(html_content, url)
            
    return {'error': 'extraction_failed'}

def setup_stealth_session(proxy_type: str = 'random') -> requests.Session:
    """Setup stealth session with proxy and headers"""
    session = requests.Session()
    
    # Setup headers
    uax = UAXExtractor()
    session.headers.update(uax.get_stealth_headers())
    
    # Setup proxy
    proxy_manager = ProxyManager()
    proxy_manager.rotate_proxy(session, proxy_type)
    
    return session

if __name__ == "__main__":
    # Example usage
    test_url = "https://www.bbc.com/news"
    result = quick_extract_content(test_url)
    print(f"Extracted content quality: {result.get('quality_analysis', {}).get('quality_score', 0)}")
