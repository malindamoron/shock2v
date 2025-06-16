
import asyncio
import aiohttp
import json
import time
import random
import hashlib
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import sqlite3
from datetime import datetime, timedelta
import logging
import re
from fake_useragent import UserAgent
import threading
from queue import Queue
import base64
from cryptography.fernet import Fernet

class DeepWebScanner:
    def __init__(self):
        self.ua = UserAgent()
        self.session = self._create_session()
        self.db_path = 'shock2/data/raw/deepweb_cache.db'
        self._init_database()
        self.logger = self._setup_logger()
        self.target_forums = self._load_forum_targets()
        self.ajax_endpoints = Queue()
        self.discovered_apis = set()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _create_session(self):
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'X-Requested-With': 'XMLHttpRequest',
        })
        return session
        
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS deepweb_content (
                id INTEGER PRIMARY KEY,
                url TEXT UNIQUE,
                content_type TEXT,
                encrypted_content BLOB,
                metadata TEXT,
                source_type TEXT,
                discovery_method TEXT,
                timestamp TEXT,
                content_hash TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ajax_endpoints (
                id INTEGER PRIMARY KEY,
                endpoint_url TEXT UNIQUE,
                parameters TEXT,
                response_structure TEXT,
                source_domain TEXT,
                discovery_timestamp TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forum_posts (
                id INTEGER PRIMARY KEY,
                forum_name TEXT,
                thread_id TEXT,
                post_id TEXT,
                author TEXT,
                title TEXT,
                encrypted_content BLOB,
                post_timestamp TEXT,
                crawl_timestamp TEXT,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/deepweb_scanner.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _load_forum_targets(self):
        return [
            {
                'name': '4chan_pol',
                'base_url': 'https://boards.4chan.org/pol/',
                'catalog_url': 'https://a.4cdn.org/pol/catalog.json',
                'thread_url_pattern': 'https://a.4cdn.org/pol/thread/{}.json'
            },
            {
                'name': 'reddit_conspiracy',
                'base_url': 'https://www.reddit.com/r/conspiracy/',
                'api_url': 'https://www.reddit.com/r/conspiracy/hot/.json'
            },
            {
                'name': 'hackernews',
                'base_url': 'https://news.ycombinator.com/',
                'api_url': 'https://hacker-news.firebaseio.com/v0/topstories.json'
            },
            {
                'name': 'discord_public',
                'servers': ['general', 'politics', 'news'],
                'method': 'webhook_scraping'
            }
        ]
        
    async def discover_ajax_endpoints(self, target_url):
        """Discover AJAX endpoints by analyzing JavaScript and network traffic"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--enable-logging')
            chrome_options.add_argument('--log-level=0')
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            
            driver = webdriver.Chrome(options=chrome_options)
            
            # Enable network domain
            driver.execute_cdp_cmd('Network.enable', {})
            
            # Collect network requests
            requests_log = []
            
            def log_request(request):
                if request['params']['type'] in ['XHR', 'Fetch']:
                    requests_log.append(request['params'])
                    
            driver.add_cdp_listener('Network.requestWillBeSent', log_request)
            
            # Navigate to target
            driver.get(target_url)
            
            # Interact with page to trigger AJAX calls
            await self._simulate_user_interaction(driver)
            
            # Extract JavaScript sources
            scripts = driver.find_elements(By.TAG_NAME, 'script')
            for script in scripts:
                src = script.get_attribute('src')
                if src:
                    js_content = await self._fetch_javascript(src)
                    endpoints = self._extract_endpoints_from_js(js_content)
                    for endpoint in endpoints:
                        self._store_ajax_endpoint(endpoint, target_url)
                        
            # Process collected network requests
            for request in requests_log:
                endpoint_data = {
                    'url': request['request']['url'],
                    'method': request['request']['method'],
                    'headers': request['request'].get('headers', {}),
                    'post_data': request['request'].get('postData', ''),
                    'source_domain': urlparse(target_url).netloc
                }
                self._store_ajax_endpoint(endpoint_data, target_url)
                
            driver.quit()
            return len(requests_log)
            
        except Exception as e:
            self.logger.error(f"Error discovering AJAX endpoints for {target_url}: {str(e)}")
            return 0
            
    async def _simulate_user_interaction(self, driver):
        """Simulate user interactions to trigger AJAX calls"""
        try:
            # Scroll to trigger lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            await asyncio.sleep(2)
            
            # Click on interactive elements
            clickable_elements = driver.find_elements(By.CSS_SELECTOR, 
                'button, .btn, .click, [onclick], [data-toggle], .tab, .menu-item')
            
            for element in clickable_elements[:5]:
                try:
                    if element.is_displayed() and element.is_enabled():
                        ActionChains(driver).move_to_element(element).click().perform()
                        await asyncio.sleep(1)
                except:
                    continue
                    
            # Fill and submit forms
            forms = driver.find_elements(By.TAG_NAME, 'form')
            for form in forms[:3]:
                try:
                    inputs = form.find_elements(By.CSS_SELECTOR, 'input[type="text"], input[type="search"]')
                    for inp in inputs:
                        inp.send_keys("test query")
                        
                    submit_btn = form.find_element(By.CSS_SELECTOR, 'input[type="submit"], button[type="submit"]')
                    submit_btn.click()
                    await asyncio.sleep(2)
                except:
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error in user interaction simulation: {str(e)}")
            
    async def _fetch_javascript(self, js_url):
        """Fetch and analyze JavaScript content"""
        try:
            headers = {'User-Agent': self.ua.random}
            response = self.session.get(js_url, headers=headers, timeout=15)
            return response.text if response.status_code == 200 else ""
        except:
            return ""
            
    def _extract_endpoints_from_js(self, js_content):
        """Extract API endpoints from JavaScript code"""
        endpoints = []
        
        # Regex patterns for common endpoint patterns
        patterns = [
            r'["\']\/api\/[^"\']*["\']',
            r'["\']https?:\/\/[^"\']*\/api\/[^"\']*["\']',
            r'fetch\(["\']([^"\']*)["\']',
            r'\.get\(["\']([^"\']*)["\']',
            r'\.post\(["\']([^"\']*)["\']',
            r'ajax\(\s*{\s*url\s*:\s*["\']([^"\']*)["\']',
            r'xhr\.open\(["\'][^"\']*["\'],\s*["\']([^"\']*)["\']'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, js_content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if match and not match.startswith('data:'):
                    endpoints.append(match.strip('"\''))
                    
        return list(set(endpoints))
        
    def _store_ajax_endpoint(self, endpoint_data, source_url):
        """Store discovered AJAX endpoint"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if isinstance(endpoint_data, str):
                endpoint_url = endpoint_data
                parameters = "{}"
                response_structure = "{}"
            else:
                endpoint_url = endpoint_data['url']
                parameters = json.dumps(endpoint_data.get('headers', {}))
                response_structure = endpoint_data.get('post_data', '{}')
                
            cursor.execute('''
                INSERT OR IGNORE INTO ajax_endpoints 
                (endpoint_url, parameters, response_structure, source_domain, discovery_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                endpoint_url,
                parameters,
                response_structure,
                urlparse(source_url).netloc,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing AJAX endpoint: {str(e)}")
            
    async def scan_forums(self):
        """Scan various forums for intelligence"""
        all_posts = []
        
        for forum in self.target_forums:
            try:
                if forum['name'] == '4chan_pol':
                    posts = await self._scan_4chan(forum)
                elif forum['name'].startswith('reddit'):
                    posts = await self._scan_reddit(forum)
                elif forum['name'] == 'hackernews':
                    posts = await self._scan_hackernews(forum)
                else:
                    posts = await self._scan_generic_forum(forum)
                    
                all_posts.extend(posts)
                self.logger.info(f"Scanned {len(posts)} posts from {forum['name']}")
                
                # Random delay between forums
                await asyncio.sleep(random.uniform(5, 15))
                
            except Exception as e:
                self.logger.error(f"Error scanning forum {forum['name']}: {str(e)}")
                
        return all_posts
        
    async def _scan_4chan(self, forum_config):
        """Scan 4chan /pol/ for intelligence"""
        posts = []
        
        try:
            # Get catalog
            catalog_response = self.session.get(forum_config['catalog_url'])
            catalog_data = catalog_response.json()
            
            # Get threads from first few pages
            threads_to_scan = []
            for page in catalog_data[:3]:  # First 3 pages
                for thread in page['threads'][:5]:  # Top 5 threads per page
                    threads_to_scan.append(thread['no'])
                    
            # Scan individual threads
            for thread_id in threads_to_scan:
                thread_url = forum_config['thread_url_pattern'].format(thread_id)
                thread_response = self.session.get(thread_url)
                
                if thread_response.status_code == 200:
                    thread_data = thread_response.json()
                    
                    for post in thread_data['posts']:
                        post_content = {
                            'forum_name': '4chan_pol',
                            'thread_id': str(thread_id),
                            'post_id': str(post['no']),
                            'author': 'Anonymous',
                            'title': post.get('sub', ''),
                            'content': post.get('com', ''),
                            'timestamp': post.get('time', 0),
                            'replies': post.get('replies', 0),
                            'images': post.get('images', 0)
                        }
                        
                        posts.append(post_content)
                        self._store_forum_post(post_content)
                        
                await asyncio.sleep(random.uniform(1, 3))
                
        except Exception as e:
            self.logger.error(f"Error scanning 4chan: {str(e)}")
            
        return posts
        
    async def _scan_reddit(self, forum_config):
        """Scan Reddit for intelligence"""
        posts = []
        
        try:
            headers = {
                'User-Agent': 'DeepWebScanner/1.0 (Research Purpose)',
                'Accept': 'application/json'
            }
            
            response = self.session.get(forum_config['api_url'], headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data['data']['children']:
                    post_data = post['data']
                    
                    post_content = {
                        'forum_name': forum_config['name'],
                        'thread_id': post_data['id'],
                        'post_id': post_data['id'],
                        'author': post_data['author'],
                        'title': post_data['title'],
                        'content': post_data.get('selftext', ''),
                        'score': post_data['score'],
                        'comments': post_data['num_comments'],
                        'timestamp': post_data['created_utc'],
                        'url': post_data.get('url', '')
                    }
                    
                    posts.append(post_content)
                    self._store_forum_post(post_content)
                    
        except Exception as e:
            self.logger.error(f"Error scanning Reddit: {str(e)}")
            
        return posts
        
    async def _scan_hackernews(self, forum_config):
        """Scan Hacker News for tech intelligence"""
        posts = []
        
        try:
            # Get top stories
            response = self.session.get(forum_config['api_url'])
            story_ids = response.json()[:30]  # Top 30 stories
            
            for story_id in story_ids:
                story_url = f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json"
                story_response = self.session.get(story_url)
                
                if story_response.status_code == 200:
                    story_data = story_response.json()
                    
                    post_content = {
                        'forum_name': 'hackernews',
                        'thread_id': str(story_id),
                        'post_id': str(story_id),
                        'author': story_data.get('by', ''),
                        'title': story_data.get('title', ''),
                        'content': story_data.get('text', ''),
                        'score': story_data.get('score', 0),
                        'comments': len(story_data.get('kids', [])),
                        'timestamp': story_data.get('time', 0),
                        'url': story_data.get('url', '')
                    }
                    
                    posts.append(post_content)
                    self._store_forum_post(post_content)
                    
                await asyncio.sleep(0.5)
                
        except Exception as e:
            self.logger.error(f"Error scanning Hacker News: {str(e)}")
            
        return posts
        
    async def _scan_generic_forum(self, forum_config):
        """Generic forum scanning using web scraping"""
        posts = []
        
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(forum_config['base_url'])
            
            # Look for post containers
            post_selectors = [
                '.post', '.thread', '.message', '.topic',
                '[class*="post"]', '[class*="thread"]', '[class*="message"]'
            ]
            
            for selector in post_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                for element in elements[:20]:  # Limit to 20 posts
                    try:
                        title = ""
                        content = element.text
                        author = ""
                        
                        # Try to extract title
                        title_elem = element.find_element(By.CSS_SELECTOR, 
                            'h1, h2, h3, .title, [class*="title"]')
                        if title_elem:
                            title = title_elem.text
                            
                        # Try to extract author
                        author_elem = element.find_element(By.CSS_SELECTOR,
                            '.author, .username, [class*="author"], [class*="user"]')
                        if author_elem:
                            author = author_elem.text
                            
                        post_content = {
                            'forum_name': forum_config['name'],
                            'thread_id': hashlib.md5(content.encode()).hexdigest()[:10],
                            'post_id': hashlib.md5(content.encode()).hexdigest(),
                            'author': author,
                            'title': title,
                            'content': content,
                            'timestamp': datetime.now().timestamp()
                        }
                        
                        posts.append(post_content)
                        self._store_forum_post(post_content)
                        
                    except:
                        continue
                        
                if posts:  # If we found posts with this selector, break
                    break
                    
            driver.quit()
            
        except Exception as e:
            self.logger.error(f"Error in generic forum scan: {str(e)}")
            
        return posts
        
    def _store_forum_post(self, post_data):
        """Store forum post with encryption"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Encrypt sensitive content
            content_json = json.dumps(post_data)
            encrypted_content = self.cipher.encrypt(content_json.encode())
            
            cursor.execute('''
                INSERT OR IGNORE INTO forum_posts 
                (forum_name, thread_id, post_id, author, title, encrypted_content, 
                 post_timestamp, crawl_timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                post_data['forum_name'],
                post_data['thread_id'],
                post_data['post_id'],
                post_data['author'],
                post_data['title'],
                encrypted_content,
                str(post_data.get('timestamp', datetime.now().timestamp())),
                datetime.now().isoformat(),
                json.dumps({'source': 'deepweb_scanner'})
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing forum post: {str(e)}")
            
    async def exploit_ajax_endpoints(self):
        """Exploit discovered AJAX endpoints for data extraction"""
        exploited_data = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM ajax_endpoints WHERE access_count < 10')
            endpoints = cursor.fetchall()
            conn.close()
            
            for endpoint in endpoints:
                endpoint_url = endpoint[1]
                parameters = json.loads(endpoint[2])
                
                try:
                    # Try different HTTP methods
                    methods = ['GET', 'POST']
                    
                    for method in methods:
                        headers = {
                            'User-Agent': self.ua.random,
                            'Accept': 'application/json, */*',
                            'X-Requested-With': 'XMLHttpRequest',
                            'Referer': f"https://{urlparse(endpoint_url).netloc}/"
                        }
                        
                        if method == 'GET':
                            response = self.session.get(endpoint_url, headers=headers, timeout=10)
                        else:
                            response = self.session.post(endpoint_url, headers=headers, 
                                                       json=parameters, timeout=10)
                                                       
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                exploited_data.append({
                                    'endpoint': endpoint_url,
                                    'method': method,
                                    'data': data,
                                    'timestamp': datetime.now().isoformat()
                                })
                                
                                # Update access count
                                conn = sqlite3.connect(self.db_path)
                                cursor = conn.cursor()
                                cursor.execute(
                                    'UPDATE ajax_endpoints SET access_count = access_count + 1, last_accessed = ? WHERE endpoint_url = ?',
                                    (datetime.now().isoformat(), endpoint_url)
                                )
                                conn.commit()
                                conn.close()
                                
                                break  # Success, no need to try other methods
                                
                            except json.JSONDecodeError:
                                # Not JSON response, store as text
                                exploited_data.append({
                                    'endpoint': endpoint_url,
                                    'method': method,
                                    'data': {'text': response.text[:1000]},
                                    'timestamp': datetime.now().isoformat()
                                })
                                
                        await asyncio.sleep(random.uniform(1, 3))
                        
                except Exception as e:
                    self.logger.error(f"Error exploiting endpoint {endpoint_url}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error in AJAX endpoint exploitation: {str(e)}")
            
        return exploited_data
        
    async def run_deep_scan(self, target_domains):
        """Run comprehensive deep web scanning"""
        self.logger.info("Starting deep web scanning operation")
        
        results = {
            'ajax_endpoints': 0,
            'forum_posts': 0,
            'exploited_data': 0
        }
        
        try:
            # Discover AJAX endpoints for each domain
            for domain in target_domains:
                endpoint_count = await self.discover_ajax_endpoints(f"https://{domain}")
                results['ajax_endpoints'] += endpoint_count
                await asyncio.sleep(random.uniform(3, 8))
                
            # Scan forums
            forum_posts = await self.scan_forums()
            results['forum_posts'] = len(forum_posts)
            
            # Exploit discovered AJAX endpoints
            exploited_data = await self.exploit_ajax_endpoints()
            results['exploited_data'] = len(exploited_data)
            
            self.logger.info(f"Deep scan completed: {results}")
            
        except Exception as e:
            self.logger.error(f"Error in deep scan operation: {str(e)}")
            
        return results

if __name__ == "__main__":
    scanner = DeepWebScanner()
    target_domains = ['cnn.com', 'bbc.com', 'reuters.com', 'bloomberg.com']
    asyncio.run(scanner.run_deep_scan(target_domains))
