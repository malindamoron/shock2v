
import asyncio
import aiohttp
import requests
import json
import time
import random
import hashlib
import sqlite3
from datetime import datetime, timedelta
import logging
import stem
from stem import Signal
from stem.control import Controller
import socks
import socket
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from fake_useragent import UserAgent
from cryptography.fernet import Fernet
import base64
import threading
from queue import Queue
import tempfile
import os

class DarkWebScraper:
    def __init__(self):
        self.tor_proxy = {'http': 'socks5h://127.0.0.1:9050', 'https': 'socks5h://127.0.0.1:9050'}
        self.session = None
        self.ua = UserAgent()
        self.db_path = 'shock2/data/raw/darkweb_intelligence.db'
        self._init_database()
        self.logger = self._setup_logger()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.onion_domains = self._load_onion_targets()
        self.discovered_links = set()
        self.intelligence_queue = Queue()
        self.circuit_renewal_interval = 300  # 5 minutes
        
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS darkweb_intelligence (
                id INTEGER PRIMARY KEY,
                onion_url TEXT,
                domain_name TEXT,
                content_type TEXT,
                encrypted_content BLOB,
                metadata TEXT,
                discovery_method TEXT,
                risk_level INTEGER,
                timestamp TEXT,
                content_hash TEXT UNIQUE
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS onion_services (
                id INTEGER PRIMARY KEY,
                onion_address TEXT UNIQUE,
                service_type TEXT,
                title TEXT,
                description TEXT,
                last_online TEXT,
                access_method TEXT,
                intelligence_value INTEGER,
                security_level INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dark_markets (
                id INTEGER PRIMARY KEY,
                market_name TEXT,
                onion_url TEXT,
                market_type TEXT,
                listings_count INTEGER,
                vendors_count INTEGER,
                encrypted_data BLOB,
                last_crawled TEXT,
                reputation_score REAL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS leaked_data (
                id INTEGER PRIMARY KEY,
                source TEXT,
                data_type TEXT,
                encrypted_content BLOB,
                leak_date TEXT,
                discovery_date TEXT,
                verification_status TEXT,
                sensitivity_level INTEGER
            )
        ''')
        conn.commit()
        conn.close()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/darkweb_scraper.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _load_onion_targets(self):
        """Load known onion service targets for intelligence gathering"""
        return [
            {
                'address': 'facebookwkhpilnemxj7asaniu7vnjjbiltxjqhye3mhbshg7kx5tfyd.onion',
                'type': 'social_media',
                'name': 'Facebook',
                'intelligence_value': 8
            },
            {
                'address': 'duckduckgogg42ts72.onion',
                'type': 'search_engine',
                'name': 'DuckDuckGo',
                'intelligence_value': 6
            },
            {
                'address': 'thehiddenwiki.onion',
                'type': 'directory',
                'name': 'Hidden Wiki',
                'intelligence_value': 9
            },
            {
                'address': 'darkweblink.onion',
                'type': 'link_directory',
                'name': 'Dark Web Links',
                'intelligence_value': 8
            }
        ]
        
    def _setup_tor_session(self):
        """Initialize Tor session with proper configuration"""
        try:
            self.session = requests.Session()
            self.session.proxies = self.tor_proxy
            self.session.headers.update({
                'User-Agent': self.ua.random,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            })
            
            # Test Tor connectivity
            test_response = self.session.get('http://httpbin.org/ip', timeout=30)
            if test_response.status_code == 200:
                tor_ip = test_response.json().get('origin')
                self.logger.info(f"Tor connection established. IP: {tor_ip}")
                return True
            else:
                self.logger.error("Failed to establish Tor connection")
                return False
                
        except Exception as e:
            self.logger.error(f"Error setting up Tor session: {str(e)}")
            return False
            
    def _renew_tor_circuit(self):
        """Renew Tor circuit for anonymity"""
        try:
            with Controller.from_port(port=9051) as controller:
                controller.authenticate()
                controller.signal(Signal.NEWNYM)
                self.logger.info("Tor circuit renewed")
                time.sleep(10)  # Wait for new circuit
        except Exception as e:
            self.logger.error(f"Error renewing Tor circuit: {str(e)}")
            
    async def discover_onion_services(self):
        """Discover new onion services through various methods"""
        discovered_services = []
        
        if not self._setup_tor_session():
            return discovered_services
            
        # Method 1: Crawl known directories
        for target in self.onion_domains:
            if target['type'] in ['directory', 'link_directory']:
                services = await self._crawl_onion_directory(target)
                discovered_services.extend(services)
                
        # Method 2: Search engine discovery
        search_queries = [
            'site:.onion news',
            'site:.onion intelligence',
            'site:.onion leaked',
            'site:.onion documents',
            'site:.onion government',
            'site:.onion corporate'
        ]
        
        for query in search_queries:
            services = await self._search_onion_services(query)
            discovered_services.extend(services)
            
        # Method 3: Forum and market discovery
        forum_services = await self._discover_through_forums()
        discovered_services.extend(forum_services)
        
        return discovered_services
        
    async def _crawl_onion_directory(self, directory_target):
        """Crawl onion directories for service discovery"""
        discovered = []
        
        try:
            url = f"http://{directory_target['address']}"
            response = self.session.get(url, timeout=60)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find onion links
                onion_pattern = r'[a-z2-7]{16,56}\.onion'
                links = soup.find_all('a', href=re.compile(onion_pattern))
                
                for link in links:
                    onion_url = link.get('href')
                    if onion_url and '.onion' in onion_url:
                        # Extract onion address
                        onion_match = re.search(onion_pattern, onion_url)
                        if onion_match:
                            onion_address = onion_match.group()
                            
                            service_data = {
                                'onion_address': onion_address,
                                'title': link.text.strip(),
                                'description': self._extract_description(link),
                                'discovery_method': 'directory_crawl',
                                'source': directory_target['name'],
                                'intelligence_value': 7
                            }
                            
                            discovered.append(service_data)
                            self._store_onion_service(service_data)
                            
                # Also look for direct text mentions
                text_onions = re.findall(onion_pattern, response.text)
                for onion_addr in text_onions:
                    if onion_addr not in [s['onion_address'] for s in discovered]:
                        service_data = {
                            'onion_address': onion_addr,
                            'title': 'Unknown Service',
                            'description': '',
                            'discovery_method': 'text_extraction',
                            'source': directory_target['name'],
                            'intelligence_value': 5
                        }
                        discovered.append(service_data)
                        self._store_onion_service(service_data)
                        
        except Exception as e:
            self.logger.error(f"Error crawling directory {directory_target['address']}: {str(e)}")
            
        return discovered
        
    async def _search_onion_services(self, query):
        """Search for onion services using Tor search engines"""
        discovered = []
        
        tor_search_engines = [
            'duckduckgogg42ts72.onion/html',
            'searx.onion',
            'ahmia.fi'  # Clearnet but indexes .onion
        ]
        
        for search_engine in tor_search_engines:
            try:
                if search_engine.endswith('.onion') or search_engine.endswith('.onion/html'):
                    search_url = f"http://{search_engine}?q={query.replace(' ', '+')}"
                else:
                    search_url = f"https://{search_engine}/search?q={query.replace(' ', '+')}"
                    
                response = self.session.get(search_url, timeout=60)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract onion links from search results
                    onion_pattern = r'[a-z2-7]{16,56}\.onion'
                    result_links = soup.find_all('a', href=re.compile(onion_pattern))
                    
                    for link in result_links:
                        onion_url = link.get('href')
                        onion_match = re.search(onion_pattern, onion_url)
                        
                        if onion_match:
                            service_data = {
                                'onion_address': onion_match.group(),
                                'title': link.text.strip(),
                                'description': self._extract_search_snippet(link),
                                'discovery_method': 'search_engine',
                                'source': search_engine,
                                'intelligence_value': 6
                            }
                            discovered.append(service_data)
                            self._store_onion_service(service_data)
                            
                await asyncio.sleep(random.uniform(5, 15))
                
            except Exception as e:
                self.logger.error(f"Error searching with {search_engine}: {str(e)}")
                
        return discovered
        
    async def _discover_through_forums(self):
        """Discover services through dark web forums"""
        discovered = []
        
        forum_targets = [
            'darkwebforums.onion',
            'hackerforums.onion',
            'leakbase.onion'
        ]
        
        for forum in forum_targets:
            try:
                url = f"http://{forum}"
                response = self.session.get(url, timeout=60)
                
                if response.status_code == 200:
                    # Extract onion links from forum content
                    onion_pattern = r'[a-z2-7]{16,56}\.onion'
                    onion_addresses = re.findall(onion_pattern, response.text)
                    
                    for onion_addr in set(onion_addresses):
                        service_data = {
                            'onion_address': onion_addr,
                            'title': 'Forum Discovery',
                            'description': '',
                            'discovery_method': 'forum_crawl',
                            'source': forum,
                            'intelligence_value': 7
                        }
                        discovered.append(service_data)
                        self._store_onion_service(service_data)
                        
                await asyncio.sleep(random.uniform(10, 20))
                
            except Exception as e:
                self.logger.error(f"Error discovering through forum {forum}: {str(e)}")
                
        return discovered
        
    def _extract_description(self, link_element):
        """Extract description from link context"""
        try:
            parent = link_element.parent
            if parent:
                siblings = parent.find_all(text=True)
                description = ' '.join([s.strip() for s in siblings if s.strip()])
                return description[:200]
        except:
            pass
        return ""
        
    def _extract_search_snippet(self, link_element):
        """Extract search result snippet"""
        try:
            # Look for description in nearby elements
            parent = link_element.parent
            for i in range(3):  # Check up to 3 parent levels
                if parent:
                    snippet_elem = parent.find('div', class_=['snippet', 'description', 'excerpt'])
                    if snippet_elem:
                        return snippet_elem.get_text()[:200]
                    parent = parent.parent
        except:
            pass
        return ""
        
    def _store_onion_service(self, service_data):
        """Store discovered onion service"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO onion_services 
                (onion_address, service_type, title, description, last_online, 
                 access_method, intelligence_value, security_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                service_data['onion_address'],
                service_data.get('service_type', 'unknown'),
                service_data['title'],
                service_data['description'],
                datetime.now().isoformat(),
                service_data['discovery_method'],
                service_data['intelligence_value'],
                5  # Default security level
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing onion service: {str(e)}")
            
    async def scrape_intelligence_content(self, onion_services):
        """Scrape intelligence content from discovered services"""
        intelligence_data = []
        
        if not self._setup_tor_session():
            return intelligence_data
            
        for service in onion_services:
            try:
                onion_url = f"http://{service['onion_address']}"
                
                # Attempt to access the service
                response = self.session.get(onion_url, timeout=120)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Extract content based on service type
                    content_data = {
                        'onion_url': onion_url,
                        'domain_name': service['onion_address'],
                        'title': soup.title.string if soup.title else 'No Title',
                        'content': soup.get_text()[:5000],  # First 5000 chars
                        'links': [a.get('href') for a in soup.find_all('a', href=True)],
                        'forms': len(soup.find_all('form')),
                        'images': len(soup.find_all('img')),
                        'scripts': len(soup.find_all('script')),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Analyze content for intelligence value
                    intelligence_score = self._analyze_intelligence_value(content_data)
                    content_data['intelligence_score'] = intelligence_score
                    
                    # Look for specific intelligence indicators
                    intel_indicators = self._extract_intelligence_indicators(soup)
                    content_data['indicators'] = intel_indicators
                    
                    # Check for market/forum characteristics
                    market_data = self._analyze_market_characteristics(soup)
                    if market_data:
                        content_data['market_data'] = market_data
                        
                    # Store the intelligence
                    self._store_intelligence_content(content_data)
                    intelligence_data.append(content_data)
                    
                    self.logger.info(f"Scraped intelligence from {service['onion_address']}")
                    
                else:
                    self.logger.warning(f"Failed to access {service['onion_address']}: {response.status_code}")
                    
                # Random delay and circuit renewal
                await asyncio.sleep(random.uniform(30, 60))
                
                if random.random() < 0.3:  # 30% chance to renew circuit
                    self._renew_tor_circuit()
                    
            except Exception as e:
                self.logger.error(f"Error scraping {service['onion_address']}: {str(e)}")
                continue
                
        return intelligence_data
        
    def _analyze_intelligence_value(self, content_data):
        """Analyze content for intelligence value"""
        intelligence_keywords = [
            'classified', 'confidential', 'secret', 'internal', 'leaked',
            'government', 'military', 'intelligence', 'surveillance',
            'documents', 'files', 'database', 'credentials', 'passwords',
            'corporate', 'financial', 'banking', 'insider', 'whistleblower',
            'breach', 'hack', 'exploit', 'vulnerability', 'zero-day'
        ]
        
        content_lower = content_data['content'].lower()
        score = 0
        
        for keyword in intelligence_keywords:
            if keyword in content_lower:
                score += 1
                
        # Bonus points for forms (potential data entry)
        score += content_data['forms'] * 2
        
        # Bonus for many links (potential directory/index)
        if len(content_data['links']) > 20:
            score += 3
            
        return min(score, 10)  # Cap at 10
        
    def _extract_intelligence_indicators(self, soup):
        """Extract specific intelligence indicators"""
        indicators = {
            'leaked_data': False,
            'government_docs': False,
            'corporate_intel': False,
            'financial_data': False,
            'personal_data': False,
            'technical_exploits': False
        }
        
        text_content = soup.get_text().lower()
        
        # Check for leaked data indicators
        leak_indicators = ['breach', 'leaked', 'dump', 'stolen', 'hacked']
        if any(indicator in text_content for indicator in leak_indicators):
            indicators['leaked_data'] = True
            
        # Check for government document indicators
        gov_indicators = ['classified', 'confidential', 'nsa', 'cia', 'fbi', 'government']
        if any(indicator in text_content for indicator in gov_indicators):
            indicators['government_docs'] = True
            
        # Check for corporate intelligence
        corp_indicators = ['insider', 'corporate', 'merger', 'acquisition', 'financial report']
        if any(indicator in text_content for indicator in corp_indicators):
            indicators['corporate_intel'] = True
            
        # Check for financial data
        fin_indicators = ['bank', 'credit card', 'financial', 'transaction', 'payment']
        if any(indicator in text_content for indicator in fin_indicators):
            indicators['financial_data'] = True
            
        # Check for personal data
        personal_indicators = ['ssn', 'social security', 'passport', 'driver license', 'personal']
        if any(indicator in text_content for indicator in personal_indicators):
            indicators['personal_data'] = True
            
        # Check for technical exploits
        tech_indicators = ['exploit', 'vulnerability', 'zero-day', 'malware', 'backdoor']
        if any(indicator in text_content for indicator in tech_indicators):
            indicators['technical_exploits'] = True
            
        return indicators
        
    def _analyze_market_characteristics(self, soup):
        """Analyze if the site is a dark web marketplace"""
        market_indicators = {
            'product_listings': len(soup.find_all(['div', 'li'], class_=re.compile(r'product|listing|item'))),
            'vendor_profiles': len(soup.find_all(['a', 'div'], class_=re.compile(r'vendor|seller|merchant'))),
            'price_elements': len(soup.find_all(text=re.compile(r'\$\d+|\d+\s*btc|\d+\s*bitcoin'))),
            'shopping_cart': bool(soup.find_all(['a', 'button'], text=re.compile(r'cart|buy|purchase', re.I))),
            'escrow_mentions': bool(soup.find_all(text=re.compile(r'escrow', re.I))),
            'crypto_payments': bool(soup.find_all(text=re.compile(r'bitcoin|btc|monero|xmr', re.I)))
        }
        
        # Calculate market score
        market_score = (
            market_indicators['product_listings'] +
            market_indicators['vendor_profiles'] * 2 +
            market_indicators['price_elements'] +
            (5 if market_indicators['shopping_cart'] else 0) +
            (3 if market_indicators['escrow_mentions'] else 0) +
            (2 if market_indicators['crypto_payments'] else 0)
        )
        
        if market_score > 10:
            return market_indicators
            
        return None
        
    def _store_intelligence_content(self, content_data):
        """Store intelligence content with encryption"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Encrypt sensitive content
            content_json = json.dumps(content_data)
            encrypted_content = self.cipher.encrypt(content_json.encode())
            
            # Create content hash for deduplication
            content_hash = hashlib.sha256(content_data['content'].encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR IGNORE INTO darkweb_intelligence 
                (onion_url, domain_name, content_type, encrypted_content, metadata,
                 discovery_method, risk_level, timestamp, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content_data['onion_url'],
                content_data['domain_name'],
                'scraped_content',
                encrypted_content,
                json.dumps({
                    'intelligence_score': content_data['intelligence_score'],
                    'indicators': content_data['indicators'],
                    'forms_count': content_data['forms'],
                    'links_count': len(content_data['links'])
                }),
                'direct_scraping',
                content_data['intelligence_score'],
                content_data['timestamp'],
                content_hash
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing intelligence content: {str(e)}")
            
    async def monitor_dark_markets(self):
        """Monitor dark web marketplaces for intelligence"""
        known_markets = [
            'alphabay.onion',
            'darkmarket.onion',
            'whitehouse.onion'
        ]
        
        market_intelligence = []
        
        if not self._setup_tor_session():
            return market_intelligence
            
        for market_onion in known_markets:
            try:
                url = f"http://{market_onion}"
                response = self.session.get(url, timeout=120)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    market_data = {
                        'market_name': self._extract_market_name(soup),
                        'onion_url': url,
                        'listings_count': self._count_listings(soup),
                        'vendors_count': self._count_vendors(soup),
                        'categories': self._extract_categories(soup),
                        'featured_products': self._extract_featured_products(soup),
                        'last_crawled': datetime.now().isoformat()
                    }
                    
                    market_intelligence.append(market_data)
                    self._store_market_data(market_data)
                    
                await asyncio.sleep(random.uniform(60, 120))
                
            except Exception as e:
                self.logger.error(f"Error monitoring market {market_onion}: {str(e)}")
                
        return market_intelligence
        
    def _extract_market_name(self, soup):
        """Extract marketplace name"""
        title_elem = soup.find('title')
        if title_elem:
            return title_elem.get_text().strip()
            
        # Look for brand/logo elements
        brand_selectors = ['.brand', '.logo', '.site-name', 'h1']
        for selector in brand_selectors:
            elem = soup.select_one(selector)
            if elem:
                return elem.get_text().strip()
                
        return "Unknown Market"
        
    def _count_listings(self, soup):
        """Count product listings"""
        listing_selectors = [
            '.listing', '.product', '.item',
            '[class*="listing"]', '[class*="product"]', '[class*="item"]'
        ]
        
        for selector in listing_selectors:
            elements = soup.select(selector)
            if elements and len(elements) > 5:  # Reasonable number of listings
                return len(elements)
                
        return 0
        
    def _count_vendors(self, soup):
        """Count vendors/sellers"""
        vendor_selectors = [
            '.vendor', '.seller', '.merchant',
            '[class*="vendor"]', '[class*="seller"]'
        ]
        
        vendors = set()
        for selector in vendor_selectors:
            elements = soup.select(selector)
            for elem in elements:
                vendor_name = elem.get_text().strip()
                if vendor_name:
                    vendors.add(vendor_name)
                    
        return len(vendors)
        
    def _extract_categories(self, soup):
        """Extract product categories"""
        categories = []
        
        category_selectors = [
            '.category', '.cat', '.navigation a',
            '[class*="category"]', '[class*="cat"]'
        ]
        
        for selector in category_selectors:
            elements = soup.select(selector)
            for elem in elements:
                cat_text = elem.get_text().strip()
                if cat_text and len(cat_text) < 50:
                    categories.append(cat_text)
                    
        return list(set(categories))
        
    def _extract_featured_products(self, soup):
        """Extract featured/popular products"""
        products = []
        
        product_selectors = [
            '.featured', '.popular', '.trending',
            '[class*="featured"]', '[class*="popular"]'
        ]
        
        for selector in product_selectors:
            elements = soup.select(selector)
            for elem in elements[:10]:  # Limit to 10 products
                product_title = elem.get_text().strip()
                if product_title:
                    products.append(product_title)
                    
        return products
        
    def _store_market_data(self, market_data):
        """Store marketplace data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Encrypt market data
            encrypted_data = self.cipher.encrypt(json.dumps(market_data).encode())
            
            cursor.execute('''
                INSERT OR REPLACE INTO dark_markets 
                (market_name, onion_url, market_type, listings_count, vendors_count,
                 encrypted_data, last_crawled, reputation_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                market_data['market_name'],
                market_data['onion_url'],
                'general_marketplace',
                market_data['listings_count'],
                market_data['vendors_count'],
                encrypted_data,
                market_data['last_crawled'],
                5.0  # Default reputation score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
            
    async def run_dark_intelligence_operation(self):
        """Run comprehensive dark web intelligence operation"""
        self.logger.info("Starting dark web intelligence operation")
        
        operation_results = {
            'discovered_services': 0,
            'scraped_content': 0,
            'market_intelligence': 0,
            'leaked_data': 0
        }
        
        try:
            # Phase 1: Service Discovery
            discovered_services = await self.discover_onion_services()
            operation_results['discovered_services'] = len(discovered_services)
            self.logger.info(f"Discovered {len(discovered_services)} onion services")
            
            # Phase 2: Content Scraping
            intelligence_data = await self.scrape_intelligence_content(discovered_services[:20])  # Limit to top 20
            operation_results['scraped_content'] = len(intelligence_data)
            self.logger.info(f"Scraped intelligence from {len(intelligence_data)} services")
            
            # Phase 3: Market Monitoring
            market_intel = await self.monitor_dark_markets()
            operation_results['market_intelligence'] = len(market_intel)
            self.logger.info(f"Monitored {len(market_intel)} dark markets")
            
            self.logger.info(f"Dark web operation completed: {operation_results}")
            
        except Exception as e:
            self.logger.error(f"Error in dark web intelligence operation: {str(e)}")
            
        return operation_results

if __name__ == "__main__":
    scraper = DarkWebScraper()
    asyncio.run(scraper.run_dark_intelligence_operation())
