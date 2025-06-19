
import asyncio
import aiohttp
import requests
import json
import random
import time
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
from queue import Queue
import socket
import socks
import urllib3
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import psutil
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import ssl
import certifi
import warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProxyNode:
    """Comprehensive proxy node information"""
    proxy_id: str
    host: str
    port: int
    protocol: str  # 'http', 'https', 'socks4', 'socks5'
    username: Optional[str] = None
    password: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    city: Optional[str] = None
    isp: Optional[str] = None
    speed_score: float = 0.0
    reliability_score: float = 0.0
    anonymity_level: str = 'unknown'  # 'transparent', 'anonymous', 'elite'
    last_tested: Optional[datetime] = None
    response_time: float = 0.0
    success_rate: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    is_active: bool = True
    detection_risk: float = 0.0
    bandwidth_limit: Optional[int] = None
    concurrent_limit: int = 1
    rotation_frequency: int = 100  # requests before rotation
    usage_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProxyRotationConfig:
    """Configuration for proxy rotation"""
    rotation_strategy: str = 'round_robin'  # 'round_robin', 'weighted', 'geographic', 'performance'
    max_requests_per_proxy: int = 100
    max_concurrent_per_proxy: int = 5
    health_check_interval: int = 300  # seconds
    retry_failed_proxies: bool = True
    geographic_distribution: bool = True
    performance_optimization: bool = True
    stealth_level: float = 0.8
    target_anonymity: str = 'elite'
    backup_proxy_count: int = 10
    blacklist_threshold: float = 0.3
    auto_discovery: bool = True

class AdvancedProxyRotator:
    """
    Advanced proxy rotation system with intelligent selection, health monitoring,
    geographic distribution, and stealth optimization capabilities.
    """
    
    def __init__(self, config: Optional[ProxyRotationConfig] = None, db_path: str = 'shock2_proxies.db'):
        self.config = config or ProxyRotationConfig()
        self.db_path = db_path
        self.setup_database()
        self.initialize_components()
        self.load_proxy_nodes()
        
        # Rotation state
        self.active_proxies = {}
        self.proxy_queue = deque()
        self.current_proxy = None
        self.rotation_lock = threading.Lock()
        
        # Monitoring
        self.health_monitor = ProxyHealthMonitor(self)
        self.performance_tracker = ProxyPerformanceTracker(self)
        self.geographic_manager = GeographicProxyManager(self)
        
        # Discovery and acquisition
        self.proxy_discoverer = ProxyDiscoverer(self)
        self.proxy_validator = ProxyValidator(self)
        
        # Start background tasks
        self.start_background_tasks()
        
    def setup_database(self):
        """Setup SQLite database for proxy management"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proxy_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proxy_id TEXT UNIQUE,
                host TEXT,
                port INTEGER,
                protocol TEXT,
                username TEXT,
                password TEXT,
                country TEXT,
                region TEXT,
                city TEXT,
                isp TEXT,
                speed_score REAL DEFAULT 0.0,
                reliability_score REAL DEFAULT 0.0,
                anonymity_level TEXT DEFAULT 'unknown',
                last_tested DATETIME,
                response_time REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                total_requests INTEGER DEFAULT 0,
                failed_requests INTEGER DEFAULT 0,
                is_active BOOLEAN DEFAULT 1,
                detection_risk REAL DEFAULT 0.0,
                bandwidth_limit INTEGER,
                concurrent_limit INTEGER DEFAULT 1,
                rotation_frequency INTEGER DEFAULT 100,
                usage_count INTEGER DEFAULT 0,
                created_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proxy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proxy_id TEXT,
                test_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                response_time REAL,
                success BOOLEAN,
                target_url TEXT,
                error_message TEXT,
                performance_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proxy_blacklist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proxy_id TEXT UNIQUE,
                reason TEXT,
                blacklisted_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                retry_after DATETIME,
                blacklist_count INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS proxy_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                proxy_id TEXT,
                start_time DATETIME,
                end_time DATETIME,
                request_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                average_response_time REAL,
                data_transferred INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS geographic_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                proxy_id TEXT,
                latitude REAL,
                longitude REAL,
                country_code TEXT,
                region_code TEXT,
                timezone TEXT,
                asn INTEGER,
                organization TEXT,
                last_geolocated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def initialize_components(self):
        """Initialize proxy rotation components"""
        self.session_manager = ProxySessionManager(self)
        self.load_balancer = ProxyLoadBalancer(self)
        self.circuit_breaker = ProxyCircuitBreaker(self)
        
        # Threading components
        self.request_queue = Queue()
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        # Metrics tracking
        self.metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'proxy_rotations': 0,
            'average_response_time': 0.0
        }
        
    def load_proxy_nodes(self):
        """Load proxy nodes from database and external sources"""
        # Load from database
        self._load_proxies_from_db()
        
        # Initialize with some default proxies if empty
        if not self.active_proxies and self.config.auto_discovery:
            self._initialize_default_proxies()
            
        # Discover additional proxies
        if self.config.auto_discovery:
            asyncio.create_task(self._discover_additional_proxies())
            
    def _load_proxies_from_db(self):
        """Load existing proxies from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT proxy_id, host, port, protocol, username, password, country,
                       region, city, isp, speed_score, reliability_score, anonymity_level,
                       last_tested, response_time, success_rate, total_requests, 
                       failed_requests, is_active, detection_risk, bandwidth_limit,
                       concurrent_limit, rotation_frequency, usage_count
                FROM proxy_nodes
                WHERE is_active = 1
                ORDER BY reliability_score DESC, speed_score DESC
            ''')
            
            for row in cursor.fetchall():
                proxy = ProxyNode(
                    proxy_id=row[0],
                    host=row[1],
                    port=row[2],
                    protocol=row[3],
                    username=row[4],
                    password=row[5],
                    country=row[6],
                    region=row[7],
                    city=row[8],
                    isp=row[9],
                    speed_score=row[10],
                    reliability_score=row[11],
                    anonymity_level=row[12],
                    last_tested=datetime.fromisoformat(row[13]) if row[13] else None,
                    response_time=row[14],
                    success_rate=row[15],
                    total_requests=row[16],
                    failed_requests=row[17],
                    is_active=bool(row[18]),
                    detection_risk=row[19],
                    bandwidth_limit=row[20],
                    concurrent_limit=row[21],
                    rotation_frequency=row[22],
                    usage_count=row[23]
                )
                
                self.active_proxies[proxy.proxy_id] = proxy
                self.proxy_queue.append(proxy.proxy_id)
                
            conn.close()
            logger.info(f"Loaded {len(self.active_proxies)} active proxies from database")
            
        except Exception as e:
            logger.error(f"Error loading proxies from database: {e}")
            
    def _initialize_default_proxies(self):
        """Initialize with default proxy sources"""
        default_sources = [
            # Free proxy sources (be careful with these in production)
            'https://www.proxy-list.download/api/v1/get?type=http',
            'https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all',
        ]
        
        # Note: In production, you'd want paid/reliable proxy services
        logger.info("Initializing default proxy sources...")
        
    async def _discover_additional_proxies(self):
        """Discover additional proxies from various sources"""
        try:
            discovered_proxies = await self.proxy_discoverer.discover_proxies()
            
            for proxy_info in discovered_proxies:
                proxy = self._create_proxy_node(proxy_info)
                if await self.proxy_validator.validate_proxy(proxy):
                    await self.add_proxy(proxy)
                    
            logger.info(f"Discovered and validated {len(discovered_proxies)} new proxies")
            
        except Exception as e:
            logger.error(f"Error discovering additional proxies: {e}")
            
    def _create_proxy_node(self, proxy_info: Dict) -> ProxyNode:
        """Create ProxyNode from proxy information"""
        proxy_id = f"{proxy_info['host']}:{proxy_info['port']}"
        
        return ProxyNode(
            proxy_id=proxy_id,
            host=proxy_info['host'],
            port=proxy_info['port'],
            protocol=proxy_info.get('protocol', 'http'),
            username=proxy_info.get('username'),
            password=proxy_info.get('password'),
            country=proxy_info.get('country'),
            region=proxy_info.get('region'),
            city=proxy_info.get('city'),
            isp=proxy_info.get('isp'),
            anonymity_level=proxy_info.get('anonymity_level', 'unknown')
        )
        
    async def add_proxy(self, proxy: ProxyNode):
        """Add new proxy to the rotation pool"""
        try:
            with self.rotation_lock:
                self.active_proxies[proxy.proxy_id] = proxy
                self.proxy_queue.append(proxy.proxy_id)
                
            # Store in database
            await self._store_proxy_in_db(proxy)
            
            logger.info(f"Added proxy {proxy.proxy_id} to rotation pool")
            
        except Exception as e:
            logger.error(f"Error adding proxy: {e}")
            
    async def _store_proxy_in_db(self, proxy: ProxyNode):
        """Store proxy in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO proxy_nodes
                (proxy_id, host, port, protocol, username, password, country,
                 region, city, isp, speed_score, reliability_score, anonymity_level,
                 last_tested, response_time, success_rate, total_requests, 
                 failed_requests, is_active, detection_risk, bandwidth_limit,
                 concurrent_limit, rotation_frequency, usage_count, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                proxy.proxy_id, proxy.host, proxy.port, proxy.protocol,
                proxy.username, proxy.password, proxy.country, proxy.region,
                proxy.city, proxy.isp, proxy.speed_score, proxy.reliability_score,
                proxy.anonymity_level, proxy.last_tested, proxy.response_time,
                proxy.success_rate, proxy.total_requests, proxy.failed_requests,
                proxy.is_active, proxy.detection_risk, proxy.bandwidth_limit,
                proxy.concurrent_limit, proxy.rotation_frequency, proxy.usage_count,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing proxy in database: {e}")
            
    async def get_next_proxy(self, target_url: Optional[str] = None) -> Optional[ProxyNode]:
        """Get next proxy based on rotation strategy"""
        try:
            with self.rotation_lock:
                if not self.proxy_queue:
                    logger.warning("No proxies available in rotation queue")
                    return None
                    
                # Apply rotation strategy
                if self.config.rotation_strategy == 'round_robin':
                    proxy_id = self._get_round_robin_proxy()
                elif self.config.rotation_strategy == 'weighted':
                    proxy_id = self._get_weighted_proxy()
                elif self.config.rotation_strategy == 'geographic':
                    proxy_id = self._get_geographic_proxy(target_url)
                elif self.config.rotation_strategy == 'performance':
                    proxy_id = self._get_performance_proxy()
                else:
                    proxy_id = self._get_round_robin_proxy()
                    
                if proxy_id and proxy_id in self.active_proxies:
                    proxy = self.active_proxies[proxy_id]
                    
                    # Check if proxy needs rotation
                    if self._should_rotate_proxy(proxy):
                        return await self.get_next_proxy(target_url)
                        
                    # Update usage
                    proxy.usage_count += 1
                    self.current_proxy = proxy
                    self.metrics['proxy_rotations'] += 1
                    
                    return proxy
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting next proxy: {e}")
            return None
            
    def _get_round_robin_proxy(self) -> Optional[str]:
        """Get next proxy using round-robin strategy"""
        if self.proxy_queue:
            proxy_id = self.proxy_queue.popleft()
            self.proxy_queue.append(proxy_id)
            return proxy_id
        return None
        
    def _get_weighted_proxy(self) -> Optional[str]:
        """Get proxy using weighted selection based on performance"""
        if not self.active_proxies:
            return None
            
        # Calculate weights based on performance metrics
        proxy_weights = []
        proxy_ids = list(self.active_proxies.keys())
        
        for proxy_id in proxy_ids:
            proxy = self.active_proxies[proxy_id]
            # Weight based on reliability and speed
            weight = proxy.reliability_score * 0.7 + proxy.speed_score * 0.3
            # Penalize high usage
            weight *= max(0.1, 1.0 - (proxy.usage_count / proxy.rotation_frequency))
            proxy_weights.append(weight)
            
        if proxy_weights and sum(proxy_weights) > 0:
            import random
            selected_proxy = random.choices(proxy_ids, weights=proxy_weights)[0]
            return selected_proxy
            
        return self._get_round_robin_proxy()
        
    def _get_geographic_proxy(self, target_url: Optional[str]) -> Optional[str]:
        """Get proxy based on geographic optimization"""
        if not target_url:
            return self._get_weighted_proxy()
            
        # Extract domain from URL
        from urllib.parse import urlparse
        domain = urlparse(target_url).netloc
        
        # Get optimal geographic location for domain
        optimal_country = self.geographic_manager.get_optimal_country(domain)
        
        # Filter proxies by country
        country_proxies = [
            proxy_id for proxy_id, proxy in self.active_proxies.items()
            if proxy.country == optimal_country
        ]
        
        if country_proxies:
            # Use weighted selection within country
            return random.choice(country_proxies)
        else:
            # Fallback to weighted selection
            return self._get_weighted_proxy()
            
    def _get_performance_proxy(self) -> Optional[str]:
        """Get proxy optimized for performance"""
        if not self.active_proxies:
            return None
            
        # Sort by performance score (combination of speed and reliability)
        sorted_proxies = sorted(
            self.active_proxies.items(),
            key=lambda x: (x[1].speed_score + x[1].reliability_score) / 2,
            reverse=True
        )
        
        # Select from top performers with some randomization
        top_count = min(5, len(sorted_proxies))
        top_proxies = sorted_proxies[:top_count]
        
        if top_proxies:
            return random.choice(top_proxies)[0]
            
        return None
        
    def _should_rotate_proxy(self, proxy: ProxyNode) -> bool:
        """Check if proxy should be rotated"""
        # Check usage limit
        if proxy.usage_count >= proxy.rotation_frequency:
            proxy.usage_count = 0  # Reset counter
            return True
            
        # Check performance degradation
        if proxy.success_rate < 0.5:
            return True
            
        # Check detection risk
        if proxy.detection_risk > self.config.blacklist_threshold:
            return True
            
        return False
        
    def create_proxied_session(self, proxy: ProxyNode) -> requests.Session:
        """Create requests session with proxy configuration"""
        session = requests.Session()
        
        try:
            # Configure proxy
            proxy_dict = self._get_proxy_dict(proxy)
            session.proxies.update(proxy_dict)
            
            # Configure retry strategy
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Set timeout
            session.timeout = (10, 30)
            
            # Configure SSL verification
            if self.config.stealth_level > 0.7:
                session.verify = False
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                
            return session
            
        except Exception as e:
            logger.error(f"Error creating proxied session: {e}")
            return session
            
    def _get_proxy_dict(self, proxy: ProxyNode) -> Dict[str, str]:
        """Get proxy dictionary for requests"""
        if proxy.username and proxy.password:
            auth = f"{proxy.username}:{proxy.password}@"
        else:
            auth = ""
            
        proxy_url = f"{proxy.protocol}://{auth}{proxy.host}:{proxy.port}"
        
        return {
            'http': proxy_url,
            'https': proxy_url
        }
        
    async def test_proxy_health(self, proxy: ProxyNode, test_urls: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test proxy health and performance"""
        if not test_urls:
            test_urls = [
                'http://httpbin.org/ip',
                'https://httpbin.org/user-agent',
                'http://www.google.com'
            ]
            
        results = {
            'proxy_id': proxy.proxy_id,
            'success_rate': 0.0,
            'average_response_time': 0.0,
            'anonymity_level': 'unknown',
            'working': False,
            'errors': []
        }
        
        try:
            session = self.create_proxied_session(proxy)
            successful_tests = 0
            total_response_time = 0
            
            for test_url in test_urls:
                try:
                    start_time = time.time()
                    response = session.get(test_url, timeout=15)
                    response_time = time.time() - start_time
                    
                    if response.status_code == 200:
                        successful_tests += 1
                        total_response_time += response_time
                        
                        # Check anonymity level
                        if 'httpbin.org/ip' in test_url:
                            results['anonymity_level'] = self._check_anonymity_level(response.json())
                            
                    await self._record_performance(proxy.proxy_id, test_url, response_time, True, None)
                    
                except Exception as e:
                    await self._record_performance(proxy.proxy_id, test_url, 0, False, str(e))
                    results['errors'].append(f"{test_url}: {str(e)}")
                    
            # Calculate metrics
            results['success_rate'] = successful_tests / len(test_urls)
            results['average_response_time'] = total_response_time / max(1, successful_tests)
            results['working'] = successful_tests > 0
            
            # Update proxy metrics
            proxy.success_rate = results['success_rate']
            proxy.response_time = results['average_response_time']
            proxy.anonymity_level = results['anonymity_level']
            proxy.last_tested = datetime.now()
            
            # Calculate performance scores
            proxy.speed_score = self._calculate_speed_score(results['average_response_time'])
            proxy.reliability_score = results['success_rate']
            
            # Update in database
            await self._store_proxy_in_db(proxy)
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing proxy health: {e}")
            results['errors'].append(f"Health check failed: {str(e)}")
            return results
            
    def _check_anonymity_level(self, ip_response: Dict) -> str:
        """Check anonymity level based on IP response"""
        # This is a simplified check - in production you'd want more sophisticated detection
        origin_ip = ip_response.get('origin', '')
        
        if ',' in origin_ip:  # Multiple IPs indicate transparent proxy
            return 'transparent'
        else:
            return 'anonymous'  # Simplified - elite detection requires more checks
            
    def _calculate_speed_score(self, response_time: float) -> float:
        """Calculate speed score from response time"""
        if response_time <= 1.0:
            return 1.0
        elif response_time <= 3.0:
            return 0.8
        elif response_time <= 5.0:
            return 0.6
        elif response_time <= 10.0:
            return 0.4
        else:
            return 0.2
            
    async def _record_performance(self, proxy_id: str, test_url: str, response_time: float, 
                                 success: bool, error_message: Optional[str]):
        """Record proxy performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            performance_score = response_time if success else 0
            
            cursor.execute('''
                INSERT INTO proxy_performance
                (proxy_id, response_time, success, target_url, error_message, performance_score)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (proxy_id, response_time, success, test_url, error_message, performance_score))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
            
    async def blacklist_proxy(self, proxy_id: str, reason: str, duration_hours: int = 24):
        """Blacklist a problematic proxy"""
        try:
            with self.rotation_lock:
                if proxy_id in self.active_proxies:
                    self.active_proxies[proxy_id].is_active = False
                    
                    # Remove from queue
                    self.proxy_queue = deque([pid for pid in self.proxy_queue if pid != proxy_id])
                    
            # Record in blacklist
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            retry_after = datetime.now() + timedelta(hours=duration_hours)
            
            cursor.execute('''
                INSERT OR REPLACE INTO proxy_blacklist
                (proxy_id, reason, retry_after, blacklist_count)
                VALUES (?, ?, ?, 
                    COALESCE((SELECT blacklist_count FROM proxy_blacklist WHERE proxy_id = ?), 0) + 1)
            ''', (proxy_id, reason, retry_after, proxy_id))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"Blacklisted proxy {proxy_id}: {reason}")
            
        except Exception as e:
            logger.error(f"Error blacklisting proxy: {e}")
            
    async def unblacklist_expired_proxies(self):
        """Unblacklist proxies whose blacklist period has expired"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT proxy_id FROM proxy_blacklist
                WHERE retry_after <= ?
            ''', (datetime.now(),))
            
            expired_proxies = [row[0] for row in cursor.fetchall()]
            
            for proxy_id in expired_proxies:
                # Remove from blacklist
                cursor.execute('DELETE FROM proxy_blacklist WHERE proxy_id = ?', (proxy_id,))
                
                # Reactivate proxy if it exists
                if proxy_id in self.active_proxies:
                    proxy = self.active_proxies[proxy_id]
                    proxy.is_active = True
                    
                    with self.rotation_lock:
                        self.proxy_queue.append(proxy_id)
                        
                    logger.info(f"Unblacklisted proxy {proxy_id}")
                    
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error unblacklisting proxies: {e}")
            
    def start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        def health_check_worker():
            """Background health checking"""
            while True:
                try:
                    asyncio.run(self.health_monitor.check_all_proxies())
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    time.sleep(60)
                    
        def maintenance_worker():
            """Background maintenance tasks"""
            while True:
                try:
                    asyncio.run(self.unblacklist_expired_proxies())
                    asyncio.run(self._cleanup_old_performance_data())
                    time.sleep(3600)  # Run hourly
                except Exception as e:
                    logger.error(f"Maintenance error: {e}")
                    time.sleep(300)
                    
        # Start background threads
        health_thread = threading.Thread(target=health_check_worker, daemon=True)
        health_thread.start()
        
        maintenance_thread = threading.Thread(target=maintenance_worker, daemon=True)
        maintenance_thread.start()
        
    async def _cleanup_old_performance_data(self):
        """Clean up old performance data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Keep only last 30 days of performance data
            cutoff_date = datetime.now() - timedelta(days=30)
            
            cursor.execute('''
                DELETE FROM proxy_performance
                WHERE test_timestamp < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old performance records")
                
        except Exception as e:
            logger.error(f"Error cleaning up performance data: {e}")
            
    def get_rotation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rotation statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute('SELECT COUNT(*) FROM proxy_nodes WHERE is_active = 1')
            active_proxies = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM proxy_blacklist')
            blacklisted_proxies = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(success_rate) FROM proxy_nodes WHERE is_active = 1')
            avg_success_rate = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT AVG(response_time) FROM proxy_nodes WHERE is_active = 1')
            avg_response_time = cursor.fetchone()[0] or 0
            
            # Geographic distribution
            cursor.execute('''
                SELECT country, COUNT(*) FROM proxy_nodes 
                WHERE is_active = 1 AND country IS NOT NULL
                GROUP BY country
            ''')
            country_distribution = dict(cursor.fetchall())
            
            # Protocol distribution
            cursor.execute('''
                SELECT protocol, COUNT(*) FROM proxy_nodes 
                WHERE is_active = 1
                GROUP BY protocol
            ''')
            protocol_distribution = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'active_proxies': active_proxies,
                'blacklisted_proxies': blacklisted_proxies,
                'average_success_rate': avg_success_rate,
                'average_response_time': avg_response_time,
                'country_distribution': country_distribution,
                'protocol_distribution': protocol_distribution,
                'current_proxy': self.current_proxy.proxy_id if self.current_proxy else None,
                'rotation_strategy': self.config.rotation_strategy,
                'total_rotations': self.metrics['proxy_rotations']
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

# Supporting classes
class ProxyHealthMonitor:
    """Monitor proxy health and performance"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        
    async def check_all_proxies(self):
        """Check health of all active proxies"""
        try:
            tasks = []
            for proxy in self.rotator.active_proxies.values():
                if proxy.is_active:
                    task = self.rotator.test_proxy_health(proxy)
                    tasks.append(task)
                    
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                failed_proxies = []
                for result in results:
                    if isinstance(result, dict) and not result.get('working'):
                        failed_proxies.append(result['proxy_id'])
                        
                # Blacklist consistently failing proxies
                for proxy_id in failed_proxies:
                    await self.rotator.blacklist_proxy(proxy_id, "Health check failure")
                    
                logger.info(f"Health check completed. {len(failed_proxies)} proxies failed.")
                
        except Exception as e:
            logger.error(f"Error in health monitoring: {e}")

class ProxyPerformanceTracker:
    """Track and analyze proxy performance"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        
    def track_request_performance(self, proxy_id: str, response_time: float, success: bool):
        """Track individual request performance"""
        if proxy_id in self.rotator.active_proxies:
            proxy = self.rotator.active_proxies[proxy_id]
            proxy.total_requests += 1
            
            if not success:
                proxy.failed_requests += 1
                
            # Update running averages
            proxy.success_rate = 1.0 - (proxy.failed_requests / proxy.total_requests)
            
            # Update response time (exponential moving average)
            alpha = 0.1
            proxy.response_time = alpha * response_time + (1 - alpha) * proxy.response_time

class GeographicProxyManager:
    """Manage geographic distribution of proxies"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        self.country_preferences = {}
        
    def get_optimal_country(self, domain: str) -> Optional[str]:
        """Get optimal country for target domain"""
        # Simple heuristics - in production you'd want more sophisticated geolocation
        domain_country_map = {
            'google.com': 'US',
            'baidu.com': 'CN',
            'yandex.ru': 'RU',
            'bbc.co.uk': 'GB'
        }
        
        for domain_pattern, country in domain_country_map.items():
            if domain_pattern in domain:
                return country
                
        return 'US'  # Default

class ProxySessionManager:
    """Manage proxy sessions and connection pooling"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        self.active_sessions = {}
        
    def get_session(self, proxy_id: str) -> requests.Session:
        """Get or create session for proxy"""
        if proxy_id not in self.active_sessions:
            proxy = self.rotator.active_proxies[proxy_id]
            session = self.rotator.create_proxied_session(proxy)
            self.active_sessions[proxy_id] = session
            
        return self.active_sessions[proxy_id]

class ProxyLoadBalancer:
    """Load balance requests across proxies"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        self.proxy_loads = defaultdict(int)
        
    def get_least_loaded_proxy(self) -> Optional[str]:
        """Get proxy with least current load"""
        if not self.rotator.active_proxies:
            return None
            
        return min(self.proxy_loads, key=self.proxy_loads.get, default=None)

class ProxyCircuitBreaker:
    """Circuit breaker pattern for proxy failure handling"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        self.circuit_states = {}  # proxy_id -> state
        
    def is_circuit_open(self, proxy_id: str) -> bool:
        """Check if circuit is open for proxy"""
        return self.circuit_states.get(proxy_id, 'closed') == 'open'

class ProxyDiscoverer:
    """Discover new proxies from various sources"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        
    async def discover_proxies(self) -> List[Dict]:
        """Discover proxies from various sources"""
        # In production, implement actual discovery from reliable sources
        discovered = []
        
        # Example: Parse proxy lists, API calls, etc.
        # This is a placeholder implementation
        
        return discovered

class ProxyValidator:
    """Validate proxy functionality and anonymity"""
    
    def __init__(self, rotator: AdvancedProxyRotator):
        self.rotator = rotator
        
    async def validate_proxy(self, proxy: ProxyNode) -> bool:
        """Validate proxy functionality"""
        try:
            health_result = await self.rotator.test_proxy_health(proxy)
            return health_result.get('working', False)
        except:
            return False

# Main execution and testing
if __name__ == "__main__":
    async def test_proxy_rotation():
        """Test proxy rotation system"""
        config = ProxyRotationConfig(
            rotation_strategy='weighted',
            max_requests_per_proxy=50,
            health_check_interval=60,
            stealth_level=0.8
        )
        
        rotator = AdvancedProxyRotator(config)
        
        # Add some test proxies (you'd use real proxies in production)
        test_proxies = [
            {'host': '127.0.0.1', 'port': 8080, 'protocol': 'http'},
            {'host': '127.0.0.1', 'port': 8081, 'protocol': 'http'},
        ]
        
        for proxy_info in test_proxies:
            proxy = rotator._create_proxy_node(proxy_info)
            await rotator.add_proxy(proxy)
            
        # Test proxy selection
        for i in range(5):
            proxy = await rotator.get_next_proxy()
            if proxy:
                print(f"Selected proxy {i+1}: {proxy.proxy_id}")
                
                # Test health
                health_result = await rotator.test_proxy_health(proxy)
                print(f"Health check: {health_result}")
            else:
                print(f"No proxy available for request {i+1}")
                
        # Get statistics
        stats = rotator.get_rotation_statistics()
        print(f"Rotation statistics: {stats}")
        
    # Run test
    asyncio.run(test_proxy_rotation())

