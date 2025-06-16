
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import json
import sqlite3
import asyncio
import aiohttp
import time
import random
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import hashlib
from cryptography.fernet import Fernet
import pickle
import re
from collections import defaultdict, Counter
import threading
from queue import Queue
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, automated_readability_index
import networkx as nx
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from dataclasses import dataclass
import yaml
import requests
import feedparser
from newspaper import Article
import pandas as pd
from scipy import stats
import seaborn as sns

@dataclass
class AnalysisContext:
    source_url: str
    content_type: str
    priority_level: float
    analysis_depth: str
    target_metrics: List[str]
    temporal_window: str

@dataclass
class NewsIntelligence:
    topic_clusters: Dict[str, List[str]]
    sentiment_trends: Dict[str, float]
    narrative_patterns: Dict[str, Any]
    influence_networks: Dict[str, List[str]]
    manipulation_indicators: Dict[str, float]
    credibility_scores: Dict[str, float]
    temporal_patterns: Dict[str, Any]
    contradiction_matrix: np.ndarray

class AdvancedNewsAnalyzer:
    """Production-grade news intelligence analyzer with sophisticated manipulation detection"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.logger = self._setup_logger()
        
        # Initialize NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
        
        # Database connections
        self.db_path = 'shock2/data/raw/news_analysis.db'
        self._init_database()
        
        # Analysis models
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))
        self.topic_model = LatentDirichletAllocation(n_components=20, random_state=42)
        
        # Intelligence components
        self.entity_network = nx.DiGraph()
        self.narrative_graph = nx.Graph()
        self.influence_graph = nx.DiGraph()
        
        # Memory systems
        self.analysis_cache = {}
        self.pattern_memory = defaultdict(list)
        self.temporal_patterns = defaultdict(list)
        
        # Configuration
        self.analysis_config = self._load_analysis_config()
        self.manipulation_indicators = self._load_manipulation_indicators()
        self.bias_patterns = self._load_bias_detection_patterns()
        
        # Real-time monitoring
        self.monitoring_threads = []
        self.analysis_queue = Queue()
        self.alert_thresholds = self._load_alert_thresholds()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/analyzer_llm.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize analysis database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_analysis (
                id INTEGER PRIMARY KEY,
                content_hash TEXT UNIQUE,
                source_url TEXT,
                analysis_type TEXT,
                intelligence_data TEXT,
                credibility_score REAL,
                manipulation_score REAL,
                influence_score REAL,
                temporal_significance REAL,
                analysis_timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id INTEGER PRIMARY KEY,
                entity_a TEXT,
                entity_b TEXT,
                relationship_type TEXT,
                strength REAL,
                context TEXT,
                first_observed TEXT,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS narrative_patterns (
                id INTEGER PRIMARY KEY,
                pattern_id TEXT UNIQUE,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER,
                effectiveness_score REAL,
                first_detected TEXT,
                last_seen TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS influence_metrics (
                id INTEGER PRIMARY KEY,
                source_identifier TEXT,
                influence_type TEXT,
                influence_score REAL,
                reach_estimate INTEGER,
                credibility_rating REAL,
                manipulation_tendency REAL,
                temporal_data TEXT,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_analysis_config(self):
        """Load analysis configuration"""
        return {
            'sentiment_thresholds': {
                'extreme_positive': 0.8,
                'extreme_negative': -0.8,
                'neutral_range': (-0.1, 0.1)
            },
            'credibility_factors': {
                'source_authority': 0.3,
                'fact_checking': 0.25,
                'citation_quality': 0.2,
                'temporal_consistency': 0.15,
                'linguistic_markers': 0.1
            },
            'manipulation_detection': {
                'emotional_manipulation': 0.4,
                'logical_fallacies': 0.3,
                'bias_injection': 0.2,
                'information_warfare': 0.1
            },
            'clustering_params': {
                'min_samples': 3,
                'eps': 0.3,
                'metric': 'cosine'
            }
        }
        
    def _load_manipulation_indicators(self):
        """Load manipulation detection indicators"""
        return {
            'emotional_triggers': {
                'fear_words': ['crisis', 'disaster', 'catastrophe', 'emergency', 'threat', 'danger'],
                'anger_words': ['outrage', 'scandal', 'betrayal', 'corruption', 'abuse'],
                'hope_words': ['breakthrough', 'solution', 'victory', 'success', 'progress'],
                'urgency_words': ['breaking', 'urgent', 'immediate', 'critical', 'now']
            },
            'logical_fallacies': {
                'ad_hominem': ['attacks', 'character', 'person', 'individual'],
                'straw_man': ['misrepresent', 'distort', 'exaggerate'],
                'false_dichotomy': ['only two', 'either or', 'black and white'],
                'appeal_to_authority': ['experts say', 'officials claim', 'studies show']
            },
            'bias_markers': {
                'confirmation_bias': ['confirms', 'proves', 'validates', 'supports'],
                'selection_bias': ['cherry-pick', 'selective', 'ignore', 'omit'],
                'framing_bias': ['spin', 'angle', 'perspective', 'interpretation']
            },
            'deception_patterns': {
                'misleading_statistics': r'\d+%?\s+(?:more|less|increase|decrease)',
                'vague_attributions': ['sources say', 'reports suggest', 'it is believed'],
                'emotional_appeals': ['think of the', 'for the sake of', 'protect our']
            }
        }
        
    def _load_bias_detection_patterns(self):
        """Load sophisticated bias detection patterns"""
        return {
            'political_bias': {
                'left_indicators': ['progressive', 'social justice', 'inequality', 'systemic'],
                'right_indicators': ['traditional', 'conservative', 'law and order', 'security'],
                'neutral_indicators': ['bipartisan', 'across the aisle', 'both sides']
            },
            'corporate_bias': {
                'pro_business': ['innovation', 'growth', 'efficiency', 'competition'],
                'anti_business': ['exploitation', 'greed', 'monopoly', 'unfair'],
                'regulatory_focus': ['oversight', 'compliance', 'standards', 'protection']
            },
            'cultural_bias': {
                'traditional': ['heritage', 'values', 'customs', 'established'],
                'progressive': ['diversity', 'inclusion', 'modern', 'evolving'],
                'generational': ['millennial', 'boomer', 'gen z', 'generation']
            }
        }
        
    def _load_alert_thresholds(self):
        """Load alert threshold configurations"""
        return {
            'high_manipulation': 0.8,
            'low_credibility': 0.3,
            'extreme_sentiment': 0.9,
            'viral_potential': 0.7,
            'contradiction_detected': 0.6,
            'influence_spike': 2.0
        }
        
    async def analyze_content_comprehensive(self, content: str, context: AnalysisContext) -> NewsIntelligence:
        """Perform comprehensive content analysis"""
        try:
            analysis_start = time.time()
            
            # Create content hash for caching
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Check cache first
            if content_hash in self.analysis_cache:
                self.logger.info(f"Returning cached analysis for {content_hash[:8]}")
                return self.analysis_cache[content_hash]
            
            # Parallel analysis components
            analysis_tasks = [
                self._analyze_topic_clusters(content),
                self._analyze_sentiment_trends(content),
                self._detect_narrative_patterns(content),
                self._map_influence_networks(content),
                self._detect_manipulation_indicators(content),
                self._assess_credibility(content, context),
                self._analyze_temporal_patterns(content),
                self._detect_contradictions(content)
            ]
            
            results = await asyncio.gather(*analysis_tasks)
            
            # Combine results into intelligence object
            intelligence = NewsIntelligence(
                topic_clusters=results[0],
                sentiment_trends=results[1],
                narrative_patterns=results[2],
                influence_networks=results[3],
                manipulation_indicators=results[4],
                credibility_scores=results[5],
                temporal_patterns=results[6],
                contradiction_matrix=results[7]
            )
            
            # Cache the results
            self.analysis_cache[content_hash] = intelligence
            
            # Store in database
            await self._store_analysis_results(content_hash, context, intelligence)
            
            analysis_time = time.time() - analysis_start
            self.logger.info(f"Comprehensive analysis completed in {analysis_time:.2f}s")
            
            return intelligence
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
            
    async def _analyze_topic_clusters(self, content: str) -> Dict[str, List[str]]:
        """Analyze and cluster topics in content"""
        try:
            # Extract entities and keywords
            doc = self.nlp(content)
            entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']]
            
            # Extract noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
            
            # Combine all topics
            all_topics = entities + noun_phrases
            topic_counts = Counter(all_topics)
            
            # Filter by frequency
            significant_topics = [topic for topic, count in topic_counts.items() if count >= 2]
            
            if not significant_topics:
                return {'general': all_topics[:10]}
                
            # Create topic vectors for clustering
            topic_texts = [f"topic about {topic}" for topic in significant_topics]
            
            if len(topic_texts) > 5:
                # Vectorize topics
                vectors = self.tfidf.fit_transform(topic_texts)
                
                # Perform clustering
                clustering = DBSCAN(**self.analysis_config['clustering_params'])
                cluster_labels = clustering.fit_predict(vectors.toarray())
                
                # Group topics by cluster
                clusters = defaultdict(list)
                for topic, label in zip(significant_topics, cluster_labels):
                    cluster_name = f"cluster_{label}" if label != -1 else "miscellaneous"
                    clusters[cluster_name].append(topic)
                    
                return dict(clusters)
            else:
                return {'main_topics': significant_topics}
                
        except Exception as e:
            self.logger.error(f"Error in topic clustering: {str(e)}")
            return {'error': ['clustering_failed']}
            
    async def _analyze_sentiment_trends(self, content: str) -> Dict[str, float]:
        """Analyze sentiment trends and patterns"""
        try:
            # Overall sentiment
            overall_sentiment = self.sentiment_analyzer.polarity_scores(content)
            
            # Sentence-level sentiment analysis
            sentences = content.split('.')
            sentence_sentiments = []
            
            for sentence in sentences:
                if len(sentence.strip()) > 10:
                    sent_score = self.sentiment_analyzer.polarity_scores(sentence)
                    sentence_sentiments.append(sent_score['compound'])
                    
            # Calculate trends
            sentiment_trends = {
                'overall_compound': overall_sentiment['compound'],
                'overall_positive': overall_sentiment['pos'],
                'overall_negative': overall_sentiment['neg'],
                'overall_neutral': overall_sentiment['neu'],
                'sentiment_variance': np.var(sentence_sentiments) if sentence_sentiments else 0,
                'sentiment_progression': self._calculate_sentiment_progression(sentence_sentiments),
                'emotional_intensity': abs(overall_sentiment['compound']),
                'sentiment_consistency': 1 - np.var(sentence_sentiments) if sentence_sentiments else 1
            }
            
            # Detect emotional manipulation
            manipulation_score = self._detect_emotional_manipulation(content, sentiment_trends)
            sentiment_trends['manipulation_likelihood'] = manipulation_score
            
            return sentiment_trends
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {'error': -999}
            
    def _calculate_sentiment_progression(self, sentiments: List[float]) -> str:
        """Calculate how sentiment progresses through the content"""
        if len(sentiments) < 3:
            return 'insufficient_data'
            
        # Divide into thirds
        third_size = len(sentiments) // 3
        first_third = np.mean(sentiments[:third_size])
        middle_third = np.mean(sentiments[third_size:2*third_size])
        last_third = np.mean(sentiments[2*third_size:])
        
        # Determine progression pattern
        if first_third < middle_third < last_third:
            return 'positive_progression'
        elif first_third > middle_third > last_third:
            return 'negative_progression'
        elif abs(first_third - last_third) < 0.1:
            return 'stable'
        elif first_third < 0 and last_third > 0:
            return 'recovery_narrative'
        elif first_third > 0 and last_third < 0:
            return 'decline_narrative'
        else:
            return 'volatile'
            
    def _detect_emotional_manipulation(self, content: str, sentiment_data: Dict) -> float:
        """Detect likelihood of emotional manipulation"""
        manipulation_score = 0.0
        content_lower = content.lower()
        
        # Check for emotional trigger words
        for emotion_type, words in self.manipulation_indicators['emotional_triggers'].items():
            trigger_count = sum(1 for word in words if word in content_lower)
            manipulation_score += trigger_count * 0.1
            
        # Check for extreme sentiment with high intensity
        if sentiment_data['emotional_intensity'] > 0.8:
            manipulation_score += 0.3
            
        # Check for sentiment volatility (emotional rollercoaster)
        if sentiment_data['sentiment_variance'] > 0.5:
            manipulation_score += 0.2
            
        # Check for specific manipulation patterns
        manipulation_patterns = [
            'you should be afraid',
            'this will destroy',
            'unprecedented crisis',
            'never seen before',
            'shocking revelation'
        ]
        
        pattern_count = sum(1 for pattern in manipulation_patterns if pattern in content_lower)
        manipulation_score += pattern_count * 0.15
        
        return min(1.0, manipulation_score)
        
    async def _detect_narrative_patterns(self, content: str) -> Dict[str, Any]:
        """Detect narrative patterns and framing techniques"""
        try:
            patterns = {
                'framing_techniques': {},
                'narrative_structure': '',
                'persuasion_techniques': [],
                'story_elements': {},
                'bias_indicators': {}
            }
            
            content_lower = content.lower()
            
            # Detect framing techniques
            framing_indicators = {
                'problem_solution': ['problem', 'solution', 'fix', 'resolve'],
                'hero_villain': ['hero', 'villain', 'good', 'evil', 'fight'],
                'victim_perpetrator': ['victim', 'perpetrator', 'abuse', 'exploit'],
                'us_vs_them': ['us', 'them', 'our', 'their', 'we', 'they']
            }
            
            for frame_type, indicators in framing_indicators.items():
                score = sum(1 for indicator in indicators if indicator in content_lower)
                patterns['framing_techniques'][frame_type] = score / len(indicators)
                
            # Detect narrative structure
            structure_indicators = {
                'chronological': ['first', 'then', 'next', 'finally', 'after'],
                'cause_effect': ['because', 'therefore', 'as a result', 'consequently'],
                'compare_contrast': ['however', 'but', 'on the other hand', 'whereas'],
                'problem_solution': ['problem', 'issue', 'solution', 'answer']
            }
            
            max_score = 0
            dominant_structure = 'unclear'
            
            for structure, indicators in structure_indicators.items():
                score = sum(1 for indicator in indicators if indicator in content_lower)
                if score > max_score:
                    max_score = score
                    dominant_structure = structure
                    
            patterns['narrative_structure'] = dominant_structure
            
            # Detect persuasion techniques
            persuasion_techniques = {
                'authority_appeal': ['expert', 'professor', 'official', 'authority'],
                'emotion_appeal': ['feel', 'emotion', 'heart', 'passionate'],
                'logic_appeal': ['evidence', 'proof', 'data', 'statistics'],
                'social_proof': ['everyone', 'most people', 'popular', 'trending']
            }
            
            for technique, indicators in persuasion_techniques.items():
                if any(indicator in content_lower for indicator in indicators):
                    patterns['persuasion_techniques'].append(technique)
                    
            # Detect story elements
            story_elements = {
                'conflict': bool(re.search(r'\b(conflict|fight|battle|struggle|dispute)\b', content_lower)),
                'tension': bool(re.search(r'\b(tension|stress|pressure|strain)\b', content_lower)),
                'resolution': bool(re.search(r'\b(resolve|solution|answer|fix)\b', content_lower)),
                'character_development': len(re.findall(r'\b(he|she|they)\s+(?:said|stated|claimed|argued)\b', content_lower)) > 3
            }
            
            patterns['story_elements'] = story_elements
            
            # Detect bias indicators
            for bias_type, indicators in self.bias_patterns.items():
                bias_scores = {}
                for subtype, words in indicators.items():
                    score = sum(1 for word in words if word in content_lower)
                    bias_scores[subtype] = score
                patterns['bias_indicators'][bias_type] = bias_scores
                
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error in narrative pattern detection: {str(e)}")
            return {'error': 'pattern_detection_failed'}
            
    async def _map_influence_networks(self, content: str) -> Dict[str, List[str]]:
        """Map influence networks and entity relationships"""
        try:
            doc = self.nlp(content)
            
            # Extract entities
            persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
            locations = [ent.text for ent in doc.ents if ent.label_ == 'GPE']
            
            # Find relationships through co-occurrence and dependency parsing
            relationships = {
                'person_org_connections': [],
                'org_org_connections': [],
                'location_associations': [],
                'influence_indicators': []
            }
            
            # Person-Organization connections
            for person in persons:
                for org in organizations:
                    # Check if they appear in same sentence
                    sentences = content.split('.')
                    for sentence in sentences:
                        if person in sentence and org in sentence:
                            relationships['person_org_connections'].append({
                                'person': person,
                                'organization': org,
                                'context': sentence.strip(),
                                'relationship_strength': self._calculate_relationship_strength(sentence, person, org)
                            })
                            
            # Organization-Organization connections
            for i, org1 in enumerate(organizations):
                for org2 in organizations[i+1:]:
                    sentences = content.split('.')
                    for sentence in sentences:
                        if org1 in sentence and org2 in sentence:
                            relationships['org_org_connections'].append({
                                'org1': org1,
                                'org2': org2,
                                'context': sentence.strip(),
                                'relationship_type': self._classify_org_relationship(sentence)
                            })
                            
            # Location associations
            for location in locations:
                associated_entities = []
                sentences = content.split('.')
                for sentence in sentences:
                    if location in sentence:
                        sent_doc = self.nlp(sentence)
                        for ent in sent_doc.ents:
                            if ent.label_ in ['PERSON', 'ORG'] and ent.text != location:
                                associated_entities.append(ent.text)
                                
                if associated_entities:
                    relationships['location_associations'].append({
                        'location': location,
                        'associated_entities': list(set(associated_entities))
                    })
                    
            # Influence indicators
            influence_patterns = [
                r'(\w+)\s+(?:influence|control|lead|direct|manage)\s+(\w+)',
                r'(\w+)\s+(?:fund|support|back|sponsor)\s+(\w+)',
                r'(\w+)\s+(?:work with|partner with|collaborate with)\s+(\w+)'
            ]
            
            for pattern in influence_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    relationships['influence_indicators'].append({
                        'influencer': match[0],
                        'influenced': match[1],
                        'type': 'direct_influence'
                    })
                    
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error in influence network mapping: {str(e)}")
            return {'error': ['network_mapping_failed']}
            
    def _calculate_relationship_strength(self, sentence: str, entity1: str, entity2: str) -> float:
        """Calculate relationship strength between entities"""
        strength_indicators = {
            'strong': ['partnership', 'alliance', 'merger', 'acquisition'],
            'medium': ['collaboration', 'cooperation', 'agreement'],
            'weak': ['meeting', 'discussion', 'mention']
        }
        
        sentence_lower = sentence.lower()
        
        for strength, indicators in strength_indicators.items():
            if any(indicator in sentence_lower for indicator in indicators):
                return {'strong': 1.0, 'medium': 0.6, 'weak': 0.3}[strength]
                
        # Default strength based on proximity
        words = sentence.split()
        try:
            pos1 = next(i for i, word in enumerate(words) if entity1.lower() in word.lower())
            pos2 = next(i for i, word in enumerate(words) if entity2.lower() in word.lower())
            distance = abs(pos1 - pos2)
            return max(0.1, 1.0 - (distance / len(words)))
        except StopIteration:
            return 0.1
            
    def _classify_org_relationship(self, sentence: str) -> str:
        """Classify the type of relationship between organizations"""
        sentence_lower = sentence.lower()
        
        relationship_types = {
            'competitive': ['compete', 'rival', 'against', 'versus'],
            'collaborative': ['partner', 'alliance', 'cooperation', 'joint'],
            'hierarchical': ['parent', 'subsidiary', 'owns', 'acquired'],
            'regulatory': ['regulate', 'oversee', 'monitor', 'compliance']
        }
        
        for rel_type, indicators in relationship_types.items():
            if any(indicator in sentence_lower for indicator in indicators):
                return rel_type
                
        return 'unspecified'
        
    async def _detect_manipulation_indicators(self, content: str) -> Dict[str, float]:
        """Detect various manipulation indicators"""
        try:
            manipulation_scores = {
                'emotional_manipulation': 0.0,
                'logical_fallacies': 0.0,
                'bias_injection': 0.0,
                'information_warfare': 0.0,
                'credibility_undermining': 0.0,
                'urgency_manufacturing': 0.0
            }
            
            content_lower = content.lower()
            
            # Emotional manipulation detection
            emotion_score = 0.0
            for emotion_type, words in self.manipulation_indicators['emotional_triggers'].items():
                count = sum(1 for word in words if word in content_lower)
                emotion_score += count * 0.1
                
            manipulation_scores['emotional_manipulation'] = min(1.0, emotion_score)
            
            # Logical fallacies detection
            fallacy_score = 0.0
            for fallacy_type, indicators in self.manipulation_indicators['logical_fallacies'].items():
                count = sum(1 for indicator in indicators if indicator in content_lower)
                fallacy_score += count * 0.15
                
            manipulation_scores['logical_fallacies'] = min(1.0, fallacy_score)
            
            # Bias injection detection
            bias_score = 0.0
            for bias_type, indicators in self.manipulation_indicators['bias_markers'].items():
                count = sum(1 for indicator in indicators if indicator in content_lower)
                bias_score += count * 0.12
                
            manipulation_scores['bias_injection'] = min(1.0, bias_score)
            
            # Information warfare indicators
            warfare_patterns = [
                'fake news', 'disinformation', 'propaganda', 'psyop',
                'narrative warfare', 'influence operation'
            ]
            warfare_score = sum(0.2 for pattern in warfare_patterns if pattern in content_lower)
            manipulation_scores['information_warfare'] = min(1.0, warfare_score)
            
            # Credibility undermining
            undermining_patterns = [
                'unreliable source', 'questionable claims', 'unverified reports',
                'anonymous sources', 'alleged', 'supposedly'
            ]
            undermining_score = sum(0.15 for pattern in undermining_patterns if pattern in content_lower)
            manipulation_scores['credibility_undermining'] = min(1.0, undermining_score)
            
            # Urgency manufacturing
            urgency_patterns = [
                'breaking', 'urgent', 'immediate action required', 'crisis',
                'emergency', 'time is running out', 'act now'
            ]
            urgency_score = sum(0.18 for pattern in urgency_patterns if pattern in content_lower)
            manipulation_scores['urgency_manufacturing'] = min(1.0, urgency_score)
            
            return manipulation_scores
            
        except Exception as e:
            self.logger.error(f"Error in manipulation detection: {str(e)}")
            return {'error': -1}
            
    async def _assess_credibility(self, content: str, context: AnalysisContext) -> Dict[str, float]:
        """Assess content credibility using multiple factors"""
        try:
            credibility_factors = {
                'source_authority': 0.0,
                'fact_checking_indicators': 0.0,
                'citation_quality': 0.0,
                'temporal_consistency': 0.0,
                'linguistic_markers': 0.0,
                'overall_credibility': 0.0
            }
            
            # Source authority assessment
            authority_indicators = [
                'according to official sources',
                'government statement',
                'verified information',
                'confirmed by authorities',
                'official announcement'
            ]
            
            authority_score = sum(0.2 for indicator in authority_indicators 
                                if indicator.lower() in content.lower())
            credibility_factors['source_authority'] = min(1.0, authority_score)
            
            # Fact-checking indicators
            fact_check_indicators = [
                'fact check', 'verified', 'confirmed', 'cross-referenced',
                'multiple sources', 'independent verification'
            ]
            
            fact_score = sum(0.15 for indicator in fact_check_indicators 
                           if indicator.lower() in content.lower())
            credibility_factors['fact_checking_indicators'] = min(1.0, fact_score)
            
            # Citation quality
            citation_patterns = [
                r'according to \w+',
                r'source: \w+',
                r'\[\d+\]',  # Reference numbers
                r'https?://[^\s]+'  # URLs
            ]
            
            citation_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) 
                               for pattern in citation_patterns)
            credibility_factors['citation_quality'] = min(1.0, citation_count * 0.1)
            
            # Temporal consistency
            date_patterns = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            dates = re.findall(date_patterns, content)
            
            if dates:
                # Check if dates are recent and consistent
                temporal_score = 0.8 if len(dates) <= 3 else 0.4  # Too many dates might indicate confusion
            else:
                temporal_score = 0.2  # Low score for no temporal anchoring
                
            credibility_factors['temporal_consistency'] = temporal_score
            
            # Linguistic markers of credibility
            credible_language = [
                'evidence suggests', 'research shows', 'data indicates',
                'study found', 'analysis reveals', 'investigation uncovered'
            ]
            
            language_score = sum(0.1 for marker in credible_language 
                               if marker.lower() in content.lower())
            credibility_factors['linguistic_markers'] = min(1.0, language_score)
            
            # Calculate overall credibility
            weights = self.analysis_config['credibility_factors']
            overall = sum(credibility_factors[factor] * weights[factor] 
                         for factor in weights.keys())
            credibility_factors['overall_credibility'] = overall
            
            return credibility_factors
            
        except Exception as e:
            self.logger.error(f"Error in credibility assessment: {str(e)}")
            return {'error': -1}
            
    async def _analyze_temporal_patterns(self, content: str) -> Dict[str, Any]:
        """Analyze temporal patterns and timing significance"""
        try:
            temporal_analysis = {
                'time_references': [],
                'temporal_density': 0.0,
                'urgency_indicators': 0.0,
                'historical_context': False,
                'future_projections': False,
                'temporal_manipulation': 0.0
            }
            
            # Extract time references
            time_patterns = [
                r'\b(?:today|yesterday|tomorrow|now|recently|soon)\b',
                r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
                r'\b\d{4}\b',  # Years
                r'\b(?:last|next|this)\s+(?:week|month|year|decade)\b'
            ]
            
            all_time_refs = []
            for pattern in time_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                all_time_refs.extend(matches)
                
            temporal_analysis['time_references'] = all_time_refs
            temporal_analysis['temporal_density'] = len(all_time_refs) / len(content.split()) * 100
            
            # Urgency indicators
            urgency_words = ['urgent', 'immediate', 'now', 'emergency', 'crisis', 'breaking']
            urgency_count = sum(1 for word in urgency_words if word.lower() in content.lower())
            temporal_analysis['urgency_indicators'] = urgency_count / len(content.split()) * 100
            
            # Historical context
            historical_indicators = ['history', 'past', 'previous', 'before', 'historically']
            temporal_analysis['historical_context'] = any(
                indicator in content.lower() for indicator in historical_indicators
            )
            
            # Future projections
            future_indicators = ['will', 'future', 'predict', 'forecast', 'expect']
            temporal_analysis['future_projections'] = any(
                indicator in content.lower() for indicator in future_indicators
            )
            
            # Temporal manipulation detection
            manipulation_patterns = [
                'time is running out', 'act now', 'limited time', 'urgent action needed',
                'before it\'s too late', 'window of opportunity', 'now or never'
            ]
            
            manipulation_score = sum(0.2 for pattern in manipulation_patterns 
                                   if pattern.lower() in content.lower())
            temporal_analysis['temporal_manipulation'] = min(1.0, manipulation_score)
            
            return temporal_analysis
            
        except Exception as e:
            self.logger.error(f"Error in temporal analysis: {str(e)}")
            return {'error': 'temporal_analysis_failed'}
            
    async def _detect_contradictions(self, content: str) -> np.ndarray:
        """Detect contradictions and inconsistencies in content"""
        try:
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
            
            if len(sentences) < 2:
                return np.array([[1.0]])  # Single sentence, no contradictions
                
            # Create similarity matrix
            similarity_matrix = np.zeros((len(sentences), len(sentences)))
            
            # Calculate semantic similarity between sentences
            sentence_vectors = []
            for sentence in sentences:
                # Simple word-based vectorization
                words = sentence.lower().split()
                vector = np.zeros(1000)  # Fixed size vector
                
                for i, word in enumerate(words[:100]):  # Limit to first 100 words
                    vector[hash(word) % 1000] = 1
                    
                sentence_vectors.append(vector)
                
            # Calculate contradictions
            contradiction_matrix = np.zeros((len(sentences), len(sentences)))
            
            contradiction_indicators = [
                ('not', 'is'), ('never', 'always'), ('impossible', 'possible'),
                ('false', 'true'), ('wrong', 'right'), ('bad', 'good')
            ]
            
            for i, sent1 in enumerate(sentences):
                for j, sent2 in enumerate(sentences):
                    if i != j:
                        # Check for explicit contradictions
                        contradiction_score = 0.0
                        
                        for neg_word, pos_word in contradiction_indicators:
                            if (neg_word in sent1.lower() and pos_word in sent2.lower()) or \
                               (pos_word in sent1.lower() and neg_word in sent2.lower()):
                                contradiction_score += 0.3
                                
                        # Check for semantic contradictions
                        sent1_words = set(sent1.lower().split())
                        sent2_words = set(sent2.lower().split())
                        
                        # Look for negation patterns
                        if 'not' in sent1_words and not ('not' in sent2_words):
                            common_words = sent1_words.intersection(sent2_words)
                            if len(common_words) > 2:
                                contradiction_score += 0.2
                                
                        contradiction_matrix[i][j] = min(1.0, contradiction_score)
                        
            return contradiction_matrix
            
        except Exception as e:
            self.logger.error(f"Error in contradiction detection: {str(e)}")
            return np.array([[0.0]])
            
    async def _store_analysis_results(self, content_hash: str, context: AnalysisContext, 
                                    intelligence: NewsIntelligence):
        """Store analysis results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate aggregate scores
            manipulation_score = np.mean(list(intelligence.manipulation_indicators.values()))
            credibility_score = intelligence.credibility_scores.get('overall_credibility', 0.0)
            influence_score = len(intelligence.influence_networks.get('person_org_connections', []))
            
            cursor.execute('''
                INSERT OR REPLACE INTO news_analysis 
                (content_hash, source_url, analysis_type, intelligence_data, 
                 credibility_score, manipulation_score, influence_score, 
                 temporal_significance, analysis_timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                content_hash,
                context.source_url,
                context.analysis_depth,
                json.dumps({
                    'topic_clusters': intelligence.topic_clusters,
                    'sentiment_trends': intelligence.sentiment_trends,
                    'narrative_patterns': intelligence.narrative_patterns,
                    'temporal_patterns': intelligence.temporal_patterns
                }),
                credibility_score,
                manipulation_score,
                influence_score,
                intelligence.temporal_patterns.get('urgency_indicators', 0.0),
                datetime.now().isoformat(),
                json.dumps({
                    'priority_level': context.priority_level,
                    'target_metrics': context.target_metrics
                })
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored analysis results for {content_hash[:8]}")
            
        except Exception as e:
            self.logger.error(f"Error storing analysis results: {str(e)}")
            
    async def continuous_monitoring_cycle(self):
        """Continuous monitoring and analysis cycle"""
        self.logger.info("Starting continuous monitoring cycle")
        
        while True:
            try:
                # Process analysis queue
                while not self.analysis_queue.empty():
                    content, context = self.analysis_queue.get()
                    intelligence = await self.analyze_content_comprehensive(content, context)
                    
                    # Check for alerts
                    await self._check_analysis_alerts(intelligence, context)
                    
                # Sleep for monitoring interval
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring cycle: {str(e)}")
                await asyncio.sleep(30)  # 30 seconds before retry
                
    async def _check_analysis_alerts(self, intelligence: NewsIntelligence, context: AnalysisContext):
        """Check analysis results against alert thresholds"""
        try:
            alerts = []
            
            # Check manipulation score
            avg_manipulation = np.mean(list(intelligence.manipulation_indicators.values()))
            if avg_manipulation > self.alert_thresholds['high_manipulation']:
                alerts.append({
                    'type': 'high_manipulation_detected',
                    'score': avg_manipulation,
                    'source': context.source_url
                })
                
            # Check credibility score
            credibility = intelligence.credibility_scores.get('overall_credibility', 1.0)
            if credibility < self.alert_thresholds['low_credibility']:
                alerts.append({
                    'type': 'low_credibility_content',
                    'score': credibility,
                    'source': context.source_url
                })
                
            # Check for contradictions
            if intelligence.contradiction_matrix.size > 1:
                max_contradiction = np.max(intelligence.contradiction_matrix)
                if max_contradiction > self.alert_thresholds['contradiction_detected']:
                    alerts.append({
                        'type': 'contradiction_detected',
                        'score': max_contradiction,
                        'source': context.source_url
                    })
                    
            # Log alerts
            for alert in alerts:
                self.logger.warning(f"ALERT: {alert['type']} - Score: {alert['score']:.3f} - Source: {alert['source']}")
                
        except Exception as e:
            self.logger.error(f"Error checking alerts: {str(e)}")

# Example usage and testing
async def main():
    """Main execution function for testing"""
    analyzer = AdvancedNewsAnalyzer()
    
    # Example content analysis
    test_content = """
    Breaking news: Government officials announced today a shocking new policy that will fundamentally 
    change how citizens interact with technology. Sources close to the investigation reveal that this 
    unprecedented move comes after months of secret negotiations. 
    
    Critics argue this confirms long-standing concerns about government overreach, while supporters 
    claim it's necessary for national security. The emotional impact on families cannot be understated,
    as this could affect millions of people immediately.
    
    However, some experts question the timing of this announcement, suggesting it may be designed to 
    distract from other pressing issues. The full implications remain unclear, but urgent action 
    may be required from citizens to protect their rights.
    """
    
    context = AnalysisContext(
        source_url="https://example.com/news",
        content_type="breaking_news",
        priority_level=0.9,
        analysis_depth="comprehensive",
        target_metrics=["manipulation", "credibility", "influence"],
        temporal_window="24h"
    )
    
    # Perform analysis
    intelligence = await analyzer.analyze_content_comprehensive(test_content, context)
    
    print("Analysis Results:")
    print("=" * 50)
    print(f"Topic Clusters: {intelligence.topic_clusters}")
    print(f"Sentiment Trends: {intelligence.sentiment_trends}")
    print(f"Manipulation Indicators: {intelligence.manipulation_indicators}")
    print(f"Credibility Scores: {intelligence.credibility_scores}")
    print(f"Narrative Patterns: {intelligence.narrative_patterns}")

if __name__ == "__main__":
    asyncio.run(main())
