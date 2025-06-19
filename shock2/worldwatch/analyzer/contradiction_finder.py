
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import sqlite3
import asyncio
import aiohttp
import json
import logging
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import hashlib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import networkx as nx
from collections import defaultdict, Counter
import difflib
from itertools import combinations
import pickle
from scipy import stats
from textstat import flesch_reading_ease
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ContradictionContext:
    source_a: str
    source_b: str
    content_type: str
    temporal_window: str
    analysis_depth: str
    priority_level: float

@dataclass
class ContradictionResult:
    contradiction_score: float
    contradiction_type: str
    confidence: float
    source_statements: List[str]
    conflicting_claims: List[Dict[str, Any]]
    semantic_distance: float
    temporal_factors: Dict[str, Any]
    severity_level: str
    evidence_quality: Dict[str, Any]
    recommendations: List[str]

class SemanticContradictionDetector:
    """Advanced semantic contradiction detection using transformers"""
    
    def __init__(self, model_name='microsoft/DialoGPT-medium'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Initialize NLI model for contradiction detection
        self.nli_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-large-mnli')
        self.nli_model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-large-mnli').to(self.device)
        
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        
    def extract_semantic_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract semantic embeddings for text comparison"""
        embeddings = []
        
        for text in texts:
            try:
                tokens = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                      max_length=512, padding=True).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**tokens)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    
                embeddings.append(embedding)
            except Exception as e:
                # Fallback to zero embedding
                embeddings.append(np.zeros(768))
                
        return np.array(embeddings)
    
    def detect_nli_contradiction(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Use NLI model to detect contradictions"""
        try:
            inputs = self.nli_tokenizer(premise, hypothesis, return_tensors='pt', 
                                      truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
                
            # Labels: 0=contradiction, 1=neutral, 2=entailment
            contradiction_prob = probabilities[0][0].item()
            neutral_prob = probabilities[0][1].item()
            entailment_prob = probabilities[0][2].item()
            
            return {
                'contradiction': contradiction_prob,
                'neutral': neutral_prob,
                'entailment': entailment_prob,
                'confidence': max(contradiction_prob, neutral_prob, entailment_prob)
            }
        except Exception as e:
            return {
                'contradiction': 0.0,
                'neutral': 1.0,
                'entailment': 0.0,
                'confidence': 0.0
            }
    
    def find_semantic_contradictions(self, statements: List[str]) -> List[Dict]:
        """Find semantic contradictions between statements"""
        contradictions = []
        
        if len(statements) < 2:
            return contradictions
            
        # Extract embeddings
        embeddings = self.extract_semantic_embeddings(statements)
        
        # Compare all pairs of statements
        for i, j in combinations(range(len(statements)), 2):
            stmt_a = statements[i]
            stmt_b = statements[j]
            
            # Calculate semantic similarity
            emb_a = embeddings[i].reshape(1, -1)
            emb_b = embeddings[j].reshape(1, -1)
            semantic_similarity = cosine_similarity(emb_a, emb_b)[0][0]
            
            # Use NLI model for contradiction detection
            nli_result_ab = self.detect_nli_contradiction(stmt_a, stmt_b)
            nli_result_ba = self.detect_nli_contradiction(stmt_b, stmt_a)
            
            # Calculate contradiction score
            contradiction_score = max(nli_result_ab['contradiction'], nli_result_ba['contradiction'])
            
            # If high contradiction probability or low semantic similarity with conflicting content
            if contradiction_score > 0.7 or (semantic_similarity < 0.3 and contradiction_score > 0.5):
                contradictions.append({
                    'statement_a': stmt_a,
                    'statement_b': stmt_b,
                    'statement_a_index': i,
                    'statement_b_index': j,
                    'contradiction_score': contradiction_score,
                    'semantic_similarity': semantic_similarity,
                    'nli_ab': nli_result_ab,
                    'nli_ba': nli_result_ba,
                    'confidence': max(nli_result_ab['confidence'], nli_result_ba['confidence'])
                })
                
        return sorted(contradictions, key=lambda x: x['contradiction_score'], reverse=True)

class FactualContradictionDetector:
    """Detect factual contradictions in claims and statements"""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.fact_patterns = self._load_fact_patterns()
        self.number_patterns = self._load_number_patterns()
        self.temporal_patterns = self._load_temporal_patterns()
        
    def _load_fact_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for different types of factual claims"""
        return {
            'quantitative': [
                r'(\d+(?:\.\d+)?)\s*(percent|%|percentage)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(people|individuals|citizens)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(dollars|USD|\$)',
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(miles|kilometers|km)'
            ],
            'temporal': [
                r'((?:19|20)\d{2})\s*(?:year|yr)',
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s*(\d{1,2}),?\s*((?:19|20)\d{2})',
                r'(\d{1,2})/(\d{1,2})/(\d{2,4})'
            ],
            'categorical': [
                r'(is|was|are|were)\s+(the\s+)?(first|second|third|last|only)',
                r'(always|never|all|none|every|no)',
                r'(highest|lowest|best|worst|greatest|smallest)'
            ]
        }
        
    def _load_number_patterns(self) -> List[str]:
        """Load patterns for extracting numbers and quantities"""
        return [
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            r'\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b',
            r'\b(?:first|second|third|fourth|fifth)\b',
            r'\d+(?:\.\d+)?%',
            r'\$\d+(?:,\d{3})*(?:\.\d+)?'
        ]
        
    def _load_temporal_patterns(self) -> List[str]:
        """Load patterns for extracting temporal information"""
        return [
            r'(?:19|20)\d{2}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*(?:19|20)?\d{2}',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'(?:yesterday|today|tomorrow|last\s+week|next\s+week|last\s+month|next\s+month)'
        ]
    
    def extract_factual_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual claims from text"""
        claims = []
        doc = self.nlp(text)
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'PERCENT', 'DATE', 'CARDINAL']:
                claims.append({
                    'type': 'entity',
                    'entity_type': ent.label_,
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'context': text[max(0, ent.start_char-50):ent.end_char+50]
                })
                
        # Extract quantitative claims
        for pattern in self.fact_patterns['quantitative']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claims.append({
                    'type': 'quantitative',
                    'text': match.group(),
                    'value': match.group(1),
                    'unit': match.group(2) if len(match.groups()) > 1 else '',
                    'start': match.start(),
                    'end': match.end(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
                
        # Extract temporal claims
        for pattern in self.fact_patterns['temporal']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claims.append({
                    'type': 'temporal',
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
                
        # Extract categorical claims
        for pattern in self.fact_patterns['categorical']:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claims.append({
                    'type': 'categorical',
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'context': text[max(0, match.start()-50):match.end()+50]
                })
                
        return claims
    
    def compare_factual_claims(self, claims_a: List[Dict], claims_b: List[Dict]) -> List[Dict]:
        """Compare factual claims to find contradictions"""
        contradictions = []
        
        for claim_a in claims_a:
            for claim_b in claims_b:
                # Compare claims of the same type
                if claim_a['type'] == claim_b['type']:
                    contradiction = self._analyze_claim_contradiction(claim_a, claim_b)
                    if contradiction:
                        contradictions.append(contradiction)
                        
        return contradictions
    
    def _analyze_claim_contradiction(self, claim_a: Dict, claim_b: Dict) -> Optional[Dict]:
        """Analyze if two claims contradict each other"""
        if claim_a['type'] == 'quantitative':
            return self._analyze_quantitative_contradiction(claim_a, claim_b)
        elif claim_a['type'] == 'temporal':
            return self._analyze_temporal_contradiction(claim_a, claim_b)
        elif claim_a['type'] == 'categorical':
            return self._analyze_categorical_contradiction(claim_a, claim_b)
        elif claim_a['type'] == 'entity':
            return self._analyze_entity_contradiction(claim_a, claim_b)
            
        return None
    
    def _analyze_quantitative_contradiction(self, claim_a: Dict, claim_b: Dict) -> Optional[Dict]:
        """Analyze contradictions in quantitative claims"""
        try:
            # Extract numerical values
            value_a = float(re.sub(r'[,%]', '', claim_a.get('value', '0')))
            value_b = float(re.sub(r'[,%]', '', claim_b.get('value', '0')))
            
            # Check if units are comparable
            unit_a = claim_a.get('unit', '').lower()
            unit_b = claim_b.get('unit', '').lower()
            
            if unit_a == unit_b:
                # Calculate relative difference
                if value_a != 0:
                    relative_diff = abs(value_a - value_b) / value_a
                elif value_b != 0:
                    relative_diff = abs(value_a - value_b) / value_b
                else:
                    relative_diff = 0
                    
                # Significant difference indicates potential contradiction
                if relative_diff > 0.2:  # 20% threshold
                    return {
                        'type': 'quantitative_contradiction',
                        'claim_a': claim_a,
                        'claim_b': claim_b,
                        'contradiction_score': min(relative_diff, 1.0),
                        'difference': abs(value_a - value_b),
                        'relative_difference': relative_diff
                    }
        except ValueError:
            pass
            
        return None
    
    def _analyze_temporal_contradiction(self, claim_a: Dict, claim_b: Dict) -> Optional[Dict]:
        """Analyze contradictions in temporal claims"""
        # Extract dates/times and compare
        text_a = claim_a.get('text', '').lower()
        text_b = claim_b.get('text', '').lower()
        
        # Simple contradiction patterns
        contradiction_patterns = [
            ('before', 'after'),
            ('earlier', 'later'),
            ('first', 'last'),
            ('beginning', 'end')
        ]
        
        for pattern_a, pattern_b in contradiction_patterns:
            if pattern_a in text_a and pattern_b in text_b:
                return {
                    'type': 'temporal_contradiction',
                    'claim_a': claim_a,
                    'claim_b': claim_b,
                    'contradiction_score': 0.8,
                    'pattern': f"{pattern_a} vs {pattern_b}"
                }
                
        return None
    
    def _analyze_categorical_contradiction(self, claim_a: Dict, claim_b: Dict) -> Optional[Dict]:
        """Analyze contradictions in categorical claims"""
        text_a = claim_a.get('text', '').lower()
        text_b = claim_b.get('text', '').lower()
        
        # Direct contradiction patterns
        contradiction_pairs = [
            ('always', 'never'),
            ('all', 'none'),
            ('every', 'no'),
            ('is', 'is not'),
            ('was', 'was not'),
            ('first', 'last'),
            ('highest', 'lowest'),
            ('best', 'worst')
        ]
        
        for pair_a, pair_b in contradiction_pairs:
            if pair_a in text_a and pair_b in text_b:
                return {
                    'type': 'categorical_contradiction',
                    'claim_a': claim_a,
                    'claim_b': claim_b,
                    'contradiction_score': 0.9,
                    'pattern': f"{pair_a} vs {pair_b}"
                }
                
        return None
    
    def _analyze_entity_contradiction(self, claim_a: Dict, claim_b: Dict) -> Optional[Dict]:
        """Analyze contradictions in entity claims"""
        if claim_a['entity_type'] == claim_b['entity_type']:
            text_a = claim_a.get('text', '').lower()
            text_b = claim_b.get('text', '').lower()
            
            # If same entity type but different text, check for contradiction
            if text_a != text_b:
                # Check if contexts suggest they refer to the same thing
                context_similarity = self._calculate_context_similarity(
                    claim_a.get('context', ''),
                    claim_b.get('context', '')
                )
                
                if context_similarity > 0.7:  # High context similarity
                    return {
                        'type': 'entity_contradiction',
                        'claim_a': claim_a,
                        'claim_b': claim_b,
                        'contradiction_score': 0.7,
                        'context_similarity': context_similarity
                    }
                    
        return None
    
    def _calculate_context_similarity(self, context_a: str, context_b: str) -> float:
        """Calculate similarity between contexts"""
        if not context_a or not context_b:
            return 0.0
            
        # Simple word overlap calculation
        words_a = set(context_a.lower().split())
        words_b = set(context_b.lower().split())
        
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        
        if len(union) == 0:
            return 0.0
            
        return len(intersection) / len(union)

class TemporalContradictionDetector:
    """Detect contradictions across different time periods"""
    
    def __init__(self):
        self.temporal_db = defaultdict(list)
        self.contradiction_cache = {}
        
    def add_temporal_statement(self, statement: str, timestamp: datetime, source: str):
        """Add a statement with timestamp for temporal analysis"""
        self.temporal_db[source].append({
            'statement': statement,
            'timestamp': timestamp,
            'source': source
        })
    
    def find_temporal_contradictions(self, source: str, time_window: timedelta = None) -> List[Dict]:
        """Find contradictions in statements from the same source over time"""
        if source not in self.temporal_db:
            return []
            
        statements = self.temporal_db[source]
        if len(statements) < 2:
            return []
            
        contradictions = []
        
        for i, stmt_a in enumerate(statements):
            for j, stmt_b in enumerate(statements[i+1:], i+1):
                # Check if within time window
                if time_window:
                    time_diff = abs(stmt_a['timestamp'] - stmt_b['timestamp'])
                    if time_diff > time_window:
                        continue
                        
                # Analyze contradiction
                contradiction = self._analyze_temporal_contradiction(stmt_a, stmt_b)
                if contradiction:
                    contradictions.append(contradiction)
                    
        return sorted(contradictions, key=lambda x: x['contradiction_score'], reverse=True)
    
    def _analyze_temporal_contradiction(self, stmt_a: Dict, stmt_b: Dict) -> Optional[Dict]:
        """Analyze contradiction between two temporal statements"""
        # Use semantic similarity and factual analysis
        semantic_detector = SemanticContradictionDetector()
        factual_detector = FactualContradictionDetector()
        
        # Semantic analysis
        semantic_result = semantic_detector.find_semantic_contradictions([
            stmt_a['statement'], stmt_b['statement']
        ])
        
        # Factual analysis
        claims_a = factual_detector.extract_factual_claims(stmt_a['statement'])
        claims_b = factual_detector.extract_factual_claims(stmt_b['statement'])
        factual_contradictions = factual_detector.compare_factual_claims(claims_a, claims_b)
        
        # Calculate overall contradiction score
        semantic_score = semantic_result[0]['contradiction_score'] if semantic_result else 0
        factual_score = max([fc.get('contradiction_score', 0) for fc in factual_contradictions], default=0)
        
        overall_score = max(semantic_score, factual_score)
        
        if overall_score > 0.5:
            time_diff = abs(stmt_a['timestamp'] - stmt_b['timestamp'])
            
            return {
                'type': 'temporal_contradiction',
                'statement_a': stmt_a,
                'statement_b': stmt_b,
                'contradiction_score': overall_score,
                'time_difference': time_diff,
                'semantic_analysis': semantic_result,
                'factual_analysis': factual_contradictions
            }
            
        return None

class ComprehensiveContradictionFinder:
    """Master contradiction detection system"""
    
    def __init__(self, db_path='shock2/data/raw/contradictions.db'):
        self.db_path = db_path
        self.logger = self._setup_logger()
        
        # Initialize component detectors
        self.semantic_detector = SemanticContradictionDetector()
        self.factual_detector = FactualContradictionDetector()
        self.temporal_detector = TemporalContradictionDetector()
        
        # Analysis cache
        self.contradiction_cache = {}
        self.statement_embeddings = {}
        
        # Database setup
        self._init_database()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/contradiction_finder.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize contradiction detection database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY,
                source_a TEXT,
                source_b TEXT,
                statement_a TEXT,
                statement_b TEXT,
                contradiction_type TEXT,
                contradiction_score REAL,
                confidence REAL,
                semantic_distance REAL,
                temporal_factor REAL,
                evidence_quality TEXT,
                detection_timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statement_cache (
                id INTEGER PRIMARY KEY,
                statement_hash TEXT UNIQUE,
                statement_text TEXT,
                extracted_claims TEXT,
                embedding_data BLOB,
                analysis_timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def find_comprehensive_contradictions(self, context: ContradictionContext, 
                                              statements: List[Dict]) -> List[ContradictionResult]:
        """Find comprehensive contradictions using multiple detection methods"""
        self.logger.info(f"Starting comprehensive contradiction analysis")
        
        all_contradictions = []
        
        try:
            # Extract text statements
            text_statements = [stmt.get('content', '') for stmt in statements if stmt.get('content')]
            
            if len(text_statements) < 2:
                return []
                
            # Semantic contradiction detection
            semantic_contradictions = await self._find_semantic_contradictions(
                text_statements, context
            )
            all_contradictions.extend(semantic_contradictions)
            
            # Factual contradiction detection
            factual_contradictions = await self._find_factual_contradictions(
                statements, context
            )
            all_contradictions.extend(factual_contradictions)
            
            # Temporal contradiction detection
            if context.temporal_window:
                temporal_contradictions = await self._find_temporal_contradictions(
                    statements, context
                )
                all_contradictions.extend(temporal_contradictions)
                
            # Cross-source contradiction detection
            cross_contradictions = await self._find_cross_source_contradictions(
                statements, context
            )
            all_contradictions.extend(cross_contradictions)
            
            # Rank and filter contradictions
            ranked_contradictions = self._rank_contradictions(all_contradictions, context)
            
            # Store results
            await self._store_contradiction_results(ranked_contradictions, context)
            
            self.logger.info(f"Found {len(ranked_contradictions)} contradictions")
            return ranked_contradictions
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive contradiction detection: {str(e)}")
            return []
            
    async def _find_semantic_contradictions(self, statements: List[str], 
                                          context: ContradictionContext) -> List[ContradictionResult]:
        """Find semantic contradictions"""
        results = []
        
        contradictions = self.semantic_detector.find_semantic_contradictions(statements)
        
        for contradiction in contradictions:
            result = ContradictionResult(
                contradiction_score=contradiction['contradiction_score'],
                contradiction_type='semantic',
                confidence=contradiction['confidence'],
                source_statements=[
                    contradiction['statement_a'],
                    contradiction['statement_b']
                ],
                conflicting_claims=[{
                    'claim_a': contradiction['statement_a'],
                    'claim_b': contradiction['statement_b'],
                    'nli_result': contradiction['nli_ab']
                }],
                semantic_distance=1 - contradiction['semantic_similarity'],
                temporal_factors={},
                severity_level=self._calculate_severity(contradiction['contradiction_score']),
                evidence_quality={'confidence': contradiction['confidence']},
                recommendations=['verify_sources', 'check_context']
            )
            results.append(result)
            
        return results
        
    async def _find_factual_contradictions(self, statements: List[Dict], 
                                         context: ContradictionContext) -> List[ContradictionResult]:
        """Find factual contradictions"""
        results = []
        
        # Extract claims from all statements
        all_claims = []
        for stmt in statements:
            content = stmt.get('content', '')
            claims = self.factual_detector.extract_factual_claims(content)
            all_claims.append((stmt, claims))
            
        # Compare claims between statements
        for i, (stmt_a, claims_a) in enumerate(all_claims):
            for j, (stmt_b, claims_b) in enumerate(all_claims[i+1:], i+1):
                contradictions = self.factual_detector.compare_factual_claims(claims_a, claims_b)
                
                for contradiction in contradictions:
                    result = ContradictionResult(
                        contradiction_score=contradiction['contradiction_score'],
                        contradiction_type='factual',
                        confidence=0.8,
                        source_statements=[
                            stmt_a.get('content', ''),
                            stmt_b.get('content', '')
                        ],
                        conflicting_claims=[contradiction],
                        semantic_distance=0.0,
                        temporal_factors={},
                        severity_level=self._calculate_severity(contradiction['contradiction_score']),
                        evidence_quality={'type': contradiction['type']},
                        recommendations=['fact_check', 'verify_sources']
                    )
                    results.append(result)
                    
        return results
        
    async def _find_temporal_contradictions(self, statements: List[Dict], 
                                          context: ContradictionContext) -> List[ContradictionResult]:
        """Find temporal contradictions"""
        results = []
        
        # Add statements to temporal detector
        for stmt in statements:
            if 'timestamp' in stmt:
                timestamp = datetime.fromisoformat(stmt['timestamp'])
                source = stmt.get('source', 'unknown')
                content = stmt.get('content', '')
                
                self.temporal_detector.add_temporal_statement(content, timestamp, source)
                
        # Find contradictions for each source
        sources = set(stmt.get('source', 'unknown') for stmt in statements)
        
        for source in sources:
            time_window = timedelta(days=30)  # 30-day window
            contradictions = self.temporal_detector.find_temporal_contradictions(source, time_window)
            
            for contradiction in contradictions:
                result = ContradictionResult(
                    contradiction_score=contradiction['contradiction_score'],
                    contradiction_type='temporal',
                    confidence=0.7,
                    source_statements=[
                        contradiction['statement_a']['statement'],
                        contradiction['statement_b']['statement']
                    ],
                    conflicting_claims=[],
                    semantic_distance=0.0,
                    temporal_factors={
                        'time_difference': str(contradiction['time_difference']),
                        'source': source
                    },
                    severity_level=self._calculate_severity(contradiction['contradiction_score']),
                    evidence_quality={'temporal_analysis': True},
                    recommendations=['timeline_verification', 'source_consistency_check']
                )
                results.append(result)
                
        return results
        
    async def _find_cross_source_contradictions(self, statements: List[Dict], 
                                              context: ContradictionContext) -> List[ContradictionResult]:
        """Find contradictions across different sources"""
        results = []
        
        # Group statements by source
        source_groups = defaultdict(list)
        for stmt in statements:
            source = stmt.get('source', 'unknown')
            source_groups[source].append(stmt)
            
        # Compare statements across sources
        sources = list(source_groups.keys())
        
        for i, source_a in enumerate(sources):
            for source_b in sources[i+1:]:
                statements_a = [s.get('content', '') for s in source_groups[source_a]]
                statements_b = [s.get('content', '') for s in source_groups[source_b]]
                
                # Find contradictions between sources
                cross_contradictions = []
                
                for stmt_a in statements_a:
                    for stmt_b in statements_b:
                        semantic_result = self.semantic_detector.find_semantic_contradictions([stmt_a, stmt_b])
                        
                        if semantic_result and semantic_result[0]['contradiction_score'] > 0.6:
                            cross_contradictions.append(semantic_result[0])
                            
                for contradiction in cross_contradictions:
                    result = ContradictionResult(
                        contradiction_score=contradiction['contradiction_score'],
                        contradiction_type='cross_source',
                        confidence=contradiction['confidence'],
                        source_statements=[
                            contradiction['statement_a'],
                            contradiction['statement_b']
                        ],
                        conflicting_claims=[],
                        semantic_distance=1 - contradiction['semantic_similarity'],
                        temporal_factors={
                            'source_a': source_a,
                            'source_b': source_b
                        },
                        severity_level=self._calculate_severity(contradiction['contradiction_score']),
                        evidence_quality={'cross_source': True},
                        recommendations=['source_bias_analysis', 'independent_verification']
                    )
                    results.append(result)
                    
        return results
        
    def _calculate_severity(self, contradiction_score: float) -> str:
        """Calculate severity level based on contradiction score"""
        if contradiction_score > 0.8:
            return 'critical'
        elif contradiction_score > 0.6:
            return 'high'
        elif contradiction_score > 0.4:
            return 'medium'
        elif contradiction_score > 0.2:
            return 'low'
        else:
            return 'minimal'
            
    def _rank_contradictions(self, contradictions: List[ContradictionResult], 
                           context: ContradictionContext) -> List[ContradictionResult]:
        """Rank contradictions by importance and relevance"""
        # Apply context-specific weighting
        for contradiction in contradictions:
            # Adjust score based on priority level
            contradiction.contradiction_score *= context.priority_level
            
            # Boost cross-source contradictions
            if contradiction.contradiction_type == 'cross_source':
                contradiction.contradiction_score *= 1.2
                
            # Boost factual contradictions
            if contradiction.contradiction_type == 'factual':
                contradiction.contradiction_score *= 1.1
                
        # Sort by contradiction score
        return sorted(contradictions, key=lambda x: x.contradiction_score, reverse=True)
        
    async def _store_contradiction_results(self, contradictions: List[ContradictionResult], 
                                         context: ContradictionContext):
        """Store contradiction results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for contradiction in contradictions:
                cursor.execute('''
                    INSERT INTO contradictions 
                    (source_a, source_b, statement_a, statement_b, contradiction_type,
                     contradiction_score, confidence, semantic_distance, temporal_factor,
                     evidence_quality, detection_timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    context.source_a,
                    context.source_b,
                    contradiction.source_statements[0] if len(contradiction.source_statements) > 0 else '',
                    contradiction.source_statements[1] if len(contradiction.source_statements) > 1 else '',
                    contradiction.contradiction_type,
                    contradiction.contradiction_score,
                    contradiction.confidence,
                    contradiction.semantic_distance,
                    0.0,  # temporal_factor placeholder
                    json.dumps(contradiction.evidence_quality),
                    datetime.now().isoformat(),
                    json.dumps(contradiction.temporal_factors)
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing contradiction results: {str(e)}")

# Main execution and testing
if __name__ == "__main__":
    finder = ComprehensiveContradictionFinder()
    
    # Example usage
    sample_statements = [
        {
            'content': 'The unemployment rate decreased to 3.5% last month.',
            'source': 'government',
            'timestamp': '2023-12-01T10:00:00'
        },
        {
            'content': 'Unemployment has risen significantly, reaching 7% according to latest data.',
            'source': 'news_outlet',
            'timestamp': '2023-12-01T14:00:00'
        }
    ]
    
    context = ContradictionContext(
        source_a='government',
        source_b='news_outlet',
        content_type='economic_data',
        temporal_window='1_day',
        analysis_depth='deep',
        priority_level=1.0
    )
    
    # Run detection
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(finder.find_comprehensive_contradictions(context, sample_statements))
    
    print(f"Found {len(results)} contradictions")
    for result in results:
        print(f"- {result.contradiction_type}: {result.contradiction_score:.3f} ({result.severity_level})")

