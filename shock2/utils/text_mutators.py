
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sqlite3
import logging
import re
import spacy
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import random
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field
import threading
from queue import Queue
import asyncio
import aiohttp
import requests
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForMaskedLM,
    pipeline, GPT2LMHeadModel, BertModel, RobertaModel
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
from textstat import flesch_reading_ease, automated_readability_index
from textblob import TextBlob
import langdetect
from fake_useragent import UserAgent
import base64
import gzip
import zlib
from cryptography.fernet import Fernet
import pickle
import warnings
warnings.filterwarnings('ignore')

# Advanced linguistic libraries
from sentence_transformers import SentenceTransformer
import openai
import anthropic
import google.generativeai as genai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import psutil
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import itertools
import math
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MutationConfig:
    """Configuration for text mutation operations"""
    mutation_intensity: float = 0.7  # 0.0 to 1.0
    preserve_meaning: bool = True
    target_readability: Optional[float] = None
    linguistic_complexity: str = 'adaptive'  # 'simple', 'complex', 'adaptive'
    mutation_types: List[str] = field(default_factory=lambda: [
        'synonym_replacement', 'paraphrasing', 'sentence_restructuring',
        'discourse_markers', 'register_shifting', 'stylistic_variation'
    ])
    evasion_focus: List[str] = field(default_factory=lambda: [
        'ai_detection', 'plagiarism_detection', 'style_fingerprinting'
    ])
    preserve_entities: bool = True
    maintain_sentiment: bool = True
    output_format: str = 'original'  # 'original', 'formal', 'casual', 'academic'

@dataclass
class MutationResult:
    """Result of text mutation operation"""
    original_text: str
    mutated_text: str
    mutation_score: float
    confidence: float
    mutations_applied: List[str]
    readability_change: float
    semantic_similarity: float
    detection_evasion_score: float
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedTextMutator:
    """
    Advanced text mutation system for evading AI detection while preserving meaning.
    Implements sophisticated linguistic transformations and adversarial techniques.
    """
    
    def __init__(self, config: Optional[MutationConfig] = None):
        self.config = config or MutationConfig()
        self.initialize_components()
        self.setup_database()
        self.load_linguistic_resources()
        self.initialize_models()
        
    def initialize_components(self):
        """Initialize core components"""
        self.nlp = spacy.load('en_core_web_sm')
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.ua = UserAgent()
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
        except:
            pass
            
        # Initialize encryption
        self.cipher_suite = Fernet(Fernet.generate_key())
        
        # Threading components
        self.mutation_queue = Queue()
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
    def setup_database(self):
        """Setup SQLite database for mutation tracking"""
        self.db_path = 'shock2_mutations.db'
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mutations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_hash TEXT UNIQUE,
                original_text TEXT,
                mutated_text TEXT,
                mutation_config TEXT,
                mutation_score REAL,
                semantic_similarity REAL,
                evasion_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mutation_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                source_pattern TEXT,
                target_pattern TEXT,
                success_rate REAL,
                usage_count INTEGER DEFAULT 1,
                effectiveness_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS linguistic_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE,
                writing_style TEXT,
                complexity_metrics TEXT,
                linguistic_features TEXT,
                author_signature TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def load_linguistic_resources(self):
        """Load comprehensive linguistic resources"""
        self.synonym_database = self._build_synonym_database()
        self.paraphrase_patterns = self._load_paraphrase_patterns()
        self.discourse_markers = self._load_discourse_markers()
        self.register_variations = self._load_register_variations()
        self.structural_templates = self._load_structural_templates()
        self.evasion_strategies = self._load_evasion_strategies()
        
    def initialize_models(self):
        """Initialize transformer models"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.masked_lm = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.paraphrase_model = pipeline('text2text-generation', 
                                            model='google/flan-t5-base')
        except Exception as e:
            logger.warning(f"Model initialization failed: {e}")
            self.tokenizer = None
            self.masked_lm = None
            self.sentence_transformer = None
            self.paraphrase_model = None
            
    def _build_synonym_database(self) -> Dict[str, List[str]]:
        """Build comprehensive synonym database"""
        synonym_db = defaultdict(list)
        
        # Base synonym sets for common words
        base_synonyms = {
            'said': ['stated', 'declared', 'mentioned', 'noted', 'remarked', 'observed', 
                    'commented', 'expressed', 'articulated', 'conveyed', 'proclaimed'],
            'important': ['significant', 'crucial', 'vital', 'essential', 'critical',
                         'paramount', 'fundamental', 'key', 'major', 'substantial'],
            'good': ['excellent', 'outstanding', 'remarkable', 'exceptional', 'superb',
                    'admirable', 'commendable', 'praiseworthy', 'exemplary', 'superior'],
            'bad': ['poor', 'inadequate', 'substandard', 'inferior', 'unsatisfactory',
                   'deficient', 'disappointing', 'problematic', 'concerning', 'troubling'],
            'big': ['large', 'enormous', 'massive', 'substantial', 'considerable',
                   'significant', 'extensive', 'vast', 'immense', 'colossal'],
            'small': ['tiny', 'minute', 'minimal', 'modest', 'limited', 'compact',
                     'negligible', 'insignificant', 'minor', 'diminutive'],
            'many': ['numerous', 'countless', 'multiple', 'various', 'several',
                    'abundant', 'extensive', 'widespread', 'proliferating', 'myriad'],
            'think': ['believe', 'consider', 'suppose', 'assume', 'presume',
                     'contemplate', 'ponder', 'reflect', 'deliberate', 'reason'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'display', 'exhibit',
                    'present', 'indicate', 'manifest', 'exemplify', 'showcase'],
            'make': ['create', 'produce', 'generate', 'construct', 'build',
                    'manufacture', 'establish', 'form', 'develop', 'craft']
        }
        
        # Add WordNet synonyms
        for word, base_syns in base_synonyms.items():
            synonym_db[word].extend(base_syns)
            
            try:
                synsets = wordnet.synsets(word)
                for synset in synsets[:5]:  # Limit to avoid noise
                    for lemma in synset.lemmas():
                        syn = lemma.name().replace('_', ' ')
                        if syn != word and syn not in synonym_db[word] and len(syn) > 2:
                            synonym_db[word].append(syn)
            except:
                pass
                
        return dict(synonym_db)
        
    def _load_paraphrase_patterns(self) -> Dict[str, List[Dict]]:
        """Load paraphrasing patterns"""
        return {
            'sentence_starters': [
                {'pattern': r'^It is (.*)', 'replacement': r'One can observe that \1'},
                {'pattern': r'^This shows (.*)', 'replacement': r'This demonstrates \1'},
                {'pattern': r'^The result is (.*)', 'replacement': r'The outcome indicates \1'},
                {'pattern': r'^We can see (.*)', 'replacement': r'Evidence suggests \1'},
                {'pattern': r'^There is (.*)', 'replacement': r'One finds \1'},
            ],
            'passive_to_active': [
                {'pattern': r'was (.*) by (.*)', 'replacement': r'\2 \1'},
                {'pattern': r'is (.*) by (.*)', 'replacement': r'\2 \1'},
                {'pattern': r'were (.*) by (.*)', 'replacement': r'\2 \1'},
            ],
            'complex_structures': [
                {'pattern': r'Because (.*), (.*)', 'replacement': r'Given that \1, it follows that \2'},
                {'pattern': r'Although (.*), (.*)', 'replacement': r'Despite \1, \2'},
                {'pattern': r'When (.*), (.*)', 'replacement': r'Upon \1, \2'},
            ]
        }
        
    def _load_discourse_markers(self) -> Dict[str, List[str]]:
        """Load discourse markers for different functions"""
        return {
            'addition': ['furthermore', 'moreover', 'additionally', 'in addition',
                        'what is more', 'besides', 'also', 'likewise'],
            'contrast': ['however', 'nevertheless', 'nonetheless', 'on the contrary',
                        'in contrast', 'conversely', 'alternatively', 'yet'],
            'causation': ['therefore', 'consequently', 'as a result', 'thus',
                         'hence', 'accordingly', 'for this reason', 'given this'],
            'emphasis': ['indeed', 'certainly', 'undoubtedly', 'clearly',
                        'obviously', 'evidently', 'naturally', 'of course'],
            'sequence': ['first', 'second', 'third', 'next', 'then', 'subsequently',
                        'following this', 'in the next stage', 'proceeding to'],
            'conclusion': ['in conclusion', 'to summarize', 'in summary', 'finally',
                          'ultimately', 'in the end', 'all things considered']
        }
        
    def _load_register_variations(self) -> Dict[str, Dict[str, List[str]]]:
        """Load register variations (formal/informal)"""
        return {
            'formal_to_casual': {
                'therefore': ['so', 'that\'s why', 'because of this'],
                'furthermore': ['also', 'plus', 'and'],
                'subsequently': ['then', 'after that', 'next'],
                'nevertheless': ['but', 'still', 'even so'],
                'approximately': ['about', 'around', 'roughly'],
                'demonstrate': ['show', 'prove', 'make clear'],
                'facilitate': ['help', 'make easier', 'assist'],
                'utilize': ['use', 'employ', 'apply']
            },
            'casual_to_formal': {
                'so': ['therefore', 'consequently', 'thus'],
                'but': ['however', 'nevertheless', 'nonetheless'],
                'also': ['furthermore', 'additionally', 'moreover'],
                'show': ['demonstrate', 'illustrate', 'exhibit'],
                'help': ['facilitate', 'assist', 'support'],
                'use': ['utilize', 'employ', 'implement'],
                'get': ['obtain', 'acquire', 'receive'],
                'big': ['substantial', 'considerable', 'significant']
            }
        }
        
    def _load_structural_templates(self) -> Dict[str, List[str]]:
        """Load sentence structural templates"""
        return {
            'simple_to_complex': [
                '{subject} {verb} {object}' + ' → ' + 'It is {subject} that {verb} {object}',
                '{subject} {verb} {object}' + ' → ' + 'What {subject} does is {verb} {object}',
                '{subject} is {adjective}' + ' → ' + 'One characteristic of {subject} is being {adjective}',
            ],
            'complex_to_simple': [
                'It is evident that {clause}' + ' → ' + '{clause}',
                'What is important to note is that {clause}' + ' → ' + '{clause}',
                'One can observe that {clause}' + ' → ' + '{clause}',
            ],
            'embedding_patterns': [
                '{clause1}. {clause2}' + ' → ' + '{clause1}, which means {clause2}',
                '{clause1}. {clause2}' + ' → ' + 'Given that {clause1}, {clause2}',
                '{clause1}. {clause2}' + ' → ' + '{clause1}, thereby {clause2}',
            ]
        }
        
    def _load_evasion_strategies(self) -> Dict[str, List[Dict]]:
        """Load AI detection evasion strategies"""
        return {
            'pattern_breaking': [
                {'name': 'punctuation_variation', 'intensity': 0.3},
                {'name': 'sentence_length_variation', 'intensity': 0.5},
                {'name': 'vocabulary_complexity_shift', 'intensity': 0.4},
                {'name': 'syntactic_pattern_disruption', 'intensity': 0.6}
            ],
            'stylistic_masking': [
                {'name': 'register_mixing', 'intensity': 0.4},
                {'name': 'discourse_marker_insertion', 'intensity': 0.3},
                {'name': 'rhetorical_device_injection', 'intensity': 0.5},
                {'name': 'colloquialism_integration', 'intensity': 0.3}
            ],
            'semantic_preservation': [
                {'name': 'synonym_clustering', 'intensity': 0.6},
                {'name': 'paraphrase_chaining', 'intensity': 0.7},
                {'name': 'context_aware_substitution', 'intensity': 0.8},
                {'name': 'meaning_preserving_restructuring', 'intensity': 0.9}
            ]
        }
        
    async def mutate_text(self, text: str, 
                         config: Optional[MutationConfig] = None) -> MutationResult:
        """
        Perform comprehensive text mutation with multiple strategies
        """
        start_time = time.time()
        config = config or self.config
        
        # Check cache first
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cached_result = self._get_cached_result(text_hash, config)
        if cached_result:
            return cached_result
            
        # Analyze original text
        original_analysis = await self._analyze_text(text)
        
        # Apply mutation strategies
        mutated_text = text
        mutations_applied = []
        
        # 1. Synonym replacement
        if 'synonym_replacement' in config.mutation_types:
            mutated_text, synonmym_mutations = await self._apply_synonym_replacement(
                mutated_text, config.mutation_intensity
            )
            mutations_applied.extend(synonmym_mutations)
            
        # 2. Paraphrasing
        if 'paraphrasing' in config.mutation_types:
            mutated_text, para_mutations = await self._apply_paraphrasing(
                mutated_text, config.mutation_intensity
            )
            mutations_applied.extend(para_mutations)
            
        # 3. Sentence restructuring
        if 'sentence_restructuring' in config.mutation_types:
            mutated_text, struct_mutations = await self._apply_sentence_restructuring(
                mutated_text, config.mutation_intensity
            )
            mutations_applied.extend(struct_mutations)
            
        # 4. Discourse markers
        if 'discourse_markers' in config.mutation_types:
            mutated_text, discourse_mutations = await self._apply_discourse_markers(
                mutated_text, config.mutation_intensity
            )
            mutations_applied.extend(discourse_mutations)
            
        # 5. Register shifting
        if 'register_shifting' in config.mutation_types:
            mutated_text, register_mutations = await self._apply_register_shifting(
                mutated_text, config.mutation_intensity
            )
            mutations_applied.extend(register_mutations)
            
        # 6. Stylistic variation
        if 'stylistic_variation' in config.mutation_types:
            mutated_text, style_mutations = await self._apply_stylistic_variation(
                mutated_text, config.mutation_intensity
            )
            mutations_applied.extend(style_mutations)
            
        # Apply evasion strategies
        for focus in config.evasion_focus:
            mutated_text, evasion_mutations = await self._apply_evasion_strategy(
                mutated_text, focus, config.mutation_intensity
            )
            mutations_applied.extend(evasion_mutations)
            
        # Analyze mutated text
        mutated_analysis = await self._analyze_text(mutated_text)
        
        # Calculate metrics
        mutation_score = self._calculate_mutation_score(original_analysis, mutated_analysis)
        semantic_similarity = self._calculate_semantic_similarity(text, mutated_text)
        evasion_score = await self._calculate_evasion_score(text, mutated_text)
        readability_change = mutated_analysis['readability'] - original_analysis['readability']
        
        # Create result
        result = MutationResult(
            original_text=text,
            mutated_text=mutated_text,
            mutation_score=mutation_score,
            confidence=min(semantic_similarity, evasion_score),
            mutations_applied=mutations_applied,
            readability_change=readability_change,
            semantic_similarity=semantic_similarity,
            detection_evasion_score=evasion_score,
            processing_time=time.time() - start_time,
            metadata={
                'original_analysis': original_analysis,
                'mutated_analysis': mutated_analysis,
                'config': config.__dict__
            }
        )
        
        # Cache result
        self._cache_result(text_hash, config, result)
        
        # Store in database
        await self._store_mutation_result(result)
        
        return result
        
    async def _apply_synonym_replacement(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply intelligent synonym replacement"""
        mutations = []
        doc = self.nlp(text)
        words = []
        
        for token in doc:
            if (token.is_alpha and not token.is_stop and 
                token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and
                random.random() < intensity):
                
                # Get context-aware synonyms
                synonyms = self._get_contextual_synonyms(token.text, token.pos_)
                if synonyms:
                    replacement = random.choice(synonyms)
                    words.append(replacement)
                    mutations.append(f"synonym: {token.text} → {replacement}")
                else:
                    words.append(token.text)
            else:
                words.append(token.text)
                
        # Reconstruct text preserving original spacing
        mutated_text = self._reconstruct_text_with_spacing(text, words)
        return mutated_text, mutations
        
    async def _apply_paraphrasing(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply sentence-level paraphrasing"""
        mutations = []
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            if random.random() < intensity:
                # Try different paraphrasing strategies
                paraphrased = await self._paraphrase_sentence(sentence)
                if paraphrased and paraphrased != sentence:
                    paraphrased_sentences.append(paraphrased)
                    mutations.append(f"paraphrase: sentence level")
                else:
                    paraphrased_sentences.append(sentence)
            else:
                paraphrased_sentences.append(sentence)
                
        return ' '.join(paraphrased_sentences), mutations
        
    async def _apply_sentence_restructuring(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply syntactic restructuring"""
        mutations = []
        sentences = sent_tokenize(text)
        restructured_sentences = []
        
        for sentence in sentences:
            if random.random() < intensity:
                restructured = self._restructure_sentence(sentence)
                if restructured != sentence:
                    restructured_sentences.append(restructured)
                    mutations.append(f"restructure: syntactic change")
                else:
                    restructured_sentences.append(sentence)
            else:
                restructured_sentences.append(sentence)
                
        return ' '.join(restructured_sentences), mutations
        
    async def _apply_discourse_markers(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Add discourse markers for coherence"""
        mutations = []
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return text, mutations
            
        enhanced_sentences = [sentences[0]]  # First sentence unchanged
        
        for i in range(1, len(sentences)):
            if random.random() < intensity:
                # Determine relationship type
                relationship = self._detect_sentence_relationship(
                    sentences[i-1], sentences[i]
                )
                
                marker = self._select_discourse_marker(relationship)
                if marker:
                    enhanced_sentence = f"{marker}, {sentences[i].lower()}"
                    enhanced_sentences.append(enhanced_sentence)
                    mutations.append(f"discourse_marker: {marker}")
                else:
                    enhanced_sentences.append(sentences[i])
            else:
                enhanced_sentences.append(sentences[i])
                
        return ' '.join(enhanced_sentences), mutations
        
    async def _apply_register_shifting(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Shift between formal and informal registers"""
        mutations = []
        
        # Detect current register
        current_register = self._detect_register(text)
        
        # Choose target register
        if current_register == 'formal':
            target_variations = self.register_variations['formal_to_casual']
        else:
            target_variations = self.register_variations['casual_to_formal']
            
        words = word_tokenize(text)
        shifted_words = []
        
        for word in words:
            if word.lower() in target_variations and random.random() < intensity:
                replacement = random.choice(target_variations[word.lower()])
                shifted_words.append(replacement)
                mutations.append(f"register_shift: {word} → {replacement}")
            else:
                shifted_words.append(word)
                
        return self._reconstruct_text_with_spacing(text, shifted_words), mutations
        
    async def _apply_stylistic_variation(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply various stylistic transformations"""
        mutations = []
        
        # Vary sentence lengths
        if random.random() < intensity:
            text, length_mutations = self._vary_sentence_lengths(text)
            mutations.extend(length_mutations)
            
        # Add rhetorical devices
        if random.random() < intensity:
            text, rhetorical_mutations = self._add_rhetorical_devices(text)
            mutations.extend(rhetorical_mutations)
            
        # Adjust punctuation patterns
        if random.random() < intensity:
            text, punct_mutations = self._vary_punctuation(text)
            mutations.extend(punct_mutations)
            
        return text, mutations
        
    async def _apply_evasion_strategy(self, text: str, focus: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply specific evasion strategies"""
        mutations = []
        
        if focus == 'ai_detection':
            text, ai_mutations = await self._evade_ai_detection(text, intensity)
            mutations.extend(ai_mutations)
        elif focus == 'plagiarism_detection':
            text, plag_mutations = await self._evade_plagiarism_detection(text, intensity)
            mutations.extend(plag_mutations)
        elif focus == 'style_fingerprinting':
            text, style_mutations = await self._evade_style_fingerprinting(text, intensity)
            mutations.extend(style_mutations)
            
        return text, mutations
        
    def _get_contextual_synonyms(self, word: str, pos: str) -> List[str]:
        """Get context-appropriate synonyms"""
        synonyms = []
        
        # Check our database first
        if word.lower() in self.synonym_database:
            synonyms.extend(self.synonym_database[word.lower()])
            
        # Use WordNet for additional synonyms
        try:
            for synset in wordnet.synsets(word, pos=self._pos_to_wordnet(pos)):
                for lemma in synset.lemmas():
                    syn = lemma.name().replace('_', ' ')
                    if syn != word and syn not in synonyms:
                        synonyms.append(syn)
        except:
            pass
            
        return synonyms[:5]  # Limit to best matches
        
    def _pos_to_wordnet(self, pos: str) -> str:
        """Convert spacy POS to wordnet POS"""
        pos_map = {
            'NOUN': wordnet.NOUN,
            'VERB': wordnet.VERB,
            'ADJ': wordnet.ADJ,
            'ADV': wordnet.ADV
        }
        return pos_map.get(pos, wordnet.NOUN)
        
    async def _paraphrase_sentence(self, sentence: str) -> str:
        """Paraphrase a sentence using multiple strategies"""
        # Try pattern-based paraphrasing first
        for pattern_type, patterns in self.paraphrase_patterns.items():
            for pattern_dict in patterns:
                if re.search(pattern_dict['pattern'], sentence, re.IGNORECASE):
                    paraphrased = re.sub(
                        pattern_dict['pattern'], 
                        pattern_dict['replacement'], 
                        sentence, 
                        flags=re.IGNORECASE
                    )
                    if paraphrased != sentence:
                        return paraphrased
                        
        # Try model-based paraphrasing if available
        if self.paraphrase_model:
            try:
                prompt = f"Paraphrase this sentence: {sentence}"
                result = self.paraphrase_model(prompt, max_length=100)
                if result and len(result) > 0:
                    paraphrased = result[0]['generated_text'].replace(prompt, '').strip()
                    if paraphrased and paraphrased != sentence:
                        return paraphrased
            except:
                pass
                
        return sentence
        
    def _restructure_sentence(self, sentence: str) -> str:
        """Restructure sentence syntax"""
        doc = self.nlp(sentence)
        
        # Try passive to active voice conversion
        if self._is_passive_voice(doc):
            active = self._convert_to_active(sentence)
            if active != sentence:
                return active
                
        # Try complex to simple conversion
        if self._is_complex_sentence(doc):
            simplified = self._simplify_sentence(sentence)
            if simplified != sentence:
                return simplified
                
        return sentence
        
    def _is_passive_voice(self, doc) -> bool:
        """Check if sentence is in passive voice"""
        for token in doc:
            if token.dep_ == 'nsubjpass':
                return True
        return False
        
    def _convert_to_active(self, sentence: str) -> str:
        """Convert passive to active voice"""
        # Simple pattern-based conversion
        passive_patterns = [
            (r'was (.*) by (.*)', r'\2 \1'),
            (r'were (.*) by (.*)', r'\2 \1'),
            (r'is (.*) by (.*)', r'\2 \1'),
            (r'are (.*) by (.*)', r'\2 \1')
        ]
        
        for pattern, replacement in passive_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                
        return sentence
        
    def _is_complex_sentence(self, doc) -> bool:
        """Check if sentence has complex structure"""
        clause_count = 0
        for token in doc:
            if token.dep_ in ['ccomp', 'xcomp', 'advcl', 'acl']:
                clause_count += 1
        return clause_count > 1
        
    def _simplify_sentence(self, sentence: str) -> str:
        """Simplify complex sentence structure"""
        # Pattern-based simplification
        simplification_patterns = [
            (r'It is (.*) that (.*)', r'\2'),
            (r'What (.*) is (.*)', r'\1 \2'),
            (r'The fact that (.*) means (.*)', r'\1, so \2')
        ]
        
        for pattern, replacement in simplification_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                
        return sentence
        
    def _detect_sentence_relationship(self, sent1: str, sent2: str) -> str:
        """Detect logical relationship between sentences"""
        # Simple heuristic-based detection
        sent2_lower = sent2.lower()
        
        if any(word in sent2_lower for word in ['however', 'but', 'although', 'despite']):
            return 'contrast'
        elif any(word in sent2_lower for word in ['because', 'since', 'due to', 'as a result']):
            return 'causation'
        elif any(word in sent2_lower for word in ['also', 'furthermore', 'moreover', 'additionally']):
            return 'addition'
        elif any(word in sent2_lower for word in ['therefore', 'thus', 'consequently']):
            return 'conclusion'
        else:
            return 'sequence'
            
    def _select_discourse_marker(self, relationship: str) -> Optional[str]:
        """Select appropriate discourse marker"""
        markers = self.discourse_markers.get(relationship, [])
        return random.choice(markers) if markers else None
        
    def _detect_register(self, text: str) -> str:
        """Detect formal vs informal register"""
        formal_indicators = ['furthermore', 'nevertheless', 'consequently', 'thereby']
        informal_indicators = ['so', 'but', 'and', 'also']
        
        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())
        
        return 'formal' if formal_count > informal_count else 'informal'
        
    def _vary_sentence_lengths(self, text: str) -> Tuple[str, List[str]]:
        """Vary sentence lengths for stylistic diversity"""
        mutations = []
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return text, mutations
            
        varied_sentences = []
        i = 0
        
        while i < len(sentences):
            sentence = sentences[i]
            
            # Randomly decide to combine or split
            if random.random() < 0.3 and i < len(sentences) - 1:
                # Combine with next sentence
                combined = f"{sentence} {self._get_connector()} {sentences[i+1].lower()}"
                varied_sentences.append(combined)
                mutations.append("sentence_combine")
                i += 2
            elif random.random() < 0.2 and len(sentence.split()) > 15:
                # Split long sentence
                split_sentences = self._split_sentence(sentence)
                varied_sentences.extend(split_sentences)
                mutations.append("sentence_split")
                i += 1
            else:
                varied_sentences.append(sentence)
                i += 1
                
        return ' '.join(varied_sentences), mutations
        
    def _get_connector(self) -> str:
        """Get random sentence connector"""
        connectors = ['and', 'while', 'as', 'whereas', 'since']
        return random.choice(connectors)
        
    def _split_sentence(self, sentence: str) -> List[str]:
        """Split long sentence into shorter ones"""
        # Simple splitting at conjunctions
        conjunctions = [' and ', ' but ', ' or ', ' while ', ' as ']
        
        for conj in conjunctions:
            if conj in sentence:
                parts = sentence.split(conj, 1)
                if len(parts) == 2:
                    return [parts[0].strip() + '.', parts[1].strip().capitalize()]
                    
        return [sentence]  # Return original if can't split
        
    def _add_rhetorical_devices(self, text: str) -> Tuple[str, List[str]]:
        """Add rhetorical devices"""
        mutations = []
        
        # Add emphatic structures
        if random.random() < 0.3:
            text = self._add_emphasis(text)
            mutations.append("emphasis_added")
            
        # Add rhetorical questions
        if random.random() < 0.2:
            text = self._add_rhetorical_question(text)
            mutations.append("rhetorical_question")
            
        return text, mutations
        
    def _add_emphasis(self, text: str) -> str:
        """Add emphatic structures"""
        emphatic_phrases = [
            "It is important to note that",
            "What is particularly significant is that",
            "One must consider that",
            "It should be emphasized that"
        ]
        
        sentences = sent_tokenize(text)
        if sentences:
            # Add emphasis to first sentence
            first_sentence = sentences[0]
            if not any(phrase in first_sentence for phrase in emphatic_phrases):
                emphatic_phrase = random.choice(emphatic_phrases)
                sentences[0] = f"{emphatic_phrase} {first_sentence.lower()}"
                
        return ' '.join(sentences)
        
    def _add_rhetorical_question(self, text: str) -> str:
        """Add rhetorical question"""
        rhetorical_starters = [
            "But what does this mean?",
            "How significant is this?",
            "What are the implications?",
            "Why is this important?"
        ]
        
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            # Insert rhetorical question in middle
            insert_pos = len(sentences) // 2
            question = random.choice(rhetorical_starters)
            sentences.insert(insert_pos, question)
            
        return ' '.join(sentences)
        
    def _vary_punctuation(self, text: str) -> Tuple[str, List[str]]:
        """Vary punctuation patterns"""
        mutations = []
        
        # Replace some periods with semicolons
        if random.random() < 0.3:
            # Find sentences that could be connected
            sentences = sent_tokenize(text)
            if len(sentences) > 1:
                for i in range(len(sentences) - 1):
                    if random.random() < 0.2:
                        sentences[i] = sentences[i].rstrip('.') + ';'
                        sentences[i+1] = sentences[i+1].lower()
                        mutations.append("semicolon_substitution")
                        break
                        
                text = ' '.join(sentences)
                
        return text, mutations
        
    async def _evade_ai_detection(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply AI detection evasion techniques"""
        mutations = []
        
        # Add subtle misspellings and corrections
        if random.random() < intensity:
            text, spell_mutations = self._add_subtle_variations(text)
            mutations.extend(spell_mutations)
            
        # Vary sentence structures unpredictably
        if random.random() < intensity:
            text, struct_mutations = self._randomize_structures(text)
            mutations.extend(struct_mutations)
            
        return text, mutations
        
    async def _evade_plagiarism_detection(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply plagiarism detection evasion"""
        mutations = []
        
        # Deep paraphrasing
        if random.random() < intensity:
            text, para_mutations = await self._deep_paraphrase(text)
            mutations.extend(para_mutations)
            
        # Reorder information
        if random.random() < intensity:
            text, order_mutations = self._reorder_information(text)
            mutations.extend(order_mutations)
            
        return text, mutations
        
    async def _evade_style_fingerprinting(self, text: str, intensity: float) -> Tuple[str, List[str]]:
        """Apply style fingerprinting evasion"""
        mutations = []
        
        # Mix writing styles
        if random.random() < intensity:
            text, style_mutations = self._mix_writing_styles(text)
            mutations.extend(style_mutations)
            
        # Vary complexity patterns
        if random.random() < intensity:
            text, complexity_mutations = self._vary_complexity_patterns(text)
            mutations.extend(complexity_mutations)
            
        return text, mutations
        
    def _add_subtle_variations(self, text: str) -> Tuple[str, List[str]]:
        """Add subtle character-level variations"""
        mutations = []
        
        # Replace some characters with visually similar ones
        char_replacements = {
            'a': ['à', 'á', 'â', 'ä'],
            'e': ['è', 'é', 'ê', 'ë'],
            'i': ['ì', 'í', 'î', 'ï'],
            'o': ['ò', 'ó', 'ô', 'ö'],
            'u': ['ù', 'ú', 'û', 'ü']
        }
        
        words = text.split()
        modified_words = []
        
        for word in words:
            if random.random() < 0.05:  # Low probability
                for char, replacements in char_replacements.items():
                    if char in word.lower():
                        replacement = random.choice(replacements)
                        word = word.replace(char, replacement)
                        mutations.append(f"char_variation: {char} → {replacement}")
                        break
                        
            modified_words.append(word)
            
        return ' '.join(modified_words), mutations
        
    def _randomize_structures(self, text: str) -> Tuple[str, List[str]]:
        """Randomize sentence structures"""
        mutations = []
        sentences = sent_tokenize(text)
        randomized_sentences = []
        
        for sentence in sentences:
            if random.random() < 0.3:
                # Apply random structural change
                doc = self.nlp(sentence)
                if len(list(doc)) > 5:  # Only for longer sentences
                    randomized = self._apply_random_transformation(sentence)
                    randomized_sentences.append(randomized)
                    mutations.append("structure_randomization")
                else:
                    randomized_sentences.append(sentence)
            else:
                randomized_sentences.append(sentence)
                
        return ' '.join(randomized_sentences), mutations
        
    def _apply_random_transformation(self, sentence: str) -> str:
        """Apply random syntactic transformation"""
        transformations = [
            self._fronting_transformation,
            self._clefting_transformation,
            self._inversion_transformation
        ]
        
        transformation = random.choice(transformations)
        return transformation(sentence)
        
    def _fronting_transformation(self, sentence: str) -> str:
        """Move elements to front of sentence"""
        doc = self.nlp(sentence)
        
        # Look for prepositional phrases to front
        for token in doc:
            if token.dep_ == 'prep' and token.head.pos_ == 'VERB':
                prep_phrase = self._extract_prep_phrase(token, doc)
                if prep_phrase:
                    remaining = sentence.replace(prep_phrase, '').strip()
                    return f"{prep_phrase}, {remaining.lower()}"
                    
        return sentence
        
    def _clefting_transformation(self, sentence: str) -> str:
        """Transform to cleft construction"""
        doc = self.nlp(sentence)
        
        # Simple subject cleft
        for token in doc:
            if token.dep_ == 'nsubj':
                subject = token.text
                rest = sentence.replace(subject, 'that', 1)
                return f"It is {subject} {rest}"
                
        return sentence
        
    def _inversion_transformation(self, sentence: str) -> str:
        """Apply subject-verb inversion where appropriate"""
        # Only for sentences starting with negative adverbs
        negative_adverbs = ['never', 'rarely', 'seldom', 'hardly', 'scarcely']
        
        for adverb in negative_adverbs:
            if sentence.lower().startswith(adverb):
                # This is a simplified inversion
                return sentence  # Return as-is for now
                
        return sentence
        
    def _extract_prep_phrase(self, prep_token, doc) -> Optional[str]:
        """Extract prepositional phrase"""
        phrase_tokens = [prep_token]
        
        # Get all children of the preposition
        for child in prep_token.children:
            phrase_tokens.extend(self._get_subtree(child))
            
        if len(phrase_tokens) > 1:
            phrase_tokens.sort(key=lambda x: x.i)
            return ' '.join([token.text for token in phrase_tokens])
            
        return None
        
    def _get_subtree(self, token):
        """Get all tokens in subtree"""
        subtree = [token]
        for child in token.children:
            subtree.extend(self._get_subtree(child))
        return subtree
        
    async def _deep_paraphrase(self, text: str) -> Tuple[str, List[str]]:
        """Perform deep paraphrasing"""
        mutations = []
        
        # Multiple rounds of paraphrasing
        current_text = text
        for round_num in range(3):
            sentences = sent_tokenize(current_text)
            paraphrased_sentences = []
            
            for sentence in sentences:
                paraphrased = await self._paraphrase_sentence(sentence)
                paraphrased_sentences.append(paraphrased)
                if paraphrased != sentence:
                    mutations.append(f"deep_paraphrase_round_{round_num}")
                    
            current_text = ' '.join(paraphrased_sentences)
            
        return current_text, mutations
        
    def _reorder_information(self, text: str) -> Tuple[str, List[str]]:
        """Reorder information while preserving meaning"""
        mutations = []
        sentences = sent_tokenize(text)
        
        if len(sentences) < 3:
            return text, mutations
            
        # Identify moveable sentences (not first or last)
        moveable_indices = list(range(1, len(sentences) - 1))
        
        if moveable_indices and random.random() < 0.5:
            # Move one sentence to different position
            source_idx = random.choice(moveable_indices)
            sentence_to_move = sentences[source_idx]
            
            # Remove from original position
            sentences.pop(source_idx)
            
            # Insert at new position
            new_idx = random.choice(range(1, len(sentences)))
            sentences.insert(new_idx, sentence_to_move)
            
            mutations.append("information_reordering")
            
        return ' '.join(sentences), mutations
        
    def _mix_writing_styles(self, text: str) -> Tuple[str, List[str]]:
        """Mix different writing styles"""
        mutations = []
        sentences = sent_tokenize(text)
        mixed_sentences = []
        
        styles = ['academic', 'journalistic', 'conversational', 'technical']
        
        for sentence in sentences:
            if random.random() < 0.4:
                target_style = random.choice(styles)
                styled_sentence = self._apply_style(sentence, target_style)
                mixed_sentences.append(styled_sentence)
                mutations.append(f"style_mix: {target_style}")
            else:
                mixed_sentences.append(sentence)
                
        return ' '.join(mixed_sentences), mutations
        
    def _apply_style(self, sentence: str, style: str) -> str:
        """Apply specific writing style"""
        if style == 'academic':
            return self._academicize(sentence)
        elif style == 'journalistic':
            return self._journalisticize(sentence)
        elif style == 'conversational':
            return self._conversationalize(sentence)
        elif style == 'technical':
            return self._technicalize(sentence)
        else:
            return sentence
            
    def _academicize(self, sentence: str) -> str:
        """Make sentence more academic"""
        academic_replacements = {
            'shows': 'demonstrates',
            'proves': 'establishes',
            'uses': 'utilizes',
            'helps': 'facilitates',
            'big': 'substantial',
            'small': 'minimal'
        }
        
        for informal, formal in academic_replacements.items():
            sentence = re.sub(r'\b' + informal + r'\b', formal, sentence, flags=re.IGNORECASE)
            
        return sentence
        
    def _journalisticize(self, sentence: str) -> str:
        """Make sentence more journalistic"""
        # Add attribution where appropriate
        if 'said' not in sentence.lower() and random.random() < 0.3:
            sentence += ', according to sources.'
            
        return sentence
        
    def _conversationalize(self, sentence: str) -> str:
        """Make sentence more conversational"""
        conversational_replacements = {
            'demonstrates': 'shows',
            'utilizes': 'uses',
            'facilitates': 'helps',
            'substantial': 'big',
            'minimal': 'small'
        }
        
        for formal, informal in conversational_replacements.items():
            sentence = re.sub(r'\b' + formal + r'\b', informal, sentence, flags=re.IGNORECASE)
            
        return sentence
        
    def _technicalize(self, sentence: str) -> str:
        """Make sentence more technical"""
        # Add technical precision
        technical_additions = [
            'specifically', 'precisely', 'systematically', 'methodologically'
        ]
        
        if random.random() < 0.3:
            addition = random.choice(technical_additions)
            # Insert after first few words
            words = sentence.split()
            if len(words) > 3:
                words.insert(2, addition)
                sentence = ' '.join(words)
                
        return sentence
        
    def _vary_complexity_patterns(self, text: str) -> Tuple[str, List[str]]:
        """Vary complexity patterns to avoid detection"""
        mutations = []
        sentences = sent_tokenize(text)
        
        # Calculate current complexity distribution
        complexities = [self._calculate_sentence_complexity(sent) for sent in sentences]
        avg_complexity = sum(complexities) / len(complexities) if complexities else 0
        
        # Adjust some sentences to create variation
        adjusted_sentences = []
        for i, sentence in enumerate(sentences):
            current_complexity = complexities[i]
            
            if random.random() < 0.3:
                if current_complexity > avg_complexity:
                    # Simplify
                    adjusted = self._simplify_sentence(sentence)
                    mutations.append("complexity_reduction")
                else:
                    # Complexify
                    adjusted = self._complexify_sentence(sentence)
                    mutations.append("complexity_increase")
                    
                adjusted_sentences.append(adjusted)
            else:
                adjusted_sentences.append(sentence)
                
        return ' '.join(adjusted_sentences), mutations
        
    def _calculate_sentence_complexity(self, sentence: str) -> float:
        """Calculate sentence complexity score"""
        doc = self.nlp(sentence)
        
        # Factors: length, clause count, vocabulary complexity
        length_score = min(len(sentence.split()) / 20, 1.0)
        
        clause_count = sum(1 for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl'])
        clause_score = min(clause_count / 3, 1.0)
        
        # Vocabulary complexity (syllable count)
        words = sentence.split()
        avg_syllables = sum(self._count_syllables(word) for word in words) / len(words) if words else 0
        vocab_score = min(avg_syllables / 3, 1.0)
        
        return (length_score + clause_score + vocab_score) / 3
        
    def _count_syllables(self, word: str) -> int:
        """Count syllables in word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = True
            else:
                previous_was_vowel = False
                
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        return max(syllable_count, 1)
        
    def _complexify_sentence(self, sentence: str) -> str:
        """Make sentence more complex"""
        # Add subordinate clauses
        subordinators = ['although', 'because', 'since', 'while', 'whereas']
        
        if random.random() < 0.5:
            subordinator = random.choice(subordinators)
            # Simple complexification by adding clause
            return f"{sentence.rstrip('.')} {subordinator} this demonstrates the complexity of the situation."
            
        return sentence
        
    def _reconstruct_text_with_spacing(self, original: str, words: List[str]) -> str:
        """Reconstruct text preserving original spacing and punctuation"""
        # This is a simplified version - in practice, you'd want more sophisticated
        # preservation of the original formatting
        return ' '.join(words)
        
    async def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Comprehensive text analysis"""
        doc = self.nlp(text)
        
        analysis = {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sent_tokenize(text)),
            'readability': flesch_reading_ease(text),
            'sentiment': self.sentiment_analyzer.polarity_scores(text),
            'pos_distribution': self._get_pos_distribution(doc),
            'complexity': self._calculate_text_complexity(text),
            'entities': [ent.text for ent in doc.ents],
            'language': self._detect_language(text)
        }
        
        return analysis
        
    def _get_pos_distribution(self, doc) -> Dict[str, int]:
        """Get part-of-speech distribution"""
        pos_counts = Counter(token.pos_ for token in doc)
        return dict(pos_counts)
        
    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate overall text complexity"""
        sentences = sent_tokenize(text)
        complexities = [self._calculate_sentence_complexity(sent) for sent in sentences]
        return sum(complexities) / len(complexities) if complexities else 0
        
    def _detect_language(self, text: str) -> str:
        """Detect text language"""
        try:
            return langdetect.detect(text)
        except:
            return 'en'  # Default to English
            
    def _calculate_mutation_score(self, original_analysis: Dict, mutated_analysis: Dict) -> float:
        """Calculate how much the text was mutated"""
        # Compare various metrics
        length_change = abs(mutated_analysis['length'] - original_analysis['length']) / original_analysis['length']
        word_change = abs(mutated_analysis['word_count'] - original_analysis['word_count']) / original_analysis['word_count']
        complexity_change = abs(mutated_analysis['complexity'] - original_analysis['complexity'])
        
        return min((length_change + word_change + complexity_change) / 3, 1.0)
        
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        if self.sentence_transformer:
            try:
                embeddings = self.sentence_transformer.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            except:
                pass
                
        # Fallback to simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
        
    async def _calculate_evasion_score(self, original: str, mutated: str) -> float:
        """Calculate detection evasion score"""
        # This would integrate with actual detection models in practice
        # For now, we'll use heuristics
        
        structural_change = self._measure_structural_change(original, mutated)
        lexical_change = self._measure_lexical_change(original, mutated)
        stylistic_change = self._measure_stylistic_change(original, mutated)
        
        return (structural_change + lexical_change + stylistic_change) / 3
        
    def _measure_structural_change(self, text1: str, text2: str) -> float:
        """Measure structural changes"""
        doc1 = self.nlp(text1)
        doc2 = self.nlp(text2)
        
        # Compare dependency structures
        deps1 = [token.dep_ for token in doc1]
        deps2 = [token.dep_ for token in doc2]
        
        if not deps1 or not deps2:
            return 0.0
            
        # Calculate structural similarity
        common_deps = set(deps1).intersection(set(deps2))
        all_deps = set(deps1).union(set(deps2))
        
        return 1.0 - (len(common_deps) / len(all_deps)) if all_deps else 0.0
        
    def _measure_lexical_change(self, text1: str, text2: str) -> float:
        """Measure lexical changes"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1:
            return 0.0
            
        changed_words = words1.symmetric_difference(words2)
        return len(changed_words) / len(words1.union(words2))
        
    def _measure_stylistic_change(self, text1: str, text2: str) -> float:
        """Measure stylistic changes"""
        # Compare readability scores
        readability1 = flesch_reading_ease(text1)
        readability2 = flesch_reading_ease(text2)
        
        readability_change = abs(readability2 - readability1) / 100  # Normalize
        
        # Compare sentence lengths
        sents1 = sent_tokenize(text1)
        sents2 = sent_tokenize(text2)
        
        avg_len1 = sum(len(s.split()) for s in sents1) / len(sents1) if sents1 else 0
        avg_len2 = sum(len(s.split()) for s in sents2) / len(sents2) if sents2 else 0
        
        length_change = abs(avg_len2 - avg_len1) / max(avg_len1, avg_len2, 1)
        
        return (readability_change + length_change) / 2
        
    def _get_cached_result(self, text_hash: str, config: MutationConfig) -> Optional[MutationResult]:
        """Get cached mutation result"""
        with self.cache_lock:
            cache_key = f"{text_hash}_{hash(str(config.__dict__))}"
            return self.result_cache.get(cache_key)
            
    def _cache_result(self, text_hash: str, config: MutationConfig, result: MutationResult):
        """Cache mutation result"""
        with self.cache_lock:
            cache_key = f"{text_hash}_{hash(str(config.__dict__))}"
            self.result_cache[cache_key] = result
            
            # Limit cache size
            if len(self.result_cache) > 1000:
                # Remove oldest entries
                oldest_keys = list(self.result_cache.keys())[:100]
                for key in oldest_keys:
                    del self.result_cache[key]
                    
    async def _store_mutation_result(self, result: MutationResult):
        """Store mutation result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            original_hash = hashlib.sha256(result.original_text.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO mutations 
                (original_hash, original_text, mutated_text, mutation_config, 
                 mutation_score, semantic_similarity, evasion_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                original_hash,
                result.original_text,
                result.mutated_text,
                json.dumps(result.metadata.get('config', {})),
                result.mutation_score,
                result.semantic_similarity,
                result.detection_evasion_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store mutation result: {e}")
            
    async def batch_mutate_texts(self, texts: List[str], 
                                config: Optional[MutationConfig] = None) -> List[MutationResult]:
        """Batch mutate multiple texts"""
        config = config or self.config
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(texts), mp.cpu_count())) as executor:
            tasks = []
            for text in texts:
                task = asyncio.create_task(self.mutate_text(text, config))
                tasks.append(task)
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        # Filter out exceptions and return successful results
        successful_results = []
        for result in results:
            if isinstance(result, MutationResult):
                successful_results.append(result)
            else:
                logger.error(f"Mutation failed: {result}")
                
        return successful_results
        
    def get_mutation_statistics(self) -> Dict[str, Any]:
        """Get mutation operation statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute('SELECT COUNT(*) FROM mutations')
            total_mutations = cursor.fetchone()[0]
            
            cursor.execute('SELECT AVG(mutation_score) FROM mutations')
            avg_mutation_score = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT AVG(semantic_similarity) FROM mutations')
            avg_semantic_similarity = cursor.fetchone()[0] or 0
            
            cursor.execute('SELECT AVG(evasion_score) FROM mutations')
            avg_evasion_score = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_mutations': total_mutations,
                'average_mutation_score': avg_mutation_score,
                'average_semantic_similarity': avg_semantic_similarity,
                'average_evasion_score': avg_evasion_score,
                'cache_size': len(self.result_cache)
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
            
    async def optimize_for_target(self, text: str, target_metrics: Dict[str, float]) -> MutationResult:
        """Optimize mutation for specific target metrics"""
        best_result = None
        best_score = 0
        
        # Try different configurations
        for intensity in [0.3, 0.5, 0.7, 0.9]:
            for mutation_types in [
                ['synonym_replacement', 'paraphrasing'],
                ['sentence_restructuring', 'discourse_markers'],
                ['register_shifting', 'stylistic_variation'],
                ['synonym_replacement', 'sentence_restructuring', 'stylistic_variation']
            ]:
                config = MutationConfig(
                    mutation_intensity=intensity,
                    mutation_types=mutation_types
                )
                
                result = await self.mutate_text(text, config)
                score = self._calculate_target_fitness(result, target_metrics)
                
                if score > best_score:
                    best_score = score
                    best_result = result
                    
        return best_result or await self.mutate_text(text)
        
    def _calculate_target_fitness(self, result: MutationResult, targets: Dict[str, float]) -> float:
        """Calculate fitness score against target metrics"""
        fitness = 0
        count = 0
        
        if 'mutation_score' in targets:
            fitness += 1 - abs(result.mutation_score - targets['mutation_score'])
            count += 1
            
        if 'semantic_similarity' in targets:
            fitness += 1 - abs(result.semantic_similarity - targets['semantic_similarity'])
            count += 1
            
        if 'evasion_score' in targets:
            fitness += 1 - abs(result.detection_evasion_score - targets['evasion_score'])
            count += 1
            
        return fitness / count if count > 0 else 0

# Advanced mutation utilities
class MutationPipeline:
    """Pipeline for complex mutation workflows"""
    
    def __init__(self):
        self.mutator = AdvancedTextMutator()
        self.pipeline_steps = []
        
    def add_step(self, step_name: str, config: MutationConfig):
        """Add mutation step to pipeline"""
        self.pipeline_steps.append((step_name, config))
        
    async def execute_pipeline(self, text: str) -> List[MutationResult]:
        """Execute complete mutation pipeline"""
        results = []
        current_text = text
        
        for step_name, config in self.pipeline_steps:
            result = await self.mutator.mutate_text(current_text, config)
            result.metadata['pipeline_step'] = step_name
            results.append(result)
            current_text = result.mutated_text
            
        return results

# Evaluation utilities
class MutationEvaluator:
    """Evaluate mutation quality and effectiveness"""
    
    def __init__(self):
        self.evaluations = []
        
    async def evaluate_mutation(self, result: MutationResult) -> Dict[str, float]:
        """Comprehensive mutation evaluation"""
        evaluation = {
            'semantic_preservation': result.semantic_similarity,
            'structural_diversity': self._evaluate_structural_diversity(result),
            'lexical_diversity': self._evaluate_lexical_diversity(result),
            'readability_impact': abs(result.readability_change) / 100,
            'evasion_effectiveness': result.detection_evasion_score,
            'overall_quality': 0.0
        }
        
        # Calculate overall quality
        weights = [0.3, 0.2, 0.2, 0.1, 0.2]  # Weighted importance
        evaluation['overall_quality'] = sum(
            score * weight for score, weight in zip(evaluation.values(), weights)
        )
        
        self.evaluations.append(evaluation)
        return evaluation
        
    def _evaluate_structural_diversity(self, result: MutationResult) -> float:
        """Evaluate structural diversity of mutations"""
        # Count different types of mutations applied
        unique_mutation_types = set()
        for mutation in result.mutations_applied:
            mutation_type = mutation.split(':')[0] if ':' in mutation else mutation
            unique_mutation_types.add(mutation_type)
            
        # Normalize by maximum possible diversity
        max_types = 10  # Approximate maximum
        return len(unique_mutation_types) / max_types
        
    def _evaluate_lexical_diversity(self, result: MutationResult) -> float:
        """Evaluate lexical diversity of mutations"""
        original_words = set(result.original_text.lower().split())
        mutated_words = set(result.mutated_text.lower().split())
        
        if not original_words:
            return 0.0
            
        changed_words = original_words.symmetric_difference(mutated_words)
        return len(changed_words) / len(original_words)

# Main execution and testing
if __name__ == "__main__":
    async def test_mutation_system():
        """Test the mutation system"""
        mutator = AdvancedTextMutator()
        
        sample_text = """
        Artificial intelligence represents a significant technological advancement that has the potential 
        to transform various industries. The implementation of AI systems requires careful consideration 
        of ethical implications and societal impact. Furthermore, the development of robust AI models 
        necessitates extensive training data and computational resources.
        """
        
        # Test basic mutation
        config = MutationConfig(
            mutation_intensity=0.7,
            mutation_types=['synonym_replacement', 'paraphrasing', 'sentence_restructuring'],
            evasion_focus=['ai_detection', 'style_fingerprinting']
        )
        
        result = await mutator.mutate_text(sample_text, config)
        
        print("=== MUTATION RESULT ===")
        print(f"Original: {result.original_text[:100]}...")
        print(f"Mutated: {result.mutated_text[:100]}...")
        print(f"Mutation Score: {result.mutation_score:.3f}")
        print(f"Semantic Similarity: {result.semantic_similarity:.3f}")
        print(f"Evasion Score: {result.detection_evasion_score:.3f}")
        print(f"Mutations Applied: {len(result.mutations_applied)}")
        
        # Test pipeline
        pipeline = MutationPipeline()
        pipeline.add_step("light_mutation", MutationConfig(mutation_intensity=0.3))
        pipeline.add_step("heavy_mutation", MutationConfig(mutation_intensity=0.8))
        
        pipeline_results = await pipeline.execute_pipeline(sample_text)
        print(f"\n=== PIPELINE RESULTS ===")
        for i, result in enumerate(pipeline_results):
            print(f"Step {i+1}: {result.metadata.get('pipeline_step', 'unknown')}")
            print(f"  Mutation Score: {result.mutation_score:.3f}")
            
        # Test evaluation
        evaluator = MutationEvaluator()
        evaluation = await evaluator.evaluate_mutation(result)
        print(f"\n=== EVALUATION ===")
        for metric, score in evaluation.items():
            print(f"{metric}: {score:.3f}")
            
        # Get statistics
        stats = mutator.get_mutation_statistics()
        print(f"\n=== STATISTICS ===")
        for stat, value in stats.items():
            print(f"{stat}: {value}")
            
    # Run test
    asyncio.run(test_mutation_system())

