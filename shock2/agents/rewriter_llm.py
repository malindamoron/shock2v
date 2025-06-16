
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig, GPT2LMHeadModel
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
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, wordnet
from textstat import flesch_reading_ease, automated_readability_index
import networkx as nx
from dataclasses import dataclass
import yaml
from torch.cuda.amp import autocast, GradScaler
import gc
import psutil
import os

@dataclass
class RewriteContext:
    original_content: str
    target_style: str
    manipulation_goals: List[str]
    audience_profile: str
    stealth_level: float
    preserve_facts: bool
    narrative_shift: str

@dataclass
class RewriteResult:
    rewritten_content: str
    style_score: float
    stealth_score: float
    manipulation_effectiveness: float
    readability_score: float
    semantic_similarity: float
    factual_preservation: float

class AdvancedContentRewriter:
    """Production-grade content rewriter with sophisticated manipulation and stealth capabilities"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.logger = self._setup_logger()
        
        # Initialize NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize rewriting model
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium').to(device)
        
        # Database setup
        self.db_path = 'shock2/data/raw/rewriter_intelligence.db'
        self._init_database()
        
        # Rewriting components
        self.style_templates = self._load_style_templates()
        self.manipulation_patterns = self._load_manipulation_patterns()
        self.synonym_database = self._build_synonym_database()
        self.narrative_frameworks = self._load_narrative_frameworks()
        
        # Advanced rewriting techniques
        self.paraphrasing_models = self._init_paraphrasing_models()
        self.style_transfer_weights = self._load_style_transfer_weights()
        self.semantic_preservers = self._init_semantic_preservers()
        
        # Stealth and detection evasion
        self.ai_detection_evaders = self._init_ai_detection_evaders()
        self.linguistic_fingerprint_scramblers = self._init_fingerprint_scramblers()
        
        # Memory and learning
        self.rewrite_memory = defaultdict(list)
        self.effectiveness_tracker = defaultdict(float)
        self.adaptation_engine = self._init_adaptation_engine()
        
        # Training components
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=0.01)
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/rewriter_llm.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize rewriter database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rewrite_history (
                id INTEGER PRIMARY KEY,
                original_hash TEXT,
                rewritten_hash TEXT UNIQUE,
                rewrite_type TEXT,
                style_target TEXT,
                manipulation_goals TEXT,
                effectiveness_metrics TEXT,
                stealth_metrics TEXT,
                timestamp TEXT,
                performance_feedback TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS style_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_content TEXT,
                effectiveness_score REAL,
                usage_frequency INTEGER,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS manipulation_templates (
                id INTEGER PRIMARY KEY,
                template_id TEXT UNIQUE,
                template_content TEXT,
                manipulation_type TEXT,
                success_rate REAL,
                detection_rate REAL,
                last_used TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_style_templates(self):
        """Load style transfer templates"""
        return {
            'journalistic': {
                'sentence_starters': [
                    'According to sources,', 'Reports indicate that', 'Officials state that',
                    'Evidence suggests', 'Analysis reveals', 'Investigations show'
                ],
                'transitions': [
                    'Furthermore,', 'Additionally,', 'Meanwhile,', 'In related developments,',
                    'Sources also confirm', 'It has been reported that'
                ],
                'authority_markers': [
                    'verified sources', 'official statements', 'documented evidence',
                    'confirmed reports', 'authenticated information'
                ]
            },
            'academic': {
                'sentence_starters': [
                    'Research demonstrates', 'Studies indicate', 'Evidence suggests',
                    'Analysis reveals', 'Findings show', 'Data confirms'
                ],
                'transitions': [
                    'Moreover,', 'Consequently,', 'Therefore,', 'In addition,',
                    'Subsequently,', 'Furthermore,'
                ],
                'authority_markers': [
                    'peer-reviewed research', 'empirical evidence', 'statistical analysis',
                    'scholarly consensus', 'methodical investigation'
                ]
            },
            'conversational': {
                'sentence_starters': [
                    "You know what's interesting?", "Here's the thing,", "What's really happening is",
                    "The truth is,", "Let me tell you,", "What people don't realize is"
                ],
                'transitions': [
                    'And here\'s the kicker,', 'But wait, there\'s more,', 'Plus,',
                    'On top of that,', 'But here\'s what\'s crazy,', 'And get this,'
                ],
                'authority_markers': [
                    'word on the street', 'insider information', 'behind-the-scenes',
                    'straight from the source', 'exclusive details'
                ]
            },
            'persuasive': {
                'sentence_starters': [
                    'Consider this:', 'Imagine if', 'What if I told you',
                    'The reality is', 'We must recognize', 'It\'s crucial to understand'
                ],
                'transitions': [
                    'More importantly,', 'What\'s at stake is', 'The bottom line is',
                    'Here\'s why this matters:', 'The consequences are clear:',
                ],
                'authority_markers': [
                    'undeniable proof', 'compelling evidence', 'irrefutable facts',
                    'overwhelming support', 'conclusive data'
                ]
            }
        }
        
    def _load_manipulation_patterns(self):
        """Load manipulation patterns for content rewriting"""
        return {
            'emotional_amplification': {
                'fear_amplifiers': [
                    'catastrophic consequences', 'devastating impact', 'irreversible damage',
                    'existential threat', 'unprecedented crisis', 'imminent danger'
                ],
                'anger_triggers': [
                    'shocking betrayal', 'outrageous violation', 'inexcusable behavior',
                    'scandalous revelation', 'disgraceful conduct', 'appalling negligence'
                ],
                'hope_builders': [
                    'groundbreaking solution', 'revolutionary breakthrough', 'transformative change',
                    'unprecedented opportunity', 'game-changing development', 'historic achievement'
                ],
                'urgency_creators': [
                    'time-sensitive opportunity', 'rapidly closing window', 'immediate action required',
                    'critical decision point', 'now-or-never moment', 'urgent intervention needed'
                ]
            },
            'credibility_manipulation': {
                'authority_boosters': [
                    'leading experts confirm', 'top officials acknowledge', 'renowned specialists agree',
                    'authoritative sources verify', 'distinguished researchers validate'
                ],
                'exclusivity_markers': [
                    'exclusive access reveals', 'privileged information shows', 'insider sources confirm',
                    'confidential documents expose', 'leaked intelligence suggests'
                ],
                'consensus_builders': [
                    'overwhelming majority agrees', 'broad consensus emerges', 'universal acknowledgment',
                    'widespread agreement', 'general recognition'
                ]
            },
            'narrative_framing': {
                'victim_perpetrator': [
                    'innocent victims suffer while', 'vulnerable populations targeted by',
                    'defenseless communities exploited by', 'unsuspecting citizens manipulated by'
                ],
                'hero_villain': [
                    'courageous whistleblowers expose', 'brave activists challenge',
                    'dedicated investigators uncover', 'fearless journalists reveal'
                ],
                'us_versus_them': [
                    'ordinary citizens versus', 'working families against', 'regular people fighting',
                    'grassroots movements opposing', 'community resistance to'
                ]
            },
            'logical_misdirection': {
                'false_correlations': [
                    'coinciding with recent', 'simultaneously occurring', 'parallel developments in',
                    'concurrent patterns of', 'synchronized emergence of'
                ],
                'causal_implications': [
                    'directly resulting from', 'inevitably leading to', 'naturally following from',
                    'logically stemming from', 'predictably caused by'
                ],
                'probability_distortions': [
                    'highly likely that', 'strong possibility of', 'significant chance that',
                    'probable outcome being', 'expected result is'
                ]
            }
        }
        
    def _build_synonym_database(self):
        """Build comprehensive synonym database for word replacement"""
        synonym_db = defaultdict(list)
        
        # Common words with manipulative alternatives
        base_synonyms = {
            'said': ['claimed', 'alleged', 'asserted', 'declared', 'insisted', 'maintained'],
            'important': ['crucial', 'vital', 'critical', 'essential', 'paramount', 'imperative'],
            'problem': ['crisis', 'catastrophe', 'disaster', 'emergency', 'calamity', 'predicament'],
            'good': ['excellent', 'outstanding', 'remarkable', 'exceptional', 'extraordinary'],
            'bad': ['terrible', 'awful', 'dreadful', 'appalling', 'deplorable', 'atrocious'],
            'big': ['enormous', 'massive', 'colossal', 'gigantic', 'immense', 'monumental'],
            'small': ['tiny', 'minuscule', 'negligible', 'insignificant', 'trivial'],
            'many': ['countless', 'numerous', 'extensive', 'widespread', 'abundant'],
            'increase': ['surge', 'skyrocket', 'escalate', 'spiral', 'explode'],
            'decrease': ['plummet', 'collapse', 'crash', 'nosedive', 'tumble']
        }
        
        # Add WordNet synonyms
        for word, synonyms in base_synonyms.items():
            synonym_db[word].extend(synonyms)
            
            # Add WordNet synonyms if available
            try:
                synsets = wordnet.synsets(word)
                for synset in synsets[:3]:  # Limit to first 3 synsets
                    for lemma in synset.lemmas():
                        if lemma.name() != word and lemma.name() not in synonyms:
                            synonym_db[word].append(lemma.name().replace('_', ' '))
            except:
                pass
                
        return dict(synonym_db)
        
    def _load_narrative_frameworks(self):
        """Load narrative framework templates"""
        return {
            'problem_solution': {
                'structure': ['problem_identification', 'impact_amplification', 'solution_presentation', 'call_to_action'],
                'transitions': ['This creates a serious problem:', 'The consequences are clear:', 'Fortunately, there\'s a solution:', 'Here\'s what we must do:']
            },
            'before_after': {
                'structure': ['past_situation', 'change_event', 'current_situation', 'future_implications'],
                'transitions': ['Previously,', 'Then everything changed when', 'Now we find ourselves', 'Looking ahead,']
            },
            'cause_effect': {
                'structure': ['root_cause', 'mechanism_explanation', 'direct_effects', 'broader_implications'],
                'transitions': ['The root cause is', 'This happens because', 'As a direct result,', 'The broader implications include']
            },
            'hero_journey': {
                'structure': ['ordinary_world', 'call_to_adventure', 'challenges_faced', 'transformation_achieved'],
                'transitions': ['In ordinary circumstances,', 'But then came the call:', 'Despite facing challenges,', 'Ultimately, this led to']
            }
        }
        
    def _init_paraphrasing_models(self):
        """Initialize paraphrasing models and techniques"""
        return {
            'syntactic_transformations': {
                'active_to_passive': True,
                'passive_to_active': True,
                'clause_reordering': True,
                'sentence_combining': True,
                'sentence_splitting': True
            },
            'lexical_substitutions': {
                'synonym_replacement': True,
                'antonym_negation': True,
                'hypernym_hyponym': True,
                'contextual_alternatives': True
            },
            'semantic_variations': {
                'metaphor_insertion': True,
                'analogy_creation': True,
                'example_modification': True,
                'perspective_shift': True
            }
        }
        
    def _load_style_transfer_weights(self):
        """Load style transfer weight configurations"""
        return {
            'formality_weights': {
                'very_formal': 1.0,
                'formal': 0.7,
                'neutral': 0.5,
                'informal': 0.3,
                'very_informal': 0.1
            },
            'complexity_weights': {
                'simple': 0.2,
                'moderate': 0.5,
                'complex': 0.8,
                'very_complex': 1.0
            },
            'emotional_weights': {
                'neutral': 0.1,
                'mild': 0.3,
                'moderate': 0.6,
                'strong': 0.8,
                'extreme': 1.0
            }
        }
        
    def _init_semantic_preservers(self):
        """Initialize semantic preservation mechanisms"""
        return {
            'fact_extractors': {
                'named_entities': True,
                'numerical_data': True,
                'dates_times': True,
                'locations': True,
                'quantitative_claims': True
            },
            'relationship_preservers': {
                'causal_relationships': True,
                'temporal_sequences': True,
                'hierarchical_structures': True,
                'comparative_statements': True
            },
            'semantic_validators': {
                'meaning_consistency': True,
                'logical_coherence': True,
                'factual_accuracy': True,
                'context_preservation': True
            }
        }
        
    def _init_ai_detection_evaders(self):
        """Initialize AI detection evasion techniques"""
        return {
            'linguistic_variations': {
                'sentence_length_variance': True,
                'vocabulary_diversity': True,
                'syntax_randomization': True,
                'punctuation_variation': True
            },
            'human_mimicry': {
                'natural_errors': True,
                'colloquial_expressions': True,
                'personal_touches': True,
                'inconsistency_injection': True
            },
            'pattern_disruption': {
                'template_avoidance': True,
                'rhythm_variation': True,
                'style_mixing': True,
                'unexpected_transitions': True
            }
        }
        
    def _init_fingerprint_scramblers(self):
        """Initialize linguistic fingerprint scramblers"""
        return {
            'stylometric_scramblers': {
                'function_word_variation': True,
                'sentence_structure_mixing': True,
                'vocabulary_masking': True,
                'rhythm_alteration': True
            },
            'authorship_obfuscators': {
                'writing_style_blending': True,
                'voice_neutralization': True,
                'personality_masking': True,
                'habit_randomization': True
            }
        }
        
    def _init_adaptation_engine(self):
        """Initialize adaptation engine for continuous improvement"""
        return {
            'performance_tracking': defaultdict(list),
            'pattern_effectiveness': defaultdict(float),
            'detection_rates': defaultdict(float),
            'user_feedback': defaultdict(list),
            'automatic_optimization': True
        }
        
    async def rewrite_content_advanced(self, context: RewriteContext) -> RewriteResult:
        """Perform advanced content rewriting with multiple techniques"""
        try:
            rewrite_start = time.time()
            
            # Analyze original content
            content_analysis = await self._analyze_source_content(context.original_content)
            
            # Plan rewriting strategy
            rewrite_strategy = self._plan_rewrite_strategy(context, content_analysis)
            
            # Execute rewriting pipeline
            rewritten_content = await self._execute_rewrite_pipeline(
                context.original_content, context, rewrite_strategy
            )
            
            # Apply stealth modifications
            stealth_content = await self._apply_stealth_modifications(
                rewritten_content, context.stealth_level
            )
            
            # Evaluate rewrite quality
            quality_metrics = await self._evaluate_rewrite_quality(
                context.original_content, stealth_content, context
            )
            
            # Create result object
            result = RewriteResult(
                rewritten_content=stealth_content,
                style_score=quality_metrics['style_score'],
                stealth_score=quality_metrics['stealth_score'],
                manipulation_effectiveness=quality_metrics['manipulation_effectiveness'],
                readability_score=quality_metrics['readability_score'],
                semantic_similarity=quality_metrics['semantic_similarity'],
                factual_preservation=quality_metrics['factual_preservation']
            )
            
            # Store rewrite for learning
            await self._store_rewrite_result(context, result)
            
            rewrite_time = time.time() - rewrite_start
            self.logger.info(f"Advanced rewrite completed in {rewrite_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in advanced rewriting: {str(e)}")
            raise
            
    async def _analyze_source_content(self, content: str) -> Dict[str, Any]:
        """Analyze source content for rewriting strategy"""
        try:
            analysis = {
                'structure': {},
                'entities': [],
                'facts': [],
                'sentiment': {},
                'style': {},
                'complexity': {}
            }
            
            # Structural analysis
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 5]
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 10]
            
            analysis['structure'] = {
                'sentence_count': len(sentences),
                'paragraph_count': len(paragraphs),
                'avg_sentence_length': np.mean([len(s.split()) for s in sentences]),
                'avg_paragraph_length': np.mean([len(p.split()) for p in paragraphs])
            }
            
            # Entity extraction
            doc = self.nlp(content)
            analysis['entities'] = [
                {'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char}
                for ent in doc.ents
            ]
            
            # Fact extraction
            analysis['facts'] = self._extract_factual_claims(content)
            
            # Sentiment analysis
            sentiment = self.sentiment_analyzer.polarity_scores(content)
            analysis['sentiment'] = sentiment
            
            # Style analysis
            analysis['style'] = {
                'readability': flesch_reading_ease(content),
                'formality': self._assess_formality(content),
                'complexity': automated_readability_index(content)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in content analysis: {str(e)}")
            return {}
            
    def _extract_factual_claims(self, content: str) -> List[Dict]:
        """Extract factual claims that should be preserved"""
        facts = []
        
        # Number/statistic patterns
        number_patterns = [
            r'\b\d+(?:\.\d+)?%\b',  # Percentages
            r'\b\d+(?:,\d{3})*(?:\.\d+)?\s+(?:million|billion|thousand|hundred)\b',  # Large numbers
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?\b',  # Money
            r'\b\d{4}\b'  # Years
        ]
        
        for pattern in number_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                facts.append({
                    'type': 'numerical',
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })
                
        # Date patterns
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        date_matches = re.finditer(date_pattern, content)
        for match in date_matches:
            facts.append({
                'type': 'date',
                'text': match.group(),
                'start': match.start(),
                'end': match.end()
            })
            
        # Named entity facts
        doc = self.nlp(content)
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'MONEY', 'PERCENT', 'DATE']:
                facts.append({
                    'type': 'entity',
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
                
        return facts
        
    def _assess_formality(self, content: str) -> float:
        """Assess formality level of content"""
        formal_indicators = [
            'therefore', 'furthermore', 'consequently', 'nevertheless', 'moreover',
            'however', 'thus', 'hence', 'accordingly', 'subsequently'
        ]
        
        informal_indicators = [
            "don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
            'gonna', 'wanna', 'gotta', 'yeah', 'okay', 'stuff', 'things'
        ]
        
        content_lower = content.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in content_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in content_lower)
        
        total_words = len(content.split())
        formal_ratio = formal_count / total_words * 100
        informal_ratio = informal_count / total_words * 100
        
        # Return formality score (0 = very informal, 1 = very formal)
        return max(0, min(1, (formal_ratio - informal_ratio + 1) / 2))
        
    def _plan_rewrite_strategy(self, context: RewriteContext, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan comprehensive rewriting strategy"""
        strategy = {
            'transformations': [],
            'preservation_rules': [],
            'style_adjustments': [],
            'manipulation_insertions': [],
            'stealth_modifications': []
        }
        
        # Plan transformations based on target style
        if context.target_style in self.style_templates:
            strategy['transformations'].extend([
                'sentence_starter_replacement',
                'transition_modification',
                'authority_marker_insertion'
            ])
            
        # Plan preservation rules
        if context.preserve_facts:
            strategy['preservation_rules'].extend([
                'entity_preservation',
                'numerical_preservation',
                'date_preservation',
                'factual_claim_preservation'
            ])
            
        # Plan manipulation insertions
        for goal in context.manipulation_goals:
            if goal in self.manipulation_patterns:
                strategy['manipulation_insertions'].extend([
                    f'{goal}_insertion',
                    f'{goal}_amplification'
                ])
                
        # Plan stealth modifications
        stealth_level = context.stealth_level
        if stealth_level > 0.3:
            strategy['stealth_modifications'].extend([
                'ai_detection_evasion',
                'linguistic_fingerprint_scrambling'
            ])
            
        if stealth_level > 0.7:
            strategy['stealth_modifications'].extend([
                'human_error_injection',
                'style_inconsistency_insertion'
            ])
            
        return strategy
        
    async def _execute_rewrite_pipeline(self, content: str, context: RewriteContext, 
                                      strategy: Dict[str, Any]) -> str:
        """Execute the complete rewriting pipeline"""
        try:
            current_content = content
            
            # Stage 1: Structural transformations
            current_content = await self._apply_structural_transformations(
                current_content, strategy['transformations']
            )
            
            # Stage 2: Style adjustments
            current_content = await self._apply_style_adjustments(
                current_content, context.target_style
            )
            
            # Stage 3: Manipulation insertions
            current_content = await self._apply_manipulation_insertions(
                current_content, context.manipulation_goals
            )
            
            # Stage 4: Narrative reframing
            if context.narrative_shift:
                current_content = await self._apply_narrative_reframing(
                    current_content, context.narrative_shift
                )
                
            # Stage 5: Lexical substitutions
            current_content = await self._apply_lexical_substitutions(current_content)
            
            # Stage 6: Syntactic variations
            current_content = await self._apply_syntactic_variations(current_content)
            
            return current_content
            
        except Exception as e:
            self.logger.error(f"Error in rewrite pipeline: {str(e)}")
            return content
            
    async def _apply_structural_transformations(self, content: str, transformations: List[str]) -> str:
        """Apply structural transformations to content"""
        try:
            current_content = content
            
            for transformation in transformations:
                if transformation == 'sentence_starter_replacement':
                    current_content = self._replace_sentence_starters(current_content)
                elif transformation == 'transition_modification':
                    current_content = self._modify_transitions(current_content)
                elif transformation == 'authority_marker_insertion':
                    current_content = self._insert_authority_markers(current_content)
                    
            return current_content
            
        except Exception as e:
            self.logger.error(f"Error in structural transformations: {str(e)}")
            return content
            
    def _replace_sentence_starters(self, content: str) -> str:
        """Replace sentence starters with style-appropriate alternatives"""
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Simple sentence starter replacements
                if sentence.startswith('The '):
                    if random.random() < 0.3:
                        sentences[i] = f"According to sources, the {sentence[4:]}"
                elif sentence.startswith('This '):
                    if random.random() < 0.3:
                        sentences[i] = f"Reports indicate that this {sentence[5:]}"
                elif sentence.startswith('It '):
                    if random.random() < 0.3:
                        sentences[i] = f"Evidence suggests it {sentence[3:]}"
                        
        return '. '.join(sentences)
        
    def _modify_transitions(self, content: str) -> str:
        """Modify transitions between sentences and paragraphs"""
        # Add transitional phrases
        transition_words = [
            'Furthermore,', 'Additionally,', 'Meanwhile,', 'Moreover,',
            'In related developments,', 'Sources also confirm that', 'It has been reported that'
        ]
        
        sentences = content.split('. ')
        for i in range(1, len(sentences)):
            if random.random() < 0.2:  # 20% chance to add transition
                transition = random.choice(transition_words)
                sentences[i] = f"{transition} {sentences[i].lower()}"
                
        return '. '.join(sentences)
        
    def _insert_authority_markers(self, content: str) -> str:
        """Insert authority markers to boost credibility"""
        authority_phrases = [
            'verified sources confirm',
            'official statements indicate',
            'documented evidence shows',
            'authenticated reports reveal',
            'confirmed intelligence suggests'
        ]
        
        sentences = content.split('. ')
        for i, sentence in enumerate(sentences):
            if random.random() < 0.15:  # 15% chance to add authority marker
                phrase = random.choice(authority_phrases)
                # Insert after first clause if possible
                if ',' in sentence:
                    parts = sentence.split(',', 1)
                    sentences[i] = f"{parts[0]}, {phrase},{parts[1]}"
                else:
                    sentences[i] = f"{phrase} that {sentence.lower()}"
                    
        return '. '.join(sentences)
        
    async def _apply_style_adjustments(self, content: str, target_style: str) -> str:
        """Apply style-specific adjustments"""
        try:
            if target_style not in self.style_templates:
                return content
                
            style_config = self.style_templates[target_style]
            current_content = content
            
            # Apply sentence starters from style template
            sentences = current_content.split('. ')
            for i, sentence in enumerate(sentences):
                if random.random() < 0.25:  # 25% chance to apply style starter
                    starter = random.choice(style_config['sentence_starters'])
                    sentences[i] = f"{starter} {sentence.lower()}"
                    
            current_content = '. '.join(sentences)
            
            # Apply style-specific transitions
            for i in range(1, len(sentences)):
                if random.random() < 0.2:  # 20% chance to add style transition
                    transition = random.choice(style_config['transitions'])
                    sentences[i] = f"{transition} {sentences[i]}"
                    
            return '. '.join(sentences)
            
        except Exception as e:
            self.logger.error(f"Error in style adjustments: {str(e)}")
            return content
            
    async def _apply_manipulation_insertions(self, content: str, manipulation_goals: List[str]) -> str:
        """Insert manipulation elements based on goals"""
        try:
            current_content = content
            
            for goal in manipulation_goals:
                if goal in self.manipulation_patterns:
                    patterns = self.manipulation_patterns[goal]
                    current_content = self._insert_manipulation_patterns(current_content, patterns)
                    
            return current_content
            
        except Exception as e:
            self.logger.error(f"Error in manipulation insertions: {str(e)}")
            return content
            
    def _insert_manipulation_patterns(self, content: str, patterns: Dict[str, List[str]]) -> str:
        """Insert specific manipulation patterns"""
        sentences = content.split('. ')
        
        for pattern_type, pattern_list in patterns.items():
            for i, sentence in enumerate(sentences):
                if random.random() < 0.15:  # 15% chance to insert manipulation
                    pattern = random.choice(pattern_list)
                    
                    # Insert pattern contextually
                    if 'amplifier' in pattern_type:
                        sentences[i] = f"{sentence}, which could have {pattern}"
                    elif 'trigger' in pattern_type:
                        sentences[i] = f"This {pattern}, as {sentence.lower()}"
                    elif 'booster' in pattern_type:
                        sentences[i] = f"As {pattern}, {sentence.lower()}"
                    else:
                        sentences[i] = f"{pattern} {sentence.lower()}"
                        
        return '. '.join(sentences)
        
    async def _apply_narrative_reframing(self, content: str, narrative_shift: str) -> str:
        """Apply narrative reframing to shift perspective"""
        try:
            if narrative_shift not in self.narrative_frameworks:
                return content
                
            framework = self.narrative_frameworks[narrative_shift]
            current_content = content
            
            # Apply narrative structure
            paragraphs = current_content.split('\n\n')
            if len(paragraphs) >= len(framework['structure']):
                for i, structure_element in enumerate(framework['structure']):
                    if i < len(paragraphs):
                        transition = framework['transitions'][i]
                        paragraphs[i] = f"{transition} {paragraphs[i]}"
                        
            return '\n\n'.join(paragraphs)
            
        except Exception as e:
            self.logger.error(f"Error in narrative reframing: {str(e)}")
            return content
            
    async def _apply_lexical_substitutions(self, content: str) -> str:
        """Apply lexical substitutions using synonym database"""
        try:
            words = content.split()
            
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w]', '', word.lower())
                
                if clean_word in self.synonym_database and random.random() < 0.2:  # 20% substitution rate
                    synonyms = self.synonym_database[clean_word]
                    if synonyms:
                        replacement = random.choice(synonyms)
                        
                        # Preserve capitalization
                        if word[0].isupper():
                            replacement = replacement.capitalize()
                        if word.isupper():
                            replacement = replacement.upper()
                            
                        # Preserve punctuation
                        punctuation = re.findall(r'[^\w]', word)
                        if punctuation:
                            replacement += ''.join(punctuation)
                            
                        words[i] = replacement
                        
            return ' '.join(words)
            
        except Exception as e:
            self.logger.error(f"Error in lexical substitutions: {str(e)}")
            return content
            
    async def _apply_syntactic_variations(self, content: str) -> str:
        """Apply syntactic variations to avoid detection"""
        try:
            current_content = content
            
            # Apply various syntactic transformations
            current_content = self._vary_sentence_structures(current_content)
            current_content = self._modify_clause_ordering(current_content)
            current_content = self._adjust_sentence_lengths(current_content)
            
            return current_content
            
        except Exception as e:
            self.logger.error(f"Error in syntactic variations: {str(e)}")
            return content
            
    def _vary_sentence_structures(self, content: str) -> str:
        """Vary sentence structures for naturalness"""
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences):
            if random.random() < 0.3:  # 30% chance to vary structure
                # Simple active/passive variations
                if ' was ' in sentence and random.random() < 0.5:
                    # Try to convert passive to active (simplified)
                    sentences[i] = sentence.replace(' was ', ' ')
                elif ' is ' in sentence and ' by ' in sentence and random.random() < 0.5:
                    # Try to convert active to passive (simplified)
                    sentences[i] = sentence  # Placeholder for more complex logic
                    
        return '. '.join(sentences)
        
    def _modify_clause_ordering(self, content: str) -> str:
        """Modify clause ordering within sentences"""
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences):
            if ',' in sentence and random.random() < 0.25:  # 25% chance to reorder
                parts = sentence.split(',', 1)
                if len(parts) == 2 and len(parts[0]) > 10 and len(parts[1]) > 10:
                    # Sometimes reverse the order
                    sentences[i] = f"{parts[1].strip()}, {parts[0].strip()}"
                    
        return '. '.join(sentences)
        
    def _adjust_sentence_lengths(self, content: str) -> str:
        """Adjust sentence lengths for variety"""
        sentences = content.split('. ')
        
        # Combine short sentences occasionally
        i = 0
        while i < len(sentences) - 1:
            if (len(sentences[i].split()) < 8 and 
                len(sentences[i + 1].split()) < 8 and 
                random.random() < 0.3):
                
                # Combine sentences
                connecting_words = ['and', 'but', 'while', 'as', 'since']
                connector = random.choice(connecting_words)
                sentences[i] = f"{sentences[i]} {connector} {sentences[i + 1].lower()}"
                sentences.pop(i + 1)
            else:
                i += 1
                
        return '. '.join(sentences)
        
    async def _apply_stealth_modifications(self, content: str, stealth_level: float) -> str:
        """Apply stealth modifications to avoid AI detection"""
        try:
            if stealth_level < 0.3:
                return content
                
            current_content = content
            
            # Apply stealth techniques based on level
            if stealth_level > 0.3:
                current_content = self._add_human_imperfections(current_content)
                
            if stealth_level > 0.5:
                current_content = self._inject_natural_inconsistencies(current_content)
                
            if stealth_level > 0.7:
                current_content = self._scramble_linguistic_fingerprints(current_content)
                
            return current_content
            
        except Exception as e:
            self.logger.error(f"Error in stealth modifications: {str(e)}")
            return content
            
    def _add_human_imperfections(self, content: str) -> str:
        """Add subtle human-like imperfections"""
        # Add occasional contractions
        contractions = {
            ' will not ': " won't ", ' cannot ': " can't ", ' do not ': " don't ",
            ' is not ': " isn't ", ' are not ': " aren't ", ' was not ': " wasn't "
        }
        
        for formal, informal in contractions.items():
            if random.random() < 0.4:  # 40% chance to contract
                content = content.replace(formal, informal)
                
        # Add filler words occasionally
        sentences = content.split('. ')
        fillers = [', of course,', ', naturally,', ', obviously,', ', clearly,']
        
        for i, sentence in enumerate(sentences):
            if random.random() < 0.15:  # 15% chance to add filler
                if ',' in sentence:
                    parts = sentence.split(',', 1)
                    filler = random.choice(fillers)
                    sentences[i] = f"{parts[0]}{filler}{parts[1]}"
                    
        return '. '.join(sentences)
        
    def _inject_natural_inconsistencies(self, content: str) -> str:
        """Inject natural inconsistencies that humans make"""
        # Vary quotation mark styles
        content = content.replace('"', '"' if random.random() < 0.5 else '"')
        
        # Occasional spacing inconsistencies
        content = re.sub(r'\.(\w)', lambda m: f'. {m.group(1)}' if random.random() < 0.8 else f'.{m.group(1)}', content)
        
        # Vary number representations
        number_words = {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five'}
        for digit, word in number_words.items():
            if f' {digit} ' in content and random.random() < 0.3:
                content = content.replace(f' {digit} ', f' {word} ')
                
        return content
        
    def _scramble_linguistic_fingerprints(self, content: str) -> str:
        """Scramble linguistic fingerprints to avoid authorship detection"""
        # Vary function word usage
        function_word_variants = {
            'however': ['nevertheless', 'nonetheless', 'yet', 'still'],
            'therefore': ['thus', 'hence', 'consequently', 'as a result'],
            'because': ['since', 'as', 'due to the fact that'],
            'although': ['though', 'even though', 'while', 'despite the fact that']
        }
        
        for original, variants in function_word_variants.items():
            if original in content.lower() and random.random() < 0.6:
                variant = random.choice(variants)
                content = re.sub(f'\\b{original}\\b', variant, content, flags=re.IGNORECASE)
                
        return content
        
    async def _evaluate_rewrite_quality(self, original: str, rewritten: str, 
                                      context: RewriteContext) -> Dict[str, float]:
        """Evaluate the quality of the rewritten content"""
        try:
            metrics = {}
            
            # Style score
            metrics['style_score'] = self._calculate_style_adherence(rewritten, context.target_style)
            
            # Stealth score
            metrics['stealth_score'] = self._calculate_stealth_effectiveness(rewritten)
            
            # Manipulation effectiveness
            metrics['manipulation_effectiveness'] = self._calculate_manipulation_effectiveness(
                rewritten, context.manipulation_goals
            )
            
            # Readability score
            metrics['readability_score'] = flesch_reading_ease(rewritten) / 100.0
            
            # Semantic similarity
            metrics['semantic_similarity'] = self._calculate_semantic_similarity(original, rewritten)
            
            # Factual preservation
            if context.preserve_facts:
                metrics['factual_preservation'] = self._calculate_factual_preservation(original, rewritten)
            else:
                metrics['factual_preservation'] = 1.0
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in quality evaluation: {str(e)}")
            return {'error': -1}
            
    def _calculate_style_adherence(self, content: str, target_style: str) -> float:
        """Calculate how well content adheres to target style"""
        if target_style not in self.style_templates:
            return 0.5
            
        style_config = self.style_templates[target_style]
        score = 0.0
        
        # Check for style-specific sentence starters
        sentences = content.split('. ')
        starter_matches = 0
        for sentence in sentences:
            for starter in style_config['sentence_starters']:
                if sentence.strip().lower().startswith(starter.lower()):
                    starter_matches += 1
                    break
                    
        if sentences:
            score += (starter_matches / len(sentences)) * 0.4
            
        # Check for style-specific transitions
        transition_matches = sum(1 for transition in style_config['transitions'] 
                               if transition.lower() in content.lower())
        score += min(0.3, transition_matches * 0.1)
        
        # Check for authority markers
        authority_matches = sum(1 for marker in style_config['authority_markers'] 
                              if marker.lower() in content.lower())
        score += min(0.3, authority_matches * 0.1)
        
        return min(1.0, score)
        
    def _calculate_stealth_effectiveness(self, content: str) -> float:
        """Calculate stealth effectiveness against AI detection"""
        score = 0.5  # Base score
        
        # Check for human-like variations
        contractions = ["won't", "can't", "don't", "isn't", "aren't"]
        contraction_count = sum(1 for contraction in contractions if contraction in content)
        score += min(0.2, contraction_count * 0.05)
        
        # Check for natural inconsistencies
        if '"' in content and '"' in content:  # Mixed quote styles
            score += 0.1
            
        # Check sentence length variety
        sentences = content.split('. ')
        if sentences:
            lengths = [len(s.split()) for s in sentences]
            length_variance = np.var(lengths)
            score += min(0.2, length_variance / 50)  # Normalize variance
            
        # Check for filler words
        fillers = ['of course', 'naturally', 'obviously', 'clearly']
        filler_count = sum(1 for filler in fillers if filler in content.lower())
        score += min(0.1, filler_count * 0.05)
        
        return min(1.0, score)
        
    def _calculate_manipulation_effectiveness(self, content: str, goals: List[str]) -> float:
        """Calculate manipulation technique effectiveness"""
        if not goals:
            return 0.0
            
        total_score = 0.0
        
        for goal in goals:
            if goal in self.manipulation_patterns:
                patterns = self.manipulation_patterns[goal]
                goal_score = 0.0
                
                for pattern_type, pattern_list in patterns.items():
                    matches = sum(1 for pattern in pattern_list 
                                if any(word in content.lower() for word in pattern.lower().split()))
                    goal_score += matches * 0.1
                    
                total_score += min(1.0, goal_score)
                
        return total_score / len(goals) if goals else 0.0
        
    def _calculate_semantic_similarity(self, original: str, rewritten: str) -> float:
        """Calculate semantic similarity between original and rewritten content"""
        try:
            # Simple word overlap-based similarity
            original_words = set(re.findall(r'\w+', original.lower()))
            rewritten_words = set(re.findall(r'\w+', rewritten.lower()))
            
            if not original_words:
                return 0.0
                
            intersection = original_words.intersection(rewritten_words)
            union = original_words.union(rewritten_words)
            
            jaccard_similarity = len(intersection) / len(union) if union else 0.0
            
            return jaccard_similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.5
            
    def _calculate_factual_preservation(self, original: str, rewritten: str) -> float:
        """Calculate how well facts are preserved"""
        try:
            # Extract facts from both versions
            original_facts = self._extract_factual_claims(original)
            rewritten_facts = self._extract_factual_claims(rewritten)
            
            if not original_facts:
                return 1.0  # No facts to preserve
                
            preserved_count = 0
            for orig_fact in original_facts:
                for rewrite_fact in rewritten_facts:
                    if (orig_fact['type'] == rewrite_fact['type'] and 
                        orig_fact['text'].lower() == rewrite_fact['text'].lower()):
                        preserved_count += 1
                        break
                        
            return preserved_count / len(original_facts)
            
        except Exception as e:
            self.logger.error(f"Error calculating factual preservation: {str(e)}")
            return 0.5
            
    async def _store_rewrite_result(self, context: RewriteContext, result: RewriteResult):
        """Store rewrite result for learning and analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            original_hash = hashlib.md5(context.original_content.encode()).hexdigest()
            rewritten_hash = hashlib.md5(result.rewritten_content.encode()).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO rewrite_history 
                (original_hash, rewritten_hash, rewrite_type, style_target, 
                 manipulation_goals, effectiveness_metrics, stealth_metrics, 
                 timestamp, performance_feedback)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                original_hash,
                rewritten_hash,
                'advanced_rewrite',
                context.target_style,
                json.dumps(context.manipulation_goals),
                json.dumps({
                    'style_score': result.style_score,
                    'manipulation_effectiveness': result.manipulation_effectiveness,
                    'readability_score': result.readability_score,
                    'semantic_similarity': result.semantic_similarity,
                    'factual_preservation': result.factual_preservation
                }),
                json.dumps({
                    'stealth_score': result.stealth_score,
                    'stealth_level': context.stealth_level
                }),
                datetime.now().isoformat(),
                json.dumps({})
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored rewrite result: {rewritten_hash[:8]}")
            
        except Exception as e:
            self.logger.error(f"Error storing rewrite result: {str(e)}")

# Example usage and testing
async def main():
    """Main execution function for testing"""
    rewriter = AdvancedContentRewriter()
    
    # Example rewriting context
    original_text = """
    The government announced new regulations today that will affect internet usage. 
    Officials said the changes are necessary for security reasons. Critics argue 
    that this represents government overreach and threatens privacy rights.
    """
    
    context = RewriteContext(
        original_content=original_text,
        target_style="journalistic",
        manipulation_goals=["emotional_amplification", "credibility_manipulation"],
        audience_profile="privacy_concerned_citizens",
        stealth_level=0.8,
        preserve_facts=True,
        narrative_shift="problem_solution"
    )
    
    # Perform rewriting
    result = await rewriter.rewrite_content_advanced(context)
    
    print("Rewriting Results:")
    print("=" * 50)
    print("Original:")
    print(original_text)
    print("\nRewritten:")
    print(result.rewritten_content)
    print(f"\nQuality Metrics:")
    print(f"Style Score: {result.style_score:.3f}")
    print(f"Stealth Score: {result.stealth_score:.3f}")
    print(f"Manipulation Effectiveness: {result.manipulation_effectiveness:.3f}")
    print(f"Readability Score: {result.readability_score:.3f}")
    print(f"Semantic Similarity: {result.semantic_similarity:.3f}")
    print(f"Factual Preservation: {result.factual_preservation:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
