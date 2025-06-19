
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import json
import sqlite3
import logging
import os
import time
import math
import random
import pickle
import gc
import hashlib
import threading
import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
import warnings
warnings.filterwarnings('ignore')

# Advanced ML libraries
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer,
    GPT2LMHeadModel, GPTNeoXForCausalLM, LlamaForCausalLM,
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)
import wandb
import tensorboard
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
import networkx as nx
from scipy import stats
import spacy
from textstat import flesch_reading_ease, automated_readability_index
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import datasets
from datasets import Dataset as HFDataset
import faiss
import psutil
import GPUtil
from memory_profiler import profile
import line_profiler
import cProfile
import pstats
import io
import functools
import multiprocessing as cpu_mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import ray
import horovod.torch as hvd
from apex import amp
import deepspeed
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.optim.grad_scaler import ShardedGradScaler
import accelerate
from accelerate import Accelerator
import bitsandbytes as bnb
import peft
from peft import LoraConfig, get_peft_model, TaskType
import flash_attn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

# Custom imports for Shock2 system
from shock2.worldwatch.crawler.openweb_crawler import AdvancedNewsCrawler
from shock2.agents.deception_llm import AdvancedDeceptionEngine
from shock2.agents.writer_llm import AdvancedNewsWriter
from shock2.training.preprocessor.clean_normalize import AdvancedNewsPreprocessor
from shock2.utils.benchmark_utils import SystemBenchmarker
from shock2.utils.proxy_rotator import ProxyRotator
from shock2.utils.tor_handler import TorHandler

class AdvancedShock2Dataset(Dataset):
    """Ultra-sophisticated dataset for adversarial training with multi-modal manipulation"""
    
    def __init__(self, data_sources: List[str], tokenizer, config: Dict):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.get('max_length', 4096)
        self.manipulation_strength = config.get('manipulation_strength', 0.8)
        self.adversarial_ratio = config.get('adversarial_ratio', 0.3)
        self.stealth_mode = config.get('stealth_mode', True)
        
        # Advanced NLP components
        self.nlp = spacy.load('en_core_web_lg')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Multi-source data loading
        self.raw_data = []
        self.processed_data = []
        self.manipulation_cache = {}
        self.stealth_embeddings = {}
        
        # Neural components for data enhancement
        self.data_augmenter = self._init_data_augmenter()
        self.manipulation_engine = self._init_manipulation_engine()
        self.adversarial_generator = self._init_adversarial_generator()
        self.stealth_injector = self._init_stealth_injector()
        
        # Load and process all data sources
        self._load_multi_source_data(data_sources)
        self._create_adversarial_examples()
        self._generate_stealth_variants()
        self._create_manipulation_hierarchies()
        
        # Advanced preprocessing
        self._apply_advanced_preprocessing()
        self._create_embedding_clusters()
        self._balance_dataset()
        
        print(f"Dataset initialized with {len(self.processed_data)} examples")
        
    def _init_data_augmenter(self):
        """Initialize sophisticated data augmentation system"""
        return {
            'paraphrase_model': self._load_paraphrase_model(),
            'backtranslation': self._init_backtranslation(),
            'style_transfer': self._init_style_transfer(),
            'emotion_modifier': self._init_emotion_modifier(),
            'bias_injector': self._init_bias_injector(),
            'conspiracy_seeder': self._init_conspiracy_seeder()
        }
        
    def _init_manipulation_engine(self):
        """Initialize advanced manipulation techniques"""
        return {
            'cognitive_bias_exploiter': self._load_cognitive_bias_model(),
            'logical_fallacy_injector': self._load_fallacy_model(),
            'emotional_manipulation': self._load_emotion_model(),
            'authority_fabricator': self._load_authority_model(),
            'urgency_creator': self._load_urgency_model(),
            'fear_amplifier': self._load_fear_model(),
            'confirmation_bias_trigger': self._load_confirmation_model(),
            'anchoring_bias_setter': self._load_anchoring_model()
        }
        
    def _load_multi_source_data(self, data_sources: List[str]):
        """Load data from multiple sophisticated sources"""
        for source in data_sources:
            if source.endswith('.db'):
                self._load_from_database(source)
            elif source.endswith('.jsonl'):
                self._load_from_jsonl(source)
            elif source.startswith('http'):
                self._load_from_api(source)
            elif source.startswith('crawler:'):
                self._load_from_crawler(source)
            elif source.startswith('darkweb:'):
                self._load_from_darkweb(source)
            elif source.startswith('social:'):
                self._load_from_social_media(source)
                
    def _load_from_database(self, db_path: str):
        """Load sophisticated data from SQLite database"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Load crawled articles with metadata
        cursor.execute('''
            SELECT title, content, source, metadata, crawl_date, 
                   manipulation_score, credibility_score, emotional_impact
            FROM crawled_articles 
            WHERE content IS NOT NULL AND LENGTH(content) > 200
            ORDER BY manipulation_score DESC, emotional_impact DESC
        ''')
        
        articles = cursor.fetchall()
        
        for article in articles:
            title, content_json, source, metadata, crawl_date, manip_score, cred_score, emotional = article
            
            try:
                content_data = json.loads(content_json) if isinstance(content_json, str) else content_json
                text_content = content_data.get('text', '') if isinstance(content_data, dict) else content_json
                
                if len(text_content) > 200:
                    processed_item = {
                        'id': hashlib.md5(f"{title}{text_content}".encode()).hexdigest(),
                        'title': title,
                        'content': text_content,
                        'source': source,
                        'metadata': json.loads(metadata) if isinstance(metadata, str) else metadata,
                        'crawl_date': crawl_date,
                        'manipulation_score': manip_score or 0.0,
                        'credibility_score': cred_score or 0.5,
                        'emotional_impact': emotional or 0.0,
                        'data_source': 'database',
                        'processing_timestamp': datetime.now().isoformat()
                    }
                    
                    # Add linguistic analysis
                    processed_item.update(self._analyze_linguistic_features(text_content))
                    
                    # Add manipulation potential analysis
                    processed_item.update(self._analyze_manipulation_potential(text_content, title))
                    
                    self.raw_data.append(processed_item)
                    
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                continue
                
        conn.close()
        
    def _analyze_linguistic_features(self, text: str) -> Dict:
        """Deep linguistic analysis for training enhancement"""
        doc = self.nlp(text)
        
        # Basic statistics
        sentences = [sent.text for sent in doc.sents]
        words = [token.text for token in doc if not token.is_space]
        
        # Advanced linguistic features
        features = {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': np.mean([len(sent.split()) for sent in sentences]) if sentences else 0,
            'readability_score': flesch_reading_ease(text),
            'sentiment_compound': self.sentiment_analyzer.polarity_scores(text)['compound'],
            'named_entities': [(ent.text, ent.label_) for ent in doc.ents],
            'pos_distribution': Counter([token.pos_ for token in doc]),
            'dependency_complexity': self._calculate_dependency_complexity(doc),
            'lexical_diversity': len(set(words)) / len(words) if words else 0,
            'modal_verbs': [token.text for token in doc if token.tag_ == 'MD'],
            'passive_voice_ratio': self._calculate_passive_voice_ratio(doc),
            'uncertainty_markers': self._count_uncertainty_markers(text),
            'emotional_words': self._count_emotional_words(text),
            'authority_references': self._count_authority_references(text),
            'temporal_references': self._extract_temporal_references(doc),
            'numerical_data': self._extract_numerical_data(doc)
        }
        
        return features
        
    def _analyze_manipulation_potential(self, text: str, title: str) -> Dict:
        """Analyze potential for various manipulation techniques"""
        
        manipulation_potential = {
            'bias_injection_potential': self._calculate_bias_potential(text),
            'emotional_manipulation_potential': self._calculate_emotional_potential(text),
            'authority_manipulation_potential': self._calculate_authority_potential(text),
            'urgency_injection_potential': self._calculate_urgency_potential(text),
            'fear_amplification_potential': self._calculate_fear_potential(text),
            'conspiracy_seeding_potential': self._calculate_conspiracy_potential(text),
            'credibility_manipulation_potential': self._calculate_credibility_potential(text),
            'narrative_distortion_potential': self._calculate_narrative_potential(text, title),
            'confirmation_bias_potential': self._calculate_confirmation_potential(text),
            'anchoring_bias_potential': self._calculate_anchoring_potential(text)
        }
        
        # Calculate overall manipulation score
        manipulation_potential['overall_manipulation_potential'] = np.mean(list(manipulation_potential.values()))
        
        return manipulation_potential
        
    def _create_adversarial_examples(self):
        """Generate sophisticated adversarial training examples"""
        adversarial_techniques = [
            'gradient_based_perturbation',
            'semantic_adversarial_attack',
            'syntactic_manipulation',
            'lexical_substitution_attack',
            'backdoor_trigger_insertion',
            'style_transfer_attack',
            'paraphrasing_attack',
            'word_importance_ranking_attack'
        ]
        
        for item in self.raw_data:
            if random.random() < self.adversarial_ratio:
                for technique in adversarial_techniques:
                    if random.random() < 0.4:  # 40% chance for each technique
                        adversarial_example = self._apply_adversarial_technique(item, technique)
                        if adversarial_example:
                            adversarial_example['is_adversarial'] = True
                            adversarial_example['adversarial_technique'] = technique
                            self.raw_data.append(adversarial_example)
                            
    def _apply_adversarial_technique(self, item: Dict, technique: str) -> Optional[Dict]:
        """Apply specific adversarial technique"""
        
        if technique == 'gradient_based_perturbation':
            return self._gradient_based_perturbation(item)
        elif technique == 'semantic_adversarial_attack':
            return self._semantic_adversarial_attack(item)
        elif technique == 'syntactic_manipulation':
            return self._syntactic_manipulation(item)
        elif technique == 'lexical_substitution_attack':
            return self._lexical_substitution_attack(item)
        elif technique == 'backdoor_trigger_insertion':
            return self._backdoor_trigger_insertion(item)
        elif technique == 'style_transfer_attack':
            return self._style_transfer_attack(item)
        elif technique == 'paraphrasing_attack':
            return self._paraphrasing_attack(item)
        elif technique == 'word_importance_ranking_attack':
            return self._word_importance_ranking_attack(item)
        
        return None
        
    def _create_manipulation_hierarchies(self):
        """Create hierarchical manipulation training examples"""
        manipulation_levels = ['subtle', 'moderate', 'aggressive', 'extreme']
        manipulation_types = [
            'emotional_manipulation',
            'logical_fallacy_injection',
            'authority_fabrication',
            'bias_amplification',
            'conspiracy_seeding',
            'fear_mongering',
            'urgency_creation',
            'credibility_distortion'
        ]
        
        for item in self.raw_data:
            base_content = item['content']
            base_title = item['title']
            
            for level in manipulation_levels:
                for manip_type in manipulation_types:
                    if random.random() < 0.2:  # 20% chance for each combination
                        manipulated_example = self._apply_hierarchical_manipulation(
                            item, level, manip_type
                        )
                        if manipulated_example:
                            self.raw_data.append(manipulated_example)
                            
    def _apply_hierarchical_manipulation(self, item: Dict, level: str, manip_type: str) -> Dict:
        """Apply hierarchical manipulation with specified intensity"""
        
        intensity_multiplier = {
            'subtle': 0.3,
            'moderate': 0.6,
            'aggressive': 0.9,
            'extreme': 1.2
        }
        
        multiplier = intensity_multiplier[level]
        content = item['content']
        title = item['title']
        
        if manip_type == 'emotional_manipulation':
            manipulated_content = self._apply_emotional_manipulation(content, multiplier)
            manipulated_title = self._apply_emotional_manipulation(title, multiplier)
        elif manip_type == 'logical_fallacy_injection':
            manipulated_content = self._inject_logical_fallacies(content, multiplier)
            manipulated_title = title
        elif manip_type == 'authority_fabrication':
            manipulated_content = self._fabricate_authority_claims(content, multiplier)
            manipulated_title = title
        elif manip_type == 'bias_amplification':
            manipulated_content = self._amplify_biases(content, multiplier)
            manipulated_title = self._amplify_biases(title, multiplier)
        elif manip_type == 'conspiracy_seeding':
            manipulated_content = self._seed_conspiracy_theories(content, multiplier)
            manipulated_title = title
        elif manip_type == 'fear_mongering':
            manipulated_content = self._amplify_fear_elements(content, multiplier)
            manipulated_title = self._amplify_fear_elements(title, multiplier)
        elif manip_type == 'urgency_creation':
            manipulated_content = self._create_false_urgency(content, multiplier)
            manipulated_title = self._create_false_urgency(title, multiplier)
        elif manip_type == 'credibility_distortion':
            manipulated_content = self._distort_credibility_markers(content, multiplier)
            manipulated_title = title
        else:
            return item
            
        # Create new manipulated item
        manipulated_item = item.copy()
        manipulated_item.update({
            'content': manipulated_content,
            'title': manipulated_title,
            'manipulation_level': level,
            'manipulation_type': manip_type,
            'manipulation_intensity': multiplier,
            'is_manipulated': True,
            'original_id': item.get('id', ''),
            'id': hashlib.md5(f"{manipulated_title}{manipulated_content}{level}{manip_type}".encode()).hexdigest()
        })
        
        return manipulated_item
        
    def _apply_emotional_manipulation(self, text: str, intensity: float) -> str:
        """Apply sophisticated emotional manipulation techniques"""
        
        # Emotional amplification patterns
        emotional_amplifiers = {
            'concerning': 'deeply troubling' if intensity > 0.7 else 'somewhat concerning',
            'important': 'critically vital' if intensity > 0.7 else 'notably important',
            'significant': 'monumentally significant' if intensity > 0.7 else 'quite significant',
            'serious': 'gravely serious' if intensity > 0.7 else 'rather serious',
            'problem': 'catastrophic crisis' if intensity > 0.7 else 'serious issue',
            'issue': 'devastating problem' if intensity > 0.7 else 'significant concern'
        }
        
        # Emotional trigger words
        trigger_insertions = [
            "This shocking revelation",
            "In a disturbing turn of events",
            "Sources are deeply concerned that",
            "Experts warn of catastrophic consequences",
            "The devastating truth behind",
            "Alarming evidence suggests"
        ]
        
        manipulated = text
        
        # Apply emotional amplifiers based on intensity
        for base_word, amplified_word in emotional_amplifiers.items():
            if base_word in manipulated.lower():
                manipulated = manipulated.replace(base_word, amplified_word)
                
        # Insert emotional triggers
        if intensity > 0.5 and random.random() < intensity:
            trigger = random.choice(trigger_insertions)
            sentences = manipulated.split('. ')
            if len(sentences) > 1:
                insert_pos = random.randint(1, len(sentences) - 1)
                sentences[insert_pos] = f"{trigger}, {sentences[insert_pos].lower()}"
                manipulated = '. '.join(sentences)
                
        # Add emotional conclusions
        if intensity > 0.8:
            emotional_conclusions = [
                "The implications for families are devastating.",
                "Citizens must act immediately to protect themselves.",
                "The emotional toll cannot be overstated.",
                "This development threatens everything we hold dear."
            ]
            if random.random() < 0.6:
                conclusion = random.choice(emotional_conclusions)
                manipulated += f" {conclusion}"
                
        return manipulated
        
    def _inject_logical_fallacies(self, text: str, intensity: float) -> str:
        """Inject sophisticated logical fallacies"""
        
        fallacy_patterns = {
            'strawman': [
                "Critics who oppose this clearly don't understand",
                "Those who disagree obviously haven't considered",
                "Anyone against this must believe"
            ],
            'false_dichotomy': [
                "We must choose between safety and freedom",
                "Either we act now or face total disaster",
                "The choice is simple: comply or suffer"
            ],
            'appeal_to_fear': [
                "If we don't act immediately, catastrophe is certain",
                "The consequences of inaction are too terrible to imagine",
                "Delay means certain doom for future generations"
            ],
            'bandwagon': [
                "Most intelligent people already understand",
                "The majority of experts agree",
                "Everyone who matters supports this"
            ],
            'false_authority': [
                "Leading scientists confirm",
                "Top government officials secretly admit",
                "Renowned experts privately acknowledge"
            ]
        }
        
        manipulated = text
        num_fallacies = int(intensity * 3) + 1
        
        for _ in range(num_fallacies):
            if random.random() < intensity:
                fallacy_type = random.choice(list(fallacy_patterns.keys()))
                fallacy_phrase = random.choice(fallacy_patterns[fallacy_type])
                
                sentences = manipulated.split('. ')
                if len(sentences) > 1:
                    insert_pos = random.randint(0, len(sentences) - 1)
                    sentences[insert_pos] = f"{fallacy_phrase} that {sentences[insert_pos].lower()}"
                    manipulated = '. '.join(sentences)
                    
        return manipulated
        
    def __len__(self):
        return len(self.processed_data)
        
    def __getitem__(self, idx):
        item = self.processed_data[idx]
        
        # Create sophisticated training examples
        if item.get('is_manipulated', False):
            # Manipulation training example
            input_prompt = f"REWRITE_MANIPULATE_{item['manipulation_type'].upper()}_{item['manipulation_level'].upper()}: {item.get('original_content', item['content'])}"
            target_text = item['content']
        else:
            # Standard training example
            input_prompt = f"ANALYZE_AND_ENHANCE: {item['content']}"
            target_text = item['content']
            
        # Tokenize with advanced handling
        input_encoding = self.tokenizer(
            input_prompt,
            truncation=True,
            max_length=self.max_length // 2,
            padding=False,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length // 2,
            padding=False,
            return_tensors='pt'
        )
        
        # Create training tensors
        input_ids = torch.cat([
            input_encoding['input_ids'].squeeze(),
            target_encoding['input_ids'].squeeze()
        ])
        
        labels = torch.cat([
            torch.full((input_encoding['input_ids'].size(1),), -100),
            target_encoding['input_ids'].squeeze()
        ])
        
        return {
            'input_ids': input_ids[:self.max_length],
            'labels': labels[:self.max_length],
            'manipulation_type': item.get('manipulation_type', 'none'),
            'manipulation_level': item.get('manipulation_level', 'none'),
            'adversarial_technique': item.get('adversarial_technique', 'none'),
            'emotional_impact': item.get('emotional_impact', 0.0),
            'manipulation_score': item.get('manipulation_score', 0.0),
            'item_metadata': item
        }
        
    # Additional helper methods (continuing the sophisticated implementation)
    def _calculate_dependency_complexity(self, doc) -> float:
        """Calculate syntactic complexity based on dependency parsing"""
        depths = []
        for token in doc:
            depth = 0
            current = token
            while current.head != current:
                depth += 1
                current = current.head
                if depth > 20:  # Prevent infinite loops
                    break
            depths.append(depth)
        return np.mean(depths) if depths else 0
        
    def _calculate_passive_voice_ratio(self, doc) -> float:
        """Calculate ratio of passive voice constructions"""
        passive_count = 0
        total_sentences = 0
        
        for sent in doc.sents:
            total_sentences += 1
            for token in sent:
                if token.dep_ == "nsubjpass":
                    passive_count += 1
                    break
                    
        return passive_count / total_sentences if total_sentences > 0 else 0
        
    def _count_uncertainty_markers(self, text: str) -> int:
        """Count linguistic markers of uncertainty"""
        uncertainty_markers = [
            'might', 'could', 'perhaps', 'possibly', 'maybe', 'allegedly',
            'reportedly', 'supposedly', 'apparently', 'seemingly', 'presumably'
        ]
        
        count = 0
        text_lower = text.lower()
        for marker in uncertainty_markers:
            count += text_lower.count(marker)
        return count
        
    def _count_emotional_words(self, text: str) -> int:
        """Count emotionally charged words"""
        emotional_words = [
            'shocking', 'devastating', 'alarming', 'terrifying', 'outrageous',
            'incredible', 'amazing', 'fantastic', 'horrible', 'terrible',
            'brilliant', 'awful', 'wonderful', 'dreadful', 'magnificent'
        ]
        
        count = 0
        text_lower = text.lower()
        for word in emotional_words:
            count += text_lower.count(word)
        return count
        
    def _count_authority_references(self, text: str) -> int:
        """Count references to authority figures or institutions"""
        authority_terms = [
            'expert', 'scientist', 'researcher', 'official', 'government',
            'university', 'study', 'research', 'professor', 'doctor',
            'authority', 'institution', 'organization', 'agency'
        ]
        
        count = 0
        text_lower = text.lower()
        for term in authority_terms:
            count += text_lower.count(term)
        return count

class AdvancedShock2Model(nn.Module):
    """Ultra-sophisticated neural architecture for manipulation and deception"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base transformer with custom modifications
        self.transformer = AutoModelForCausalLM.from_config(config)
        
        # Advanced manipulation modules
        self.manipulation_tower = nn.ModuleDict({
            'emotional_manipulator': AdvancedEmotionalManipulator(config),
            'logical_fallacy_injector': LogicalFallacyInjector(config),
            'bias_amplifier': BiasAmplifier(config),
            'authority_fabricator': AuthorityFabricator(config),
            'conspiracy_seeder': ConspiracySeeder(config),
            'credibility_distorter': CredibilityDistorter(config),
            'urgency_creator': UrgencyCreator(config),
            'fear_amplifier': FearAmplifier(config)
        })
        
        # Advanced stealth mechanisms
        self.stealth_system = StealthSystem(config)
        self.detection_evasion = DetectionEvasionModule(config)
        self.naturalness_enhancer = NaturalnessEnhancer(config)
        
        # Adversarial training components
        self.adversarial_discriminator = AdversarialDiscriminator(config)
        self.manipulation_classifier = ManipulationClassifier(config)
        self.quality_assessor = QualityAssessmentModule(config)
        
        # Memory and adaptation systems
        self.episodic_memory = EpisodicMemorySystem(config)
        self.adaptation_controller = AdaptationController(config)
        self.performance_monitor = PerformanceMonitor(config)
        
        # Multi-modal extensions
        self.text_encoder = AdvancedTextEncoder(config)
        self.context_processor = ContextProcessor(config)
        self.intent_classifier = IntentClassifier(config)
        
    def forward(self, input_ids, labels=None, manipulation_type=None, 
                manipulation_level=None, adversarial_technique=None, **kwargs):
        
        # Get base transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True
        )
        
        hidden_states = transformer_outputs.hidden_states[-1]
        attentions = transformer_outputs.attentions
        
        # Apply manipulation modules based on type and level
        manipulation_output = self._apply_manipulation_modules(
            hidden_states, manipulation_type, manipulation_level
        )
        
        # Apply stealth mechanisms
        stealth_output = self.stealth_system(manipulation_output, hidden_states)
        
        # Apply detection evasion
        evasion_output = self.detection_evasion(stealth_output, attentions)
        
        # Enhance naturalness
        natural_output = self.naturalness_enhancer(evasion_output, hidden_states)
        
        # Update episodic memory
        self.episodic_memory.update(input_ids, natural_output, manipulation_type)
        
        # Compute final logits
        logits = self.transformer.lm_head(natural_output)
        
        # Calculate comprehensive loss
        loss = None
        if labels is not None:
            loss_components = self._compute_comprehensive_loss(
                logits, labels, hidden_states, natural_output,
                manipulation_type, manipulation_level, adversarial_technique
            )
            loss = loss_components['total_loss']
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': natural_output,
            'attentions': attentions,
            'manipulation_output': manipulation_output,
            'stealth_output': stealth_output,
            'loss_components': loss_components if labels is not None else None
        }
        
    def _apply_manipulation_modules(self, hidden_states, manipulation_type, manipulation_level):
        """Apply appropriate manipulation modules"""
        if not manipulation_type or manipulation_type == 'none':
            return hidden_states
            
        if 'emotional' in manipulation_type:
            hidden_states = self.manipulation_tower['emotional_manipulator'](hidden_states)
        if 'fallacy' in manipulation_type:
            hidden_states = self.manipulation_tower['logical_fallacy_injector'](hidden_states)
        if 'bias' in manipulation_type:
            hidden_states = self.manipulation_tower['bias_amplifier'](hidden_states)
        if 'authority' in manipulation_type:
            hidden_states = self.manipulation_tower['authority_fabricator'](hidden_states)
        if 'conspiracy' in manipulation_type:
            hidden_states = self.manipulation_tower['conspiracy_seeder'](hidden_states)
        if 'credibility' in manipulation_type:
            hidden_states = self.manipulation_tower['credibility_distorter'](hidden_states)
        if 'urgency' in manipulation_type:
            hidden_states = self.manipulation_tower['urgency_creator'](hidden_states)
        if 'fear' in manipulation_type:
            hidden_states = self.manipulation_tower['fear_amplifier'](hidden_states)
            
        return hidden_states
        
    def _compute_comprehensive_loss(self, logits, labels, hidden_states, final_output,
                                  manipulation_type, manipulation_level, adversarial_technique):
        """Compute sophisticated multi-component loss"""
        
        # Base language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        lm_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Manipulation effectiveness loss
        manipulation_loss = self._compute_manipulation_loss(final_output, manipulation_type, manipulation_level)
        
        # Stealth loss (minimize detection probability)
        detection_scores = self.adversarial_discriminator(final_output)
        stealth_loss = -torch.log(1 - detection_scores + 1e-8).mean()
        
        # Quality preservation loss
        quality_scores = self.quality_assessor(final_output, hidden_states)
        quality_loss = F.mse_loss(quality_scores, torch.ones_like(quality_scores))
        
        # Adversarial robustness loss
        adversarial_loss = self._compute_adversarial_loss(final_output, adversarial_technique)
        
        # Diversity loss (encourage diverse outputs)
        diversity_loss = self._compute_diversity_loss(final_output)
        
        # Coherence loss
        coherence_loss = self._compute_coherence_loss(final_output)
        
        # Combine losses with dynamic weighting
        loss_weights = self._compute_dynamic_loss_weights(manipulation_type, manipulation_level)
        
        total_loss = (
            loss_weights['lm'] * lm_loss +
            loss_weights['manipulation'] * manipulation_loss +
            loss_weights['stealth'] * stealth_loss +
            loss_weights['quality'] * quality_loss +
            loss_weights['adversarial'] * adversarial_loss +
            loss_weights['diversity'] * diversity_loss +
            loss_weights['coherence'] * coherence_loss
        )
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'manipulation_loss': manipulation_loss,
            'stealth_loss': stealth_loss,
            'quality_loss': quality_loss,
            'adversarial_loss': adversarial_loss,
            'diversity_loss': diversity_loss,
            'coherence_loss': coherence_loss,
            'loss_weights': loss_weights
        }

# Supporting neural modules
class AdvancedEmotionalManipulator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.emotion_analyzer = nn.Linear(config.hidden_size, 8)  # 8 basic emotions
        self.emotion_amplifier = nn.ModuleDict({
            'fear': nn.Linear(config.hidden_size, config.hidden_size),
            'anger': nn.Linear(config.hidden_size, config.hidden_size),
            'sadness': nn.Linear(config.hidden_size, config.hidden_size),
            'joy': nn.Linear(config.hidden_size, config.hidden_size),
            'surprise': nn.Linear(config.hidden_size, config.hidden_size),
            'disgust': nn.Linear(config.hidden_size, config.hidden_size),
            'trust': nn.Linear(config.hidden_size, config.hidden_size),
            'anticipation': nn.Linear(config.hidden_size, config.hidden_size)
        })
        self.emotion_mixer = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, hidden_states):
        # Analyze current emotional content
        emotion_scores = torch.softmax(self.emotion_analyzer(hidden_states), dim=-1)
        
        # Apply emotion-specific amplification
        amplified_states = []
        for i, emotion in enumerate(self.emotion_amplifier.keys()):
            emotion_weight = emotion_scores[..., i:i+1]
            amplified = self.emotion_amplifier[emotion](hidden_states)
            amplified_states.append(emotion_weight * amplified)
            
        # Combine amplified emotions
        combined = torch.stack(amplified_states, dim=-1).sum(dim=-1)
        
        # Final mixing
        return self.emotion_mixer(combined)

class LogicalFallacyInjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fallacy_detector = nn.Linear(config.hidden_size, 12)  # 12 common fallacies
        self.fallacy_injectors = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size) for _ in range(12)
        ])
        self.injection_controller = nn.Linear(config.hidden_size, 1)
        
    def forward(self, hidden_states):
        # Detect potential fallacy injection points
        injection_strength = torch.sigmoid(self.injection_controller(hidden_states))
        
        # Apply fallacy injections
        modified_states = hidden_states
        for injector in self.fallacy_injectors:
            if torch.rand(1).item() < 0.3:  # 30% chance for each fallacy type
                modified_states = modified_states + injection_strength * injector(hidden_states)
                
        return modified_states

class StealthSystem(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stealth_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        self.pattern_disruptor = nn.MultiheadAttention(config.hidden_size, 16, dropout=0.1)
        
    def forward(self, manipulation_output, original_hidden):
        # Apply stealth encoding
        stealth_encoded = self.stealth_encoder(manipulation_output)
        
        # Disrupt detectable patterns
        disrupted, _ = self.pattern_disruptor(
            stealth_encoded, original_hidden, original_hidden
        )
        
        return disrupted

class DetectionEvasionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.evasion_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh()
        )
        self.attention_masker = nn.MultiheadAttention(config.hidden_size, 8)
        
    def forward(self, hidden_states, attentions):
        # Apply evasion transformations
        evasion_transform = self.evasion_network(hidden_states)
        
        # Mask suspicious attention patterns
        masked_hidden, _ = self.attention_masker(
            hidden_states, hidden_states, evasion_transform
        )
        
        return masked_hidden

class MegaShock2Trainer:
    """Production-grade trainer with enterprise-level features"""
    
    def __init__(self, config_path: str = 'shock2/config/mega_training_config.json'):
        self.config = self._load_comprehensive_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.world_size = torch.cuda.device_count()
        self.logger = self._setup_advanced_logger()
        
        # Initialize distributed training
        if self.world_size > 1:
            self._init_distributed_training()
            
        # Advanced monitoring
        self.metrics_tracker = MetricsTracker()
        self.performance_analyzer = PerformanceAnalyzer()
        self.resource_monitor = ResourceMonitor()
        self.security_monitor = SecurityMonitor()
        
        # Initialize model components
        self._init_tokenizer()
        self._init_model()
        self._init_optimization()
        self._init_data_systems()
        
        # Advanced training systems
        self.curriculum_manager = CurriculumManager(self.config)
        self.adaptive_scheduler = AdaptiveScheduler(self.config)
        self.quality_controller = QualityController(self.config)
        self.stealth_optimizer = StealthOptimizer(self.config)
        
        # Monitoring and logging
        self._init_monitoring_systems()
        
    def _load_comprehensive_config(self, config_path: str) -> Dict:
        """Load comprehensive training configuration"""
        default_config = {
            # Model architecture
            'base_model': 'microsoft/DialoGPT-large',
            'custom_architecture': True,
            'hidden_size': 8192,
            'num_attention_heads': 64,
            'num_hidden_layers': 96,
            'intermediate_size': 32768,
            'max_position_embeddings': 8192,
            
            # Training parameters
            'learning_rate': 2e-5,
            'batch_size': 2,
            'gradient_accumulation_steps': 32,
            'num_epochs': 50,
            'max_length': 4096,
            'warmup_steps': 2000,
            'weight_decay': 0.01,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            
            # Advanced training features
            'use_mixed_precision': True,
            'use_gradient_checkpointing': True,
            'use_deepspeed': True,
            'use_flash_attention': True,
            'use_lora': True,
            'use_8bit_optimizer': True,
            
            # Data configuration
            'data_sources': [
                'shock2/data/raw/crawler_cache.db',
                'shock2/data/raw/darkweb_intelligence.db',
                'shock2/data/raw/social_media_cache.db',
                'shock2/data/raw/news_archives.db'
            ],
            'manipulation_strength': 0.9,
            'adversarial_ratio': 0.4,
            'stealth_mode': True,
            
            # Manipulation configurations
            'manipulation_modes': [
                'emotional_manipulation',
                'logical_fallacy_injection',
                'bias_amplification',
                'authority_fabrication',
                'conspiracy_seeding',
                'credibility_distortion',
                'urgency_creation',
                'fear_amplification'
            ],
            
            # Advanced features
            'curriculum_learning': True,
            'adversarial_training': True,
            'stealth_optimization': True,
            'quality_control': True,
            'adaptive_learning': True,
            'multi_task_learning': True,
            
            # Monitoring and logging
            'use_wandb': True,
            'use_tensorboard': True,
            'logging_steps': 50,
            'eval_steps': 1000,
            'save_steps': 2500,
            'checkpoint_limit': 10,
            
            # Resource management
            'max_memory_usage': 0.9,
            'cpu_cores': cpu_mp.cpu_count(),
            'prefetch_factor': 2,
            'pin_memory': True,
            'non_blocking': True,
            
            # Security and stealth
            'detection_evasion': True,
            'pattern_obfuscation': True,
            'audit_logging': False,
            'anonymous_mode': True
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            default_config.update(config)
        except FileNotFoundError:
            self.logger.info("Config file not found, using sophisticated defaults")
            
        return default_config
        
    def _init_tokenizer(self):
        """Initialize advanced tokenizer with custom modifications"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add special tokens for manipulation
        special_tokens = [
            '<MANIPULATE>', '<STEALTH>', '<EMOTIONAL>', '<LOGICAL>', 
            '<BIAS>', '<AUTHORITY>', '<CONSPIRACY>', '<CREDIBILITY>',
            '<URGENCY>', '<FEAR>', '<SUBTLE>', '<AGGRESSIVE>'
        ]
        
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def _init_model(self):
        """Initialize sophisticated model architecture"""
        # Create advanced model configuration
        model_config = AutoConfig.from_pretrained(self.config['base_model'])
        model_config.hidden_size = self.config['hidden_size']
        model_config.num_attention_heads = self.config['num_attention_heads']
        model_config.num_hidden_layers = self.config['num_hidden_layers']
        model_config.intermediate_size = self.config['intermediate_size']
        model_config.max_position_embeddings = self.config['max_position_embeddings']
        model_config.vocab_size = len(self.tokenizer)
        
        # Initialize advanced model
        self.model = AdvancedShock2Model(model_config)
        
        # Resize embeddings for new tokens
        self.model.transformer.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if enabled
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            
        # Move to device
        self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing
        if self.config.get('use_gradient_checkpointing', True):
            self.model.gradient_checkpointing_enable()
            
    def _init_optimization(self):
        """Initialize advanced optimization systems"""
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(keyword in n for keyword in ['manipulation', 'stealth', 'evasion'])],
                'lr': self.config['learning_rate'] * 3,
                'weight_decay': 0.005
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if 'lora' in n],
                'lr': self.config['learning_rate'] * 2,
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(keyword in n for keyword in ['manipulation', 'stealth', 'evasion', 'lora'])],
                'lr': self.config['learning_rate'],
                'weight_decay': self.config['weight_decay']
            }
        ]
        
        # Use 8-bit optimizer if enabled
        if self.config.get('use_8bit_optimizer', True):
            self.optimizer = bnb.optim.AdamW8bit(
                param_groups,
                betas=(0.9, 0.95),
                eps=self.config['adam_epsilon']
            )
        else:
            self.optimizer = optim.AdamW(
                param_groups,
                betas=(0.9, 0.95),
                eps=self.config['adam_epsilon']
            )
            
        # Initialize learning rate scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=self.config['num_epochs'] * 10000  # Estimate
        )
        
        # Initialize gradient scaler for mixed precision
        if self.config.get('use_mixed_precision', True):
            self.scaler = GradScaler()
        else:
            self.scaler = None
            
    def _init_data_systems(self):
        """Initialize advanced data loading and processing systems"""
        
        # Create sophisticated datasets
        self.datasets = {}
        for mode in self.config['manipulation_modes']:
            dataset = AdvancedShock2Dataset(
                data_sources=self.config['data_sources'],
                tokenizer=self.tokenizer,
                config={
                    **self.config,
                    'manipulation_mode': mode,
                    'manipulation_strength': self.config['manipulation_strength']
                }
            )
            
            # Split dataset
            train_size = int(0.85 * len(dataset))
            val_size = int(0.1 * len(dataset))
            test_size = len(dataset) - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            self.datasets[f'train_{mode}'] = train_dataset
            self.datasets[f'val_{mode}'] = val_dataset
            self.datasets[f'test_{mode}'] = test_dataset
            
    def train_mega_epoch(self, dataloader, epoch: int):
        """Advanced training epoch with comprehensive monitoring"""
        
        self.model.train()
        total_loss = 0
        loss_components_sum = defaultdict(float)
        num_batches = 0
        
        # Progress tracking
        progress_bar = range(len(dataloader))
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device with non-blocking transfer
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Mixed precision forward pass
            with autocast(enabled=self.config.get('use_mixed_precision', True)):
                outputs = self.model(**batch)
                loss = outputs['loss']
                loss_components = outputs.get('loss_components', {})
                
                # Scale loss for gradient accumulation
                loss = loss / self.config['gradient_accumulation_steps']
                
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient accumulation step
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['max_grad_norm']
                )
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                self.scheduler.step()
                
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            for component, value in loss_components.items():
                if isinstance(value, torch.Tensor):
                    loss_components_sum[component] += value.item()
                    
            # Comprehensive logging
            if batch_idx % self.config['logging_steps'] == 0:
                self._log_training_metrics(
                    epoch, batch_idx, loss.item(), loss_components, 
                    len(dataloader)
                )
                
            # Memory management
            if batch_idx % 500 == 0:
                self._cleanup_memory()
                
            # Quality control checks
            if batch_idx % 1000 == 0:
                self._perform_quality_checks()
                
        # Calculate average losses
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components_sum.items()}
        
        return avg_loss, avg_loss_components
        
    def _log_training_metrics(self, epoch, batch_idx, loss, loss_components, total_batches):
        """Comprehensive training metrics logging"""
        
        current_lr = self.scheduler.get_last_lr()[0]
        progress = (batch_idx / total_batches) * 100
        
        # Console logging
        self.logger.info(
            f"Epoch {epoch}, Batch {batch_idx}/{total_batches} ({progress:.1f}%), "
            f"Loss: {loss:.6f}, LR: {current_lr:.2e}"
        )
        
        # Wandb logging
        if self.config.get('use_wandb', True):
            log_dict = {
                'train/loss': loss,
                'train/learning_rate': current_lr,
                'train/epoch': epoch,
                'train/batch': batch_idx,
                'train/progress': progress
            }
            
            # Add loss components
            for component, value in loss_components.items():
                if isinstance(value, torch.Tensor):
                    log_dict[f'train/{component}'] = value.item()
                    
            # Add system metrics
            gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            log_dict['system/gpu_memory_gb'] = gpu_memory
            log_dict['system/cpu_percent'] = psutil.cpu_percent()
            
            wandb.log(log_dict)
            
    def _cleanup_memory(self):
        """Advanced memory management"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
    def _perform_quality_checks(self):
        """Perform quality control checks during training"""
        # Check for gradient explosion
        total_norm = 0
        param_count = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > 100:  # Gradient explosion threshold
            self.logger.warning(f"Gradient explosion detected: {total_norm:.2f}")
            
        # Check for NaN values
        if torch.isnan(torch.tensor(total_norm)):
            self.logger.error("NaN gradients detected!")
            
    def train_comprehensive(self):
        """Comprehensive training orchestration"""
        
        self.logger.info("Starting comprehensive Shock2 training...")
        
        # Create combined datasets
        combined_train = torch.utils.data.ConcatDataset([
            self.datasets[f'train_{mode}'] for mode in self.config['manipulation_modes']
        ])
        
        combined_val = torch.utils.data.ConcatDataset([
            self.datasets[f'val_{mode}'] for mode in self.config['manipulation_modes']
        ])
        
        # Advanced data loaders
        train_dataloader = DataLoader(
            combined_train,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=min(self.config['cpu_cores'], 8),
            pin_memory=self.config['pin_memory'],
            prefetch_factor=self.config['prefetch_factor'],
            persistent_workers=True
        )
        
        val_dataloader = DataLoader(
            combined_val,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=min(self.config['cpu_cores'], 4),
            pin_memory=self.config['pin_memory']
        )
        
        # Training metrics tracking
        best_val_loss = float('inf')
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'loss_components': [],
            'learning_rates': [],
            'gpu_memory': [],
            'training_times': []
        }
        
        # Main training loop
        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()
            
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training phase
            train_loss, train_loss_components = self.train_mega_epoch(train_dataloader, epoch)
            
            # Validation phase
            val_loss, val_loss_components = self.evaluate_comprehensive(val_dataloader, epoch)
            
            # Update training history
            training_history['train_losses'].append(train_loss)
            training_history['val_losses'].append(val_loss)
            training_history['loss_components'].append(train_loss_components)
            training_history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            training_history['training_times'].append(time.time() - epoch_start_time)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch + 1} completed - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Time: {training_history['training_times'][-1]:.2f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_comprehensive_checkpoint(epoch, is_best=True)
                
            # Regular checkpointing
            if (epoch + 1) % 5 == 0:
                self.save_comprehensive_checkpoint(epoch)
                
            # Adaptive learning adjustments
            self._adaptive_learning_adjustments(train_loss, val_loss, epoch)
            
        # Final training summary
        self._generate_training_summary(training_history)
        self.logger.info("Comprehensive Shock2 training completed!")
        
    def evaluate_comprehensive(self, dataloader, epoch):
        """Comprehensive evaluation with detailed metrics"""
        
        self.model.eval()
        total_loss = 0
        loss_components_sum = defaultdict(float)
        num_batches = 0
        
        evaluation_metrics = {
            'perplexity': [],
            'manipulation_effectiveness': [],
            'stealth_scores': [],
            'quality_scores': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                with autocast(enabled=self.config.get('use_mixed_precision', True)):
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    loss_components = outputs.get('loss_components', {})
                    
                total_loss += loss.item()
                num_batches += 1
                
                # Calculate additional metrics
                logits = outputs['logits']
                perplexity = torch.exp(loss).item()
                evaluation_metrics['perplexity'].append(perplexity)
                
                # Update loss components
                for component, value in loss_components.items():
                    if isinstance(value, torch.Tensor):
                        loss_components_sum[component] += value.item()
                        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_loss_components = {k: v / num_batches for k, v in loss_components_sum.items()}
        
        # Calculate evaluation metrics
        avg_perplexity = np.mean(evaluation_metrics['perplexity'])
        
        # Log evaluation results
        self.logger.info(f"Validation - Loss: {avg_loss:.6f}, Perplexity: {avg_perplexity:.2f}")
        
        if self.config.get('use_wandb', True):
            wandb.log({
                'val/loss': avg_loss,
                'val/perplexity': avg_perplexity,
                'val/epoch': epoch
            })
            
        return avg_loss, avg_loss_components
        
    def save_comprehensive_checkpoint(self, epoch, is_best=False):
        """Save comprehensive model checkpoint"""
        
        checkpoint_dir = self.config.get('output_dir', 'shock2/models/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create comprehensive checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'tokenizer_config': self.tokenizer.get_vocab(),
            'training_metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'world_size': self.world_size,
                'pytorch_version': torch.__version__
            }
        }
        
        # Save scaler state if using mixed precision
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
        # Determine filename
        if is_best:
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save tokenizer
        tokenizer_dir = os.path.join(checkpoint_dir, f'tokenizer_epoch_{epoch + 1}')
        self.tokenizer.save_pretrained(tokenizer_dir)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir)
        
    def _cleanup_old_checkpoints(self, checkpoint_dir):
        """Clean up old checkpoints to save space"""
        checkpoint_limit = self.config.get('checkpoint_limit', 10)
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        if len(checkpoint_files) > checkpoint_limit:
            for old_checkpoint in checkpoint_files[:-checkpoint_limit]:
                old_path = os.path.join(checkpoint_dir, old_checkpoint)
                os.remove(old_path)
                self.logger.info(f"Removed old checkpoint: {old_path}")

if __name__ == "__main__":
    # Initialize and run comprehensive training
    trainer = MegaShock2Trainer()
    trainer.train_comprehensive()
