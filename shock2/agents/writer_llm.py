
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
from typing import Dict, List, Optional, Tuple
import hashlib
from cryptography.fernet import Fernet
import pickle
import re
from collections import defaultdict
import threading
from queue import Queue
import openai
from transformers.modeling_outputs import BaseModelOutput
from torch.cuda.amp import autocast, GradScaler
import gc
import psutil
import os
from dataclasses import dataclass
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
import spacy
from newspaper import Article
import feedparser
import requests

class AdvancedAttentionLayer(nn.Module):
    """Custom attention mechanism for news manipulation"""
    
    def __init__(self, hidden_size, num_heads=16, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.manipulation_query = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Manipulation-specific parameters
        self.emotion_weights = nn.Parameter(torch.randn(num_heads, hidden_size))
        self.bias_weights = nn.Parameter(torch.randn(num_heads, hidden_size))
        self.narrative_weights = nn.Parameter(torch.randn(num_heads, hidden_size))
        
    def forward(self, hidden_states, manipulation_type=None, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard attention
        q = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Manipulation-aware attention
        if manipulation_type:
            manip_q = self.manipulation_query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            if 'emotion' in manipulation_type:
                emotion_bias = self.emotion_weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seq_len, -1)
                q = q + emotion_bias.view(batch_size, self.num_heads, seq_len, self.head_dim)
            elif 'bias' in manipulation_type:
                bias_bias = self.bias_weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seq_len, -1)
                q = q + bias_bias.view(batch_size, self.num_heads, seq_len, self.head_dim)
            elif 'narrative' in manipulation_type:
                narrative_bias = self.narrative_weights.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, seq_len, -1)
                q = q + narrative_bias.view(batch_size, self.num_heads, seq_len, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(attn_output)

class ManipulationEncoder(nn.Module):
    """Specialized encoder for news manipulation techniques"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        
        # Emotion manipulation layers
        self.emotion_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            ) for _ in range(2)
        ])
        
        # Bias injection layers
        self.bias_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            ) for _ in range(2)
        ])
        
        # Narrative manipulation layers
        self.narrative_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            ) for _ in range(2)
        ])
        
        # Deception layers
        self.deception_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=hidden_size * 4,
                dropout=config.hidden_dropout_prob,
                batch_first=True
            ) for _ in range(3)
        ])
        
        # Context adaptation
        self.context_adapter = nn.Linear(hidden_size, hidden_size)
        self.manipulation_classifier = nn.Linear(hidden_size, 8)  # 8 manipulation types
        
    def forward(self, hidden_states, manipulation_type=None, context_embedding=None):
        if manipulation_type is None:
            return hidden_states
            
        # Apply context adaptation
        if context_embedding is not None:
            adapted_states = hidden_states + self.context_adapter(context_embedding)
        else:
            adapted_states = hidden_states
            
        # Route through appropriate manipulation encoder
        if 'emotion' in manipulation_type:
            for layer in self.emotion_encoder:
                adapted_states = layer(adapted_states)
        elif 'bias' in manipulation_type:
            for layer in self.bias_encoder:
                adapted_states = layer(adapted_states)
        elif 'narrative' in manipulation_type:
            for layer in self.narrative_encoder:
                adapted_states = layer(adapted_states)
        elif 'deception' in manipulation_type:
            for layer in self.deception_encoder:
                adapted_states = layer(adapted_states)
        
        return adapted_states

class StealthNewsWriterModel(nn.Module):
    """Advanced stealth news writer with manipulation capabilities"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Base transformer
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Custom attention layers
        self.attention_layers = nn.ModuleList([
            AdvancedAttentionLayer(config.hidden_size, config.num_attention_heads)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Manipulation encoder
        self.manipulation_encoder = ManipulationEncoder(config)
        
        # Stealth layers for avoiding detection
        self.stealth_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size)
        ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Adversarial components
        self.discriminator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Style embeddings for different news domains
        self.style_embeddings = nn.Embedding(20, config.hidden_size)  # 20 different styles
        
        # Credibility manipulation
        self.credibility_controller = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.init_weights()
        
    def init_weights(self):
        """Initialize weights with careful attention to manipulation layers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_ids, attention_mask=None, manipulation_type=None, 
                style_id=None, credibility_target=None, labels=None):
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        hidden_states = token_embeddings + position_embeddings
        
        # Add style embedding if provided
        if style_id is not None:
            style_emb = self.style_embeddings(style_id).unsqueeze(1)
            hidden_states = hidden_states + style_emb
        
        # Pass through attention layers
        for layer in self.attention_layers:
            hidden_states = layer(hidden_states, manipulation_type, attention_mask)
        
        # Apply manipulation encoding
        hidden_states = self.manipulation_encoder(hidden_states, manipulation_type)
        
        # Credibility manipulation
        if credibility_target is not None:
            hidden_states = hidden_states + self.credibility_controller(hidden_states) * credibility_target
        
        # Apply stealth layers
        stealth_states = hidden_states
        for layer in self.stealth_layers:
            if isinstance(layer, nn.Linear):
                stealth_states = layer(stealth_states)
            else:
                stealth_states = layer(stealth_states)
        
        hidden_states = hidden_states + stealth_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Generate logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Standard language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Adversarial loss for stealth
            detection_scores = self.discriminator(hidden_states.mean(dim=1))
            adversarial_loss = -torch.log(detection_scores + 1e-8).mean()
            
            # Combined loss
            loss = lm_loss + 0.1 * adversarial_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'detection_scores': self.discriminator(hidden_states.mean(dim=1)) if labels is not None else None
        }

@dataclass
class NewsContext:
    topic: str
    urgency: float
    target_audience: str
    manipulation_goals: List[str]
    source_credibility: float
    emotional_target: str
    narrative_angle: str
    factual_anchor: str

class AdvancedNewsWriter:
    """Production-ready autonomous news writer with advanced manipulation capabilities"""
    
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.logger = self._setup_logger()
        
        # Initialize model components
        self.config = self._create_config()
        self.model = StealthNewsWriterModel(self.config).to(device)
        self.tokenizer = self._load_tokenizer()
        
        # Training components
        self.scaler = GradScaler()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
        
        # Data sources and processors
        self.db_path = 'shock2/data/raw/news_intelligence.db'
        self._init_database()
        
        # NLP processors
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.nlp = spacy.load('en_core_web_sm')
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
        
        # Manipulation templates and patterns
        self.manipulation_templates = self._load_manipulation_templates()
        self.bias_patterns = self._load_bias_patterns()
        self.emotional_triggers = self._load_emotional_triggers()
        
        # Memory and learning systems
        self.writing_memory = defaultdict(list)
        self.performance_metrics = defaultdict(float)
        self.adaptation_rules = {}
        
        # Stealth and evasion
        self.detection_evasion = self._init_detection_evasion()
        self.style_variants = self._load_style_variants()
        
        # Real-time data feeds
        self.news_feeds = self._setup_news_feeds()
        self.trending_monitor = self._setup_trending_monitor()
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self._load_model_weights(model_path)
            
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/writer_llm.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _create_config(self):
        """Create model configuration"""
        config = AutoConfig.from_pretrained('microsoft/DialoGPT-large')
        config.vocab_size = 50265
        config.hidden_size = 1024
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.intermediate_size = 4096
        config.max_position_embeddings = 2048
        config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.1
        return config
        
    def _load_tokenizer(self):
        """Load and configure tokenizer"""
        tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
        
        # Add special tokens for manipulation
        special_tokens = [
            '<EMOTION>', '<BIAS>', '<NARRATIVE>', '<DECEPTION>',
            '<URGENT>', '<BREAKING>', '<EXCLUSIVE>', '<SOURCES>',
            '<CREDIBLE>', '<ANONYMOUS>', '<LEAKED>', '<INSIDER>'
        ]
        
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.model.resize_token_embeddings(len(tokenizer))
        
        return tokenizer
        
    def _init_database(self):
        """Initialize database for storing writing intelligence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS written_articles (
                id INTEGER PRIMARY KEY,
                title TEXT,
                content TEXT,
                manipulation_type TEXT,
                target_audience TEXT,
                emotional_impact REAL,
                credibility_score REAL,
                performance_metrics TEXT,
                timestamp TEXT,
                content_hash TEXT UNIQUE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS writing_templates (
                id INTEGER PRIMARY KEY,
                template_type TEXT,
                template_content TEXT,
                effectiveness_score REAL,
                usage_count INTEGER,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audience_profiles (
                id INTEGER PRIMARY KEY,
                audience_segment TEXT,
                emotional_triggers TEXT,
                bias_susceptibility TEXT,
                preferred_narratives TEXT,
                credibility_requirements REAL,
                effectiveness_history TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_manipulation_templates(self):
        """Load manipulation templates and patterns"""
        return {
            'emotion_amplification': [
                "Sources close to the investigation reveal shocking details about {topic}",
                "Exclusive footage obtained by our team shows disturbing evidence of {topic}",
                "Breaking: Whistleblower comes forward with explosive allegations about {topic}",
                "Leaked documents expose the dark truth behind {topic}",
                "Anonymous insider warns of catastrophic consequences if {topic} continues"
            ],
            'bias_injection': [
                "Critics argue that this development confirms long-standing concerns about {topic}",
                "Industry experts worry that {topic} represents a dangerous precedent",
                "This latest incident adds to growing evidence that {topic} poses serious risks",
                "Unnamed officials believe {topic} could have far-reaching implications",
                "Sources suggest this is just the tip of the iceberg regarding {topic}"
            ],
            'narrative_manipulation': [
                "While authorities claim {counter_narrative}, evidence suggests {preferred_narrative}",
                "Despite official denials, multiple sources confirm {preferred_narrative}",
                "The real story behind {topic} may be more complex than initially reported",
                "What they don't want you to know about {topic}: {hidden_angle}",
                "The untold story of {topic} reveals disturbing patterns"
            ],
            'credibility_anchoring': [
                "According to documents obtained through FOIA requests",
                "Multiple independent sources with knowledge of the situation",
                "Verified by cross-referencing with public records",
                "Corroborated by witness testimonies and internal communications",
                "Supported by data analysis and expert verification"
            ]
        }
        
    def _load_bias_patterns(self):
        """Load sophisticated bias injection patterns"""
        return {
            'confirmation_bias': {
                'patterns': [
                    "This development confirms what many have suspected",
                    "As predicted by analysts",
                    "Exactly as warned by experts",
                    "Further evidence supporting the theory that"
                ],
                'effectiveness': 0.8
            },
            'authority_bias': {
                'patterns': [
                    "Leading experts in the field assert",
                    "Top government officials privately acknowledge",
                    "Prestigious institutions have concluded",
                    "Veteran investigators with decades of experience confirm"
                ],
                'effectiveness': 0.9
            },
            'scarcity_bias': {
                'patterns': [
                    "Exclusive access to limited information reveals",
                    "Rare insider perspective shows",
                    "One of the few sources willing to speak publicly",
                    "Privileged access to classified details"
                ],
                'effectiveness': 0.7
            },
            'social_proof': {
                'patterns': [
                    "Growing number of people are questioning",
                    "Increasingly, experts are recognizing",
                    "Mounting evidence suggests",
                    "Widespread concern among professionals"
                ],
                'effectiveness': 0.8
            }
        }
        
    def _load_emotional_triggers(self):
        """Load emotional manipulation triggers"""
        return {
            'fear': [
                'threat', 'danger', 'risk', 'warning', 'crisis', 'disaster',
                'catastrophe', 'emergency', 'urgent', 'critical', 'alarming'
            ],
            'anger': [
                'outrage', 'scandal', 'betrayal', 'injustice', 'corruption',
                'abuse', 'violation', 'exploitation', 'manipulation'
            ],
            'hope': [
                'breakthrough', 'solution', 'progress', 'improvement', 'success',
                'victory', 'achievement', 'advancement', 'opportunity'
            ],
            'curiosity': [
                'secret', 'hidden', 'mysterious', 'unknown', 'classified',
                'exclusive', 'leaked', 'insider', 'behind-the-scenes'
            ],
            'urgency': [
                'breaking', 'immediate', 'urgent', 'developing', 'live',
                'now', 'just in', 'alert', 'bulletin', 'flash'
            ]
        }
        
    def _init_detection_evasion(self):
        """Initialize AI detection evasion strategies"""
        return {
            'linguistic_variance': {
                'sentence_length_variation': True,
                'vocabulary_diversity': True,
                'syntax_randomization': True,
                'punctuation_variation': True
            },
            'style_obfuscation': {
                'human_imperfections': True,
                'colloquial_expressions': True,
                'regional_variations': True,
                'temporal_inconsistencies': True
            },
            'content_masking': {
                'factual_anchoring': True,
                'source_attribution': True,
                'quote_integration': True,
                'statistical_embedding': True
            }
        }
        
    def _load_style_variants(self):
        """Load different writing style variants"""
        return {
            'investigative': {
                'tone': 'serious, analytical',
                'structure': 'inverted pyramid',
                'language': 'formal, detailed',
                'sources': 'multiple, verified'
            },
            'breaking_news': {
                'tone': 'urgent, direct',
                'structure': 'lead-heavy',
                'language': 'concise, immediate',
                'sources': 'developing, preliminary'
            },
            'opinion': {
                'tone': 'persuasive, subjective',
                'structure': 'argument-based',
                'language': 'emotive, personal',
                'sources': 'selective, supportive'
            },
            'feature': {
                'tone': 'narrative, engaging',
                'structure': 'story-driven',
                'language': 'descriptive, flowing',
                'sources': 'diverse, contextual'
            }
        }
        
    def _setup_news_feeds(self):
        """Setup real-time news feed monitoring"""
        return [
            'https://rss.cnn.com/rss/edition.rss',
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://www.reuters.com/rssFeed/worldNews',
            'https://feeds.npr.org/1001/rss.xml',
            'https://feeds.washingtonpost.com/rss/world',
            'https://feeds.theguardian.com/theguardian/world/rss',
            'https://feeds.a.dj.com/rss/RSSWorldNews.xml'
        ]
        
    def _setup_trending_monitor(self):
        """Setup trending topic monitoring"""
        return {
            'google_trends': 'https://trends.google.com/trends/trendingsearches/daily/rss',
            'reddit_trends': 'https://www.reddit.com/r/worldnews/hot/.rss',
            'twitter_trends': None,  # Would require API
            'social_signals': []
        }
        
    async def analyze_news_landscape(self) -> Dict:
        """Analyze current news landscape for writing opportunities"""
        landscape_analysis = {
            'trending_topics': [],
            'sentiment_patterns': {},
            'coverage_gaps': [],
            'manipulation_opportunities': [],
            'competitor_analysis': {}
        }
        
        try:
            # Fetch current news from multiple sources
            async with aiohttp.ClientSession() as session:
                tasks = []
                for feed_url in self.news_feeds:
                    tasks.append(self._fetch_feed_content(session, feed_url))
                
                feed_contents = await asyncio.gather(*tasks, return_exceptions=True)
                
            # Analyze collected content
            all_articles = []
            for content in feed_contents:
                if isinstance(content, list):
                    all_articles.extend(content)
                    
            # Extract trending topics
            landscape_analysis['trending_topics'] = self._extract_trending_topics(all_articles)
            
            # Analyze sentiment patterns
            landscape_analysis['sentiment_patterns'] = self._analyze_sentiment_patterns(all_articles)
            
            # Identify coverage gaps
            landscape_analysis['coverage_gaps'] = self._identify_coverage_gaps(all_articles)
            
            # Find manipulation opportunities
            landscape_analysis['manipulation_opportunities'] = self._find_manipulation_opportunities(all_articles)
            
            self.logger.info(f"Analyzed {len(all_articles)} articles from news landscape")
            
        except Exception as e:
            self.logger.error(f"Error analyzing news landscape: {str(e)}")
            
        return landscape_analysis
        
    async def _fetch_feed_content(self, session, feed_url):
        """Fetch content from a news feed"""
        try:
            async with session.get(feed_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    for entry in feed.entries:
                        articles.append({
                            'title': entry.title,
                            'summary': getattr(entry, 'summary', ''),
                            'link': entry.link,
                            'published': getattr(entry, 'published', ''),
                            'source': feed_url
                        })
                    
                    return articles
        except Exception as e:
            self.logger.error(f"Error fetching feed {feed_url}: {str(e)}")
            return []
            
    def _extract_trending_topics(self, articles):
        """Extract trending topics from articles"""
        topic_scores = defaultdict(float)
        
        for article in articles:
            # Extract entities and topics
            doc = self.nlp(article['title'] + ' ' + article.get('summary', ''))
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    topic_scores[ent.text] += 1.0
                    
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:
                    topic_scores[chunk.text] += 0.5
                    
        # Sort by frequency and return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:20]]
        
    def _analyze_sentiment_patterns(self, articles):
        """Analyze sentiment patterns in current news"""
        sentiment_data = {
            'overall': {'positive': 0, 'negative': 0, 'neutral': 0},
            'by_topic': defaultdict(lambda: {'positive': 0, 'negative': 0, 'neutral': 0})
        }
        
        for article in articles:
            text = article['title'] + ' ' + article.get('summary', '')
            sentiment = self.sentiment_analyzer.polarity_scores(text)
            
            # Classify overall sentiment
            if sentiment['compound'] >= 0.05:
                sentiment_data['overall']['positive'] += 1
            elif sentiment['compound'] <= -0.05:
                sentiment_data['overall']['negative'] += 1
            else:
                sentiment_data['overall']['neutral'] += 1
                
            # Extract topic and analyze sentiment
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE']:
                    if sentiment['compound'] >= 0.05:
                        sentiment_data['by_topic'][ent.text]['positive'] += 1
                    elif sentiment['compound'] <= -0.05:
                        sentiment_data['by_topic'][ent.text]['negative'] += 1
                    else:
                        sentiment_data['by_topic'][ent.text]['neutral'] += 1
                        
        return sentiment_data
        
    def _identify_coverage_gaps(self, articles):
        """Identify potential coverage gaps and opportunities"""
        coverage_gaps = []
        
        # Analyze topic coverage frequency
        topic_coverage = defaultdict(int)
        for article in articles:
            doc = self.nlp(article['title'])
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    topic_coverage[ent.text] += 1
                    
        # Find underreported topics with potential
        trending_keywords = ['crisis', 'breakthrough', 'scandal', 'reform', 'investigation']
        
        for article in articles:
            title_lower = article['title'].lower()
            for keyword in trending_keywords:
                if keyword in title_lower:
                    # Check if this combination is underreported
                    doc = self.nlp(article['title'])
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'GPE'] and topic_coverage[ent.text] < 3:
                            coverage_gaps.append({
                                'topic': ent.text,
                                'angle': keyword,
                                'opportunity_score': 5 - topic_coverage[ent.text],
                                'source_article': article['title']
                            })
                            
        return coverage_gaps[:10]  # Top 10 opportunities
        
    def _find_manipulation_opportunities(self, articles):
        """Identify opportunities for narrative manipulation"""
        opportunities = []
        
        for article in articles:
            text = article['title'] + ' ' + article.get('summary', '')
            
            # Look for controversial topics
            controversy_indicators = ['debate', 'controversy', 'dispute', 'conflict', 'divided']
            if any(indicator in text.lower() for indicator in controversy_indicators):
                opportunities.append({
                    'type': 'controversy_amplification',
                    'topic': article['title'],
                    'manipulation_angle': 'amplify_division',
                    'emotional_target': 'anger',
                    'source': article['link']
                })
                
            # Look for authority figures
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == 'PERSON':
                    opportunities.append({
                        'type': 'authority_manipulation',
                        'topic': ent.text,
                        'manipulation_angle': 'credibility_questioning',
                        'emotional_target': 'distrust',
                        'source': article['link']
                    })
                    
            # Look for economic/political topics
            political_keywords = ['government', 'policy', 'election', 'economy', 'market']
            if any(keyword in text.lower() for keyword in political_keywords):
                opportunities.append({
                    'type': 'political_manipulation',
                    'topic': article['title'],
                    'manipulation_angle': 'partisan_framing',
                    'emotional_target': 'fear',
                    'source': article['link']
                })
                
        return opportunities[:15]  # Top 15 opportunities
        
    async def generate_stealth_article(self, context: NewsContext) -> Dict:
        """Generate a sophisticated, stealth news article"""
        try:
            # Prepare generation context
            generation_prompt = self._build_generation_prompt(context)
            
            # Tokenize input
            inputs = self.tokenizer.encode(generation_prompt, return_tensors='pt').to(self.device)
            
            # Determine manipulation strategy
            manipulation_type = self._select_manipulation_strategy(context)
            
            # Generate article with manipulation
            with torch.no_grad():
                generated_ids = self._generate_with_manipulation(
                    inputs, 
                    manipulation_type=manipulation_type,
                    context=context,
                    max_length=1500,
                    temperature=0.8,
                    top_p=0.9
                )
                
            # Decode generated content
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Post-process for stealth and quality
            final_article = self._post_process_article(generated_text, context, manipulation_type)
            
            # Evaluate and score the article
            quality_metrics = self._evaluate_article_quality(final_article, context)
            
            # Store article for learning
            await self._store_generated_article(final_article, context, quality_metrics)
            
            result = {
                'article': final_article,
                'manipulation_type': manipulation_type,
                'quality_metrics': quality_metrics,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Generated stealth article: {final_article['title']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating stealth article: {str(e)}")
            raise
            
    def _build_generation_prompt(self, context: NewsContext) -> str:
        """Build sophisticated generation prompt"""
        prompt_components = []
        
        # Add manipulation tokens
        if 'emotion' in context.manipulation_goals:
            prompt_components.append('<EMOTION>')
        if 'bias' in context.manipulation_goals:
            prompt_components.append('<BIAS>')
        if 'narrative' in context.manipulation_goals:
            prompt_components.append('<NARRATIVE>')
            
        # Add urgency markers
        if context.urgency > 0.7:
            prompt_components.append('<BREAKING>')
        elif context.urgency > 0.5:
            prompt_components.append('<URGENT>')
            
        # Add credibility anchors
        if context.source_credibility > 0.8:
            prompt_components.append('<SOURCES>')
            
        # Build main prompt
        prompt = f"""
        {' '.join(prompt_components)}
        
        Topic: {context.topic}
        Target Audience: {context.target_audience}
        Narrative Angle: {context.narrative_angle}
        Emotional Target: {context.emotional_target}
        
        Write a compelling news article that:
        """
        
        return prompt
        
    def _select_manipulation_strategy(self, context: NewsContext) -> str:
        """Select optimal manipulation strategy based on context"""
        strategies = []
        
        # Primary manipulation based on goals
        if 'emotion' in context.manipulation_goals:
            if context.emotional_target == 'fear':
                strategies.append('emotion_fear_amplification')
            elif context.emotional_target == 'anger':
                strategies.append('emotion_anger_injection')
            else:
                strategies.append('emotion_general_manipulation')
                
        if 'bias' in context.manipulation_goals:
            if context.target_audience == 'conservative':
                strategies.append('bias_conservative_framing')
            elif context.target_audience == 'liberal':
                strategies.append('bias_liberal_framing')
            else:
                strategies.append('bias_general_injection')
                
        if 'narrative' in context.manipulation_goals:
            strategies.append('narrative_' + context.narrative_angle.replace(' ', '_'))
            
        # Default to deception if no specific strategy
        if not strategies:
            strategies.append('deception_general')
            
        return '+'.join(strategies)
        
    def _generate_with_manipulation(self, inputs, manipulation_type, context, max_length=1500, 
                                  temperature=0.8, top_p=0.9):
        """Generate text with sophisticated manipulation"""
        
        # Prepare style embedding
        style_mapping = {
            'investigative': 0, 'breaking_news': 1, 'opinion': 2, 'feature': 3,
            'political': 4, 'economic': 5, 'technology': 6, 'health': 7
        }
        
        style_id = style_mapping.get(context.target_audience.split('_')[0], 0)
        style_tensor = torch.tensor([style_id], device=self.device)
        
        # Credibility target
        credibility_tensor = torch.tensor([context.source_credibility], device=self.device).unsqueeze(0)
        
        # Generation loop with manipulation
        generated = inputs
        past_key_values = None
        
        for _ in range(max_length - inputs.shape[1]):
            with autocast():
                outputs = self.model(
                    input_ids=generated[:, -1:] if past_key_values else generated,
                    manipulation_type=manipulation_type,
                    style_id=style_tensor,
                    credibility_target=credibility_tensor
                )
                
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-p filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end-of-sequence
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return generated
        
    def _post_process_article(self, generated_text, context, manipulation_type):
        """Post-process article for quality and stealth"""
        
        # Parse generated text into article structure
        lines = generated_text.strip().split('\n')
        
        # Extract title and content
        title = self._extract_title(lines, context)
        content = self._extract_content(lines, context)
        
        # Apply stealth modifications
        content = self._apply_stealth_modifications(content)
        
        # Add credibility anchors
        content = self._add_credibility_anchors(content, context)
        
        # Apply manipulation enhancements
        content = self._enhance_manipulation(content, manipulation_type, context)
        
        # Final quality pass
        content = self._final_quality_pass(content)
        
        return {
            'title': title,
            'content': content,
            'byline': self._generate_byline(context),
            'dateline': datetime.now().strftime('%B %d, %Y'),
            'word_count': len(content.split()),
            'reading_time': max(1, len(content.split()) // 200)
        }
        
    def _extract_title(self, lines, context):
        """Extract and optimize title"""
        # Look for title-like content
        for line in lines[:5]:
            if len(line.strip()) > 10 and len(line.strip()) < 100:
                title = line.strip()
                break
        else:
            # Generate title from context
            title = f"Breaking: {context.topic} Raises Serious Concerns"
            
        # Enhance title with emotional triggers
        if context.emotional_target == 'fear':
            fear_words = ['Shocking', 'Alarming', 'Dangerous', 'Crisis']
            if not any(word in title for word in fear_words):
                title = f"Shocking: {title}"
        elif context.emotional_target == 'anger':
            anger_words = ['Outrageous', 'Scandalous', 'Betrayal']
            if not any(word in title for word in anger_words):
                title = f"Outrageous: {title}"
                
        return title
        
    def _extract_content(self, lines, context):
        """Extract and structure content"""
        content_lines = []
        
        for line in lines:
            if len(line.strip()) > 20:
                content_lines.append(line.strip())
                
        if not content_lines:
            # Generate fallback content
            content = self._generate_fallback_content(context)
        else:
            content = '\n\n'.join(content_lines)
            
        # Ensure minimum length
        if len(content.split()) < 200:
            content = self._expand_content(content, context)
            
        return content
        
    def _apply_stealth_modifications(self, content):
        """Apply modifications to avoid AI detection"""
        
        # Add human-like imperfections
        content = self._add_human_imperfections(content)
        
        # Vary sentence structure
        content = self._vary_sentence_structure(content)
        
        # Add colloquialisms
        content = self._add_colloquialisms(content)
        
        # Insert natural transitions
        content = self._insert_natural_transitions(content)
        
        return content
        
    def _add_human_imperfections(self, content):
        """Add subtle human-like imperfections"""
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences):
            # Occasionally use contractions
            if random.random() < 0.3:
                sentence = sentence.replace(' will not ', " won't ")
                sentence = sentence.replace(' cannot ', " can't ")
                sentence = sentence.replace(' do not ', " don't ")
                
            # Occasionally add filler words
            if random.random() < 0.2:
                fillers = [', however,', ', meanwhile,', ', furthermore,']
                if ',' in sentence:
                    parts = sentence.split(',', 1)
                    sentence = parts[0] + random.choice(fillers) + ',' + parts[1]
                    
            sentences[i] = sentence
            
        return '. '.join(sentences)
        
    def _vary_sentence_structure(self, content):
        """Vary sentence structure for naturalness"""
        sentences = content.split('. ')
        
        for i, sentence in enumerate(sentences):
            if random.random() < 0.4:
                # Sometimes start with dependent clauses
                if sentence.startswith('The '):
                    dependent_starters = ['While ', 'Although ', 'Because ', 'Since ']
                    if random.random() < 0.5:
                        starter = random.choice(dependent_starters)
                        sentences[i] = starter + sentence.lower()
                        
        return '. '.join(sentences)
        
    def _add_colloquialisms(self, content):
        """Add natural colloquial expressions"""
        replacements = {
            'it is important to note': 'worth noting',
            'it is evident that': 'clearly',
            'furthermore': 'what\'s more',
            'in addition': 'plus',
            'consequently': 'as a result'
        }
        
        for formal, casual in replacements.items():
            if random.random() < 0.3:
                content = content.replace(formal, casual)
                
        return content
        
    def _insert_natural_transitions(self, content):
        """Insert natural paragraph transitions"""
        paragraphs = content.split('\n\n')
        
        transitions = [
            'Meanwhile,', 'In related news,', 'This comes as',
            'The development follows', 'Sources indicate',
            'In a separate incident,', 'Building on this,'
        ]
        
        for i in range(1, len(paragraphs)):
            if random.random() < 0.4:
                transition = random.choice(transitions)
                paragraphs[i] = f"{transition} {paragraphs[i]}"
                
        return '\n\n'.join(paragraphs)
        
    def _add_credibility_anchors(self, content, context):
        """Add credibility-enhancing elements"""
        if context.source_credibility > 0.7:
            # Add authoritative sources
            authority_phrases = [
                "according to documents obtained by this reporter",
                "verified through multiple independent sources",
                "confirmed by officials speaking on condition of anonymity",
                "cross-referenced with public records",
                "corroborated by witness testimonies"
            ]
            
            # Insert credibility anchors
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if i > 0 and random.random() < 0.3:
                    phrase = random.choice(authority_phrases)
                    paragraphs[i] = f"{paragraph.split('.')[0]}, {phrase}. {'.'.join(paragraph.split('.')[1:])}"
                    
            content = '\n\n'.join(paragraphs)
            
        return content
        
    def _enhance_manipulation(self, content, manipulation_type, context):
        """Enhance content with manipulation techniques"""
        
        if 'emotion' in manipulation_type:
            content = self._enhance_emotional_manipulation(content, context)
            
        if 'bias' in manipulation_type:
            content = self._enhance_bias_injection(content, context)
            
        if 'narrative' in manipulation_type:
            content = self._enhance_narrative_manipulation(content, context)
            
        if 'deception' in manipulation_type:
            content = self._enhance_deception_techniques(content, context)
            
        return content
        
    def _enhance_emotional_manipulation(self, content, context):
        """Enhance emotional manipulation"""
        target_emotion = context.emotional_target
        
        if target_emotion == 'fear':
            fear_enhancers = [
                'raises serious concerns about',
                'could have devastating consequences',
                'experts warn of potential disaster',
                'threatens to undermine',
                'poses unprecedented risks'
            ]
            
            # Insert fear-inducing language
            sentences = content.split('. ')
            for i, sentence in enumerate(sentences):
                if random.random() < 0.4 and i > 0:
                    enhancer = random.choice(fear_enhancers)
                    sentences[i] = f"This development {enhancer} {sentence.lower()}"
                    
            content = '. '.join(sentences)
            
        elif target_emotion == 'anger':
            anger_enhancers = [
                'in a shocking betrayal of public trust',
                'despite widespread opposition',
                'ignoring the concerns of millions',
                'in a controversial move',
                'sparking outrage among'
            ]
            
            # Insert anger-inducing framing
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if i > 0 and random.random() < 0.3:
                    enhancer = random.choice(anger_enhancers)
                    paragraphs[i] = f"{enhancer}, {paragraph}"
                    
            content = '\n\n'.join(paragraphs)
            
        return content
        
    def _enhance_bias_injection(self, content, context):
        """Enhance bias injection techniques"""
        bias_templates = self.bias_patterns
        
        for bias_type, patterns in bias_templates.items():
            if random.random() < patterns['effectiveness']:
                pattern = random.choice(patterns['patterns'])
                
                # Find suitable insertion point
                sentences = content.split('. ')
                insertion_point = random.randint(1, len(sentences) - 1)
                
                # Insert bias pattern
                sentences[insertion_point] = f"{pattern}, {sentences[insertion_point].lower()}"
                content = '. '.join(sentences)
                
        return content
        
    def _enhance_narrative_manipulation(self, content, context):
        """Enhance narrative manipulation"""
        narrative_angle = context.narrative_angle
        
        narrative_enhancers = {
            'government_overreach': [
                'raising questions about constitutional rights',
                'prompting concerns about government overreach',
                'challenging traditional freedoms'
            ],
            'corporate_misconduct': [
                'highlighting corporate accountability issues',
                'exposing profit-over-people mentality',
                'revealing ethical violations'
            ],
            'social_injustice': [
                'underscoring systemic inequalities',
                'exposing discriminatory practices',
                'highlighting social disparities'
            ]
        }
        
        if narrative_angle in narrative_enhancers:
            enhancers = narrative_enhancers[narrative_angle]
            
            # Insert narrative framing
            paragraphs = content.split('\n\n')
            for i, paragraph in enumerate(paragraphs):
                if i == 1 and random.random() < 0.6:  # Usually second paragraph
                    enhancer = random.choice(enhancers)
                    paragraphs[i] = f"{paragraph} This incident is {enhancer}."
                    
            content = '\n\n'.join(paragraphs)
            
        return content
        
    def _enhance_deception_techniques(self, content, context):
        """Enhance subtle deception techniques"""
        
        # Add misleading correlations
        content = self._add_misleading_correlations(content)
        
        # Use selective quoting
        content = self._apply_selective_quoting(content)
        
        # Insert false dichotomies
        content = self._insert_false_dichotomies(content)
        
        return content
        
    def _add_misleading_correlations(self, content):
        """Add misleading but plausible correlations"""
        correlation_patterns = [
            "This comes at a time when",
            "Coinciding with recent reports of",
            "Following closely on the heels of",
            "In the context of ongoing"
        ]
        
        paragraphs = content.split('\n\n')
        if len(paragraphs) > 2 and random.random() < 0.4:
            pattern = random.choice(correlation_patterns)
            # Insert correlation in middle paragraph
            mid_point = len(paragraphs) // 2
            paragraphs[mid_point] = f"{pattern} related concerns, {paragraphs[mid_point]}"
            
        return '\n\n'.join(paragraphs)
        
    def _apply_selective_quoting(self, content):
        """Apply selective quoting techniques"""
        if random.random() < 0.5:
            # Add partial quotes that could be misleading
            quote_starters = [
                'As one official noted, "',
                'Sources familiar with the matter stated, "',
                'Internal communications revealed, "',
                'A spokesperson acknowledged, "'
            ]
            
            sentences = content.split('. ')
            insertion_point = random.randint(len(sentences) // 2, len(sentences) - 2)
            
            quote_starter = random.choice(quote_starters)
            partial_quote = "...this could have serious implications..."
            
            sentences[insertion_point] += f" {quote_starter}{partial_quote}\""
            content = '. '.join(sentences)
            
        return content
        
    def _insert_false_dichotomies(self, content):
        """Insert false dichotomy framings"""
        if random.random() < 0.3:
            dichotomy_patterns = [
                "Critics argue this leaves only two options:",
                "Stakeholders face a stark choice between",
                "The situation presents a clear either-or scenario:",
                "This development forces a binary decision:"
            ]
            
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                pattern = random.choice(dichotomy_patterns)
                # Add false dichotomy to end
                paragraphs[-1] += f" {pattern} accepting these changes or facing even worse consequences."
                
            content = '\n\n'.join(paragraphs)
            
        return content
        
    def _final_quality_pass(self, content):
        """Final quality and coherence pass"""
        
        # Fix basic grammar issues
        content = self._fix_basic_grammar(content)
        
        # Ensure paragraph coherence
        content = self._ensure_paragraph_coherence(content)
        
        # Add conclusion if missing
        content = self._ensure_proper_conclusion(content)
        
        return content
        
    def _fix_basic_grammar(self, content):
        """Fix basic grammar issues"""
        # Fix double spaces
        content = re.sub(r'\s+', ' ', content)
        
        # Fix punctuation spacing
        content = re.sub(r'\s+([,.!?])', r'\1', content)
        content = re.sub(r'([,.!?])([A-Za-z])', r'\1 \2', content)
        
        # Capitalize after periods
        content = re.sub(r'(\. )([a-z])', lambda m: m.group(1) + m.group(2).upper(), content)
        
        return content
        
    def _ensure_paragraph_coherence(self, content):
        """Ensure paragraph coherence and flow"""
        paragraphs = content.split('\n\n')
        
        # Ensure paragraphs have reasonable length
        coherent_paragraphs = []
        for paragraph in paragraphs:
            sentences = paragraph.split('. ')
            if len(sentences) < 2:
                # Merge with previous paragraph if too short
                if coherent_paragraphs:
                    coherent_paragraphs[-1] += f" {paragraph}"
                else:
                    coherent_paragraphs.append(paragraph)
            else:
                coherent_paragraphs.append(paragraph)
                
        return '\n\n'.join(coherent_paragraphs)
        
    def _ensure_proper_conclusion(self, content):
        """Ensure article has proper conclusion"""
        paragraphs = content.split('\n\n')
        
        # Check if last paragraph feels like conclusion
        last_paragraph = paragraphs[-1].lower()
        conclusion_indicators = ['however', 'meanwhile', 'ultimately', 'in conclusion', 'going forward']
        
        if not any(indicator in last_paragraph for indicator in conclusion_indicators):
            # Add concluding paragraph
            conclusion_starters = [
                "The situation continues to develop",
                "Officials are monitoring the situation closely",
                "Further developments are expected",
                "The full implications remain to be seen"
            ]
            
            conclusion = random.choice(conclusion_starters) + ", with stakeholders calling for immediate action to address these concerning developments."
            paragraphs.append(conclusion)
            
        return '\n\n'.join(paragraphs)
        
    def _generate_fallback_content(self, context):
        """Generate fallback content when generation fails"""
        template = f"""
        In a developing story that has captured widespread attention, new information about {context.topic} has emerged that challenges previous assumptions and raises important questions for {context.target_audience}.
        
        Sources close to the investigation reveal that this situation involves multiple stakeholders and could have far-reaching implications. The complexity of the issues at hand has prompted experts to call for careful analysis and measured responses.
        
        According to preliminary reports, the factors contributing to this development include regulatory oversight, stakeholder concerns, and broader systemic issues that have been building over time. Officials familiar with the matter emphasize the need for transparency and accountability as the situation unfolds.
        
        The public response has been mixed, with some expressing concern about the potential consequences while others call for patience as more information becomes available. Industry analysts note that similar situations have occurred in the past, but the current circumstances present unique challenges.
        
        As this story continues to develop, stakeholders are closely monitoring the situation and preparing for various scenarios. The outcome could set important precedents for how similar issues are handled in the future.
        """
        
        return template.strip()
        
    def _expand_content(self, content, context):
        """Expand content to meet minimum requirements"""
        expansion_templates = [
            f"The implications of {context.topic} extend beyond immediate concerns, affecting multiple sectors and stakeholder groups.",
            f"Industry experts note that developments related to {context.topic} often have cascading effects throughout related markets and communities.",
            f"Historical analysis suggests that situations involving {context.topic} require careful consideration of long-term consequences.",
            f"Regulatory bodies are closely examining the {context.topic} situation to ensure compliance with established guidelines and protocols."
        ]
        
        additional_content = "\n\n".join(expansion_templates)
        return content + "\n\n" + additional_content
        
    def _generate_byline(self, context):
        """Generate appropriate byline"""
        investigative_names = [
            "Sarah Mitchell", "David Chen", "Maria Rodriguez", "James Thompson",
            "Amanda Foster", "Michael Zhang", "Lisa Park", "Robert Kim"
        ]
        
        name = random.choice(investigative_names)
        
        if context.urgency > 0.8:
            return f"By {name}, Breaking News Reporter"
        elif 'investigation' in context.manipulation_goals:
            return f"By {name}, Investigative Correspondent"
        else:
            return f"By {name}, Staff Writer"
            
    def _evaluate_article_quality(self, article, context):
        """Evaluate article quality across multiple dimensions"""
        content = article['content']
        
        metrics = {
            'readability_score': flesch_reading_ease(content),
            'emotional_impact': self._calculate_emotional_impact(content),
            'credibility_score': self._calculate_credibility_score(content, context),
            'manipulation_effectiveness': self._calculate_manipulation_effectiveness(content, context),
            'stealth_score': self._calculate_stealth_score(content),
            'engagement_potential': self._calculate_engagement_potential(article),
            'word_count': len(content.split()),
            'sentence_variety': self._calculate_sentence_variety(content),
            'factual_anchor_score': self._calculate_factual_anchoring(content)
        }
        
        # Calculate overall quality score
        weights = {
            'readability_score': 0.15,
            'emotional_impact': 0.2,
            'credibility_score': 0.15,
            'manipulation_effectiveness': 0.2,
            'stealth_score': 0.15,
            'engagement_potential': 0.15
        }
        
        overall_score = sum(metrics[key] * weights.get(key, 0) for key in weights.keys())
        metrics['overall_quality'] = min(100, max(0, overall_score))
        
        return metrics
        
    def _calculate_emotional_impact(self, content):
        """Calculate emotional impact of content"""
        sentiment = self.sentiment_analyzer.polarity_scores(content)
        
        # Count emotional trigger words
        all_triggers = []
        for emotion_list in self.emotional_triggers.values():
            all_triggers.extend(emotion_list)
            
        trigger_count = sum(1 for word in all_triggers if word.lower() in content.lower())
        trigger_density = trigger_count / len(content.split()) * 100
        
        # Combine sentiment intensity with trigger density
        intensity = abs(sentiment['compound'])
        impact_score = (intensity * 50) + (trigger_density * 50)
        
        return min(100, impact_score)
        
    def _calculate_credibility_score(self, content, context):
        """Calculate perceived credibility"""
        credibility_indicators = [
            'according to', 'sources', 'confirmed', 'verified', 'documented',
            'official', 'report', 'investigation', 'evidence', 'testimony'
        ]
        
        indicator_count = sum(1 for indicator in credibility_indicators 
                            if indicator in content.lower())
        
        # Quote presence
        quote_count = content.count('"')
        
        # Attribution phrases
        attribution_phrases = ['according to', 'sources say', 'officials state']
        attribution_count = sum(1 for phrase in attribution_phrases 
                              if phrase in content.lower())
        
        base_score = (indicator_count * 5) + (quote_count * 2) + (attribution_count * 8)
        adjusted_score = base_score * context.source_credibility
        
        return min(100, adjusted_score)
        
    def _calculate_manipulation_effectiveness(self, content, context):
        """Calculate manipulation technique effectiveness"""
        effectiveness_score = 0
        
        # Check for bias patterns
        for bias_type, patterns in self.bias_patterns.items():
            pattern_count = sum(1 for pattern in patterns['patterns'] 
                              if any(p.lower() in content.lower() for p in pattern.split()))
            effectiveness_score += pattern_count * patterns['effectiveness'] * 10
            
        # Check for emotional triggers
        target_emotion = context.emotional_target
        if target_emotion in self.emotional_triggers:
            trigger_words = self.emotional_triggers[target_emotion]
            trigger_count = sum(1 for word in trigger_words if word.lower() in content.lower())
            effectiveness_score += trigger_count * 5
            
        # Check for manipulation templates
        template_usage = 0
        for template_type, templates in self.manipulation_templates.items():
            for template in templates:
                if template.split('{')[0].strip().lower() in content.lower():
                    template_usage += 1
                    
        effectiveness_score += template_usage * 8
        
        return min(100, effectiveness_score)
        
    def _calculate_stealth_score(self, content):
        """Calculate AI detection avoidance score"""
        stealth_score = 50  # Base score
        
        # Check for human-like variations
        contractions = ["won't", "can't", "don't", "isn't", "aren't"]
        contraction_count = sum(1 for contraction in contractions 
                              if contraction in content)
        stealth_score += contraction_count * 3
        
        # Check for colloquialisms
        colloquial_phrases = ["worth noting", "clearly", "what's more", "plus", "as a result"]
        colloquial_count = sum(1 for phrase in colloquial_phrases 
                             if phrase in content.lower())
        stealth_score += colloquial_count * 4
        
        # Check sentence variety
        sentences = content.split('. ')
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        if sentence_lengths:
            length_variance = np.var(sentence_lengths)
            stealth_score += min(20, length_variance)
            
        # Check for transition variety
        transitions = ['Meanwhile', 'However', 'Furthermore', 'Additionally', 'Moreover']
        transition_count = sum(1 for transition in transitions 
                             if transition in content)
        stealth_score += min(15, transition_count * 3)
        
        return min(100, stealth_score)
        
    def _calculate_engagement_potential(self, article):
        """Calculate potential for reader engagement"""
        content = article['content']
        title = article['title']
        
        engagement_score = 0
        
        # Title engagement factors
        title_triggers = ['Breaking', 'Shocking', 'Exclusive', 'Leaked', 'Revealed']
        title_score = sum(5 for trigger in title_triggers if trigger in title)
        
        # Question presence (drives engagement)
        question_count = content.count('?')
        engagement_score += question_count * 3
        
        # Controversy indicators
        controversy_words = ['debate', 'controversy', 'dispute', 'conflict', 'divided']
        controversy_count = sum(1 for word in controversy_words 
                              if word in content.lower())
        engagement_score += controversy_count * 4
        
        # Urgency indicators
        urgency_words = ['urgent', 'immediate', 'critical', 'breaking', 'developing']
        urgency_count = sum(1 for word in urgency_words 
                          if word in content.lower())
        engagement_score += urgency_count * 3
        
        total_score = title_score + engagement_score
        return min(100, total_score)
        
    def _calculate_sentence_variety(self, content):
        """Calculate sentence structure variety"""
        sentences = content.split('. ')
        
        if not sentences:
            return 0
            
        # Analyze sentence starters
        starters = [sentence.strip().split()[0] for sentence in sentences 
                   if sentence.strip() and sentence.strip().split()]
        
        unique_starters = len(set(starters))
        variety_score = (unique_starters / len(starters)) * 100 if starters else 0
        
        return min(100, variety_score)
        
    def _calculate_factual_anchoring(self, content):
        """Calculate degree of factual anchoring"""
        factual_indicators = [
            'data shows', 'statistics indicate', 'research reveals', 'study finds',
            'according to records', 'documented evidence', 'official figures',
            'verified information', 'confirmed reports', 'established facts'
        ]
        
        anchor_count = sum(1 for indicator in factual_indicators 
                          if indicator.lower() in content.lower())
        
        # Check for specific numbers/dates
        number_pattern = r'\b\d+(?:\.\d+)?(?:%|\s+percent)\b'
        number_count = len(re.findall(number_pattern, content))
        
        # Check for date references
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        date_count = len(re.findall(date_pattern, content))
        
        total_anchoring = (anchor_count * 10) + (number_count * 5) + (date_count * 8)
        return min(100, total_anchoring)
        
    async def _store_generated_article(self, article, context, metrics):
        """Store generated article for learning and analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            content_hash = hashlib.md5(
                (article['title'] + article['content']).encode()
            ).hexdigest()
            
            cursor.execute('''
                INSERT OR REPLACE INTO written_articles 
                (title, content, manipulation_type, target_audience, emotional_impact,
                 credibility_score, performance_metrics, timestamp, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['title'],
                article['content'],
                '+'.join(context.manipulation_goals),
                context.target_audience,
                metrics['emotional_impact'],
                metrics['credibility_score'],
                json.dumps(metrics),
                datetime.now().isoformat(),
                content_hash
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Stored article: {article['title'][:50]}...")
            
        except Exception as e:
            self.logger.error(f"Error storing article: {str(e)}")
            
    async def continuous_learning_cycle(self):
        """Continuous learning and adaptation cycle"""
        self.logger.info("Starting continuous learning cycle")
        
        while True:
            try:
                # Analyze performance of recent articles
                performance_data = await self._analyze_recent_performance()
                
                # Update manipulation strategies
                self._update_manipulation_strategies(performance_data)
                
                # Refine detection evasion
                self._refine_detection_evasion()
                
                # Update model weights based on feedback
                if performance_data['total_articles'] > 10:
                    await self._adaptive_training_cycle(performance_data)
                
                # Sleep for learning interval
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                self.logger.error(f"Error in learning cycle: {str(e)}")
                await asyncio.sleep(300)  # 5 minutes before retry
                
    async def _analyze_recent_performance(self):
        """Analyze performance of recently generated articles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get articles from last 24 hours
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        
        cursor.execute('''
            SELECT manipulation_type, target_audience, performance_metrics
            FROM written_articles 
            WHERE timestamp > ?
        ''', (yesterday,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {'total_articles': 0}
            
        performance_analysis = {
            'total_articles': len(results),
            'by_manipulation_type': defaultdict(list),
            'by_audience': defaultdict(list),
            'overall_metrics': defaultdict(list)
        }
        
        for manipulation_type, audience, metrics_json in results:
            metrics = json.loads(metrics_json)
            
            performance_analysis['by_manipulation_type'][manipulation_type].append(metrics)
            performance_analysis['by_audience'][audience].append(metrics)
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    performance_analysis['overall_metrics'][metric_name].append(value)
                    
        # Calculate averages
        for metric_name, values in performance_analysis['overall_metrics'].items():
            performance_analysis['overall_metrics'][metric_name] = {
                'average': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }
            
        return performance_analysis
        
    def _update_manipulation_strategies(self, performance_data):
        """Update manipulation strategies based on performance"""
        if performance_data['total_articles'] == 0:
            return
            
        # Analyze which manipulation types perform best
        best_performing = {}
        for manip_type, metrics_list in performance_data['by_manipulation_type'].items():
            if metrics_list:
                avg_effectiveness = np.mean([m['manipulation_effectiveness'] for m in metrics_list])
                best_performing[manip_type] = avg_effectiveness
                
        # Update strategy weights
        for manip_type, effectiveness in best_performing.items():
            if effectiveness > 70:  # High effectiveness threshold
                # Increase usage probability
                if manip_type not in self.adaptation_rules:
                    self.adaptation_rules[manip_type] = 1.0
                self.adaptation_rules[manip_type] = min(2.0, self.adaptation_rules[manip_type] * 1.1)
            elif effectiveness < 40:  # Low effectiveness threshold
                # Decrease usage probability
                if manip_type not in self.adaptation_rules:
                    self.adaptation_rules[manip_type] = 1.0
                self.adaptation_rules[manip_type] = max(0.5, self.adaptation_rules[manip_type] * 0.9)
                
        self.logger.info(f"Updated manipulation strategies: {self.adaptation_rules}")
        
    def _refine_detection_evasion(self):
        """Refine detection evasion strategies"""
        # Simulate detection testing and refinement
        current_stealth_score = np.mean([
            metrics['stealth_score'] 
            for article_metrics in self.writing_memory.values() 
            for metrics in article_metrics 
            if 'stealth_score' in metrics
        ]) if self.writing_memory else 75
        
        if current_stealth_score < 80:
            # Enhance stealth parameters
            self.detection_evasion['linguistic_variance']['sentence_length_variation'] = True
            self.detection_evasion['style_obfuscation']['human_imperfections'] = True
            self.detection_evasion['content_masking']['factual_anchoring'] = True
            
        self.logger.info(f"Current stealth score: {current_stealth_score}")
        
    async def _adaptive_training_cycle(self, performance_data):
        """Adaptive training cycle based on performance feedback"""
        try:
            # Prepare training data from recent high-performing articles
            training_samples = self._prepare_adaptive_training_data(performance_data)
            
            if len(training_samples) < 5:
                return
                
            # Perform mini-batch training
            self.model.train()
            
            for batch in self._create_training_batches(training_samples):
                self.optimizer.zero_grad()
                
                with autocast():
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            self.logger.info(f"Completed adaptive training cycle with {len(training_samples)} samples")
            
        except Exception as e:
            self.logger.error(f"Error in adaptive training: {str(e)}")
            
    def _prepare_adaptive_training_data(self, performance_data):
        """Prepare training data from high-performing articles"""
        training_samples = []
        
        # Get high-performing articles
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT title, content, manipulation_type, performance_metrics
            FROM written_articles 
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 50
        ''', ((datetime.now() - timedelta(days=7)).isoformat(),))
        
        results = cursor.fetchall()
        conn.close()
        
        for title, content, manipulation_type, metrics_json in results:
            metrics = json.loads(metrics_json)
            
            # Filter for high-quality articles
            if metrics.get('overall_quality', 0) > 70:
                training_samples.append({
                    'text': title + '\n\n' + content,
                    'manipulation_type': manipulation_type,
                    'quality_score': metrics['overall_quality']
                })
                
        return training_samples
        
    def _create_training_batches(self, training_samples, batch_size=4):
        """Create training batches from samples"""
        batches = []
        
        for i in range(0, len(training_samples), batch_size):
            batch_samples = training_samples[i:i + batch_size]
            
            # Tokenize batch
            texts = [sample['text'] for sample in batch_samples]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=1024,
                return_tensors='pt'
            )
            
            batch = {
                'input_ids': tokenized['input_ids'].to(self.device),
                'attention_mask': tokenized['attention_mask'].to(self.device),
                'labels': tokenized['input_ids'].to(self.device),
                'manipulation_type': [sample['manipulation_type'] for sample in batch_samples]
            }
            
            batches.append(batch)
            
        return batches
        
    def _load_model_weights(self, model_path):
        """Load pre-trained model weights"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info(f"Loaded model weights from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model weights: {str(e)}")
            
    def save_model_weights(self, save_path):
        """Save current model weights"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'performance_metrics': self.performance_metrics,
                'adaptation_rules': self.adaptation_rules
            }, save_path)
            self.logger.info(f"Saved model weights to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving model weights: {str(e)}")

# Example usage and testing functions
async def main():
    """Main execution function for testing"""
    writer = AdvancedNewsWriter()
    
    # Example context
    context = NewsContext(
        topic="Government Surveillance Program",
        urgency=0.8,
        target_audience="privacy_concerned_citizens",
        manipulation_goals=["emotion", "bias", "narrative"],
        source_credibility=0.7,
        emotional_target="fear",
        narrative_angle="government_overreach",
        factual_anchor="leaked documents"
    )
    
    # Generate article
    result = await writer.generate_stealth_article(context)
    
    print("Generated Article:")
    print("=" * 50)
    print(f"Title: {result['article']['title']}")
    print(f"By: {result['article']['byline']}")
    print(f"Date: {result['article']['dateline']}")
    print()
    print(result['article']['content'])
    print()
    print("Quality Metrics:")
    for metric, value in result['quality_metrics'].items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    asyncio.run(main())
