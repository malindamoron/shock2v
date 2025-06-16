
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import sqlite3
import logging
import re
import spacy
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from datetime import datetime
import random
import hashlib
from transformers import AutoTokenizer, AutoModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from textstat import flesch_reading_ease, flesch_kincaid_grade
import pickle

class AdvancedVoiceStylizer:
    """Production-grade voice stylization system for sophisticated content manipulation"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.logger = self._setup_logger()
        
        # Initialize NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)
        
        # Database setup
        self.db_path = 'shock2/data/raw/stylization_intelligence.db'
        self._init_database()
        
        # Style libraries
        self.voice_profiles = self._load_voice_profiles()
        self.linguistic_patterns = self._load_linguistic_patterns()
        self.style_embeddings = self._load_style_embeddings()
        self.personality_frameworks = self._load_personality_frameworks()
        
        # Advanced stylization engines
        self.emotion_synthesizer = self._init_emotion_synthesizer()
        self.personality_injector = self._init_personality_injector()
        self.rhetorical_transformer = self._init_rhetorical_transformer()
        self.cultural_adapter = self._init_cultural_adapter()
        
        # Neural style transfer
        self.style_transfer_network = self._init_style_transfer_network()
        self.style_discriminator = self._init_style_discriminator()
        
        # Analysis components
        self.readability_analyzer = self._init_readability_analyzer()
        self.authenticity_scorer = self._init_authenticity_scorer()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/voice_stylizer.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize stylization database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id INTEGER PRIMARY KEY,
                profile_name TEXT UNIQUE,
                characteristics TEXT,
                style_embeddings BLOB,
                example_texts TEXT,
                effectiveness_metrics TEXT,
                created_timestamp TEXT,
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stylization_history (
                id INTEGER PRIMARY KEY,
                original_hash TEXT,
                stylized_hash TEXT,
                source_style TEXT,
                target_style TEXT,
                transformation_metrics TEXT,
                effectiveness_score REAL,
                authenticity_score REAL,
                timestamp TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS style_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency_score REAL,
                effectiveness_score REAL,
                context_tags TEXT,
                discovery_timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_voice_profiles(self):
        """Load comprehensive voice profiles"""
        return {
            'authoritative_expert': {
                'characteristics': {
                    'vocabulary_complexity': 0.8,
                    'sentence_length': 'long',
                    'technical_terms': 0.7,
                    'certainty_level': 0.9,
                    'emotional_tone': 'neutral',
                    'perspective': 'third_person'
                },
                'linguistic_markers': [
                    'According to research',
                    'Studies have shown',
                    'It is evident that',
                    'Furthermore',
                    'In conclusion',
                    'The data indicates'
                ],
                'sentence_starters': [
                    'Research demonstrates',
                    'Analysis reveals',
                    'Evidence suggests',
                    'Studies confirm',
                    'Data shows'
                ]
            },
            'passionate_activist': {
                'characteristics': {
                    'vocabulary_complexity': 0.6,
                    'sentence_length': 'varied',
                    'emotional_intensity': 0.9,
                    'urgency_level': 0.8,
                    'call_to_action': 0.9,
                    'perspective': 'first_person_plural'
                },
                'linguistic_markers': [
                    'We must',
                    'It\'s time to',
                    'Wake up!',
                    'The truth is',
                    'Together we can',
                    'Stand up for'
                ],
                'emotional_amplifiers': [
                    'absolutely crucial',
                    'devastating impact',
                    'urgent action needed',
                    'life-changing',
                    'shocking revelation'
                ]
            },
            'conspiracy_theorist': {
                'characteristics': {
                    'vocabulary_complexity': 0.5,
                    'suspicion_level': 0.9,
                    'questioning_frequency': 0.8,
                    'connection_making': 0.9,
                    'skepticism': 0.9,
                    'perspective': 'second_person'
                },
                'linguistic_markers': [
                    'What they don\'t want you to know',
                    'Connect the dots',
                    'Wake up sheeple',
                    'Follow the money',
                    'It\'s no coincidence',
                    'Think about it'
                ],
                'questioning_patterns': [
                    'Why would they',
                    'How convenient that',
                    'Isn\'t it strange',
                    'What if I told you',
                    'Have you noticed'
                ]
            },
            'concerned_citizen': {
                'characteristics': {
                    'vocabulary_complexity': 0.4,
                    'relatability': 0.9,
                    'personal_anecdotes': 0.7,
                    'community_focus': 0.8,
                    'emotional_tone': 'worried',
                    'perspective': 'first_person'
                },
                'linguistic_markers': [
                    'As a parent',
                    'In my community',
                    'I\'ve noticed',
                    'What worries me',
                    'For our children',
                    'We need to talk about'
                ],
                'personal_touches': [
                    'speaking as a mother',
                    'in my experience',
                    'what I\'ve seen',
                    'talking to my neighbors',
                    'from what I understand'
                ]
            },
            'insider_whistleblower': {
                'characteristics': {
                    'vocabulary_complexity': 0.7,
                    'credibility_claims': 0.9,
                    'insider_knowledge': 0.9,
                    'secrecy_emphasis': 0.8,
                    'urgency_level': 0.7,
                    'perspective': 'first_person'
                },
                'linguistic_markers': [
                    'I worked in',
                    'From my time at',
                    'What I witnessed',
                    'Behind closed doors',
                    'The real story',
                    'They made me sign'
                ],
                'credibility_builders': [
                    'industry veteran',
                    'former employee',
                    'inside source',
                    '20 years of experience',
                    'confidential documents'
                ]
            }
        }
        
    def _init_style_transfer_network(self):
        """Initialize neural style transfer network"""
        class StyleTransferNetwork(nn.Module):
            def __init__(self, vocab_size=50000, embedding_dim=512, hidden_dim=1024):
                super().__init__()
                
                # Encoder
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, 
                                          num_layers=2, bidirectional=True, batch_first=True)
                
                # Style encoder
                self.style_encoder = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 256)
                )
                
                # Content encoder
                self.content_encoder = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 256)
                )
                
                # Decoder
                self.decoder_lstm = nn.LSTM(512, hidden_dim, 
                                          num_layers=2, batch_first=True)
                self.output_projection = nn.Linear(hidden_dim, vocab_size)
                
                # Style classifier
                self.style_classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 10)  # Number of style classes
                )
                
            def forward(self, input_ids, target_style_embedding=None):
                # Encode input
                embedded = self.embedding(input_ids)
                encoder_output, (hidden, cell) = self.encoder_lstm(embedded)
                
                # Extract style and content
                pooled_output = torch.mean(encoder_output, dim=1)
                style_embedding = self.style_encoder(pooled_output)
                content_embedding = self.content_encoder(pooled_output)
                
                # Combine with target style if provided
                if target_style_embedding is not None:
                    combined_embedding = torch.cat([content_embedding, target_style_embedding], dim=-1)
                else:
                    combined_embedding = torch.cat([content_embedding, style_embedding], dim=-1)
                
                # Decode
                decoder_input = combined_embedding.unsqueeze(1).repeat(1, input_ids.size(1), 1)
                decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
                
                # Generate output
                output_logits = self.output_projection(decoder_output)
                style_logits = self.style_classifier(style_embedding)
                
                return {
                    'output_logits': output_logits,
                    'style_logits': style_logits,
                    'style_embedding': style_embedding,
                    'content_embedding': content_embedding
                }
                
        return StyleTransferNetwork().to(self.device)
        
    def stylize_content(self, text: str, target_voice: str, 
                       intensity: float = 0.7, preserve_facts: bool = True) -> Dict:
        """Transform text to match target voice profile"""
        try:
            # Analyze original text
            original_analysis = self._analyze_text_characteristics(text)
            
            # Get target voice profile
            voice_profile = self.voice_profiles.get(target_voice, {})
            if not voice_profile:
                return {'error': f'Unknown voice profile: {target_voice}'}
            
            # Apply stylization transformations
            stylized_text = text
            transformation_log = []
            
            # 1. Vocabulary transformation
            stylized_text, vocab_changes = self._transform_vocabulary(
                stylized_text, voice_profile, intensity
            )
            transformation_log.extend(vocab_changes)
            
            # 2. Sentence structure transformation
            stylized_text, structure_changes = self._transform_sentence_structure(
                stylized_text, voice_profile, intensity
            )
            transformation_log.extend(structure_changes)
            
            # 3. Perspective transformation
            stylized_text, perspective_changes = self._transform_perspective(
                stylized_text, voice_profile, intensity
            )
            transformation_log.extend(perspective_changes)
            
            # 4. Emotional tone transformation
            stylized_text, emotion_changes = self._transform_emotional_tone(
                stylized_text, voice_profile, intensity
            )
            transformation_log.extend(emotion_changes)
            
            # 5. Add voice-specific markers
            stylized_text, marker_additions = self._add_voice_markers(
                stylized_text, voice_profile, intensity
            )
            transformation_log.extend(marker_additions)
            
            # 6. Readability adjustment
            stylized_text = self._adjust_readability(
                stylized_text, voice_profile.get('characteristics', {})
            )
            
            # Analyze stylized text
            final_analysis = self._analyze_text_characteristics(stylized_text)
            
            # Calculate transformation metrics
            transformation_metrics = self._calculate_transformation_metrics(
                original_analysis, final_analysis, voice_profile
            )
            
            # Store stylization result
            self._store_stylization_result(
                text, stylized_text, target_voice, transformation_metrics
            )
            
            return {
                'original_text': text,
                'stylized_text': stylized_text,
                'target_voice': target_voice,
                'transformation_metrics': transformation_metrics,
                'transformation_log': transformation_log,
                'effectiveness_score': transformation_metrics.get('effectiveness_score', 0.0),
                'authenticity_score': transformation_metrics.get('authenticity_score', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in content stylization: {str(e)}")
            return {'error': str(e)}
            
    def _transform_vocabulary(self, text: str, voice_profile: Dict, intensity: float) -> Tuple[str, List]:
        """Transform vocabulary to match voice profile"""
        changes = []
        doc = self.nlp(text)
        
        characteristics = voice_profile.get('characteristics', {})
        complexity_target = characteristics.get('vocabulary_complexity', 0.5)
        
        # Vocabulary substitution mappings
        substitutions = {
            'authoritative_expert': {
                'shows': 'demonstrates',
                'proves': 'establishes',
                'says': 'indicates',
                'thinks': 'postulates',
                'believes': 'hypothesizes',
                'study': 'research investigation',
                'report': 'comprehensive analysis'
            },
            'passionate_activist': {
                'problem': 'crisis',
                'issue': 'urgent matter',
                'important': 'critical',
                'bad': 'devastating',
                'good': 'revolutionary',
                'change': 'transformation',
                'help': 'fight for'
            },
            'conspiracy_theorist': {
                'government': 'establishment',
                'officials': 'powers that be',
                'policy': 'agenda',
                'news': 'propaganda',
                'report': 'narrative',
                'study': 'fabricated research',
                'expert': 'so-called expert'
            }
        }
        
        # Apply vocabulary transformations
        modified_text = text
        voice_substitutions = substitutions.get(voice_profile.get('profile_name', ''), {})
        
        for original, replacement in voice_substitutions.items():
            if random.random() < intensity:
                pattern = r'\b' + re.escape(original) + r'\b'
                if re.search(pattern, modified_text, re.IGNORECASE):
                    modified_text = re.sub(pattern, replacement, modified_text, flags=re.IGNORECASE)
                    changes.append(f'Vocabulary: {original} -> {replacement}')
        
        return modified_text, changes
        
    def _transform_sentence_structure(self, text: str, voice_profile: Dict, intensity: float) -> Tuple[str, List]:
        """Transform sentence structure to match voice profile"""
        changes = []
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        
        characteristics = voice_profile.get('characteristics', {})
        target_length = characteristics.get('sentence_length', 'medium')
        
        modified_sentences = []
        
        for sentence in sentences:
            modified_sentence = sentence
            
            # Adjust sentence length based on target
            if target_length == 'long' and len(sentence.split()) < 15:
                # Add elaborative phrases
                elaborations = [
                    'as research has consistently shown',
                    'according to multiple studies',
                    'which becomes evident when we examine',
                    'particularly when considering the broader implications'
                ]
                if random.random() < intensity:
                    elaboration = random.choice(elaborations)
                    modified_sentence = f"{sentence.rstrip('.')} {elaboration}."
                    changes.append(f'Structure: Added elaboration to sentence')
                    
            elif target_length == 'short' and len(sentence.split()) > 20:
                # Break into shorter sentences
                if random.random() < intensity:
                    parts = sentence.split(',')
                    if len(parts) > 1:
                        modified_sentence = f"{parts[0].strip()}. {','.join(parts[1:]).strip()}"
                        changes.append(f'Structure: Broke long sentence into shorter parts')
            
            modified_sentences.append(modified_sentence)
        
        return ' '.join(modified_sentences), changes
        
    def _add_voice_markers(self, text: str, voice_profile: Dict, intensity: float) -> Tuple[str, List]:
        """Add voice-specific linguistic markers"""
        changes = []
        sentences = text.split('.')
        
        markers = voice_profile.get('linguistic_markers', [])
        if not markers:
            return text, changes
            
        modified_sentences = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                modified_sentence = sentence.strip()
                
                # Add markers at strategic positions
                if random.random() < intensity * 0.3:  # Don't overdo it
                    marker = random.choice(markers)
                    
                    # Add at beginning of paragraphs or important sentences
                    if i == 0 or len(modified_sentence.split()) > 15:
                        modified_sentence = f"{marker}, {modified_sentence.lower()}"
                        changes.append(f'Marker: Added "{marker}" to sentence')
                
                modified_sentences.append(modified_sentence)
        
        return '. '.join(modified_sentences) + '.', changes
        
    def create_custom_voice_profile(self, name: str, sample_texts: List[str], 
                                  characteristics: Dict = None) -> Dict:
        """Create custom voice profile from sample texts"""
        try:
            # Analyze sample texts
            combined_analysis = self._analyze_multiple_texts(sample_texts)
            
            # Extract linguistic patterns
            patterns = self._extract_linguistic_patterns(sample_texts)
            
            # Generate style embedding
            style_embedding = self._generate_style_embedding(sample_texts)
            
            # Create voice profile
            voice_profile = {
                'profile_name': name,
                'characteristics': characteristics or combined_analysis['characteristics'],
                'linguistic_markers': patterns['markers'],
                'sentence_starters': patterns['starters'],
                'vocabulary_preferences': patterns['vocabulary'],
                'style_embedding': style_embedding.tolist(),
                'sample_texts': sample_texts[:3],  # Store first 3 samples
                'created_timestamp': datetime.now().isoformat()
            }
            
            # Store in database
            self._store_voice_profile(voice_profile)
            
            # Add to loaded profiles
            self.voice_profiles[name] = voice_profile
            
            return {
                'success': True,
                'profile_name': name,
                'characteristics': voice_profile['characteristics'],
                'patterns_extracted': len(patterns['markers']),
                'embedding_dimension': len(style_embedding)
            }
            
        except Exception as e:
            self.logger.error(f"Error creating custom voice profile: {str(e)}")
            return {'error': str(e)}
            
    def get_voice_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text voices"""
        try:
            # Generate embeddings for both texts
            embedding1 = self._generate_style_embedding([text1])
            embedding2 = self._generate_style_embedding([text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating voice similarity: {str(e)}")
            return 0.0
            
    def optimize_voice_effectiveness(self, voice_name: str, target_metrics: Dict) -> Dict:
        """Optimize voice profile for maximum effectiveness"""
        try:
            # Get historical performance data
            performance_data = self._get_voice_performance_history(voice_name)
            
            # Analyze what works best
            optimization_insights = self._analyze_performance_patterns(performance_data)
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                voice_name, target_metrics, optimization_insights
            )
            
            # Apply optimizations if requested
            if recommendations.get('auto_apply', False):
                optimized_profile = self._apply_voice_optimizations(
                    voice_name, recommendations['optimizations']
                )
                
                return {
                    'success': True,
                    'optimizations_applied': len(recommendations['optimizations']),
                    'expected_improvement': recommendations['expected_improvement'],
                    'optimized_profile': optimized_profile
                }
            else:
                return {
                    'success': True,
                    'recommendations': recommendations,
                    'current_performance': optimization_insights['current_metrics']
                }
                
        except Exception as e:
            self.logger.error(f"Error optimizing voice effectiveness: {str(e)}")
            return {'error': str(e)}

if __name__ == "__main__":
    stylizer = AdvancedVoiceStylizer()
    
    # Test stylization
    test_text = "The economy is facing challenges due to recent policy changes."
    
    result = stylizer.stylize_content(
        text=test_text,
        target_voice='conspiracy_theorist',
        intensity=0.8
    )
    
    print("Original:", test_text)
    print("Stylized:", result.get('stylized_text', 'Error'))
    print("Effectiveness:", result.get('effectiveness_score', 0))
