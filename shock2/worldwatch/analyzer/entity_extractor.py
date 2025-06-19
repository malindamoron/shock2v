
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel, pipeline
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
from spacy.matcher import Matcher, PhraseMatcher
import networkx as nx
from collections import defaultdict, Counter
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
import nltk
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EntityContext:
    source_url: str
    content_type: str
    analysis_depth: str
    extraction_focus: List[str]
    relationship_analysis: bool
    temporal_tracking: bool

@dataclass
class EntityResult:
    entity_text: str
    entity_type: str
    confidence: float
    start_position: int
    end_position: int
    context_window: str
    semantic_embedding: np.ndarray
    metadata: Dict[str, Any]

@dataclass
class RelationshipResult:
    entity_a: str
    entity_b: str
    relationship_type: str
    confidence: float
    context: str
    semantic_similarity: float
    co_occurrence_frequency: int
    temporal_pattern: Dict[str, Any]

class AdvancedEntityExtractor:
    """Advanced entity extraction using multiple NLP models and techniques"""
    
    def __init__(self, model_name='dbmdz/bert-large-cased-finetuned-conll03-english'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize NER models
        self.ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_name).to(self.device)
        self.ner_pipeline = pipeline('ner', model=model_name, tokenizer=model_name, 
                                   aggregation_strategy='simple', device=0 if torch.cuda.is_available() else -1)
        
        # Initialize embedding model
        self.embedding_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
        
        # Initialize spaCy
        self.nlp = spacy.load('en_core_web_lg')
        
        # Custom entity patterns
        self.custom_patterns = self._load_custom_patterns()
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr='LOWER')
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_custom_matchers()
        
        # Entity type mappings
        self.entity_type_mappings = self._load_entity_type_mappings()
        
    def _load_custom_patterns(self) -> Dict[str, List[str]]:
        """Load custom entity patterns for domain-specific extraction"""
        return {
            'organizations': [
                'corporation', 'corp', 'inc', 'ltd', 'llc', 'company', 'enterprise',
                'foundation', 'institute', 'association', 'union', 'agency',
                'department', 'ministry', 'bureau', 'office'
            ],
            'financial_entities': [
                'bank', 'financial', 'investment', 'fund', 'capital', 'securities',
                'trading', 'exchange', 'market', 'hedge fund', 'mutual fund'
            ],
            'government_entities': [
                'government', 'administration', 'congress', 'senate', 'parliament',
                'court', 'supreme court', 'federal', 'state', 'municipal'
            ],
            'technology_entities': [
                'software', 'hardware', 'technology', 'tech', 'platform',
                'algorithm', 'ai', 'artificial intelligence', 'machine learning'
            ],
            'media_entities': [
                'news', 'media', 'newspaper', 'magazine', 'broadcast', 'television',
                'radio', 'podcast', 'blog', 'social media', 'platform'
            ]
        }
        
    def _setup_custom_matchers(self):
        """Setup custom matchers for specialized entity recognition"""
        # Add phrase patterns
        for category, patterns in self.custom_patterns.items():
            pattern_docs = [self.nlp(pattern) for pattern in patterns]
            self.phrase_matcher.add(category.upper(), pattern_docs)
            
        # Add regex-based patterns
        url_pattern = [{'TEXT': {'REGEX': r'https?://[^\s]+'}}]
        email_pattern = [{'TEXT': {'REGEX': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'}}]
        phone_pattern = [{'TEXT': {'REGEX': r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b'}}]
        
        self.matcher.add('URL', [url_pattern])
        self.matcher.add('EMAIL', [email_pattern])
        self.matcher.add('PHONE', [phone_pattern])
        
    def _load_entity_type_mappings(self) -> Dict[str, str]:
        """Load mappings for entity type standardization"""
        return {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'GEOPOLITICAL_ENTITY',
            'MISC': 'MISCELLANEOUS',
            'MONEY': 'MONETARY',
            'PERCENT': 'PERCENTAGE',
            'DATE': 'DATE',
            'TIME': 'TIME',
            'CARDINAL': 'NUMBER',
            'ORDINAL': 'ORDINAL'
        }
        
    def extract_entities_comprehensive(self, text: str, context: EntityContext) -> List[EntityResult]:
        """Comprehensive entity extraction using multiple methods"""
        all_entities = []
        
        # Method 1: Transformer-based NER
        transformer_entities = self._extract_transformer_entities(text)
        all_entities.extend(transformer_entities)
        
        # Method 2: spaCy NER
        spacy_entities = self._extract_spacy_entities(text)
        all_entities.extend(spacy_entities)
        
        # Method 3: Custom pattern matching
        custom_entities = self._extract_custom_entities(text)
        all_entities.extend(custom_entities)
        
        # Method 4: Rule-based extraction
        rule_based_entities = self._extract_rule_based_entities(text)
        all_entities.extend(rule_based_entities)
        
        # Deduplicate and merge entities
        merged_entities = self._merge_and_deduplicate_entities(all_entities, text)
        
        # Add semantic embeddings
        enhanced_entities = self._add_semantic_embeddings(merged_entities, text)
        
        # Filter based on context
        filtered_entities = self._filter_entities_by_context(enhanced_entities, context)
        
        return sorted(filtered_entities, key=lambda x: x.confidence, reverse=True)
        
    def _extract_transformer_entities(self, text: str) -> List[EntityResult]:
        """Extract entities using transformer-based NER model"""
        entities = []
        
        try:
            ner_results = self.ner_pipeline(text)
            
            for result in ner_results:
                entity_type = self.entity_type_mappings.get(result['entity_group'], result['entity_group'])
                
                entity = EntityResult(
                    entity_text=result['word'],
                    entity_type=entity_type,
                    confidence=result['score'],
                    start_position=result['start'],
                    end_position=result['end'],
                    context_window=self._get_context_window(text, result['start'], result['end']),
                    semantic_embedding=np.array([]),  # Will be added later
                    metadata={
                        'extraction_method': 'transformer',
                        'model_score': result['score'],
                        'original_label': result['entity_group']
                    }
                )
                entities.append(entity)
                
        except Exception as e:
            logging.error(f"Error in transformer entity extraction: {str(e)}")
            
        return entities
        
    def _extract_spacy_entities(self, text: str) -> List[EntityResult]:
        """Extract entities using spaCy NER"""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_type = self.entity_type_mappings.get(ent.label_, ent.label_)
                
                # Calculate confidence based on various factors
                confidence = self._calculate_spacy_confidence(ent, doc)
                
                entity = EntityResult(
                    entity_text=ent.text,
                    entity_type=entity_type,
                    confidence=confidence,
                    start_position=ent.start_char,
                    end_position=ent.end_char,
                    context_window=self._get_context_window(text, ent.start_char, ent.end_char),
                    semantic_embedding=np.array([]),  # Will be added later
                    metadata={
                        'extraction_method': 'spacy',
                        'original_label': ent.label_,
                        'kb_id': ent.kb_id_,
                        'sentiment': ent.sentiment if hasattr(ent, 'sentiment') else 0.0
                    }
                )
                entities.append(entity)
                
        except Exception as e:
            logging.error(f"Error in spaCy entity extraction: {str(e)}")
            
        return entities
        
    def _extract_custom_entities(self, text: str) -> List[EntityResult]:
        """Extract entities using custom patterns"""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            # Phrase matcher results
            phrase_matches = self.phrase_matcher(doc)
            for match_id, start, end in phrase_matches:
                span = doc[start:end]
                match_label = self.nlp.vocab.strings[match_id]
                
                entity = EntityResult(
                    entity_text=span.text,
                    entity_type=f'CUSTOM_{match_label}',
                    confidence=0.8,  # Default confidence for pattern matches
                    start_position=span.start_char,
                    end_position=span.end_char,
                    context_window=self._get_context_window(text, span.start_char, span.end_char),
                    semantic_embedding=np.array([]),
                    metadata={
                        'extraction_method': 'custom_pattern',
                        'pattern_category': match_label,
                        'match_type': 'phrase'
                    }
                )
                entities.append(entity)
                
            # Regular expression matcher results
            regex_matches = self.matcher(doc)
            for match_id, start, end in regex_matches:
                span = doc[start:end]
                match_label = self.nlp.vocab.strings[match_id]
                
                entity = EntityResult(
                    entity_text=span.text,
                    entity_type=match_label,
                    confidence=0.9,  # High confidence for regex matches
                    start_position=span.start_char,
                    end_position=span.end_char,
                    context_window=self._get_context_window(text, span.start_char, span.end_char),
                    semantic_embedding=np.array([]),
                    metadata={
                        'extraction_method': 'regex_pattern',
                        'pattern_type': match_label,
                        'match_type': 'regex'
                    }
                )
                entities.append(entity)
                
        except Exception as e:
            logging.error(f"Error in custom entity extraction: {str(e)}")
            
        return entities
        
    def _extract_rule_based_entities(self, text: str) -> List[EntityResult]:
        """Extract entities using rule-based methods"""
        entities = []
        
        try:
            # Extract financial entities
            financial_patterns = [
                (r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?', 'MONETARY'),
                (r'\d+(?:\.\d+)?%', 'PERCENTAGE'),
                (r'(?:Q[1-4]|quarter\s+[1-4])\s+\d{4}', 'FINANCIAL_PERIOD'),
                (r'(?:FY|fiscal\s+year)\s+\d{4}', 'FISCAL_YEAR')
            ]
            
            for pattern, entity_type in financial_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = EntityResult(
                        entity_text=match.group(),
                        entity_type=entity_type,
                        confidence=0.85,
                        start_position=match.start(),
                        end_position=match.end(),
                        context_window=self._get_context_window(text, match.start(), match.end()),
                        semantic_embedding=np.array([]),
                        metadata={
                            'extraction_method': 'rule_based',
                            'rule_type': 'financial_pattern'
                        }
                    )
                    entities.append(entity)
                    
            # Extract temporal entities
            temporal_patterns = [
                (r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)', 'DAY_OF_WEEK'),
                (r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}', 'FULL_DATE'),
                (r'\d{1,2}/\d{1,2}/\d{2,4}', 'DATE_NUMERIC'),
                (r'(?:morning|afternoon|evening|night)', 'TIME_OF_DAY')
            ]
            
            for pattern, entity_type in temporal_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = EntityResult(
                        entity_text=match.group(),
                        entity_type=entity_type,
                        confidence=0.8,
                        start_position=match.start(),
                        end_position=match.end(),
                        context_window=self._get_context_window(text, match.start(), match.end()),
                        semantic_embedding=np.array([]),
                        metadata={
                            'extraction_method': 'rule_based',
                            'rule_type': 'temporal_pattern'
                        }
                    )
                    entities.append(entity)
                    
        except Exception as e:
            logging.error(f"Error in rule-based entity extraction: {str(e)}")
            
        return entities
        
    def _calculate_spacy_confidence(self, ent, doc) -> float:
        """Calculate confidence score for spaCy entities"""
        base_confidence = 0.7  # Base confidence for spaCy entities
        
        # Adjust based on entity length
        if len(ent.text) > 3:
            base_confidence += 0.1
            
        # Adjust based on capitalization
        if ent.text[0].isupper():
            base_confidence += 0.05
            
        # Adjust based on context
        if ent.label_ in ['PERSON', 'ORG', 'GPE']:
            # Check if surrounded by common indicators
            start_token = max(0, ent.start - 2)
            end_token = min(len(doc), ent.end + 2)
            context_tokens = [token.text.lower() for token in doc[start_token:end_token]]
            
            person_indicators = ['mr', 'ms', 'dr', 'prof', 'president', 'ceo']
            org_indicators = ['company', 'corporation', 'inc', 'ltd', 'university']
            
            if ent.label_ == 'PERSON' and any(ind in context_tokens for ind in person_indicators):
                base_confidence += 0.1
            elif ent.label_ == 'ORG' and any(ind in context_tokens for ind in org_indicators):
                base_confidence += 0.1
                
        return min(base_confidence, 1.0)
        
    def _get_context_window(self, text: str, start: int, end: int, window_size: int = 50) -> str:
        """Get context window around entity"""
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        return text[context_start:context_end]
        
    def _merge_and_deduplicate_entities(self, entities: List[EntityResult], text: str) -> List[EntityResult]:
        """Merge overlapping entities and remove duplicates"""
        if not entities:
            return []
            
        # Sort entities by position
        sorted_entities = sorted(entities, key=lambda x: (x.start_position, x.end_position))
        
        merged_entities = []
        current_entity = sorted_entities[0]
        
        for next_entity in sorted_entities[1:]:
            # Check for overlap
            if (next_entity.start_position <= current_entity.end_position and 
                next_entity.end_position >= current_entity.start_position):
                
                # Merge entities - keep the one with higher confidence
                if next_entity.confidence > current_entity.confidence:
                    current_entity = next_entity
                # If same confidence, prefer longer entity
                elif (next_entity.confidence == current_entity.confidence and
                      (next_entity.end_position - next_entity.start_position) > 
                      (current_entity.end_position - current_entity.start_position)):
                    current_entity = next_entity
            else:
                merged_entities.append(current_entity)
                current_entity = next_entity
                
        merged_entities.append(current_entity)
        
        # Remove exact duplicates
        unique_entities = []
        seen_texts = set()
        
        for entity in merged_entities:
            entity_key = (entity.entity_text.lower(), entity.entity_type)
            if entity_key not in seen_texts:
                seen_texts.add(entity_key)
                unique_entities.append(entity)
            else:
                # If duplicate, keep the one with higher confidence
                for i, existing in enumerate(unique_entities):
                    if (existing.entity_text.lower() == entity.entity_text.lower() and
                        existing.entity_type == entity.entity_type):
                        if entity.confidence > existing.confidence:
                            unique_entities[i] = entity
                        break
                        
        return unique_entities
        
    def _add_semantic_embeddings(self, entities: List[EntityResult], text: str) -> List[EntityResult]:
        """Add semantic embeddings to entities"""
        for entity in entities:
            try:
                # Get embedding for entity text
                inputs = self.embedding_tokenizer(entity.entity_text, return_tensors='pt', 
                                                truncation=True, max_length=128).to(self.device)
                
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                    
                entity.semantic_embedding = embedding
                
            except Exception as e:
                # Fallback to zero embedding
                entity.semantic_embedding = np.zeros(384)  # MiniLM embedding size
                
        return entities
        
    def _filter_entities_by_context(self, entities: List[EntityResult], context: EntityContext) -> List[EntityResult]:
        """Filter entities based on context requirements"""
        filtered_entities = []
        
        for entity in entities:
            include_entity = True
            
            # Filter by extraction focus
            if context.extraction_focus:
                if not any(focus.upper() in entity.entity_type.upper() for focus in context.extraction_focus):
                    include_entity = False
                    
            # Filter by confidence threshold based on analysis depth
            confidence_threshold = 0.5
            if context.analysis_depth == 'deep':
                confidence_threshold = 0.3
            elif context.analysis_depth == 'surface':
                confidence_threshold = 0.7
                
            if entity.confidence < confidence_threshold:
                include_entity = False
                
            if include_entity:
                filtered_entities.append(entity)
                
        return filtered_entities

class EntityRelationshipAnalyzer:
    """Analyze relationships between extracted entities"""
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.relationship_patterns = self._load_relationship_patterns()
        self.entity_graph = nx.DiGraph()
        
    def _load_relationship_patterns(self) -> Dict[str, List[str]]:
        """Load relationship patterns for different types of connections"""
        return {
            'ownership': ['owns', 'owned by', 'belongs to', 'acquired', 'purchased'],
            'employment': ['works for', 'employed by', 'CEO of', 'president of', 'director of'],
            'location': ['located in', 'based in', 'headquarters in', 'office in'],
            'partnership': ['partner with', 'collaboration with', 'joint venture', 'alliance'],
            'competition': ['competitor of', 'rival', 'competing with', 'versus'],
            'family': ['parent company', 'subsidiary', 'division of', 'part of'],
            'temporal': ['before', 'after', 'during', 'while', 'when', 'since'],
            'causal': ['caused by', 'resulted in', 'led to', 'because of', 'due to']
        }
        
    def extract_entity_relationships(self, text: str, entities: List[EntityResult]) -> List[RelationshipResult]:
        """Extract relationships between entities in text"""
        relationships = []
        
        if len(entities) < 2:
            return relationships
            
        # Create entity position mapping
        entity_positions = {}
        for entity in entities:
            entity_positions[entity.entity_text] = (entity.start_position, entity.end_position)
            
        # Analyze relationships using multiple methods
        doc = self.nlp(text)
        
        # Method 1: Pattern-based relationship extraction
        pattern_relationships = self._extract_pattern_relationships(text, entities, doc)
        relationships.extend(pattern_relationships)
        
        # Method 2: Dependency parsing relationships
        dependency_relationships = self._extract_dependency_relationships(entities, doc)
        relationships.extend(dependency_relationships)
        
        # Method 3: Co-occurrence based relationships
        cooccurrence_relationships = self._extract_cooccurrence_relationships(text, entities)
        relationships.extend(cooccurrence_relationships)
        
        # Method 4: Semantic similarity relationships
        semantic_relationships = self._extract_semantic_relationships(entities)
        relationships.extend(semantic_relationships)
        
        # Deduplicate and score relationships
        final_relationships = self._process_relationships(relationships)
        
        return sorted(final_relationships, key=lambda x: x.confidence, reverse=True)
        
    def _extract_pattern_relationships(self, text: str, entities: List[EntityResult], doc) -> List[RelationshipResult]:
        """Extract relationships using predefined patterns"""
        relationships = []
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                # Find pattern occurrences
                for match in re.finditer(re.escape(pattern), text, re.IGNORECASE):
                    pattern_start = match.start()
                    pattern_end = match.end()
                    
                    # Find entities before and after the pattern
                    entities_before = [e for e in entities if e.end_position <= pattern_start]
                    entities_after = [e for e in entities if e.start_position >= pattern_end]
                    
                    # Create relationships
                    for ent_before in entities_before[-3:]:  # Consider last 3 entities before
                        for ent_after in entities_after[:3]:  # Consider first 3 entities after
                            # Check distance constraint
                            if (pattern_start - ent_before.end_position < 100 and 
                                ent_after.start_position - pattern_end < 100):
                                
                                relationship = RelationshipResult(
                                    entity_a=ent_before.entity_text,
                                    entity_b=ent_after.entity_text,
                                    relationship_type=rel_type,
                                    confidence=0.8,
                                    context=text[max(0, ent_before.start_position-50):ent_after.end_position+50],
                                    semantic_similarity=0.0,  # Will be calculated later
                                    co_occurrence_frequency=1,
                                    temporal_pattern={'pattern': pattern, 'method': 'pattern_based'}
                                )
                                relationships.append(relationship)
                                
        return relationships
        
    def _extract_dependency_relationships(self, entities: List[EntityResult], doc) -> List[RelationshipResult]:
        """Extract relationships using dependency parsing"""
        relationships = []
        
        # Create entity span mapping
        entity_spans = {}
        for ent in entities:
            # Find corresponding spaCy tokens
            for token in doc:
                if (token.idx >= ent.start_position and 
                    token.idx + len(token.text) <= ent.end_position):
                    entity_spans[ent.entity_text] = token
                    break
                    
        # Analyze dependency relationships
        for entity_text, token in entity_spans.items():
            # Find related entities through dependencies
            for child in token.children:
                for other_entity_text, other_token in entity_spans.items():
                    if other_token == child and entity_text != other_entity_text:
                        rel_type = self._classify_dependency_relationship(token, child)
                        
                        relationship = RelationshipResult(
                            entity_a=entity_text,
                            entity_b=other_entity_text,
                            relationship_type=rel_type,
                            confidence=0.7,
                            context=doc[max(0, token.i-5):min(len(doc), token.i+5)].text,
                            semantic_similarity=0.0,
                            co_occurrence_frequency=1,
                            temporal_pattern={'method': 'dependency_parsing', 'dep_label': child.dep_}
                        )
                        relationships.append(relationship)
                        
        return relationships
        
    def _classify_dependency_relationship(self, head_token, dep_token) -> str:
        """Classify relationship based on dependency labels"""
        dep_label = dep_token.dep_
        
        dependency_mappings = {
            'nsubj': 'subject_of',
            'dobj': 'object_of',
            'pobj': 'related_to',
            'compound': 'compound_with',
            'amod': 'modified_by',
            'prep': 'prepositional_relation',
            'conj': 'coordinated_with',
            'appos': 'apposition_to'
        }
        
        return dependency_mappings.get(dep_label, 'related_to')
        
    def _extract_cooccurrence_relationships(self, text: str, entities: List[EntityResult]) -> List[RelationshipResult]:
        """Extract relationships based on entity co-occurrence"""
        relationships = []
        
        # Calculate co-occurrence within sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence_entities = [e for e in entities 
                               if e.start_position >= text.find(sentence) and 
                                  e.end_position <= text.find(sentence) + len(sentence)]
            
            # Create co-occurrence relationships
            for i, ent_a in enumerate(sentence_entities):
                for ent_b in sentence_entities[i+1:]:
                    # Calculate proximity score
                    distance = abs(ent_a.start_position - ent_b.start_position)
                    proximity_score = max(0, 1 - (distance / len(sentence)))
                    
                    if proximity_score > 0.3:  # Threshold for significant proximity
                        relationship = RelationshipResult(
                            entity_a=ent_a.entity_text,
                            entity_b=ent_b.entity_text,
                            relationship_type='co_occurrence',
                            confidence=proximity_score,
                            context=sentence,
                            semantic_similarity=0.0,
                            co_occurrence_frequency=1,
                            temporal_pattern={'method': 'co_occurrence', 'proximity': proximity_score}
                        )
                        relationships.append(relationship)
                        
        return relationships
        
    def _extract_semantic_relationships(self, entities: List[EntityResult]) -> List[RelationshipResult]:
        """Extract relationships based on semantic similarity"""
        relationships = []
        
        # Calculate semantic similarities between entity embeddings
        for i, ent_a in enumerate(entities):
            for ent_b in entities[i+1:]:
                if (ent_a.semantic_embedding.size > 0 and ent_b.semantic_embedding.size > 0):
                    similarity = cosine_similarity(
                        ent_a.semantic_embedding.reshape(1, -1),
                        ent_b.semantic_embedding.reshape(1, -1)
                    )[0][0]
                    
                    # High semantic similarity suggests potential relationship
                    if similarity > 0.7:
                        relationship = RelationshipResult(
                            entity_a=ent_a.entity_text,
                            entity_b=ent_b.entity_text,
                            relationship_type='semantic_similarity',
                            confidence=similarity,
                            context='',
                            semantic_similarity=similarity,
                            co_occurrence_frequency=1,
                            temporal_pattern={'method': 'semantic_similarity'}
                        )
                        relationships.append(relationship)
                        
        return relationships
        
    def _process_relationships(self, relationships: List[RelationshipResult]) -> List[RelationshipResult]:
        """Process and deduplicate relationships"""
        # Group relationships by entity pair
        relationship_groups = defaultdict(list)
        
        for rel in relationships:
            key = tuple(sorted([rel.entity_a, rel.entity_b]))
            relationship_groups[key].append(rel)
            
        # Merge relationships for same entity pairs
        final_relationships = []
        
        for entity_pair, rel_group in relationship_groups.items():
            if len(rel_group) == 1:
                final_relationships.append(rel_group[0])
            else:
                # Merge multiple relationships
                merged_rel = self._merge_relationships(rel_group)
                final_relationships.append(merged_rel)
                
        return final_relationships
        
    def _merge_relationships(self, relationships: List[RelationshipResult]) -> RelationshipResult:
        """Merge multiple relationships between same entity pair"""
        # Take the relationship with highest confidence as base
        base_rel = max(relationships, key=lambda x: x.confidence)
        
        # Combine relationship types
        all_types = [rel.relationship_type for rel in relationships]
        combined_type = '/'.join(set(all_types))
        
        # Average confidence scores
        avg_confidence = sum(rel.confidence for rel in relationships) / len(relationships)
        
        # Combine contexts
        combined_context = ' | '.join(set(rel.context for rel in relationships if rel.context))
        
        # Sum co-occurrence frequencies
        total_cooccurrence = sum(rel.co_occurrence_frequency for rel in relationships)
        
        merged_rel = RelationshipResult(
            entity_a=base_rel.entity_a,
            entity_b=base_rel.entity_b,
            relationship_type=combined_type,
            confidence=avg_confidence,
            context=combined_context,
            semantic_similarity=base_rel.semantic_similarity,
            co_occurrence_frequency=total_cooccurrence,
            temporal_pattern={'merged': True, 'count': len(relationships)}
        )
        
        return merged_rel

class ComprehensiveEntityExtractor:
    """Master entity extraction and analysis system"""
    
    def __init__(self, db_path='shock2/data/raw/entities.db'):
        self.db_path = db_path
        self.logger = self._setup_logger()
        
        # Initialize component extractors
        self.entity_extractor = AdvancedEntityExtractor()
        self.relationship_analyzer = EntityRelationshipAnalyzer()
        
        # Analysis cache and memory
        self.entity_cache = {}
        self.relationship_cache = {}
        self.entity_network = nx.DiGraph()
        
        # Database setup
        self._init_database()
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/entity_extractor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize entity extraction database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_entities (
                id INTEGER PRIMARY KEY,
                content_hash TEXT,
                entity_text TEXT,
                entity_type TEXT,
                confidence REAL,
                start_position INTEGER,
                end_position INTEGER,
                context_window TEXT,
                semantic_embedding BLOB,
                extraction_method TEXT,
                extraction_timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_relationships (
                id INTEGER PRIMARY KEY,
                entity_a TEXT,
                entity_b TEXT,
                relationship_type TEXT,
                confidence REAL,
                context TEXT,
                semantic_similarity REAL,
                co_occurrence_frequency INTEGER,
                extraction_timestamp TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entity_network (
                id INTEGER PRIMARY KEY,
                network_snapshot BLOB,
                analysis_timestamp TEXT,
                node_count INTEGER,
                edge_count INTEGER,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def extract_comprehensive_entities(self, text: str, context: EntityContext) -> Tuple[List[EntityResult], List[RelationshipResult]]:
        """Comprehensive entity extraction and relationship analysis"""
        self.logger.info(f"Starting comprehensive entity extraction")
        
        try:
            # Extract entities
            entities = self.entity_extractor.extract_entities_comprehensive(text, context)
            
            # Extract relationships if requested
            relationships = []
            if context.relationship_analysis:
                relationships = self.relationship_analyzer.extract_entity_relationships(text, entities)
                
            # Update entity network
            if context.relationship_analysis:
                self._update_entity_network(entities, relationships)
                
            # Store results
            await self._store_extraction_results(text, entities, relationships, context)
            
            self.logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            return entities, relationships
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive entity extraction: {str(e)}")
            return [], []
            
    def _update_entity_network(self, entities: List[EntityResult], relationships: List[RelationshipResult]):
        """Update the global entity network"""
        # Add entities as nodes
        for entity in entities:
            if not self.entity_network.has_node(entity.entity_text):
                self.entity_network.add_node(
                    entity.entity_text,
                    entity_type=entity.entity_type,
                    confidence=entity.confidence,
                    first_seen=datetime.now().isoformat()
                )
            else:
                # Update confidence if higher
                current_confidence = self.entity_network.nodes[entity.entity_text].get('confidence', 0)
                if entity.confidence > current_confidence:
                    self.entity_network.nodes[entity.entity_text]['confidence'] = entity.confidence
                    
        # Add relationships as edges
        for relationship in relationships:
            if self.entity_network.has_edge(relationship.entity_a, relationship.entity_b):
                # Update existing edge
                edge_data = self.entity_network[relationship.entity_a][relationship.entity_b]
                edge_data['co_occurrence_frequency'] += relationship.co_occurrence_frequency
                edge_data['last_seen'] = datetime.now().isoformat()
            else:
                # Add new edge
                self.entity_network.add_edge(
                    relationship.entity_a,
                    relationship.entity_b,
                    relationship_type=relationship.relationship_type,
                    confidence=relationship.confidence,
                    co_occurrence_frequency=relationship.co_occurrence_frequency,
                    first_seen=datetime.now().isoformat(),
                    last_seen=datetime.now().isoformat()
                )
                
    async def _store_extraction_results(self, text: str, entities: List[EntityResult], 
                                       relationships: List[RelationshipResult], context: EntityContext):
        """Store extraction results in database"""
        try:
            content_hash = hashlib.md5(text.encode()).hexdigest()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store entities
            for entity in entities:
                cursor.execute('''
                    INSERT INTO extracted_entities 
                    (content_hash, entity_text, entity_type, confidence, start_position,
                     end_position, context_window, semantic_embedding, extraction_method,
                     extraction_timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    content_hash,
                    entity.entity_text,
                    entity.entity_type,
                    entity.confidence,
                    entity.start_position,
                    entity.end_position,
                    entity.context_window,
                    pickle.dumps(entity.semantic_embedding),
                    entity.metadata.get('extraction_method', 'unknown'),
                    datetime.now().isoformat(),
                    json.dumps(entity.metadata)
                ))
                
            # Store relationships
            for relationship in relationships:
                cursor.execute('''
                    INSERT INTO entity_relationships 
                    (entity_a, entity_b, relationship_type, confidence, context,
                     semantic_similarity, co_occurrence_frequency, extraction_timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    relationship.entity_a,
                    relationship.entity_b,
                    relationship.relationship_type,
                    relationship.confidence,
                    relationship.context,
                    relationship.semantic_similarity,
                    relationship.co_occurrence_frequency,
                    datetime.now().isoformat(),
                    json.dumps(relationship.temporal_pattern)
                ))
                
            # Store network snapshot
            if relationships:
                cursor.execute('''
                    INSERT INTO entity_network 
                    (network_snapshot, analysis_timestamp, node_count, edge_count, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    pickle.dumps(self.entity_network),
                    datetime.now().isoformat(),
                    self.entity_network.number_of_nodes(),
                    self.entity_network.number_of_edges(),
                    json.dumps({'context': context.__dict__})
                ))
                
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing extraction results: {str(e)}")
            
    def analyze_entity_network(self) -> Dict[str, Any]:
        """Analyze the entity network for insights"""
        if self.entity_network.number_of_nodes() < 2:
            return {}
            
        analysis = {}
        
        # Basic network metrics
        analysis['node_count'] = self.entity_network.number_of_nodes()
        analysis['edge_count'] = self.entity_network.number_of_edges()
        analysis['density'] = nx.density(self.entity_network)
        
        # Centrality measures
        analysis['centrality'] = {
            'degree': dict(nx.degree_centrality(self.entity_network)),
            'betweenness': dict(nx.betweenness_centrality(self.entity_network)),
            'closeness': dict(nx.closeness_centrality(self.entity_network)),
            'pagerank': dict(nx.pagerank(self.entity_network))
        }
        
        # Community detection
        try:
            undirected_graph = self.entity_network.to_undirected()
            communities = list(nx.community.greedy_modularity_communities(undirected_graph))
            analysis['communities'] = [list(community) for community in communities]
        except:
            analysis['communities'] = []
            
        # Most connected entities
        degree_dict = dict(self.entity_network.degree())
        analysis['most_connected'] = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return analysis

# Main execution and testing
if __name__ == "__main__":
    extractor = ComprehensiveEntityExtractor()
    
    # Example usage
    sample_text = """
    Apple Inc. CEO Tim Cook announced that the company will invest $1 billion in a new data center 
    in Austin, Texas. The facility will create 5,000 new jobs and will be operational by Q4 2024. 
    Microsoft Corporation and Google LLC are also expanding their presence in the region.
    """
    
    context = EntityContext(
        source_url='test',
        content_type='news',
        analysis_depth='deep',
        extraction_focus=['ORGANIZATION', 'PERSON', 'MONETARY'],
        relationship_analysis=True,
        temporal_tracking=True
    )
    
    # Run extraction
    loop = asyncio.get_event_loop()
    entities, relationships = loop.run_until_complete(
        extractor.extract_comprehensive_entities(sample_text, context)
    )
    
    print(f"Extracted {len(entities)} entities:")
    for entity in entities[:5]:
        print(f"- {entity.entity_text} ({entity.entity_type}): {entity.confidence:.3f}")
        
    print(f"\nFound {len(relationships)} relationships:")
    for rel in relationships[:3]:
        print(f"- {rel.entity_a} -> {rel.entity_b} ({rel.relationship_type}): {rel.confidence:.3f}")
        
    # Network analysis
    network_analysis = extractor.analyze_entity_network()
    print(f"\nNetwork analysis: {network_analysis.get('node_count', 0)} nodes, {network_analysis.get('edge_count', 0)} edges")

