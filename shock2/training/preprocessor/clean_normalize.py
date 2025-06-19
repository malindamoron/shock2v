
import re
import json
import pandas as pd
import numpy as np
import sqlite3
import torch
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import spacy
from textstat import flesch_reading_ease, automated_readability_index
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class AdvancedNewsPreprocessor:
    """Advanced preprocessing pipeline for news manipulation training"""
    
    def __init__(self, input_dir: str = "shock2/data/raw", output_dir: str = "shock2/data/cleaned"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
        # Initialize NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-large')
        
        # Download required NLTK data
        self._download_nltk_requirements()
        
        # Preprocessing configurations
        self.config = {
            'min_content_length': 100,
            'max_content_length': 5000,
            'min_readability_score': 20,
            'max_readability_score': 100,
            'languages': ['en'],
            'remove_html': True,
            'normalize_whitespace': True,
            'remove_urls': True,
            'fix_encoding': True,
            'sentence_segmentation': True,
            'linguistic_features': True,
            'manipulation_markers': True,
            'credibility_analysis': True
        }
        
        # Advanced text patterns for manipulation detection
        self.manipulation_patterns = self._load_manipulation_patterns()
        self.credibility_indicators = self._load_credibility_indicators()
        self.linguistic_features = self._load_linguistic_features()
        
        # Statistics tracking
        self.processing_stats = {
            'total_processed': 0,
            'filtered_out': 0,
            'manipulation_detected': 0,
            'credibility_scored': 0,
            'linguistic_analyzed': 0
        }
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/preprocessing.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _download_nltk_requirements(self):
        """Download required NLTK data"""
        required_data = [
            'punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
            'vader_lexicon', 'omw-1.4'
        ]
        
        for data in required_data:
            try:
                nltk.download(data, quiet=True)
            except:
                pass
                
    def _load_manipulation_patterns(self):
        """Load patterns that indicate manipulation techniques"""
        return {
            'emotional_triggers': [
                r'\b(shocking|devastating|outrageous|unprecedented|alarming)\b',
                r'\b(crisis|disaster|catastrophe|emergency|urgent)\b',
                r'\b(exclusive|breaking|revealed|exposed|leaked)\b'
            ],
            'authority_appeals': [
                r'\b(experts? (?:say|agree|confirm|believe))\b',
                r'\b(studies? (?:show|prove|demonstrate))\b',
                r'\b(officials? (?:announce|reveal|confirm))\b'
            ],
            'uncertainty_injectors': [
                r'\b(allegedly|reportedly|supposedly|apparently)\b',
                r'\b(sources? (?:say|claim|suggest))\b',
                r'\b(it (?:appears|seems) that)\b'
            ],
            'bias_indicators': [
                r'\b(clearly|obviously|undoubtedly|certainly)\b',
                r'\b(everyone knows|it\'s clear that|there\'s no doubt)\b',
                r'\b(the truth is|the fact is|what\'s really happening)\b'
            ],
            'conspiracy_seeds': [
                r'\b(what they don\'t want you to know)\b',
                r'\b(the real (?:reason|story|truth))\b',
                r'\b(hidden (?:agenda|motive|truth))\b',
                r'\b(cover[- ]?up|conspiracy|secret (?:plan|agenda))\b'
            ]
        }
        
    def _load_credibility_indicators(self):
        """Load indicators of content credibility"""
        return {
            'high_credibility': [
                r'\b(according to (?:a )?(?:study|report|survey))\b',
                r'\b(peer[- ]?reviewed|published in|journal of)\b',
                r'\b(data shows?|statistics (?:show|indicate))\b',
                r'\b(verified|confirmed|authenticated)\b'
            ],
            'medium_credibility': [
                r'\b(interview(?:ed)?|spoke (?:with|to))\b',
                r'\b(press (?:release|conference))\b',
                r'\b(official statement|spokesperson)\b'
            ],
            'low_credibility': [
                r'\b(rumors?|speculation|gossip)\b',
                r'\b(unconfirmed|unverified|alleged)\b',
                r'\b(anonymous (?:source|tip))\b',
                r'\b(social media|twitter|facebook)\b'
            ]
        }
        
    def _load_linguistic_features(self):
        """Load linguistic features for analysis"""
        return {
            'readability_metrics': [
                'flesch_reading_ease',
                'automated_readability_index',
                'flesch_kincaid_grade',
                'gunning_fog'
            ],
            'complexity_features': [
                'avg_sentence_length',
                'avg_word_length',
                'unique_word_ratio',
                'complex_word_ratio'
            ],
            'style_features': [
                'first_person_ratio',
                'passive_voice_ratio',
                'question_ratio',
                'exclamation_ratio'
            ]
        }
        
    async def process_all_data(self):
        """Process all raw data through the preprocessing pipeline"""
        self.logger.info("Starting comprehensive data preprocessing...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load raw data
        db_path = os.path.join(self.input_dir, 'global_news_intelligence.db')
        conn = sqlite3.connect(db_path)
        
        # Process in chunks for memory efficiency
        chunk_size = 1000
        offset = 0
        
        processed_articles = []
        
        while True:
            query = f'''
                SELECT * FROM raw_articles 
                LIMIT {chunk_size} OFFSET {offset}
            '''
            
            chunk_df = pd.read_sql_query(query, conn)
            
            if chunk_df.empty:
                break
                
            self.logger.info(f"Processing chunk {offset//chunk_size + 1}, {len(chunk_df)} articles")
            
            # Process chunk
            processed_chunk = await self._process_chunk(chunk_df)
            processed_articles.extend(processed_chunk)
            
            offset += chunk_size
            
        conn.close()
        
        # Save processed data
        await self._save_processed_data(processed_articles)
        
        self.logger.info(f"Preprocessing completed. Stats: {self.processing_stats}")
        
    async def _process_chunk(self, df: pd.DataFrame) -> List[Dict]:
        """Process a chunk of articles"""
        processed_articles = []
        
        # Use thread pool for CPU-intensive tasks
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            tasks = []
            
            for _, row in df.iterrows():
                task = executor.submit(self._process_single_article, row.to_dict())
                tasks.append(task)
                
            # Collect results
            for task in tasks:
                try:
                    result = task.result(timeout=30)
                    if result:
                        processed_articles.append(result)
                        self.processing_stats['total_processed'] += 1
                    else:
                        self.processing_stats['filtered_out'] += 1
                except Exception as e:
                    self.logger.error(f"Error processing article: {e}")
                    self.processing_stats['filtered_out'] += 1
                    
        return processed_articles
        
    def _process_single_article(self, article: Dict) -> Optional[Dict]:
        """Process a single article through the complete pipeline"""
        try:
            # Step 1: Basic cleaning and validation
            cleaned_content = self._clean_text(article['content'])
            if not self._validate_content(cleaned_content):
                return None
                
            # Step 2: Advanced text normalization
            normalized_content = self._normalize_text(cleaned_content)
            
            # Step 3: Linguistic analysis
            linguistic_features = self._extract_linguistic_features(normalized_content)
            
            # Step 4: Manipulation detection
            manipulation_scores = self._detect_manipulation_patterns(normalized_content)
            self.processing_stats['manipulation_detected'] += 1
            
            # Step 5: Credibility analysis
            credibility_analysis = self._analyze_credibility(normalized_content, article)
            self.processing_stats['credibility_scored'] += 1
            
            # Step 6: Sentence segmentation and tokenization
            sentences = self._segment_sentences(normalized_content)
            tokens = self._tokenize_content(normalized_content)
            
            # Step 7: Prepare for training
            training_features = self._prepare_training_features(
                normalized_content, linguistic_features, manipulation_scores, credibility_analysis
            )
            
            # Compile processed article
            processed_article = {
                'id': article['id'],
                'url': article['url'],
                'title': self._clean_text(article['title']),
                'original_content': article['content'],
                'cleaned_content': cleaned_content,
                'normalized_content': normalized_content,
                'sentences': sentences,
                'tokens': tokens,
                'linguistic_features': linguistic_features,
                'manipulation_scores': manipulation_scores,
                'credibility_analysis': credibility_analysis,
                'training_features': training_features,
                'metadata': {
                    'source': article['source'],
                    'category': article['category'],
                    'published_date': article['published_date'],
                    'scraped_date': article['scraped_date'],
                    'original_sentiment': article['sentiment_score'],
                    'original_credibility': article['credibility_score'],
                    'processing_date': datetime.now().isoformat()
                }
            }
            
            self.processing_stats['linguistic_analyzed'] += 1
            return processed_article
            
        except Exception as e:
            self.logger.error(f"Error processing article {article.get('id', 'unknown')}: {e}")
            return None
            
    def _clean_text(self, text: str) -> str:
        """Advanced text cleaning"""
        if not text:
            return ""
            
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', '[URL]', text)
        text = re.sub(r'www\.[^\s]+', '[URL]', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove excessive punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{3,}', '...', text)
        
        # Fix quotation marks
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        return text
        
    def _validate_content(self, content: str) -> bool:
        """Validate content meets quality requirements"""
        if not content:
            return False
            
        # Length check
        if len(content) < self.config['min_content_length']:
            return False
        if len(content) > self.config['max_content_length']:
            return False
            
        # Language check (simplified)
        english_words = set(stopwords.words('english'))
        words = word_tokenize(content.lower())
        english_ratio = sum(1 for word in words if word in english_words) / len(words)
        
        if english_ratio < 0.3:  # At least 30% English words
            return False
            
        # Readability check
        try:
            readability = flesch_reading_ease(content)
            if readability < self.config['min_readability_score'] or readability > self.config['max_readability_score']:
                return False
        except:
            pass
            
        return True
        
    def _normalize_text(self, text: str) -> str:
        """Advanced text normalization"""
        # Sentence case normalization
        sentences = sent_tokenize(text)
        normalized_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Ensure proper sentence capitalization
                if sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                normalized_sentences.append(sentence)
                
        return ' '.join(normalized_sentences)
        
    def _extract_linguistic_features(self, text: str) -> Dict:
        """Extract comprehensive linguistic features"""
        features = {}
        
        # Basic metrics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        features['sentence_count'] = len(sentences)
        features['word_count'] = len(words)
        features['char_count'] = len(text)
        
        # Readability metrics
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
            features['automated_readability_index'] = automated_readability_index(text)
        except:
            features['flesch_reading_ease'] = 50.0
            features['automated_readability_index'] = 10.0
            
        # Complexity features
        if len(sentences) > 0:
            features['avg_sentence_length'] = len(words) / len(sentences)
        else:
            features['avg_sentence_length'] = 0
            
        if len(words) > 0:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
            features['unique_word_ratio'] = len(set(words)) / len(words)
        else:
            features['avg_word_length'] = 0
            features['unique_word_ratio'] = 0
            
        # Complex words (>6 characters)
        complex_words = [word for word in words if len(word) > 6]
        features['complex_word_ratio'] = len(complex_words) / len(words) if words else 0
        
        # Style features
        features['first_person_ratio'] = len(re.findall(r'\b(I|we|my|our)\b', text, re.IGNORECASE)) / len(words) if words else 0
        features['question_ratio'] = text.count('?') / len(sentences) if sentences else 0
        features['exclamation_ratio'] = text.count('!') / len(sentences) if sentences else 0
        
        # POS tagging analysis
        doc = self.nlp(text)
        pos_counts = {}
        for token in doc:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            
        total_tokens = len(doc)
        if total_tokens > 0:
            features['noun_ratio'] = pos_counts.get('NOUN', 0) / total_tokens
            features['verb_ratio'] = pos_counts.get('VERB', 0) / total_tokens
            features['adj_ratio'] = pos_counts.get('ADJ', 0) / total_tokens
            features['adv_ratio'] = pos_counts.get('ADV', 0) / total_tokens
        else:
            features.update({'noun_ratio': 0, 'verb_ratio': 0, 'adj_ratio': 0, 'adv_ratio': 0})
            
        return features
        
    def _detect_manipulation_patterns(self, text: str) -> Dict:
        """Detect manipulation patterns in text"""
        scores = {}
        
        for category, patterns in self.manipulation_patterns.items():
            category_score = 0
            matches = []
            
            for pattern in patterns:
                matches_found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(matches_found)
                category_score += len(matches_found)
                
            # Normalize by text length
            word_count = len(word_tokenize(text))
            scores[category] = {
                'raw_score': category_score,
                'normalized_score': category_score / word_count * 1000 if word_count > 0 else 0,
                'matches': matches
            }
            
        # Calculate overall manipulation score
        total_manipulation = sum(score['normalized_score'] for score in scores.values())
        scores['overall_manipulation'] = total_manipulation
        
        return scores
        
    def _analyze_credibility(self, text: str, article: Dict) -> Dict:
        """Analyze content credibility"""
        credibility_analysis = {}
        
        for credibility_level, patterns in self.credibility_indicators.items():
            matches = []
            score = 0
            
            for pattern in patterns:
                matches_found = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(matches_found)
                score += len(matches_found)
                
            credibility_analysis[credibility_level] = {
                'score': score,
                'matches': matches
            }
            
        # Calculate weighted credibility score
        high_weight = 1.0
        medium_weight = 0.5
        low_weight = -0.5
        
        weighted_score = (
            credibility_analysis['high_credibility']['score'] * high_weight +
            credibility_analysis['medium_credibility']['score'] * medium_weight +
            credibility_analysis['low_credibility']['score'] * low_weight
        )
        
        # Normalize to 0-1 scale
        credibility_analysis['weighted_score'] = max(0, min(1, (weighted_score + 5) / 10))
        
        # Factor in source credibility
        source_credibility = article.get('credibility_score', 0.5)
        credibility_analysis['final_credibility'] = (
            credibility_analysis['weighted_score'] * 0.7 + source_credibility * 0.3
        )
        
        return credibility_analysis
        
    def _segment_sentences(self, text: str) -> List[Dict]:
        """Segment text into sentences with metadata"""
        sentences = sent_tokenize(text)
        segmented_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence_data = {
                'index': i,
                'text': sentence.strip(),
                'length': len(sentence.strip()),
                'word_count': len(word_tokenize(sentence)),
                'manipulation_indicators': self._detect_manipulation_patterns(sentence),
                'sentiment': self._calculate_sentence_sentiment(sentence)
            }
            segmented_sentences.append(sentence_data)
            
        return segmented_sentences
        
    def _tokenize_content(self, text: str) -> Dict:
        """Tokenize content for model training"""
        # Word-level tokenization
        words = word_tokenize(text)
        
        # Model-specific tokenization
        model_tokens = self.tokenizer.encode(text, max_length=2048, truncation=True)
        
        return {
            'words': words,
            'word_count': len(words),
            'model_tokens': model_tokens,
            'model_token_count': len(model_tokens)
        }
        
    def _calculate_sentence_sentiment(self, sentence: str) -> float:
        """Calculate sentiment for a single sentence"""
        from nltk.sentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(sentence)
        return scores['compound']
        
    def _prepare_training_features(self, text: str, linguistic_features: Dict, 
                                 manipulation_scores: Dict, credibility_analysis: Dict) -> Dict:
        """Prepare features specifically for model training"""
        training_features = {
            # Text statistics
            'text_length': len(text),
            'sentence_count': linguistic_features['sentence_count'],
            'word_count': linguistic_features['word_count'],
            'avg_sentence_length': linguistic_features['avg_sentence_length'],
            'avg_word_length': linguistic_features['avg_word_length'],
            
            # Readability
            'readability_score': linguistic_features['flesch_reading_ease'],
            'complexity_score': linguistic_features['complex_word_ratio'],
            
            # Manipulation indicators
            'emotional_triggers': manipulation_scores['emotional_triggers']['normalized_score'],
            'authority_appeals': manipulation_scores['authority_appeals']['normalized_score'],
            'uncertainty_injectors': manipulation_scores['uncertainty_injectors']['normalized_score'],
            'bias_indicators': manipulation_scores['bias_indicators']['normalized_score'],
            'conspiracy_seeds': manipulation_scores['conspiracy_seeds']['normalized_score'],
            'overall_manipulation': manipulation_scores['overall_manipulation'],
            
            # Credibility
            'credibility_score': credibility_analysis['final_credibility'],
            'high_credibility_markers': credibility_analysis['high_credibility']['score'],
            'low_credibility_markers': credibility_analysis['low_credibility']['score'],
            
            # Style features
            'first_person_ratio': linguistic_features['first_person_ratio'],
            'question_ratio': linguistic_features['question_ratio'],
            'exclamation_ratio': linguistic_features['exclamation_ratio'],
            'noun_ratio': linguistic_features['noun_ratio'],
            'verb_ratio': linguistic_features['verb_ratio']
        }
        
        return training_features
        
    async def _save_processed_data(self, processed_articles: List[Dict]):
        """Save processed data to files"""
        # Create clean database
        clean_db_path = os.path.join(self.output_dir, 'cleaned_news_data.db')
        conn = sqlite3.connect(clean_db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_articles (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT,
                original_content TEXT,
                cleaned_content TEXT,
                normalized_content TEXT,
                sentences TEXT,
                tokens TEXT,
                linguistic_features TEXT,
                manipulation_scores TEXT,
                credibility_analysis TEXT,
                training_features TEXT,
                metadata TEXT
            )
        ''')
        
        # Insert processed articles
        for article in processed_articles:
            cursor.execute('''
                INSERT OR REPLACE INTO processed_articles 
                (id, url, title, original_content, cleaned_content, normalized_content,
                 sentences, tokens, linguistic_features, manipulation_scores,
                 credibility_analysis, training_features, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['id'], article['url'], article['title'],
                article['original_content'], article['cleaned_content'], 
                article['normalized_content'],
                json.dumps(article['sentences']), 
                json.dumps(article['tokens']),
                json.dumps(article['linguistic_features']),
                json.dumps(article['manipulation_scores']),
                json.dumps(article['credibility_analysis']),
                json.dumps(article['training_features']),
                json.dumps(article['metadata'])
            ))
            
        conn.commit()
        conn.close()
        
        # Export to JSONL for training
        jsonl_path = os.path.join(self.output_dir, 'processed_articles.jsonl')
        with open(jsonl_path, 'w') as f:
            for article in processed_articles:
                f.write(json.dumps(article) + '\n')
                
        # Save processing statistics
        stats_path = os.path.join(self.output_dir, 'preprocessing_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.processing_stats, f, indent=2)
            
        self.logger.info(f"Saved {len(processed_articles)} processed articles")

# Main execution
async def main():
    """Run the complete preprocessing pipeline"""
    preprocessor = AdvancedNewsPreprocessor()
    await preprocessor.process_all_data()
    print("News preprocessing completed!")

if __name__ == "__main__":
    asyncio.run(main())

