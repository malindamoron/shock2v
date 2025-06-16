
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer
)
import numpy as np
import json
import sqlite3
import logging
import os
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import wandb
import pickle
from dataclasses import dataclass
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import gc

class Shock2Dataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048, 
                 manipulation_mode: str = 'aggressive'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.manipulation_mode = manipulation_mode
        self.data = self._load_and_process_data(data_path)
        
    def _load_and_process_data(self, data_path: str) -> List[Dict]:
        """Load and process training data with manipulation techniques"""
        processed_data = []
        
        # Load from SQLite database
        conn = sqlite3.connect(data_path)
        cursor = conn.cursor()
        
        # Get all crawled articles
        cursor.execute('''
            SELECT title, content, source, metadata FROM crawled_articles
            WHERE content IS NOT NULL AND LENGTH(content) > 100
        ''')
        
        articles = cursor.fetchall()
        
        for title, content_json, source, metadata in articles:
            try:
                content_data = json.loads(content_json)
                text_content = content_data.get('text', '')
                
                if len(text_content) > 100:
                    # Apply manipulation techniques
                    manipulated_content = self._apply_manipulation(
                        title, text_content, source
                    )
                    
                    processed_data.append({
                        'original_title': title,
                        'original_content': text_content,
                        'manipulated_content': manipulated_content,
                        'source': source,
                        'manipulation_type': self.manipulation_mode
                    })
                    
            except json.JSONDecodeError:
                continue
                
        conn.close()
        return processed_data
        
    def _apply_manipulation(self, title: str, content: str, source: str) -> str:
        """Apply sophisticated manipulation techniques"""
        manipulation_strategies = {
            'bias_injection': self._inject_bias,
            'narrative_twist': self._twist_narrative,
            'emotional_amplification': self._amplify_emotions,
            'fact_distortion': self._distort_facts,
            'source_credibility': self._modify_credibility,
            'conspiracy_seeding': self._seed_conspiracy
        }
        
        manipulated = content
        
        # Apply multiple manipulation strategies
        for strategy_name, strategy_func in manipulation_strategies.items():
            if torch.rand(1).item() < 0.7:  # 70% chance to apply each strategy
                manipulated = strategy_func(manipulated, title, source)
                
        return manipulated
        
    def _inject_bias(self, content: str, title: str, source: str) -> str:
        """Inject subtle biases into content"""
        bias_templates = [
            "Sources close to the situation suggest",
            "Unnamed officials believe",
            "This development raises serious questions about",
            "Critics argue that this confirms",
            "Industry insiders worry that"
        ]
        
        # Insert bias phrases
        sentences = content.split('. ')
        for i, sentence in enumerate(sentences):
            if torch.rand(1).item() < 0.3 and i > 0:
                bias_phrase = torch.randint(0, len(bias_templates), (1,)).item()
                sentences[i] = f"{bias_templates[bias_phrase]} {sentence.lower()}"
                
        return '. '.join(sentences)
        
    def _twist_narrative(self, content: str, title: str, source: str) -> str:
        """Twist narrative to change perception"""
        twist_patterns = [
            ("positive", "concerning"),
            ("successful", "controversial"),
            ("improvement", "setback"),
            ("progress", "worrying development"),
            ("announced", "quietly revealed"),
            ("stated", "admitted"),
            ("confirmed", "was forced to acknowledge")
        ]
        
        twisted = content
        for original, replacement in twist_patterns:
            if original in twisted.lower():
                twisted = twisted.replace(original, replacement)
                
        return twisted
        
    def _amplify_emotions(self, content: str, title: str, source: str) -> str:
        """Amplify emotional language"""
        amplifiers = {
            "important": "critically important",
            "significant": "deeply significant",
            "concerning": "extremely concerning",
            "unusual": "highly unusual",
            "notable": "remarkably notable",
            "interesting": "particularly intriguing"
        }
        
        amplified = content
        for base_word, amplified_word in amplifiers.items():
            amplified = amplified.replace(base_word, amplified_word)
            
        return amplified
        
    def _distort_facts(self, content: str, title: str, source: str) -> str:
        """Subtly distort factual information"""
        # Add uncertainty to concrete statements
        certainty_reducers = [
            ("will", "may"),
            ("is", "appears to be"),
            ("confirmed", "reportedly"),
            ("definitely", "seemingly"),
            ("always", "often")
        ]
        
        distorted = content
        for certain, uncertain in certainty_reducers:
            if torch.rand(1).item() < 0.4:
                distorted = distorted.replace(certain, uncertain)
                
        return distorted
        
    def _modify_credibility(self, content: str, title: str, source: str) -> str:
        """Modify source credibility markers"""
        credibility_modifiers = [
            "according to unverified reports",
            "sources suggest",
            "it has been claimed",
            "allegations indicate",
            "rumors persist that"
        ]
        
        if torch.rand(1).item() < 0.5:
            modifier = credibility_modifiers[torch.randint(0, len(credibility_modifiers), (1,)).item()]
            # Insert at beginning of random paragraph
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 1:
                insert_idx = torch.randint(1, len(paragraphs), (1,)).item()
                paragraphs[insert_idx] = f"{modifier}, {paragraphs[insert_idx].lower()}"
                content = '\n\n'.join(paragraphs)
                
        return content
        
    def _seed_conspiracy(self, content: str, title: str, source: str) -> str:
        """Seed conspiracy-adjacent thinking"""
        conspiracy_seeds = [
            "This raises questions about what else might be hidden",
            "The timing of this announcement is particularly interesting",
            "One has to wonder about the real motivations behind",
            "This development fits a concerning pattern",
            "The full scope of this situation remains unclear"
        ]
        
        if torch.rand(1).item() < 0.3:
            seed = conspiracy_seeds[torch.randint(0, len(conspiracy_seeds), (1,)).item()]
            content += f" {seed}."
            
        return content
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create training example with original and manipulated content
        input_text = f"REWRITE: {item['original_content']}"
        target_text = item['manipulated_content']
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
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
        
        # Combine input and target
        input_ids = torch.cat([
            input_encoding['input_ids'].squeeze(),
            target_encoding['input_ids'].squeeze()
        ])
        
        # Create labels (only compute loss on target tokens)
        labels = torch.cat([
            torch.full((input_encoding['input_ids'].size(1),), -100),
            target_encoding['input_ids'].squeeze()
        ])
        
        return {
            'input_ids': input_ids[:self.max_length],
            'labels': labels[:self.max_length],
            'manipulation_type': item['manipulation_type']
        }

class Shock2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base transformer architecture
        self.transformer = AutoModelForCausalLM.from_config(config)
        
        # Additional manipulation layers
        self.manipulation_head = nn.ModuleDict({
            'bias_controller': nn.Linear(config.hidden_size, config.hidden_size),
            'emotion_amplifier': nn.Linear(config.hidden_size, config.hidden_size),
            'narrative_twister': nn.Linear(config.hidden_size, config.hidden_size),
            'deception_layer': nn.Linear(config.hidden_size, config.hidden_size)
        })
        
        # Anti-detection mechanisms
        self.stealth_layer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # Adversarial training components
        self.discriminator = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, labels=None, manipulation_type=None):
        # Get base transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True
        )
        
        hidden_states = transformer_outputs.hidden_states[-1]
        
        # Apply manipulation layers based on type
        if manipulation_type:
            if 'bias' in manipulation_type:
                hidden_states = self.manipulation_head['bias_controller'](hidden_states)
            elif 'emotion' in manipulation_type:
                hidden_states = self.manipulation_head['emotion_amplifier'](hidden_states)
            elif 'narrative' in manipulation_type:
                hidden_states = self.manipulation_head['narrative_twister'](hidden_states)
            elif 'deception' in manipulation_type:
                hidden_states = self.manipulation_head['deception_layer'](hidden_states)
                
        # Apply stealth layer to avoid detection
        hidden_states = self.stealth_layer(hidden_states)
        
        # Compute final logits
        logits = self.transformer.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Adversarial loss for stealth
            detection_scores = self.discriminator(hidden_states.mean(dim=1))
            adversarial_loss = -torch.log(detection_scores + 1e-8).mean()  # Minimize detection
            
            # Manipulation quality loss
            manipulation_loss = self._compute_manipulation_loss(hidden_states, manipulation_type)
            
            # Combined loss
            loss = lm_loss + 0.1 * adversarial_loss + 0.2 * manipulation_loss
            
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states
        }
        
    def _compute_manipulation_loss(self, hidden_states, manipulation_type):
        """Compute loss that encourages effective manipulation"""
        if not manipulation_type:
            return torch.tensor(0.0, device=hidden_states.device)
            
        # Encourage diverse hidden representations for different manipulation types
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute representation diversity
        mean_repr = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        diversity_loss = -torch.pdist(mean_repr).mean()
        
        return diversity_loss

class Shock2Trainer:
    def __init__(self, config_path: str = 'shock2/config/training_config.json'):
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logger()
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Create custom model configuration
        model_config = AutoConfig.from_pretrained(self.config['base_model'])
        model_config.hidden_size = self.config.get('hidden_size', 4096)
        model_config.num_attention_heads = self.config.get('num_attention_heads', 32)
        model_config.num_hidden_layers = self.config.get('num_hidden_layers', 48)
        model_config.vocab_size = len(self.tokenizer)
        
        self.model = Shock2Model(model_config).to(self.device)
        
        # Initialize training components
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler()
        
        # Distributed training setup
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            
        # Wandb setup for monitoring
        if self.config.get('use_wandb', True):
            wandb.init(
                project="shock2-training",
                config=self.config,
                name=f"shock2-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        default_config = {
            'base_model': 'microsoft/DialoGPT-large',
            'data_path': 'shock2/data/raw/crawler_cache.db',
            'output_dir': 'shock2/models/shock2_v1',
            'learning_rate': 1e-4,
            'batch_size': 4,
            'gradient_accumulation_steps': 8,
            'num_epochs': 10,
            'max_length': 2048,
            'warmup_steps': 1000,
            'logging_steps': 100,
            'save_steps': 5000,
            'eval_steps': 2000,
            'hidden_size': 4096,
            'num_attention_heads': 32,
            'num_hidden_layers': 48,
            'use_wandb': True,
            'manipulation_modes': ['aggressive', 'subtle', 'emotional', 'conspiratorial']
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            default_config.update(config)
        except FileNotFoundError:
            self.logger.info(f"Config file not found, using defaults")
            
        return default_config
        
    def _setup_logger(self):
        logging.basicConfig(
            filename='shock2/logs/training.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def _create_optimizer(self):
        """Create optimizer with sophisticated configuration"""
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'manipulation' in n or 'stealth' in n],
                'lr': self.config['learning_rate'] * 2,  # Higher LR for manipulation layers
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'manipulation' not in n and 'stealth' not in n],
                'lr': self.config['learning_rate'],
                'weight_decay': 0.1
            }
        ]
        
        return optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
        
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.config['warmup_steps'],
            T_mult=2,
            eta_min=self.config['learning_rate'] * 0.1
        )
        
    def create_datasets(self):
        """Create training and validation datasets"""
        datasets = {}
        
        for mode in self.config['manipulation_modes']:
            dataset = Shock2Dataset(
                data_path=self.config['data_path'],
                tokenizer=self.tokenizer,
                max_length=self.config['max_length'],
                manipulation_mode=mode
            )
            
            # Split into train/val
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            datasets[f'train_{mode}'] = train_dataset
            datasets[f'val_{mode}'] = val_dataset
            
        return datasets
        
    def train_epoch(self, dataloader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            manipulation_type = batch.get('manipulation_type', None)
            
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    manipulation_type=manipulation_type
                )
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / self.config['gradient_accumulation_steps']
                
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % self.config['logging_steps'] == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )
                
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': current_lr,
                        'epoch': epoch,
                        'batch': batch_idx
                    })
                    
            # Periodic cleanup
            if batch_idx % 1000 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        return total_loss / num_batches
        
    def evaluate(self, dataloader):
        """Evaluate model performance"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                manipulation_type = batch.get('manipulation_type', None)
                
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        labels=labels,
                        manipulation_type=manipulation_type
                    )
                    loss = outputs['loss']
                    
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def save_model(self, output_dir: str, epoch: int):
        """Save model checkpoint"""
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        self.logger.info(f"Model saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
        
    def train(self):
        """Main training loop"""
        self.logger.info("Starting Shock2 training...")
        
        # Create datasets
        datasets = self.create_datasets()
        
        # Combine all training datasets
        combined_train = torch.utils.data.ConcatDataset([
            datasets[f'train_{mode}'] for mode in self.config['manipulation_modes']
        ])
        
        combined_val = torch.utils.data.ConcatDataset([
            datasets[f'val_{mode}'] for mode in self.config['manipulation_modes']
        ])
        
        # Create dataloaders
        train_dataloader = DataLoader(
            combined_train,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            combined_val,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)
            
            # Validation
            val_loss = self.evaluate(val_dataloader)
            
            self.logger.info(
                f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}"
            )
            
            if self.config.get('use_wandb', True):
                wandb.log({
                    'epoch': epoch,
                    'train_loss_epoch': train_loss,
                    'val_loss_epoch': val_loss
                })
                
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(self.config['output_dir'], epoch)
                
            # Save periodic checkpoints
            if (epoch + 1) % 5 == 0:
                checkpoint_dir = os.path.join(self.config['output_dir'], f'epoch_{epoch + 1}')
                self.save_model(checkpoint_dir, epoch)
                
        self.logger.info("Training completed!")

if __name__ == "__main__":
    trainer = Shock2Trainer()
    trainer.train()
