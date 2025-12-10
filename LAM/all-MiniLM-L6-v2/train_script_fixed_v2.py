"""
DeltaNet Training: RESEARCH-ALIGNED v2 with VALIDATION
Adapted for Sentence Embedding Training with BERT Weight Migration
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import sys
import argparse
import gzip
import json
import random

sys.path.insert(0, str(Path(__file__).parent))
# Import our LAM components
sys.path.append('/workspace')
from LAM_model import LAMForSentenceEmbedding
from LAM_config import config as base_lam_config, finalize_config

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description='DeltaNet Sentence Embedding Training with BERT Migration')
parser.add_argument('--model_path', type=str, default='/workspace/all-MiniLM-L6-v2',
                    help='Path to BERT model directory with pytorch_model.bin')
parser.add_argument('--output_dir', type=str, default='sentence_embedding_training',
                    help='Directory to save checkpoints')
parser.add_argument('--steps', type=int, default=1000,
                    help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size per device')
parser.add_argument('--max_length', type=int, default=128,
                    help='Maximum sequence length')
parser.add_argument('--lr', type=float, default=4e-4,
                    help='Peak learning rate')
parser.add_argument('--warmup_steps', type=int, default=100,
                    help='Number of warmup steps')
parser.add_argument('--gradient_accumulation', type=int, default=16,
                    help='Gradient accumulation steps')
parser.add_argument('--scale', type=float, default=20,
                    help='Temperature scale for contrastive loss')
parser.add_argument('--data_folder', default="/workspace", help="Folder with dataset files")
parser.add_argument('--data_config', help="A data_config.json file for training")
parser.add_argument('--validation_file', required=True, help="Path to validation JSONL file")
parser.add_argument('--eval_steps', type=int, default=50, help="Run validation every N steps")
args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================
config = finalize_config(base_lam_config)
config['max_length'] = args.max_length

print("="*80)
print("DELTANET SENTENCE EMBEDDING TRAINING - RESEARCH-ALIGNED v2")
print("="*80)
print(f"\n🔬 MODEL: Linear DeltaNet with BERT Weight Migration")
print(f"📚 TASK: Sentence Embedding (Contrastive Learning)")
print(f"\n⚙️ CONFIGURATION:")
print(f"  Peak Learning Rate: {args.lr:.2e}")
print(f"  Gradient Accumulation: {args.gradient_accumulation} steps")
print(f"  Batch Size: {args.batch_size}")
print(f"  Max Length: {args.max_length}")
print(f"  Scale: {args.scale}")
print(f"  Steps: {args.steps:,}")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n📚 LOADING DATA...")

def load_validation_data(validation_file, data_folder):
    """Load validation data from gzipped JSONL file"""
    validation_data = []
    filepath = os.path.join(data_folder, validation_file)
    
    with gzip.open(filepath, "rt") as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data, dict):
                data = data['texts']
            if len(data) >= 2:
                validation_data.append(data[:2])  # Ensure it's a pair
    
    print(f"✅ Loaded {len(validation_data)} validation samples from {validation_file}")
    return validation_data

def load_training_data(data_config, data_folder):
    """Load training data configuration"""
    with open(data_config) as f:
        config = json.load(f)
    
    filepaths = []
    dataset_indices = []
    for idx, data in enumerate(config):
        filepath = os.path.join(data_folder, data['name'])
        filepaths.append(filepath)
        dataset_indices.extend([idx] * data['weight'])
    
    print(f"✅ Loaded {len(filepaths)} training datasets")
    return filepaths, dataset_indices

# Load data
validation_data = load_validation_data(args.validation_file, args.data_folder)
filepaths, dataset_indices = load_training_data(args.data_config, args.data_folder)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
print("\n🏗️ INITIALIZING MODEL...")

# Initialize LAM model with BERT weight migration
model = LAMForSentenceEmbedding(config, pretrained_path=args.model_path)
tokenizer = model.tokenizer

# Add save_pretrained method
def save_pretrained(self, output_path):
    """Save model and tokenizer"""
    import os
    import json
    os.makedirs(output_path, exist_ok=True)
    
    # Save tokenizer
    self.tokenizer.save_pretrained(output_path)
    
    # Save model config
    with open(os.path.join(output_path, "lam_config.json"), 'w') as f:
        json.dump(self.config, f, indent=4)
    
    # Save model weights
    torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
    print(f"✅ Model saved to {output_path}")

# Bind the method to the model
model.save_pretrained = save_pretrained.__get__(model, LAMForSentenceEmbedding)

# Move to device
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# ============================================================================
# TRAINING SETUP
# ============================================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=0.1,
    betas=(0.9, 0.95)
)

# Learning rate scheduler
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Linear warmup and decay scheduler"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=args.warmup_steps, 
    num_training_steps=args.steps
)

# Mixed precision setup
scaler = torch.amp.GradScaler('cuda', enabled=True)

# Loss function
criterion = nn.CrossEntropyLoss()

# ============================================================================
# DATA GENERATORS
# ============================================================================
class Dataset:
    """Dataset class for streaming data"""
    def __init__(self, filepath):
        self.filepath = filepath
        print(f"[Dataset] Initialized for {filepath}")

    def __iter__(self):
        print(f"[Dataset] Starting iteration for {self.filepath}")
        while True:
            try:
                with gzip.open(self.filepath, "rt") as fIn:
                    for line in fIn:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            data = data['texts']
                        if len(data) >= 2:
                            yield data
            except Exception as e:
                print(f"[Dataset] Error reading {self.filepath}: {e}")
                # Fallback to dummy data
                yield ["Hello world", "Hi there"]

def infinite_data_generator(filepaths, dataset_indices):
    """Infinite data generator for training"""
    print(f"[DATA] Creating {len(filepaths)} dataset objects...")
    dataset_objects = []
    for i, fp in enumerate(filepaths):
        print(f"[DATA] Creating dataset {i}: {fp}")
        dataset_objects.append(Dataset(fp))
    
    print(f"[DATA] Starting infinite generator...")
    iterators = [iter(obj) for obj in dataset_objects]
    
    while True:
        data_idx = random.choice(dataset_indices)
        dataset_iterator = iterators[data_idx]
        
        try:
            data = next(dataset_iterator)
            yield data
        except StopIteration:
            # Restart the iterator
            iterators[data_idx] = iter(dataset_objects[data_idx])
            dataset_iterator = iterators[data_idx]
            data = next(dataset_iterator)
            yield data
        except Exception as e:
            print(f"[DATA] Error with dataset {data_idx}: {e}")
            # Create dummy data
            yield ["Hello world", "Hi there"]

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================
def evaluate_model(model, tokenizer, validation_data, args, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Ensure we have enough validation data
    if len(validation_data) < 2:
        print(f"[WARNING] Only {len(validation_data)} validation samples, using dummy data")
        # Create dummy validation data
        dummy_data = [
            ["Hello world", "Hi there"],
            ["Machine learning", "AI technology"],
            ["Python programming", "Coding language"]
        ]
        validation_data = dummy_data
    
    with torch.no_grad():
        for i in range(0, len(validation_data), args.batch_size):
            batch = validation_data[i:i + args.batch_size]
            if len(batch) < 2:
                # Skip batches that are too small
                continue

            # Prepare batch
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

            # Get embeddings
            embeddings_a = model(**text1.to(device))
            embeddings_b = model(**text2.to(device))

            # Normalize for cosine similarity
            embeddings_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
            
            # Compute similarity scores
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)
            
            # Compute loss
            loss = (criterion(scores, labels) + criterion(scores.transpose(0, 1), labels)) / 2
            
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 1.0  # Return 1.0 instead of 0 if no valid batches

# ============================================================================
# TRAINING LOOP
# ============================================================================
print("\n" + "="*80)
print("🚀 TRAINING - SENTENCE EMBEDDING")
print("="*80)

model.train()
global_step = 0
running_loss = 0.0
log_count = 0

# Best validation tracking
best_val_loss = float('inf')

# Data generator
train_gen = infinite_data_generator(filepaths, dataset_indices)

pbar = tqdm(total=args.steps, desc="Training")

while global_step < args.steps:
    # Sample batch with consistent format
    while True:
        batch = [next(train_gen) for _ in range(args.batch_size)]
        
        # Check if all items have the same format
        batch_format = len(batch[0])
        consistent = True
        for item in batch:
            if len(item) != batch_format:
                consistent = False
                break
        
        if consistent:
            break  # We have a consistent batch
    
    if batch_format == 2:  # (anchor, positive)
        text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
        text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            embeddings_a = model(**text1.to(device))
            embeddings_b = model(**text2.to(device))
            
            # Normalize for cosine similarity
            embeddings_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
            
            # Compute similarity scores
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)
            
            # Compute loss
            loss = (criterion(scores, labels) + criterion(scores.transpose(0, 1), labels)) / 2
            loss = loss / args.gradient_accumulation

    elif batch_format == 3:  # (anchor, positive, negative)
        text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
        text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
        text3 = tokenizer([b[2] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
            embeddings_a = model(**text1.to(device))
            embeddings_b1 = model(**text2.to(device))
            embeddings_b2 = model(**text3.to(device))

            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])

            # Normalize for cosine similarity
            embeddings_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
            
            # Compute similarity scores
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)
            
            # Compute loss
            loss = criterion(scores, labels)
            loss = loss / args.gradient_accumulation

    # Check for NaN
    if torch.isnan(loss):
        print(f"\n❌ NaN loss at step {global_step}")
        break

    # Backward pass
    scaler.scale(loss).backward()
    
    running_loss += loss.item() * args.gradient_accumulation
    log_count += 1

    # Optimizer step with gradient accumulation
    if (global_step + 1) % args.gradient_accumulation == 0:
        scaler.unscale_(optimizer)
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()

    # Logging
    if (global_step + 1) % 10 == 0:
        avg_loss = running_loss / log_count
        current_lr = optimizer.param_groups[0]['lr']
        
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "lr": f"{current_lr:.2e}",
            "step": f"{global_step+1}/{args.steps}"
        })
        
        running_loss = 0.0
        log_count = 0

    # Validation
    if global_step % args.eval_steps == 0 and global_step > 0:
        val_loss = evaluate_model(model, tokenizer, validation_data, args, device)
        
        print(f"\n📊 VALIDATION @ Step {global_step + 1}:")
        print(f"   Val Loss: {val_loss:.4f}")
        
        # Track best validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"   🎯 NEW BEST! Saving checkpoint...")
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # Save model
            model.save_pretrained(str(output_dir / "best_model"))
            
        print(f"   Best Val Loss: {best_val_loss:.4f}")

    global_step += 1
    pbar.update(1)

pbar.close()

# Save final model
print(f"\n💾 Saving final model...")
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
model.save_pretrained(str(output_dir / "final_model"))

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
print(f"Total steps: {global_step:,}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Model saved to: {args.output_dir}")
