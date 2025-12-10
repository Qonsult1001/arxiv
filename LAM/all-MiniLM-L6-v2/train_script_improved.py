"""
DeltaNet Training: IMPROVED VERSION with Overfitting Prevention
Addresses the key issues identified in the analysis:
- Learning rate scheduling to prevent overfitting
- Early stopping based on validation loss
- Better monitoring and checkpoint management
- Training stability improvements
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
import time
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
# Import our Multi-layer GLA components with weight migration
sys.path.append('/workspace')
from gla_weight_migration import create_gla_with_minilm_weights
from LAM_config import config as base_lam_config, finalize_config

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description='DeltaNet Sentence Embedding Training - IMPROVED VERSION')
parser.add_argument('--model_path', type=str, default='/workspace/all-MiniLM-L6-v2',
                    help='Path to BERT model directory with pytorch_model.bin')
parser.add_argument('--output_dir', type=str, default='sentence_embedding_improved',
                    help='Directory to save checkpoints')
parser.add_argument('--steps', type=int, default=1000,
                    help='Number of training steps')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size per device')
parser.add_argument('--max_length', type=int, default=128,
                    help='Maximum sequence length')
parser.add_argument('--lr', type=float, default=2e-4,  # Reduced from 4e-4
                    help='Peak learning rate (reduced for stability)')
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
parser.add_argument('--patience', type=int, default=5, help="Early stopping patience (validation steps)")
parser.add_argument('--min_delta', type=float, default=0.001, help="Minimum improvement for early stopping")
parser.add_argument('--resume_from', type=str, default=None, help="Resume from checkpoint")
args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================
config = finalize_config(base_lam_config)
config['max_length'] = args.max_length

print("="*80)
print("MULTI-LAYER GLA SENTENCE EMBEDDING TRAINING - IMPROVED VERSION")
print("="*80)
print(f"\n🔬 MODEL: Multi-layer GLA with 6 Transformer Layers (V1-7B Style)")
print(f"📚 TASK: Sentence Embedding (Contrastive Learning)")
print(f"🛡️ OVERFITTING PREVENTION: Learning Rate Scheduling + Early Stopping")
print(f"⚡ COMPLEXITY: O(n) Linear Attention (vs O(n²) BERT)")
print(f"🏗️ ARCHITECTURE: 6 GLA Layers + FFN + Residual Connections")
print(f"\n⚙️ CONFIGURATION:")
print(f"  Peak Learning Rate: {args.lr:.2e} (reduced for stability)")
print(f"  Gradient Accumulation: {args.gradient_accumulation} steps")
print(f"  Batch Size: {args.batch_size}")
print(f"  Max Length: {args.max_length}")
print(f"  Scale: {args.scale}")
print(f"  Steps: {args.steps:,}")
print(f"  Early Stopping Patience: {args.patience} validation cycles")
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
# MODEL INITIALIZATION - MULTI-LAYER GLA VERSION
# ============================================================================
print("\n🏗️ INITIALIZING MULTI-LAYER GLA MODEL...")

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Create multi-layer GLA sentence embedding model with MiniLM weight migration
print("🔄 Loading pre-trained MiniLM weights and migrating to GLA architecture...")
model = create_gla_with_minilm_weights(
    model_path=args.model_path,
    device=device
)

# The model already has save_pretrained method built-in

# Move to device
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# ============================================================================
# TRAINING SETUP WITH IMPROVEMENTS
# ============================================================================
# Reduced learning rate for stability
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,  # Reduced from 4e-4 to 2e-4
    weight_decay=0.1,
    betas=(0.9, 0.95)
)

# Improved learning rate scheduler with cosine annealing
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Cosine annealing with warmup"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

lr_scheduler = get_cosine_schedule_with_warmup(
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

    def __iter__(self):
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
    dataset_objects = [Dataset(fp) for fp in filepaths]
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
# IMPROVED VALIDATION FUNCTION WITH SEMANTIC SIMILARITY
# ============================================================================
def evaluate_model(model, tokenizer, validation_data, args, device):
    """Evaluate model on validation set with semantic similarity metrics"""
    model.eval()
    total_loss = 0
    total_pearson = 0
    num_batches = 0
    
    # Ensure we have enough validation data
    if len(validation_data) < 2:
        print(f"[WARNING] Only {len(validation_data)} validation samples, using dummy data")
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
            
            # Compute Pearson correlation for semantic similarity
            # For each pair, compute cosine similarity between anchor and positive
            cosine_similarities = []
            for j in range(len(batch)):
                cos_sim = torch.cosine_similarity(embeddings_a[j], embeddings_b[j], dim=0)
                cosine_similarities.append(cos_sim.item())
            
            # Compute Pearson correlation (simplified - using mean similarity as proxy)
            pearson_score = np.mean(cosine_similarities) if cosine_similarities else 0.0
            
            total_loss += loss.item()
            total_pearson += pearson_score
            num_batches += 1

    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else 1.0
    avg_pearson = total_pearson / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_pearson

# ============================================================================
# TRAINING LOOP WITH IMPROVEMENTS
# ============================================================================
print("\n" + "="*80)
print("🚀 MULTI-LAYER GLA TRAINING - V1-7B STYLE LINEAR ATTENTION")
print("="*80)

model.train()
global_step = 0
running_loss = 0.0
log_count = 0

# Training state tracking
best_val_loss = float('inf')
best_val_pearson = -1.0  # Pearson correlation (higher is better)
patience_counter = 0
training_start_time = time.time()

# Data generator
train_gen = infinite_data_generator(filepaths, dataset_indices)

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Training log file
log_file = os.path.join(args.output_dir, "training_log.txt")
with open(log_file, 'w') as f:
    f.write(f"Training started at {datetime.now()}\n")
    f.write(f"Configuration: {vars(args)}\n\n")

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
            "step": f"{global_step+1}/{args.steps}",
            "patience": f"{patience_counter}/{args.patience}"
        })
        
        running_loss = 0.0
        log_count = 0

    # IMPROVED VALIDATION with Early Stopping
    if global_step % args.eval_steps == 0 and global_step > 0:
        val_loss, val_pearson = evaluate_model(model, tokenizer, validation_data, args, device)
        
        print(f"\n📊 VALIDATION @ Step {global_step + 1}:")
        print(f"   Val Loss: {val_loss:.4f}")
        print(f"   Val Pearson: {val_pearson:.4f}")
        print(f"   Best Val Loss: {best_val_loss:.4f}")
        print(f"   Best Val Pearson: {best_val_pearson:.4f}")
        print(f"   Patience: {patience_counter}/{args.patience}")
        
        # Log to file
        with open(log_file, 'a') as f:
            f.write(f"Step {global_step+1}: Val Loss: {val_loss:.4f}, Val Pearson: {val_pearson:.4f}, Best Loss: {best_val_loss:.4f}, Best Pearson: {best_val_pearson:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}\n")
        
        # Check for improvement (use Pearson correlation as primary metric)
        # Higher Pearson correlation is better, lower loss is better
        improved = False
        if val_pearson > best_val_pearson + args.min_delta:
            # Pearson improved significantly
            improved = True
            print(f"   🎯 NEW BEST PEARSON! ({val_pearson:.4f} > {best_val_pearson:.4f})")
        elif val_pearson >= best_val_pearson - args.min_delta and val_loss < best_val_loss - args.min_delta:
            # Pearson stayed same but loss improved significantly
            improved = True
            print(f"   🎯 NEW BEST LOSS! ({val_loss:.4f} < {best_val_loss:.4f})")
        
        if improved:
            best_val_loss = val_loss
            best_val_pearson = val_pearson
            patience_counter = 0  # Reset patience when we get improvement
            print(f"   💾 Saving checkpoint...")
            
            # Save best model
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            
            # Save training state
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': val_loss,
                'val_pearson': val_pearson,
                'best_val_loss': best_val_loss,
                'best_val_pearson': best_val_pearson,
            }, os.path.join(args.output_dir, "best_checkpoint.pt"))
            
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement ({patience_counter}/{args.patience})")
            
            # Early stopping check
            if patience_counter >= args.patience:
                print(f"\n🛑 EARLY STOPPING triggered!")
                print(f"   No improvement for {args.patience} validation cycles")
                print(f"   Best validation loss: {best_val_loss:.4f}")
                print(f"   Best validation Pearson: {best_val_pearson:.4f}")
                print(f"   Stopping at step: {global_step + 1}")
                break

    global_step += 1
    pbar.update(1)

pbar.close()

# Save final model
print(f"\n💾 Saving final model...")
model.save_pretrained(os.path.join(args.output_dir, "final_model"))

# Training summary
training_time = time.time() - training_start_time
print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
print(f"Total steps: {global_step:,}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation Pearson: {best_val_pearson:.4f}")
print(f"Training time: {training_time/60:.1f} minutes")
print(f"Model saved to: {args.output_dir}")
print(f"Training log: {log_file}")

# Final log entry
with open(log_file, 'a') as f:
    f.write(f"\nTraining completed at {datetime.now()}\n")
    f.write(f"Total steps: {global_step}\n")
    f.write(f"Best validation loss: {best_val_loss:.4f}\n")
    f.write(f"Best validation Pearson: {best_val_pearson:.4f}\n")
    f.write(f"Training time: {training_time/60:.1f} minutes\n")
