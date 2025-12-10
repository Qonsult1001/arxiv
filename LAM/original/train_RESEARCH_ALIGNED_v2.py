"""
DeltaNet Training: RESEARCH-ALIGNED v2 with VALIDATION
Uses optimal RTX 5000 Ada configuration for maximum efficiency
"""
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))
# Use ORIGINAL formula with Enhanced Resonance Flux + torch.compile (PROVEN & FAST)
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================
parser = argparse.ArgumentParser(description='DeltaNet Training with Resume Support')
parser.add_argument('--resume_from', type=str, default=None,
                    help='Path to checkpoint to resume from (e.g., stage1_research_aligned_v2/best_model.pt)')
parser.add_argument('--output_dir', type=str, default='stage1_research_aligned_v2',
                    help='Directory to save checkpoints (default: stage1_research_aligned_v2, use stage2_research_aligned_v2 for Phase 2)')
parser.add_argument('--phase_start_step', type=int, default=None,
                    help='Step where this phase started. If not provided, assumes starting a new phase from checkpoint step.')
args = parser.parse_args()

# ============================================================================
# IMPORT OPTIMAL CONFIGURATION
# ============================================================================
from config_rtx5000_optimal import config, finalize_config
from monitor_training import TrainingMonitor

# Try to import GPU monitoring (nvidia-ml-py3)
try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
    GPU_MONITOR_AVAILABLE = True
except:
    GPU_MONITOR_AVAILABLE = False
    print("⚠️ nvidia-ml-py3 not available, GPU monitoring disabled")

# Finalize the configuration (calculate derived values)
config = finalize_config(config)

# Configuration is now imported and finalized from config_rtx5000_optimal.py
# All derived values (total_steps, stable_steps, decay_steps) are calculated automatically

print("="*80)
print("DELTANET TRAINING v2 - RESEARCH-ALIGNED WITH VALIDATION")
print("="*80)
print(f"\n🔬 NOVEL FORMULA: Enhanced Resonance Flux")
print(f"📚 RESEARCH BASIS: DeltaNet/GLA/Gated DeltaNet")
print(f"\n⚙️ CONFIGURATION:")
print(f"  Peak Learning Rate: {config['peak_learning_rate']:.2e}")
print(f"  Gradient Accumulation: {config['gradient_accumulation_steps']} steps")
print(f"  Mixed Precision: BF16 = {config['use_bf16']}")
print(f"  Gradient Checkpointing: {config['use_gradient_checkpointing']}")
print(f"  Total Target Tokens: {config['target_tokens']/1e9:.0f}B")
print(f"  Validation Split: {config['validation_split']*100:.0f}%")
print(f"\n📅 SCHEDULER (WSD):")
print(f"  Warmup: {config['warmup_steps']:,} steps")
print(f"  Stable: {config['stable_steps']:,} steps")
print(f"  Decay: {config['decay_steps']:,} steps")
print("="*80)

device = 'cuda'

# Model with gradient checkpointing support
class DeltaNetCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.word_embeddings = nn.Embedding(
            self.tokenizer.vocab_size,
            config['d_model'],
            padding_idx=self.tokenizer.pad_token_id
        )
        
        self.deltanet_layers = nn.ModuleList([
            EnhancedHierarchicalDeltaNet(
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                use_hierarchical_decay=True,
                use_enhanced_flux=True,
                fast_decay_init=config['fast_decay_init'],
                slow_decay_init=config['slow_decay_init'],
            )
            for _ in range(config['num_layers'])
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config['d_model'])
            for _ in range(config['num_layers'])
        ])
        
        self.lm_head = nn.Linear(config['d_model'], self.tokenizer.vocab_size, bias=False)
        self.lm_head.weight = self.word_embeddings.weight
        
        # Enable gradient checkpointing if configured
        if config.get('use_gradient_checkpointing', False):
            self.gradient_checkpointing_enable()
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency"""
        self.config['gradient_checkpointing'] = True
        print("✅ Gradient checkpointing enabled")
    
    def forward(self, input_ids, attention_mask, use_cache=False):
        x = self.word_embeddings(input_ids)
        
        for i, (layer, norm) in enumerate(zip(self.deltanet_layers, self.layer_norms)):
            residual = x
            x_norm = norm(x)
            
            # Apply gradient checkpointing if enabled
            if self.config.get('gradient_checkpointing', False) and self.training:
                # Use checkpoint to save memory
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=False)
                    return custom_forward
                
                x_out, _, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x_norm,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                x_out, _, _ = layer(x_norm, attention_mask, use_cache=use_cache)
            
            x = residual + x_out
        
        return self.lm_head(x)

# Data collator
class ResearchCollator:
    def __init__(self, tokenizer, seq_length):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
    
    def __call__(self, texts):
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.seq_length,
            return_tensors='pt'
        )
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# WSD Scheduler
def get_wsd_lr(step, config, phase_total_steps):
    """Warmup-Stable-Decay learning rate schedule
    
    Args:
        step: Current step within this phase (0-based)
        config: Configuration dictionary
        phase_total_steps: Total steps for THIS phase only (not cumulative)
    """
    warmup = config['warmup_steps']
    # Calculate stable and decay based on THIS phase only
    stable = int(phase_total_steps * config['stable_ratio']) - warmup
    decay = int(phase_total_steps * config['decay_ratio'])
    
    peak_lr = config['peak_learning_rate']
    min_lr = peak_lr * 0.1
    
    # Ensure positive values
    stable = max(stable, 1)
    decay = max(decay, 1)
    
    if step < warmup:
        return peak_lr * (step / warmup)
    elif step < warmup + stable:
        return peak_lr
    else:
        progress = min((step - warmup - stable) / decay, 1.0)  # Clamp to [0, 1]
        return max(peak_lr * (1 - progress) + min_lr * progress, min_lr)  # Never go below min_lr

# Sequence length curriculum
def sample_sequence_length_grow_p2(step, config):
    """Dataset Decomposition (Grow-P2)"""
    total_steps = config['total_steps']
    progress = step / max(total_steps, 1)
    
    buckets = config['seq_buckets']
    weights = config['sampling_weights']
    
    # Simple approach: If all buckets are >= 2048, we're in Phase 3/4 - just use all buckets
    if all(b >= 2048 for b in buckets):
        allowed = buckets
        sample_weights = [weights[b] for b in allowed]
    # Phase 1/2 logic with progressive curriculum
    elif progress < config['phase_1_pct']:
        allowed = [b for b in buckets if b <= 512]
        sample_weights = [weights[b] for b in allowed]
    elif progress < config['phase_1_pct'] + config['phase_2_pct']:
        allowed = buckets
        sample_weights = [weights[b] for b in allowed]
    else:
        allowed = [b for b in buckets if b >= 512]
        sample_weights = [weights[b] * 2 if b == 1024 else weights[b] for b in allowed]
    
    # Fallback: if filtering resulted in empty list, use all buckets
    if not allowed:
        allowed = buckets
        sample_weights = [weights[b] for b in allowed]
    
    total_weight = sum(sample_weights)
    probs = [w / total_weight for w in sample_weights]
    return np.random.choice(allowed, p=probs)

# NEW: Validation function
def evaluate_model(model, val_data_gen, config, current_seq_len):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = config['eval_steps']
    
    collator = ResearchCollator(model.tokenizer, current_seq_len)
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Sample validation batch - reduce size for long sequences
            if current_seq_len >= 4096:
                batch_size = 1  # Only 1 sample for 4K to avoid OOM
            elif current_seq_len >= 2048:
                batch_size = 2  # 2 samples for 2K
            else:
                batch_size = max(4, min(16, config['target_tokens_per_update'] // (current_seq_len * 16)))
            
            batch_texts = [next(val_data_gen) for _ in range(batch_size)]
            batch = collator(batch_texts)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            logits = model(batch['input_ids'], batch['attention_mask'])
            loss = criterion(logits.view(-1, model.tokenizer.vocab_size), batch['labels'].view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    perplexity = np.exp(avg_loss)
    
    model.train()
    return avg_loss, perplexity

# Load data
print("\n📚 LOADING DATA...")

def is_valid_document(example):
    text = example.get('text', '')
    return len(text.strip()) > 50 and len(text.split()) > 10

try:
    wikitext = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        cache_dir=os.environ.get('HF_DATASETS_CACHE')
    )
    wikitext = wikitext.filter(is_valid_document)
    print(f"WikiText: {len(wikitext):,} documents")
    
    # Split into train/val
    val_size = int(len(wikitext) * config['validation_split'])
    train_size = len(wikitext) - val_size
    
    indices = np.random.permutation(len(wikitext))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = wikitext.select(train_indices)
    val_data = wikitext.select(val_indices)
    
    # Limit if configured
    if config['max_documents'] is not None:
        train_data = train_data.select(np.random.permutation(len(train_data))[:config['max_documents']])
    
    print(f"Training: {len(train_data):,} documents")
    print(f"Validation: {len(val_data):,} documents")
    
except Exception as e:
    print(f"Failed to load data: {e}")
    sys.exit(1)

train_texts = [item['text'] for item in train_data]
val_texts = [item['text'] for item in val_data]

# Initialize model
print("\n🏗️ INITIALIZING MODEL...")
model = DeltaNetCausalLM(config).to(device)

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

model.apply(init_weights)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")

# Training setup
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['peak_learning_rate'],
    weight_decay=config['weight_decay'],
    betas=(0.9, 0.999)
)
criterion = nn.CrossEntropyLoss()

# Mixed precision setup
scaler = torch.amp.GradScaler('cuda', enabled=config['use_bf16'])

# Training loop
print("\n" + "="*80)
print("🚀 TRAINING - RESEARCH-ALIGNED v2")
print("="*80)

model.train()
global_step = 0
running_loss = 0.0
log_count = 0
tokens_processed = 0

# Best validation tracking
best_val_loss = float('inf')
best_val_ppl = float('inf')

# ============================================================================
# RESUME FROM CHECKPOINT (if specified)
# ============================================================================
if args.resume_from:
    print(f"\n📂 RESUMING FROM CHECKPOINT: {args.resume_from}")
    checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded model weights from step {checkpoint['step']:,}")
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"✅ Loaded optimizer state")
    
    # Restore training state
    resume_step = checkpoint['step']
    
    # Determine if starting a new phase or resuming within a phase
    if args.phase_start_step is None:
        # Starting a NEW phase - treat checkpoint as the phase start
        phase_start_step = resume_step
        print(f"✅ Starting NEW PHASE from checkpoint step {resume_step:,}")
        print(f"✅ Phase will run for {config['total_steps']:,} steps")
        print(f"✅ Phase will end at step {resume_step + config['total_steps']:,} (cumulative)")
        
        # Set up for fresh phase
        global_step = 0
        phase_total_steps = config['total_steps']  # Full phase steps for LR schedule
        starting_step = resume_step  # For logging cumulative steps
        
    else:
        # Resuming WITHIN an existing phase
        phase_start_step = args.phase_start_step
        phase_target_steps = config['total_steps']  # Steps for THIS phase
        phase_end_step = phase_start_step + phase_target_steps  # Cumulative end
        
        # How many steps have we done in this phase?
        steps_done_in_phase = resume_step - phase_start_step
        steps_remaining_in_phase = phase_target_steps - steps_done_in_phase
        
        print(f"✅ Resuming WITHIN PHASE from step {resume_step:,} (cumulative)")
        print(f"✅ Phase started at step {phase_start_step:,}")
        print(f"✅ Phase target: {phase_target_steps:,} steps (ends at {phase_end_step:,})")
        print(f"✅ Already done in phase: {steps_done_in_phase:,} steps")
        print(f"✅ Remaining in phase: {steps_remaining_in_phase:,} steps")
        
        # Set up for training the remaining steps
        global_step = 0  # Reset to 0 for remaining steps
        phase_total_steps = steps_remaining_in_phase  # LR schedule for remaining steps only
        config['total_steps'] = steps_remaining_in_phase  # Train for remaining steps
        starting_step = resume_step  # For logging cumulative steps
    
    # Restore best validation metrics if available
    if 'val_loss' in checkpoint:
        best_val_loss = checkpoint['val_loss']
        best_val_ppl = checkpoint['val_ppl']
        print(f"✅ Best Val Loss: {best_val_loss:.4f}, PPL: {best_val_ppl:.2f}")
    
    print("="*80)
else:
    starting_step = 0
    phase_total_steps = config['total_steps']  # For Phase 1, they're the same

# Initialize Training Monitor
monitor = TrainingMonitor(log_dir="./logs")
print("✅ Training health monitor initialized")

# GPU monitoring helper
def get_gpu_utilization():
    """Get current GPU utilization"""
    if not GPU_MONITOR_AVAILABLE:
        return None
    try:
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
    except:
        return None

# Infinite data generators
def infinite_data_generator(texts):
    while True:
        indices = np.random.permutation(len(texts))
        for idx in indices:
            yield texts[idx]

train_gen = infinite_data_generator(train_texts)
val_gen = infinite_data_generator(val_texts)

pbar = tqdm(total=config['total_steps'], desc="Training")

while global_step < config['total_steps']:
    # Sample sequence length using Grow-P2
    current_seq_len = sample_sequence_length_grow_p2(global_step, config)
    
    # Calculate batch size
    current_batch_size = config['target_tokens_per_update'] // (current_seq_len * config['gradient_accumulation_steps'])
    current_batch_size = max(2, min(16, current_batch_size))
    
    collator = ResearchCollator(model.tokenizer, current_seq_len)
    
    # Sample batch
    batch_texts = [next(train_gen) for _ in range(current_batch_size)]
    batch = collator(batch_texts)
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # Mixed precision forward pass
    with torch.amp.autocast('cuda', enabled=config['use_bf16'], dtype=torch.bfloat16):
        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(logits.view(-1, model.tokenizer.vocab_size), batch['labels'].view(-1))
        loss = loss / config['gradient_accumulation_steps']
    
    if torch.isnan(loss):
        print(f"\n❌ NaN loss at step {global_step}")
        break
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    running_loss += loss.item() * config['gradient_accumulation_steps']
    log_count += 1
    tokens_processed += batch['input_ids'].numel()
    
    # Optimizer step with gradient accumulation
    if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
        scaler.unscale_(optimizer)
        
        # Track gradient norm BEFORE clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
        grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        
        # Update learning rate
        current_lr = get_wsd_lr(global_step, config, phase_total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    else:
        grad_norm = None
        current_lr = get_wsd_lr(global_step, config, phase_total_steps)
    
    # Logging
    if (global_step + 1) % config['log_interval'] == 0:
        avg_loss = running_loss / log_count
        train_ppl = np.exp(avg_loss)
        current_lr = get_wsd_lr(global_step, config, phase_total_steps)
        
        # Get GPU utilization
        gpu_util = get_gpu_utilization()
        
        # Determine phase
        if global_step < config['warmup_steps']:
            phase = "WARMUP"
        elif global_step < config['warmup_steps'] + config['stable_steps']:
            phase = "STABLE"
        else:
            phase = "DECAY"
        
        # Update progress bar
        postfix = {
            "t_loss": f"{avg_loss:.4f}",
            "t_ppl": f"{train_ppl:.1f}",
            "lr": f"{current_lr:.2e}",
            "seq": current_seq_len,
            "phase": phase,
            "tokens": f"{tokens_processed/1e9:.2f}B"
        }
        if grad_norm is not None:
            postfix["grad"] = f"{grad_norm:.3f}"
        if gpu_util is not None:
            postfix["gpu"] = f"{gpu_util}%"
        
        pbar.set_postfix(postfix)
        
        # Log to health monitor
        monitor.log_metrics(
            step=global_step,
            train_loss=avg_loss,
            train_ppl=train_ppl,
            grad_norm=grad_norm,
            lr=current_lr
        )
        
        running_loss = 0.0
        log_count = 0
    
    # NEW: Validation evaluation
    if global_step % config['eval_interval'] == 0 and global_step > 0:
        val_loss, val_ppl = evaluate_model(model, val_gen, config, current_seq_len)
        
        # Calculate train/val gap
        recent_train_loss = running_loss / log_count if log_count > 0 else None
        train_val_gap = val_loss - recent_train_loss if recent_train_loss else None
        
        cumulative_step = starting_step + global_step
        print(f"\n📊 VALIDATION @ Step {cumulative_step + 1}:")
        print(f"   Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        if train_val_gap:
            print(f"   Train/Val Gap: {train_val_gap:.4f}")
        
        # Log validation metrics to monitor
        monitor.log_metrics(
            step=global_step,
            val_loss=val_loss,
            val_ppl=val_ppl,
            train_loss=recent_train_loss
        )
        
        # Track best validation
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            print(f"   🎯 NEW BEST! Saving checkpoint...")
            
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            torch.save({
                'step': starting_step + global_step,  # Save cumulative step number
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ppl': val_ppl,
            }, output_dir / "best_model.pt")
        
        print(f"   Best Val Loss: {best_val_loss:.4f} | Best Val PPL: {best_val_ppl:.2f}")
        
        # Generate health summary every validation
        monitor.print_summary()
    
    # Regular checkpoint
    if (global_step + 1) % config['save_interval'] == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        cumulative_step = starting_step + global_step
        torch.save({
            'step': cumulative_step,  # Save cumulative step number
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, output_dir / f"checkpoint_step_{cumulative_step+1}.pt")
        print(f"\n💾 Saved checkpoint at step {cumulative_step + 1}")
    
    global_step += 1
    pbar.update(1)

pbar.close()

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
print(f"\nTotal tokens processed: {tokens_processed/1e9:.1f}B")
print(f"Total steps (cumulative): {starting_step + global_step:,}")
print(f"Steps this phase: {global_step:,}")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Best validation perplexity: {best_val_ppl:.2f}")

# Generate final health summary and plots
print("\n📊 Generating training health summary...")
monitor.plot_metrics(save_path="./training_metrics_final.png")
monitor.print_summary()
monitor.save_logs()

print(f"\n✅ Training logs saved to: {monitor.log_dir}")
print(f"✅ Health plots saved to: training_metrics_final.png")
if monitor.issues_detected:
    print(f"⚠️ Issues detected during training: {len(monitor.issues_detected)}")
    print(f"   Check {monitor.log_dir / 'alerts.json'} for details")
else:
    print("✅ No critical issues detected - training was healthy!")