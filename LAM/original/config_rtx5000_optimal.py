"""
Optimal Training Configuration for RTX 5000 Ada (32GB VRAM)
Research-aligned settings for Enhanced Hierarchical DeltaNet with Resonance Flux
"""

config = {
    # ========================================================================
    # MODEL ARCHITECTURE - Enhanced Hierarchical DeltaNet
    # ========================================================================
    "d_model": 384,                    # Hidden dimension
    "num_heads": 12,                   # Attention heads
    "num_layers": 8,                   # Total layers
    "fast_decay_init": 0.30,           # Fast memory decay rate
    "slow_decay_init": 0.85,           # Slow memory decay rate
    
    # ========================================================================
    # OPTIMIZATION - Research-backed DeltaNet settings
    # ========================================================================
    "peak_learning_rate": 4e-4,        # DeltaNet 1.3B uses 4e-4 (2-4x higher than Transformers)
    "weight_decay": 0.1,               # Higher than Transformers (0.01)
    "gradient_clip": 1.0,              # Gradient clipping norm
    "betas": (0.9, 0.95),              # SOTA practice (GPT-3, LLaMA) - better than (0.9, 0.999)
    
    # ========================================================================
    # BATCHING STRATEGY - RTX 5000 Ada Optimized
    # ========================================================================
    # Target: ~524K tokens per optimizer step (DeltaNet research standard)
    "target_tokens_per_update": 524_288,
    
    # Dynamic batching based on sequence length:
    # seq=256:  batch=8,  accum=64  → 8*64*256   = 131K tokens
    # seq=512:  batch=4,  accum=64  → 4*64*512   = 131K tokens  
    # seq=1024: batch=2,  accum=128 → 2*128*1024 = 262K tokens
    
    # Let code auto-calculate these based on sequence length
    "auto_batch_size": True,
    "min_batch_size": 2,               # Minimum micro-batch size
    "max_batch_size": 8,               # Maximum micro-batch size
    
    
    # ========================================================================
    # SEQUENCE LENGTH CURRICULUM - Dataset Decomposition (Grow-P2)
    # ========================================================================
    # PHASED TRAINING FOR RULER BENCHMARKS (4K/8K)
    # 
    # PHASE 1 (Week 3): Foundation - 10B tokens
    "seq_buckets": [256, 512, 1024],           # Max 1024 for base training
    "sampling_weights": {256: 32, 512: 16, 1024: 8},  # Sample shorter more
    "gradient_accumulation_steps": 64, # Base accumulation (adjust per seq_len)
    
    # PHASE 2 (Week 4): Extend to 2K - Additional 5B tokens
    # "seq_buckets": [512, 1024, 2048],
    # "sampling_weights": {512: 16, 1024: 8, 2048: 4},
    # "gradient_accumulation_steps": 128,  # Increase for 2048
    
    # PHASE 3 (Week 5-6): RULER 4K Prep - Additional 3B tokens
    # "seq_buckets": [2048, 4096],
    # "sampling_weights": {2048: 8, 4096: 4},
    # "gradient_accumulation_steps": 128,  # changed to 128 from 256 # Increase for 4096
    # "max_batch_size": 1,  # Must be 1 for 4096 (memory limit)
    
    # PHASE 4 (Week 7): RULER 8K - Train on 4096, test on 8192 with RoPE scaling
    # "seq_buckets": [4096],
    # "sampling_weights": {4096: 1},
    # "gradient_accumulation_steps": 256,
    # "max_batch_size": 1,
    # "use_rope_scaling": True,  # Enable for 8K inference
    # "rope_scaling_factor": 2.0,  # Scale 4096 → 8192
    
    # Training phases (based on % of total steps):
    "phase_1_pct": 0.40,  # 0-40%: Mostly short (256-512)
    "phase_2_pct": 0.40,  # 40-80%: All lengths (256-1024)  
    "phase_3_pct": 0.20,  # 80-100%: Focus longer (512-1024)
    
    # ========================================================================
    # LEARNING RATE SCHEDULE - WSD (Warmup-Stable-Decay)
    # ========================================================================
    "warmup_steps": 2000,              # 2K warmup steps for Phase 1
    "stable_ratio": 0.80,              # 80% of training at peak LR
    "decay_ratio": 0.20,               # Final 20% linear decay
    
    # ========================================================================
    # TRAINING SCALE - PHASED FOR RULER BENCHMARKS
    # ========================================================================
    # Research: Chinchilla optimal is 20:1 token-to-param ratio
    # Modern: 30:1 ratio with high-quality data
    # Your model: ~1.5B params → Need 30-45B tokens
    
    # PHASE 1 (Week 3): Foundation - Validate novel formula works
    "target_tokens": 10_000_000_000,   # 10B tokens @ 1024 max seq
    
    # PHASE 2 (Week 4): Extend to 2K context
    # "target_tokens": 5_000_000_000,   # Additional 5B tokens @ 2048 max seq
    
    # PHASE 3 (Week 5-6): RULER 4K Preparation
    # "target_tokens": 3_000_000_000,   # Additional 3B tokens @ 4096 max seq
    
    # PHASE 4 (Week 7): RULER 8K with RoPE scaling
    # "target_tokens": 2_000_000_000,   # Additional 2B tokens @ 4096 (test @ 8192)
    
    # TOTAL: 20B tokens across all phases → RULER ready!
    
    # For reference - Future scaling:
    # "target_tokens": 45_000_000_000,  # 45B tokens (Chinchilla 30:1)
    # "target_tokens": 100_000_000_000, # 100B tokens (SOTA comparison)
    
    # ========================================================================
    # DATA CONFIGURATION
    # ========================================================================
    # PHASE 1: Debug with small dataset
    "max_documents": None,          # Limit for fast iteration
    
    # PHASE 2: Production
    # "max_documents": None,            # Use full dataset
    
    "validation_split": 0.05,          # 5% held out for validation
    
    # ========================================================================
    # MEMORY OPTIMIZATION - CRITICAL for 32GB VRAM
    # ========================================================================
    "use_bf16": True,                  # BF16 mixed precision (2x memory savings)
    "use_gradient_checkpointing": True,# Activation checkpointing (30-50% memory savings)
    "checkpoint_granularity": "block", # Checkpoint at layer level
    
    # ========================================================================
    # MONITORING & CHECKPOINTING
    # ========================================================================
    "log_interval": 100,               # Log metrics every N steps
    "eval_interval": 1000,             # Validate every N steps
    "save_interval": 5000,             # Save checkpoint every N steps
    "eval_steps": 100,                 # Number of validation batches
    
    # ========================================================================
    # HARDWARE-SPECIFIC OPTIMIZATIONS
    # ========================================================================
    "num_workers": 4,                  # DataLoader workers (6 vCPU → 4 workers)
    "pin_memory": True,                # Pin memory for faster GPU transfer
    "prefetch_factor": 2,              # Prefetch batches
    "persistent_workers": True,        # Keep workers alive between epochs
    
    # ========================================================================
    # ADVANCED: Torch optimizations - WEEK 3 ENABLED FOR 2X SPEEDUP
    # ========================================================================
    "use_torch_compile": True,         # torch.compile (PyTorch 2.0+) - ENABLED!
                                       # Note: May increase startup time
    "compile_mode": "reduce-overhead", # Compilation mode
    
    # ========================================================================
    # REPRODUCIBILITY
    # ========================================================================
    "seed": 42,                        # Random seed
    "deterministic": False,            # Deterministic mode (slower but reproducible)
}

# ============================================================================
# AUTO-CALCULATED VALUES (Don't modify)
# ============================================================================
def finalize_config(config):
    """Calculate derived configuration values"""
    
    # Calculate total training steps
    avg_seq_len = sum(config['seq_buckets']) / len(config['seq_buckets'])
    config['total_steps'] = config['target_tokens'] // config['target_tokens_per_update']
    
    # Calculate WSD schedule steps
    config['stable_steps'] = int(config['total_steps'] * config['stable_ratio'])
    config['decay_steps'] = config['total_steps'] - config['warmup_steps'] - config['stable_steps']
    
    # Validate configuration
    assert config['decay_steps'] > 0, "Decay steps must be positive"
    assert config['warmup_steps'] < config['total_steps'], "Warmup longer than total steps"
    
    return config

# ============================================================================
# USAGE
# ============================================================================
if __name__ == "__main__":
    config = finalize_config(config)
    
    print("="*80)
    print("OPTIMAL CONFIGURATION FOR RTX 5000 ADA")
    print("="*80)
    print(f"\n🏗️  MODEL:")
    print(f"   Hidden size: {config['d_model']}")
    print(f"   Heads: {config['num_heads']}")
    print(f"   Layers: {config['num_layers']}")
    
    print(f"\n⚙️  OPTIMIZATION:")
    print(f"   Peak LR: {config['peak_learning_rate']:.2e}")
    print(f"   Weight Decay: {config['weight_decay']}")
    print(f"   Gradient Clip: {config['gradient_clip']}")
    
    print(f"\n📊 SCALE:")
    print(f"   Target Tokens: {config['target_tokens']/1e9:.1f}B")
    print(f"   Total Steps: {config['total_steps']:,}")
    print(f"   Tokens/Step: {config['target_tokens_per_update']:,}")
    
    print(f"\n📅 SCHEDULE:")
    print(f"   Warmup: {config['warmup_steps']:,} steps")
    print(f"   Stable: {config['stable_steps']:,} steps")
    print(f"   Decay: {config['decay_steps']:,} steps")
    
    print(f"\n💾 MEMORY:")
    print(f"   BF16: {config['use_bf16']}")
    print(f"   Gradient Checkpointing: {config['use_gradient_checkpointing']}")
    print(f"   Base Batch Size: {config['max_batch_size']}")
    print(f"   Gradient Accumulation: {config['gradient_accumulation_steps']}")
    
    print(f"\n⏱️  ESTIMATED TIME (RTX 5000 Ada):")
    # Assume 2-4 it/s with optimizations
    est_time_hours_min = config['total_steps'] / 4 / 3600
    est_time_hours_max = config['total_steps'] / 2 / 3600
    print(f"   {est_time_hours_min:.1f} - {est_time_hours_max:.1f} hours")
    print(f"   ({est_time_hours_min/24:.1f} - {est_time_hours_max/24:.1f} days)")
    
    print("\n" + "="*80)
    print("✅ Configuration validated and ready!")
    print("="*80)

# ============================================================================
# QUICK REFERENCE: Memory vs Sequence Length
# ============================================================================
"""
RTX 5000 Ada (32GB VRAM) Memory Budget with Optimizations:

Sequence  | Micro    | Grad    | Effective    | VRAM      | Speed
Length    | Batch    | Accum   | Batch Size   | Usage     | (it/s)
----------|----------|---------|--------------|-----------|--------
256       | 8        | 64      | 131K tokens  | ~22GB     | 4-5
512       | 4        | 64      | 131K tokens  | ~24GB     | 3-4  
1024      | 2        | 128     | 262K tokens  | ~28GB     | 2-3
2048      | 1        | 256     | 524K tokens  | ~30GB     | 1-2

Notes:
- VRAM includes model (4GB) + optimizer (8GB) + activations + gradients
- With BF16 + Gradient Checkpointing enabled
- Speed estimates assume no bottlenecks
"""