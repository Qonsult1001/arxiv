# Hardware and Training Configuration Summary

## Hardware Setup

### Device Configuration
- **Primary Device**: CUDA (GPU) if available, else CPU
- **TensorFloat32 (TF32)**: Enabled for Ampere+ GPUs
  - Configuration: `torch.set_float32_matmul_precision('high')` (Line 59)
  - Purpose: Optimizes matrix multiplication on Ampere+ GPUs (RTX 30xx, A100, etc.)
  - Impact: ~1.5-2x speedup on compatible hardware

### GPU Requirements (Recommended)
- **GPU Architecture**: CUDA-capable GPU (Ampere+ recommended for TF32)
- **VRAM**: Sufficient for batch size 256 with sequence length 128
- **Compute Capability**: 7.0+ (Volta) minimum, 8.0+ (Ampere) recommended
- **Training Time**: ~200K steps (varies by GPU model and VRAM)

### Actual Hardware Used (Detected)
- **GPU**: NVIDIA L40
- **VRAM**: 44.40 GB
- **Compute Capability**: 8.9 (Ada Lovelace architecture)
- **Multiprocessors**: 142
- **CUDA Version**: 12.8
- **cuDNN Version**: 9.1.0.2
- **PyTorch Version**: 2.8.0+cu128
- **TF32 Support**: ✅ Yes (Ampere+ architecture, compute capability 8.9)
- **Platform**: Linux 6.8.0-57-generic (x86_64)
- **Python Version**: 3.12.3

### Automatic Detection
The training script automatically detects and uses:
- CUDA GPU if available
- CPU as fallback
- Optimal precision settings based on GPU capabilities

---

## Training Configuration

### Batch Configuration
- **Physical Batch Size**: 256
- **Gradient Accumulation Steps**: 4
- **Effective Batch Size**: 256 (matches all-MiniLM-L6-v2 training)
- **Max Sequence Length**: 128 tokens
- **Memory Optimization**: Batch size tuned for GPU memory constraints

### Training Schedule
- **Total Steps**: 200,000
- **Warmup Steps**: 1,000
- **Peak Learning Rate**: 2e-5
- **Weight Decay**: 0.1
- **Gradient Clipping**: 1.0
- **Evaluation Interval**: Every 2,000 steps (saves best model)
- **Checkpoint Interval**: Every 1,000 steps
- **Resume Capability**: Can resume from any checkpoint by setting `resume_from_step` in config

### Current Training Run (As of Latest Execution)
- **Resuming From Step**: 116,000
- **Remaining Steps**: 84,000 (to reach 200,000 total)
- **Status**: Continuing training from checkpoint_116000.pt

### Learning Rate Schedule
- **Scheduler**: Linear warmup + linear decay
- **Warmup**: 1,000 steps (0 → 2e-5) - *Already completed if resuming*
- **Decay**: Linear from 2e-5 → 0 over remaining steps

---

## Model Architecture

### Base Configuration
- **Teacher Model**: all-MiniLM-L6-v2
- **Student Model**: 6-layer pure linear DeltaNet
- **Architecture Type**: 100% Linear (0% Attention)
- **Hidden Size (d_model)**: 384
- **Attention Heads**: 12 (for DeltaNet structure)
- **Total Layers**: 6
- **Parameters**: ~22.7M (student model only)

### DeltaNet Components
- **Enhanced Hierarchical DeltaNet**: 6 layers
- **Resonance Flux**: Enabled
- **Hierarchical Memory**: Enabled
- **Cross-timescale Coupling**: Enabled
- **Orthogonal Initialization**: Enabled
- **Orthogonal Regularization**: Enabled (W^T W ≈ I)

---

## Loss Function Configuration

### Primary Losses
1. **Contrastive Loss**: Main training objective
   - Scale: 20.0
   - Label Smoothing: 0.1

2. **Embedding Distillation**: L2 loss on final embeddings
   - Weight: 1.0

3. **Layer-wise Distillation**: L2 loss on hidden states
   - Weight: 1.5 (higher for pure linear)

### Regularization Losses
4. **Orthogonal Regularization**: W^T W ≈ I (prevents feature collapse)
   - Weight: 0.01

5. **Orthogonal State Regularization**: Prevents S_fast/S_slow correlation
   - Weight: 0.002

6. **Spearman Correlation Loss**: Optimizes rank order (MTEB main score)
   - Weight: 0.3
   - Purpose: Directly optimize for Spearman correlation

7. **Pairwise Ranking Loss**: Preserves relative ordering
   - Weight: 0.2
   - Margin: 0.05

### Additional Features
- **State Dropout**: 10% on state updates (prevents overfitting)
- **Dropout**: 15% (general regularization)

---

## Evaluation Configuration

### Primary Metric
- **MTEB Main Score**: Test Set Spearman Correlation
- **Evaluation Dataset**: STS-B test split (1,379 pairs)
- **Evaluation Frequency**: Every 2,000 steps
- **Best Model Saving**: Based on **TEST** Spearman (not validation)

### Secondary Metrics
- **Validation Set**: STS-B validation split (1,500 pairs)
- **Metrics Tracked**: Pearson, Spearman (both validation and test)
- **Best Model**: Saved as `pytorch_model.bin` when test Spearman improves

---

## Data Configuration

### Training Datasets
1. **AllNLI**: Natural Language Inference pairs (local file)
2. **QQP**: Quora Question Pairs (200K samples)
3. **MS MARCO**: Question-Answer pairs (100K samples, truncated)
4. **WikiAnswers**: Duplicate questions (1M pairs, optional)
5. **SNLI**: Stanford NLI (100K pairs)

### Data Processing
- **Total Training Pairs**: ~1.5M+ pairs
- **Shuffling**: Random seed 42
- **Tokenization**: SentencePiece (from all-MiniLM-L6-v2)
- **Max Length**: 128 tokens (truncation/padding)

---

## Performance Optimizations

### Memory Optimizations
- **Gradient Accumulation**: 4 steps (simulates larger batch)
- **Batch Size Tuning**: Adjusted for GPU memory constraints
- **Mixed Precision**: TF32 on Ampere+ GPUs

### Training Optimizations
- **Optimizer**: AdamW
- **Gradient Clipping**: 1.0 (prevents exploding gradients)
- **Efficient Data Loading**: Direct dataset access
- **Checkpoint Resuming**: Full state restoration (optimizer, scheduler, model)

---

## Output Configuration

### Saved Files
- **Best Model**: `pytorch_model.bin` (saved when test Spearman improves)
- **Checkpoints**: `checkpoint_XXXXX.pt` (every 1,000 steps)
- **Final Model**: `final_model/pytorch_model.bin` (backup)
- **Tokenizer**: Saved with model (for inference)

### Checkpoint Contents
- Model state dict (deltanet_layers, norms, FFNs)
- Optimizer state
- Scheduler state
- Training step
- Best scores (validation and test)
- Configuration

---

## Expected Performance

### Target Metrics
- **Validation Spearman**: 0.83-0.84+
- **Test Spearman**: 0.76-0.78+ (MTEB main score)
- **MTEB STSBenchmark**: 0.76-0.78+ (qualifies for efficient section)

### Training Progress
- **Early Steps (0-20K)**: Rapid improvement ✅ *Completed*
- **Mid Steps (20K-100K)**: Steady improvement ✅ *Completed*
- **Late Steps (100K-200K)**: Fine-tuning and convergence 🔄 *In Progress (116K → 200K)*
- **Current Position**: Step 116,000 (58% complete)

---

## Usage

### Training
```bash
# Start from scratch (default: resume_from_step = 0)
python train_6layer_deltanet_1.py

# Resume from specific checkpoint (edit resume_from_step in config)
# Example: Set resume_from_step = 116000 to resume from checkpoint_116000.pt
```

### Evaluation
```bash
# Check all checkpoints
python test_checkpoints.py deltanet_minilm_6layers_FIXED_FROM_SCRATCH/

# MTEB evaluation
python test_mteb_local.py --checkpoint /path/to/checkpoint.pt
```

### Save Best Checkpoint
```bash
# Save specific checkpoint as pytorch_model.bin
python save_best_checkpoint.py checkpoint_25000.pt
```

---

## Hardware Detection

To get exact hardware details, run:
```bash
python detect_hardware.py
```

This will display:
- GPU model and specifications
- CUDA and cuDNN versions
- Memory information
- TF32 support status
- Training memory estimation

---

## Notes

- **Test Set Evaluation**: The training script now evaluates on TEST set (MTEB's metric) to determine best model
- **Best Model**: `pytorch_model.bin` is saved based on TEST Spearman, matching MTEB leaderboard ranking
- **Hardware Detection**: Run `python detect_hardware.py` to get exact GPU details
- **TF32**: Automatically enabled on Ampere+ GPUs for faster training
- **Checkpoint Saving**: Use `python save_best_checkpoint.py <checkpoint_path>` to manually save a checkpoint as `pytorch_model.bin`

---

*Last Updated: Based on train_6layer_deltanet_1.py configuration*
*Hardware: NVIDIA L40 (44.40 GB VRAM, CUDA 12.8)*

