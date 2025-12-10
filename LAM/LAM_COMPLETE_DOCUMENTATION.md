# LAM (Linear Attention Model) - Complete Documentation Index

## Overview
LAM is a novel memory-driven recurrent model achieving Transformer-level semantic understanding at O(L) complexity. It achieves **0.836 Pearson correlation on STS-B**, establishing a WORLD-FIRST benchmark for linear attention models in semantic similarity tasks.

---

## 🧬 Core Formula Implementation

### Primary Formula Files

1. **`/workspace/final_solution_formula.py`** ⭐ **MAIN FORMULA**
   - **Purpose**: Core Enhanced Hierarchical DeltaNet implementation
   - **Key Components**:
     - `EnhancedHierarchicalDeltaNet` class - Main architecture
     - `EnhancedResonanceFlux` - Bilinear resonance flux for dynamic coupling
     - `_enhanced_hierarchical_delta_rule_impl` - Core delta rule with hierarchical memory
   - **Features**:
     - Hierarchical dual-state memory (S_fast, S_slow)
     - Delta rule precision (selective memory overwriting)
     - Dynamic forgetting (adaptive decay)
     - Cross-timescale interaction
     - Resonance flux (psi) for salience gating
   - **Used by**: `train_6layer_deltanet.py`, `train_6layerE5_deltanet.py`

2. **`/workspace/final_solution_formula_long.py`** ⭐ **LONG-RANGE VERSION**
   - **Purpose**: Enhanced version with Linformer support for long-range memory
   - **Key Differences**:
     - `use_linformer_proj=True` - Enables Linformer for long-range memory
     - `linformer_k=256` - Global context size
     - `linformer_max_seq_len=1572864` - 1.5M tokens exact recall window
   - **Used by**: `deltanet_finetune_6layers.py` (for advanced fine-tuning)

3. **`/workspace/final_solution_formula_pure.py`**
   - **Purpose**: Pure linear version with NO intra-chunk attention
   - **Note**: Alternative implementation for maximum linearity

4. **`/workspace/final_solution_formula_TRM.py`**
   - **Purpose**: TRM (Temporal Recursive Memory) variant
   - **Features**: Recursive depth, deep supervision, adaptive recursion

---

## 🏗️ Model Architecture Files

### Model Architecture (Used in Training)

The model architecture used in the training scripts that achieved 0.836:

1. **`train_6layer_deltanet.py`** - Defines `DeltaNetPure6Layer` class
   - Uses `EnhancedHierarchicalDeltaNet` from `final_solution_formula.py`
   - 6 DeltaNet layers with teacher model (AllMiniLM-L6-v2) for distillation

2. **`deltanet_finetune_6layers.py`** - Defines `DeltaNet6LayerWorldClass` class
   - Uses `EnhancedHierarchicalDeltaNet` from `final_solution_formula_long.py`
   - Advanced fine-tuning with multiple data sources

3. **`train_6layerE5_deltanet.py`** - Defines `ProperDistillationModel` and `LoadCheckpoint5K` classes
   - Uses `EnhancedHierarchicalDeltaNet` from `final_solution_formula.py`
   - E5-Large distillation (achieved 0.836 Pearson)

---

## 🎓 Training Scripts

### Primary Training Scripts

1. **`/workspace/train_6layer_deltanet.py`** ⭐ **BASE TRAINING**
   - **Purpose**: Trains 6-layer pure linear DeltaNet from scratch
   - **Method**: Knowledge distillation from AllMiniLM-L6-v2
   - **Features**:
     - Layer-wise distillation
     - TRUE orthogonal regularization (W^T W ≈ I)
     - Orthogonal weight initialization
   - **Output**: `deltanet_minilm_6layers_FIXED_FROM_SCRATCH/checkpoint_*.pt`
   - **Dataset**: HuggingFace `sentence-transformers/all-nli`
   - **Result**: Creates base checkpoint used by fine-tuning scripts

2. **`/workspace/deltanet_finetune_6layers.py`** ⭐ **ADVANCED FINE-TUNING**
   - **Purpose**: Fine-tunes pre-trained DeltaNet with advanced techniques
   - **Features**:
     - Hard negative mining (triplet loss)
     - In-batch negatives
     - Curriculum learning
     - Data augmentation
     - Orthogonal regularization
     - Knowledge retention
   - **Data Sources**:
     - `/workspace/data/AllNLI.jsonl.gz`
     - `/workspace/data/NQ-train_pairs.jsonl.gz`
     - `/workspace/data/squad_pairs.jsonl.gz`
     - `/workspace/data/pairs.jsonl.gz`
     - `/workspace/data/triplets.jsonl.gz`
     - HuggingFace `glue/stsb`
   - **Input**: `checkpoint_38000.pt` from base training
   - **Output**: `proper_distillation_reaccelerate/checkpoint_*.pt`
   - **Target**: 0.85+ Pearson on STS-B

3. **`/workspace/train_6layerE5_deltanet.py`** ⭐ **E5 DISTILLATION**
   - **Purpose**: Knowledge distillation from E5-Large-v2 to DeltaNet student
   - **Method**: Similarity structure matching (relational knowledge distillation)
   - **Features**:
     - KL divergence on similarity distributions
     - Contrastive margin loss
     - Embedding distillation
     - Layer-wise distillation
     - Dimension projection (1024 → 384)
   - **Input**: `checkpoint_best_3500.pt` from fine-tuning
   - **Output**: `proper_distillation_reaccelerate/checkpoint_*.pt`
   - **Result**: Achieved 0.836 Pearson (best checkpoint: `checkpoint_best_3500.pt`)


---

## 📚 Core Documentation

### Architecture & Design

1. **`/workspace/RECOMMENDED_ARCHITECTURE.md`** ⭐
   - **Content**: Memory as a Service architecture (1M token conversational memory)
   - **Key Topics**:
     - Conversational memory (S_slow)
     - Document memory (full documents, no chunking)
     - Storage strategy
     - Query priority system

2. **`/workspace/MEMORY_ARCHITECTURE.md`** ⭐
   - **Content**: Personal memory brain architecture
   - **Key Topics**:
     - S_fast (temporary, 70% decay/step)
     - S_slow (consolidated, 0.1% decay/step)
     - .pt file storage (personal knowledge base)
     - Query priority (personal memory first)

3. **`/workspace/COGNITIVE_PILLARS_VALIDATED.md`** ⭐ **CRITICAL**
   - **Content**: All six cognitive pillars validated
   - **Pillars**:
     1. Cross-Timescale Interaction (S_fast + S_slow)
     2. Pattern Separation (Hippocampus-inspired)
     3. Reconsolidation (Memory strengthening)
     4. Semantic Projection (Output space, 0.836 Pearson)
     5. Cognitive Realism (Human-like memory)
     6. Dynamic Salience Gating (Resonance flux psi)
   - **Status**: All pillars WORKING and validated

### Research Findings

4. **`/workspace/CRITICAL_FINDINGS.md`**
   - **Content**: Adaptive compression analysis
   - **Finding**: Compression adds overhead, base version is faster
   - **Recommendation**: Use `final_solution_formula.py` (no compression)

5. **`/workspace/DISCRIMINATION_ANALYSIS.md`**
   - **Content**: Semantic discrimination analysis
   - **Issue**: Hierarchical memory over-smoothing
   - **Solution**: Semantic-selective state updates

6. **`/workspace/DISTILLATION_FIX.md`**
   - **Content**: Knowledge distillation methodology
   - **Method**: MSE + cosine similarity loss
   - **Result**: GLA matches MiniLM performance

### Training Documentation

7. **`/workspace/TRAINING_SCRIPTS_FILE_DEPENDENCIES.md`** ⭐
   - **Content**: Complete file dependencies for all training scripts
   - **Lists**: All input files, data sources, checkpoints, outputs

8. **`/workspace/TRAINING_SUMMARY.md`**
   - **Content**: Training progress and results summary

9. **`/workspace/TRAINING_STATUS.md`**
   - **Content**: Current training status and checkpoints

---

## 🧪 Testing & Evaluation Files

1. **`/workspace/stsb_evaluation.py`** ⭐
   - **Purpose**: STS-B evaluation utilities
   - **Features**: Pearson/Spearman correlation computation
   - **Used by**: Training scripts for validation

2. **`/workspace/test_deltanet_loading.py`**
   - **Purpose**: Checkpoint loading verification

3. **`/workspace/test_deltanet_success.py`**
   - **Purpose**: Success criteria validation

---

## 🔬 Research & Analysis Files

### Performance Analysis

1. **`/workspace/complete_results_summary.md`**
   - **Content**: Complete results summary

2. **`/workspace/final_comparison.md`**
   - **Content**: Final model comparison

3. **`/workspace/deltanet_progression_analysis.md`**
   - **Content**: Training progression analysis

4. **`/workspace/optimal_checkpoint_analysis.md`**
   - **Content**: Best checkpoint analysis

### E5 Distillation Results

5. **`/workspace/e5_distillation_results.md`**
   - **Content**: E5 distillation training results

6. **`/workspace/e5_distillation_victory.md`**
   - **Content**: E5 distillation success summary

7. **`/workspace/e5_distillation_complete_table.md`**
   - **Content**: Complete E5 distillation results table

8. **`/workspace/e5large_from_5k_results.md`**
   - **Content**: Results from 5K checkpoint

---

## 🚀 Memory System (Related but Separate)

1. **`/workspace/memory_as_service.py`** ⭐
   - **Purpose**: Memory as a Service implementation
   - **Features**:
     - 1M token conversational memory
     - Document storage (no chunking)
     - Pattern separation
     - Reconsolidation
     - Semantic projection
   - **Note**: Uses same core principles (S_fast/S_slow) but separate system

---

## 📊 Key Achievements & Results

### Performance Metrics

- **STS-B Pearson**: **0.836** (WORLD-FIRST for linear attention models)
- **Model Size**: 22M parameters (384 dimensions)
- **Complexity**: O(L) linear complexity
- **Learning Efficiency**: 93%+ learning rate efficiency
- **Base Model**: AllMiniLM-L6-v2 (teacher for distillation)

### Unique Formula Components

1. **Hierarchical Dual-State Memory (S_fast, S_slow)**
   - S_fast: Transient, working memory (fast decay: 0.3)
   - S_slow: Long-term consolidation (slow decay: 0.85-0.96)
   - Cross-timescale interaction: `S_fast += 0.1 * S_slow`, `S_slow += 0.1 * S_fast`

2. **Delta Rule Precision**
   - Selective memory overwriting: `S_new = S_old - (k @ k.T @ S_old) + (k @ v.T)`
   - Corrects outdated associations
   - Enables associative recall

3. **Enhanced Resonance Flux (psi)**
   - Bilinear transformation: `Q @ W_bilinear @ K^T`
   - Dynamic salience gating
   - Routes information to appropriate memory store

4. **Adaptive Decay**
   - Content-dependent memory retention
   - High-importance information retained longer
   - Natural forgetting curve

5. **In-Context Learning**
   - Dynamic adaptation as sequence is read
   - State evolves causally: `S_t = f(S_{t-1}, x_t)`
   - Enables few-shot reasoning

---

## 🎯 Training Methodology Summary

### Stage 1: Base Training (`train_6layer_deltanet.py`)
- **Method**: Knowledge distillation from AllMiniLM-L6-v2
- **Loss**: Contrastive + embedding distillation + layer-wise distillation
- **Regularization**: TRUE orthogonal regularization (W^T W ≈ I)
- **Result**: Base checkpoint at step 38,000

### Stage 2: Advanced Fine-Tuning (`deltanet_finetune_6layers.py`)
- **Method**: Multi-dataset fine-tuning with advanced techniques
- **Data**: 5 local datasets + STS-B
- **Techniques**:
  - Hard negative mining (triplet loss)
  - In-batch negatives
  - Curriculum learning
  - Data augmentation
  - Knowledge retention
- **Result**: Fine-tuned checkpoint

### Stage 3: E5 Distillation (`train_6layerE5_deltanet.py`)
- **Method**: Relational knowledge distillation from E5-Large-v2
- **Loss**: Similarity structure (KL divergence) + contrastive margin + embedding + layer
- **Key Innovation**: Matches teacher's similarity structure, not just embeddings
- **Result**: **0.836 Pearson on STS-B** (checkpoint_best_3500.pt)

---

## 📁 File Organization Summary

### Core Formula (4 files)
- `final_solution_formula.py` ⭐ (main)
- `final_solution_formula_long.py` ⭐ (long-range)
- `final_solution_formula_pure.py` (pure linear)
- `final_solution_formula_TRM.py` (TRM variant)

### Model Implementation
- **Model classes defined in training scripts** ⭐
  - `DeltaNetPure6Layer` in `train_6layer_deltanet.py`
  - `DeltaNet6LayerWorldClass` in `deltanet_finetune_6layers.py`
  - `ProperDistillationModel` in `train_6layerE5_deltanet.py`

### Training Scripts (3 files)
- `train_6layer_deltanet.py` ⭐ (base training)
- `deltanet_finetune_6layers.py` ⭐ (advanced fine-tuning)
- `train_6layerE5_deltanet.py` ⭐ (E5 distillation - achieved 0.836)

### Documentation
- `RECOMMENDED_ARCHITECTURE.md` ⭐ (Memory architecture)
- `MEMORY_ARCHITECTURE.md` ⭐ (S_fast/S_slow system)
- `COGNITIVE_PILLARS_VALIDATED.md` ⭐ (6 cognitive pillars)
- `CRITICAL_FINDINGS.md` (Performance analysis)
- `DISCRIMINATION_ANALYSIS.md` (Semantic discrimination)
- `DISTILLATION_FIX.md` (Knowledge distillation method)
- `TRAINING_SCRIPTS_FILE_DEPENDENCIES.md` ⭐ (File dependencies)
- `e5_distillation_results.md` (E5 distillation results)
- `e5_distillation_victory.md` (Success summary)

### Testing Files
- `stsb_evaluation.py` ⭐ (STS-B evaluation)
- `test_deltanet_loading.py` (checkpoint verification)
- `test_deltanet_success.py` (success validation)

---

## 🔑 Key Insights from Codebase

1. **LAM is NOT a Transformer**: It's a recurrent memory-based model with causal state evolution
2. **Unique Formula**: The Enhanced Hierarchical DeltaNet with Resonance Flux doesn't exist elsewhere
3. **0.836 Achievement**: Result of 3-stage training (base → fine-tune → E5 distillation)
4. **Memory System**: S_fast/S_slow hierarchical memory enables long-range reasoning
5. **Linear Complexity**: O(L) complexity while achieving Transformer-level semantics
6. **Cognitive Validation**: All 6 cognitive pillars validated and working

---

## 📝 Notes

- ⭐ indicates critical/primary files
- All training scripts use the same core formula (`EnhancedHierarchicalDeltaNet`)
- The 0.836 Pearson result was achieved through E5 distillation (`train_6layerE5_deltanet.py`)
- The formula combines multiple innovations not found in other linear attention models
- Memory system (MaaS) is a separate but related system using the same core principles

