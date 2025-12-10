# LAM Comprehensive Evaluation Suite

**Rigorous scientific validation of all claims from the LAM research paper**

This evaluation suite provides comprehensive testing infrastructure to validate LAM's performance claims with reproducible, scientific rigor.

---

## 📋 Overview

The evaluation suite consists of 4 major test categories:

1. **Pearson Score Validation** - Verify 0.836 Pearson on STS-B benchmark
2. **Linear Scaling Validation** - Prove O(n) complexity for memory and time
3. **Long Context Processing** - Test 32K to 1M token sequences without chunking
4. **Ablation Study** - Quantify each architectural component's contribution

### What Makes This Suite Comprehensive?

✅ **Statistical Rigor**: Bootstrap confidence intervals, significance testing
✅ **Visual Proof**: High-quality charts and diagrams for every claim
✅ **Reproducible**: Deterministic tests with saved results and random seeds
✅ **Comparative**: Side-by-side with baseline (all-MiniLM-L6-v2)
✅ **Complete**: Tests every claim in the research paper

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
cd LAM-base-v1/evaluation
pip install -r requirements.txt
```

Requirements:
- `sentence-transformers >= 2.2.0`
- `datasets >= 2.14.0`
- `scipy >= 1.10.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `psutil >= 5.9.0`
- `gputil >= 1.4.0`
- `scikit-learn >= 1.3.0`

### Run All Tests

```bash
# Run complete evaluation suite
# The model path points to LAM-base-v1/ which contains both:
#   - lam_base.bin (87 MB base model)
#   - lam_tweak.pt (56 MB LAM attention)
python run_all_tests.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2
```

Expected duration: **30-60 minutes** depending on hardware

### Run Individual Tests

```bash
# Test 1: Pearson score validation (~5 minutes)
# Uses LAM-base-v1/ which contains lam_base.bin + lam_tweak.pt
# By default uses test set (official benchmark). Use --split validation for development set.
python test_pearson_score.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2 --split test

# For 100% scientific legitimacy, test on BOTH splits:
python test_pearson_score.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2 --split validation
python test_pearson_score.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2 --split test

# Test 2: Linear scaling validation (~15 minutes)
python test_linear_scaling.py --model ../LAM-base-v1 --baseline all-MiniLM-L6-v2

# Test 3: Long context processing (~10-20 minutes)
python test_long_context.py --model ../LAM-base-v1

# Test 4: Ablation study (~5 minutes)
python test_ablation_study.py --model ../LAM-base-v1
```

**Note**: The test scripts automatically load LAM using the custom `LAMEncoder` which combines:
- `lam_base.bin` (87 MB) - Base embeddings + FFN layers
- `lam_tweak.pt` (56 MB) - LAM attention weights

This is the complete LAM model (143 MB total).

---

## 📊 Test Descriptions

### Test 1: Pearson Score Validation

**File**: `test_pearson_score.py`

**Purpose**: Rigorously validate LAM's claimed 0.836 Pearson score on STS-B with full statistical analysis.

**Dataset Splits** (for 100% scientific legitimacy):
- **Test Set** (--split test): Official benchmark, 1,379 pairs, held-out during training
- **Validation Set** (--split validation): Development set, 1,500 pairs, used for checkpoint selection

**What It Tests**:
- Full STS-B evaluation on specified split
- Bootstrap confidence intervals (95%, 10K samples)
- Per-score-range performance breakdown
- Error distribution analysis
- Comparison with all-MiniLM-L6-v2 baseline

**Outputs**:
- `results/pearson_score_validation_{split}.json` - Complete metrics (split = test or validation)
- `visualizations/pearson_score_validation.png` - 6 comprehensive charts:
  - Predictions vs labels scatter (LAM)
  - Predictions vs labels scatter (Baseline)
  - Error distribution histogram
  - Residual plot
  - Model comparison bar chart
  - Error box plots by score range

**Success Criteria**:
- ✓ Pearson score between 0.826-0.846 on validation set (within 95% CI)
- ✓ Pearson score on test set matches validation within reasonable margin
- ✓ Statistically significant (p < 0.001) on both splits
- ✓ All score ranges show strong correlation

**Note**: For publication, report scores from BOTH splits to prove generalization without overfitting.

---

### Test 2: Linear Scaling Validation

**File**: `test_linear_scaling.py`

**Purpose**: Prove LAM maintains O(n) linear complexity for both memory and time vs O(n²) for transformers.

**What It Tests**:
- Memory scaling: 128 → 100K tokens
- Time scaling: 128 → 100K tokens
- Regression analysis (linear vs quadratic fit)
- Crossover point identification
- Speedup and memory reduction factors

**Outputs**:
- `results/linear_scaling_validation.json` - Scaling measurements
- `visualizations/linear_scaling_validation.png` - 6 charts:
  - Time vs length (linear scale)
  - Time vs length (log-log scale with O(n) reference)
  - Memory vs length (linear scale)
  - Memory vs length (log-log scale with O(n) reference)
  - Speedup comparison
  - Memory reduction comparison

**Success Criteria**:
- ✓ Linear regression R² > 0.95 for LAM time
- ✓ Linear regression R² > 0.95 for LAM memory
- ✓ Quadratic fit better for baseline
- ✓ 100× memory reduction at 100K tokens

---

### Test 3: Long Context Processing

**File**: `test_long_context.py`

**Purpose**: Validate LAM can process 32K, 100K, and 1M token sequences as single-pass encodings without chunking.

**What It Tests**:
- Single-pass encoding: 32K, 64K, 100K, 250K, 500K, 1M tokens
- Memory footprint at each length
- Inference time scaling
- Semantic coherence preservation across long documents
- Document-level vs section-level embedding similarity

**Outputs**:
- `results/long_context_validation.json` - All length test results
- `visualizations/long_context_validation.png` - 6 charts:
  - Inference time vs document length
  - Memory usage vs document length
  - Semantic coherence across lengths
  - Memory efficiency (MB per 1K tokens)
  - Processing speed (tokens/sec)
  - Summary table

**Success Criteria**:
- ✓ Successfully encode ≥100K tokens single-pass
- ✓ Memory usage remains linear (150-180 MB @ 100K)
- ✓ Semantic coherence ≥0.5 maintained
- ✓ No OOM errors on standard hardware

---

### Test 4: Ablation Study

**File**: `test_ablation_study.py`

**Purpose**: Quantify the contribution of each LAM architectural component to overall performance.

**Components Tested**:
1. **Dual-State Memory** (fast + slow states)
2. **Enhanced Resonance Flux** (bilinear query-key interaction)
3. **Hierarchical Decay** (position-adaptive forgetting)

**What It Tests**:
- Full LAM vs component ablations
- Performance drop when each component removed
- Statistical significance of each contribution
- Progressive component addition

**Outputs**:
- `results/ablation_study_validation.json` - Ablation metrics
- `visualizations/ablation_study_validation.png` - 6 charts:
  - Pearson scores by configuration
  - Performance drop when component removed
  - Relative contribution pie chart
  - Progressive component addition
  - Component configuration matrix
  - Statistical summary

**Success Criteria**:
- ✓ Full LAM achieves highest score
- ✓ Each component contributes ≥0.005 Pearson
- ✓ Total improvement over baseline ≥0.015
- ✓ All components statistically significant

---

## 📁 Directory Structure

```
evaluation/
├── README.md                          # This file
├── run_all_tests.py                   # Master test runner
│
├── test_pearson_score.py             # Test 1: STS-B validation
├── test_linear_scaling.py            # Test 2: O(n) complexity
├── test_long_context.py              # Test 3: Long sequences
├── test_ablation_study.py            # Test 4: Component analysis
│
├── results/                           # JSON output files
│   ├── pearson_score_validation_test.json        # Test set results
│   ├── pearson_score_validation_validation.json  # Validation set results
│   ├── linear_scaling_validation.json
│   ├── long_context_validation.json
│   ├── ablation_study_validation.json
│   ├── comprehensive_evaluation_report.json
│   └── EVALUATION_REPORT.txt         # Human-readable summary
│
├── visualizations/                    # PNG chart outputs
│   ├── pearson_score_validation.png
│   ├── linear_scaling_validation.png
│   ├── long_context_validation.png
│   └── ablation_study_validation.png
│
└── datasets/                          # Cached test data (auto-generated)
```

---

## 🔬 Scientific Rigor

### Statistical Methods

- **Bootstrap Confidence Intervals**: 10,000 resamples for robust CI estimation
- **Regression Analysis**: Both linear and quadratic fits with R² comparison
- **Significance Testing**: Pearson correlation p-values
- **Error Analysis**: MAE, RMSE, median absolute error, percentiles

### Reproducibility

- **Random Seeds**: Fixed seeds (42) for deterministic results
- **Multiple Runs**: Median of 3 runs for time/memory measurements
- **Versioned Dependencies**: requirements.txt locks package versions
- **Saved Results**: All raw data saved for re-analysis

### Visualization Quality

- **Publication-Ready**: 300 DPI, high-resolution PNGs
- **Comprehensive**: 6 charts per test (36 visualizations total)
- **Annotated**: Value labels, reference lines, statistical annotations
- **Color-Coded**: Consistent color scheme across all charts

---

## 📈 Expected Results

Based on the LAM research paper, you should observe:

### Test 1: Pearson Score
- **LAM (Validation)**: 0.836 ± 0.005 (95% CI) - 1,500 pairs
- **LAM (Test)**: ~0.83-0.84 expected (95% CI) - 1,379 pairs
- **Baseline**: 0.89 ± 0.003
- **Ratio**: ~94% of baseline quality
- **Note**: Both splits should give similar scores, proving no overfitting

### Test 2: Linear Scaling
- **LAM Time Complexity**: Linear (R² > 0.98)
- **LAM Memory Complexity**: Linear (R² > 0.98)
- **Baseline Time**: Quadratic (R² > 0.95 for quad fit)
- **Memory Reduction @ 100K**: 100× (150 MB vs 40+ GB)

### Test 3: Long Context
- **32K tokens**: ✓ Success, ~50 MB, ~1s
- **100K tokens**: ✓ Success, ~150 MB, ~2.8s
- **1M tokens**: ✓ Success, ~180 MB, ~28s
- **Semantic Coherence**: > 0.5 across all lengths

### Test 4: Ablation Study
- **Full LAM**: 0.836
- **-Resonance Flux**: 0.816 (-0.020)
- **-Dual State**: 0.821 (-0.015)
- **-Hierarchical Decay**: 0.829 (-0.007)
- **Baseline DeltaNet**: 0.820

---

## 🛠️ Troubleshooting

### Out of Memory (OOM) Errors

**Problem**: Tests fail with CUDA OOM on long sequences

**Solutions**:
```python
# Reduce test lengths in test files
self.test_lengths = [128, 256, 512, 1024, 2048, 4096]  # Stop at 4K

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
python run_all_tests.py --model ../
```

### Model Loading Errors

**Problem**: Cannot load LAM model

**Solutions**:
```bash
# Verify model path
ls -la ../lam_base.bin
ls -la ../lam_config.json

# Check model is valid SentenceTransformer format
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('../')"
```

### Slow Execution

**Problem**: Tests take too long

**Solutions**:
```bash
# Run tests individually
python test_pearson_score.py  # Only run Pearson test

# Skip long context test
python test_pearson_score.py && \
python test_linear_scaling.py && \
python test_ablation_study.py
```

### Dependency Issues

**Problem**: Import errors or version conflicts

**Solutions**:
```bash
# Create fresh environment
python -m venv venv_lam
source venv_lam/bin/activate
pip install -r requirements.txt

# Or use conda
conda create -n lam python=3.10
conda activate lam
pip install -r requirements.txt
```

---

## 📊 Interpreting Results

### Results Directory

After running tests, check `results/` directory:

```bash
# View text summary
cat results/EVALUATION_REPORT.txt

# View specific test results (split-specific files)
cat results/pearson_score_validation_test.json | python -m json.tool
cat results/pearson_score_validation_validation.json | python -m json.tool

# Open visualizations
open visualizations/*.png  # macOS
xdg-open visualizations/*.png  # Linux
```

### Validation Criteria

Tests PASS if:
1. **Pearson Score**:
   - Validation set: Within ±1% of claimed 0.836
   - Test set: Within ±2% of validation score (proves generalization)
   - Both splits statistically significant (p < 0.001)
2. **Linear Scaling**: R² > 0.95 for linear fit
3. **Long Context**: Successfully process ≥100K tokens
4. **Ablation**: All components contribute significantly

### What If Tests Fail?

1. **Check model loading**: Ensure correct model path
2. **Verify hardware**: GPU with ≥8GB VRAM recommended
3. **Review logs**: Detailed error messages in console output
4. **Compare baselines**: Baseline should match literature values
5. **Report issues**: Open GitHub issue with full error logs

---

## 🤝 Contributing

### Adding New Tests

1. Create new test file: `test_your_metric.py`
2. Follow existing test structure (class-based, JSON output, visualizations)
3. Add to `run_all_tests.py` master runner
4. Update this README

### Improving Visualizations

Visualization guidelines:
- Use 300 DPI for publication quality
- Follow existing color scheme (blue=LAM, orange=baseline)
- Add value labels and reference lines
- Include clear titles and axis labels

### Reporting Issues

When reporting issues, include:
- Full error traceback
- Hardware specs (CPU, GPU, RAM)
- Python version and package versions
- Test configuration (model path, parameters)

---

## 📚 References

- **STS-B Dataset**: [SentenceTransformers STS-B](https://huggingface.co/datasets/sentence-transformers/stsb)
- **DeltaNet Paper**: Chen et al., 2024
- **Mamba Paper**: Gu & Dao, 2023
- **LAM Paper**: LAM Research Team, 2024

---

## 📄 License

This evaluation suite is part of the LAM project and follows the same license.

---

## ✉️ Contact

For questions about the evaluation suite:
- GitHub Issues: [LAM Repository](https://github.com/lam-research/LAM)
- Email: research@lam-model.ai

---

**Last Updated**: November 2024
**Version**: 1.0.0
**Maintainers**: LAM Research Team
