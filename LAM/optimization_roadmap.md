# LAM Optimization Roadmap
## Inspired by Google Titans Architecture

### ✅ COMPLETED
1. **chunk_size 64→128**: 1.34x speedup, score preserved (0.8190)

---

### 🎯 NEXT OPTIMIZATIONS TO TEST (One at a time!)

#### **Level 1: Low-Risk Kernel Optimizations**

2. **Remove redundant normalizations in state updates**
   - Current: Normalizes states multiple times per chunk
   - Test: Remove intermediate normalizations, keep only final
   - Expected: 5-10% speedup
   - Risk: Low (mathematical equivalence)

3. **Fuse decay operations**
   - Current: Separate fast_decay and slow_decay computations
   - Test: Compute both in single pass
   - Expected: 3-5% speedup
   - Risk: Very low

4. **Optimize tensor shapes (avoid unnecessary transposes)**
   - Current: Multiple transpose operations in delta rule
   - Test: Pre-transpose and maintain optimal layout
   - Expected: 5-10% speedup
   - Risk: Low

5. **Cache flux computations**
   - Current: Recomputes flux for same tokens
   - Test: Cache flux values across fast/slow passes
   - Expected: 8-12% speedup
   - Risk: Low (flux is deterministic)

#### **Level 2: Algorithm Simplifications (Titans-inspired)**

6. **Sparse flux gating (Titans "Surprise" optimization)**
   - Current: Computes flux for ALL tokens
   - Test: Only compute flux for tokens where |k·v| > threshold
   - Expected: 10-15% speedup on long sequences
   - Risk: Medium (may affect edge cases)

7. **Simplified cross-coupling**
   - Current: Complex cross-timescale coupling
   - Test: Simplified linear mixing (like Titans MAG)
   - Expected: 5-8% speedup
   - Risk: Medium (check score carefully)

8. **Remove intra-chunk attention for fast state**
   - Current: Both states use intra-chunk attention
   - Test: Fast state = pure RNN, Slow state = full attention
   - Rationale: Titans "Core" is simpler than "Memory"
   - Expected: 15-20% speedup
   - Risk: Medium-High (architectural change)

#### **Level 3: State Management (Titans-inspired)**

9. **Separate fast/slow processing paths**
   - Current: Dual-state processed together
   - Test: Fast state in one pass, slow state conditional
   - Rationale: Titans processes Memory only on "surprise"
   - Expected: 20-30% speedup
   - Risk: High (score may drop initially)

10. **Lazy slow state updates**
    - Current: Updates slow state every chunk
    - Test: Update slow state only when flux > threshold
    - Rationale: Titans "Memory" updates are event-driven
    - Expected: 25-40% speedup on long sequences
    - Risk: High (may need retraining)

#### **Level 4: Kernel Fusion (Advanced)**

11. **Fuse projections + convolutions**
    - Current: Separate q_proj, k_proj, v_proj, then conv
    - Test: Single fused kernel for all projections + conv
    - Expected: 10-15% speedup
    - Risk: Medium (need custom CUDA)

12. **Triton kernel for full dual-state**
    - Current: Python loop for state management
    - Test: Port entire dual-state logic to Triton
    - Expected: 2-3x speedup
    - Risk: Very High (complex kernel, may need iteration)

---

### 📊 TESTING PROTOCOL (For each change)

```bash
# 1. Make change
# 2. Run quick test
python -c "
import torch, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from scipy.stats import spearmanr
import numpy as np, time
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

device = 'cuda'
teacher = AutoModel.from_pretrained('/workspace/LAM/all-MiniLM-L6-v2').to(device)
tokenizer = AutoTokenizer.from_pretrained('/workspace/LAM/all-MiniLM-L6-v2')

layers = torch.nn.ModuleList([
    EnhancedHierarchicalDeltaNet(d_model=384, num_heads=12,
                                  use_hierarchical_decay=True, use_enhanced_flux=True)
    for _ in range(6)
]).to(device)

state_dict = torch.load('/workspace/LAM/best/pytorch_model.bin', map_location=device, weights_only=False)
layer_dict = {k.replace('deltanet_layers.', ''): v for k, v in state_dict.items() if 'deltanet_layers' in k}
for i in range(6):
    layer_state = {k[2:]: v for k, v in layer_dict.items() if k.startswith(f'{i}.')}
    if layer_state:
        layers[i].load_state_dict(layer_state, strict=False)
layers.eval()

# ... rest of test ...
"

# 3. Check results
# Score must be: 0.8190 ± 0.001
# Speed must be: < previous time
# 4. If PASS: commit change, move to next
# 5. If FAIL: revert, move to next
```

---

### 🎯 TARGET METRICS

- **Conservative Goal**: 2x speedup (17.7ms → 8.8ms)
- **Stretch Goal**: 3x speedup (17.7ms → 5.9ms)
- **Score Requirement**: 0.819 ± 0.001 (EXACT preservation)

---

### 📝 IMPLEMENTATION NOTES

#### Key Insights from Google Titans:

1. **Separation of Concerns**
   - Fast state = Reactive (handles syntax, grammar)
   - Slow state = Selective (stores important facts)
   - Don't force them to do same work!

2. **Event-Driven Memory**
   - Not every token needs slow state update
   - Use flux/surprise as trigger
   - This is where BIG speedups come from

3. **Output Mixing > Input Gating**
   - Titans mixes outputs, not inputs
   - Input gating "starves" pre-trained weights
   - Output mixing preserves gradients

4. **Kernel Fusion Priorities**
   - Fuse operations with same data access pattern
   - Don't fuse if it increases memory traffic
   - Profile before committing to custom kernels

---

### 🔍 PROFILING TARGETS

Current breakdown (13.2ms total):
- DeltaNet layers: 9.5ms (72%)
- Other ops: 3.7ms (28%)

Within DeltaNet layer:
- Projections: ?
- Convolutions: ?
- Delta rule: ?
- Cross-coupling: ?
- Output gating: ?

**Next step**: Profile individual operations to find biggest targets.

