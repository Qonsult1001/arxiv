# Fast State Update Implementation Plan
# Goal: Eliminate chunk loop while preserving state dependencies

## Current Bottleneck
The chunk loop (lines 429-485 in final_solution_formula_final.py) processes chunks sequentially:
```
for i in range(num_chunks):  # Typically 2-4 iterations
    S_fast = S_fast * decay  # State update
    S_slow = S_slow * decay
    # ... compute outputs using states
    S_fast += k @ v  # Accumulate new information
    S_slow += k @ v
```

## Problem
- Python loop overhead
- Sequential dependencies prevent parallelization
- torch.compile can't optimize across loop iterations

## Solution: Block-Diagonal Recurrence Matrix

Instead of iterating, we can reformulate as a single matrix operation:

### Mathematical Reformulation
Current (sequential):
```
S[0] = S_init
S[1] = decay[0] * S[0] + update[0]
S[2] = decay[1] * S[1] + update[1]
S[3] = decay[2] * S[2] + update[2]
```

Equivalent (parallel):
```
S = [S_init, 0, 0, 0] + 
    [0, decay[0]*S_init, 0, 0] + 
    [0, 0, decay[1]*decay[0]*S_init, 0] +
    ...
    + [update[0], decay[0]*update[0], decay[1]*decay[0]*update[0], ...]
```

This can be expressed as a **lower triangular matrix multiplication**:
```
[S[0]]   [1    0    0    0  ]   [S_init  ]   [update[0]]
[S[1]] = [d[0] 1    0    0  ] @ [S_init  ] + [update[1]]
[S[2]]   [d[1] d[0] 1    0  ]   [S_init  ]   [update[2]]
[S[3]]   [d[2] d[1] d[0] 1  ]   [S_init  ]   [update[3]]
```

Where d[i] = decay[i] * decay[i-1] * ... * decay[0]

### Implementation Strategy

1. **Pre-compute cumulative decay products** using `torch.cumprod`
2. **Build lower triangular decay matrix** (sparse, can use efficient kernels)
3. **Single matrix multiplication** instead of loop
4. **All operations stay in latent space** (384D)

### Expected Speedup
- **Current**: O(n) sequential Python loop
- **New**: O(1) single matrix op with optimized CUDA kernels
- **Estimated**: 3-5x faster for typical sequence lengths

### Quality Preservation
- ✅ Exact same mathematical result
- ✅ All state dependencies preserved
- ✅ No approximations
- ✅ Maintains Delta Rule error correction

## Implementation Steps
1. Replace chunk loop with cumulative product formulation
2. Use torch.tril for efficient lower triangular operations
3. Leverage cuBLAS for matrix multiplications
4. Keep all intermediate values in latent space

## Validation
- Run verify_from_scratch.py to ensure Spearman ≥ 0.74
- Run benchmark to measure speedup
- Verify gradients flow correctly
