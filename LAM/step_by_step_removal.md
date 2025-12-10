# Step-by-Step Sequential Loop Removal

## Current Loop Structure (Lines 400-482)

### SECTION 1: Decay Computation (Lines 413-419) ❌ IN LOOP
```python
fast_decay_factor = fast_decay_i.mean(-1, keepdim=True).unsqueeze(-1)
slow_decay_factor = slow_decay_i.mean(-1, keepdim=True).unsqueeze(-1)
psi_expanded = psi_i.unsqueeze(-1).unsqueeze(-1)
fast_decay_modulated = fast_decay_factor * (1 - 0.1 * psi_expanded)
slow_decay_modulated = slow_decay_factor * (1 - 0.05 * psi_expanded)
```
**Status**: Can move OUTSIDE loop ✅

### SECTION 2: State Decay (Lines 422-423) ❌ SEQUENTIAL
```python
S_fast = S_fast * fast_decay_modulated
S_slow = S_slow * slow_decay_modulated
```
**Status**: TRUE recurrence - MUST stay ❌

### SECTION 3: State Normalization (Lines 426-427) ❌ IN LOOP
```python
S_fast_read = S_fast / (S_fast.norm(dim=(-2, -1), keepdim=True) + 1e-8)
S_slow_read = S_slow / (S_slow.norm(dim=(-2, -1), keepdim=True) + 1e-8)
```
**Status**: Depends on S, but COULD be removed and normalized only at END ✅

### SECTION 4: Delta Rule (Lines 432-437) ❌ SEQUENTIAL
```python
u_i_fast = u[:, :, i] - w[:, :, i] @ S_fast_read
u_i_slow = u[:, :, i] - w[:, :, i] @ S_slow_read
```
**Status**: Depends on S_read - MUST stay ❌

### SECTION 5: Output Readout (Lines 439-443) ❌ SEQUENTIAL
```python
o_inter_fast = q_i @ S_fast_read
o_fast = fast_gate_i * (o_inter_fast + attn @ u_i_fast)
```
**Status**: Depends on S_read and u_i - MUST stay ❌

### SECTION 6: Alpha Blending (Lines 449-454) ❌ IN LOOP
```python
alpha = 0.5 + 0.3 * token_flux_i
beta_weight = 1.0 - alpha
o_chunk = alpha * o_fast + beta_weight * o_slow
```
**Status**: Alpha computation can move OUTSIDE ✅, blending depends on o

### SECTION 7: State Update (Lines 458-467) ❌ SEQUENTIAL
```python
update_fast = k_i.transpose(-1, -2) @ u_i_fast
S_fast = S_fast + update_fast
```
**Status**: TRUE recurrence - MUST stay ❌

### SECTION 8: Cross-Coupling (Lines 469-474) ❓ MAYBE REMOVABLE
```python
cross_influence = 0.05 + 0.1 * psi_i.mean()
cross_update_fast = cross_influence * psi_expanded * S_slow
cross_update_slow = cross_influence * (1 - psi_expanded) * S_fast
S_fast = S_fast + cross_update_fast
S_slow = S_slow + cross_update_slow
```
**Status**: This is the "Titans coupling" - could be REMOVED or made conditional ✅

### SECTION 9: Final Norm (Lines 478-481) ❌ IN LOOP
```python
S_fast_norm = S_fast.norm(dim=(-2, -1), keepdim=True) + 1e-8
S_fast = S_fast / S_fast_norm
```
**Status**: Could REMOVE from loop, normalize only at END ✅

---

## 🎯 STEP-BY-STEP OPTIMIZATION PLAN

### STEP 1: Remove cross-coupling entirely (Lines 469-474)
**Why**: This is Titans-style "decoupling". Fast and Slow states don't need to exchange info every chunk.
**Risk**: Low - cross-coupling coefficient is only 0.05-0.15
**Expected gain**: Remove 4 tensor ops per chunk

### STEP 2: Remove per-chunk final normalization (Lines 478-481)
**Why**: Normalize only at the END, not every chunk
**Risk**: Medium - might cause numerical instability
**Expected gain**: Remove 4 tensor ops per chunk

### STEP 3: Remove per-chunk readout normalization (Lines 426-427)
**Why**: If we keep final norm at end, readout norm is redundant
**Risk**: High - might break delta rule semantics
**Expected gain**: Remove 2 tensor ops per chunk

### STEP 4: Make slow state update conditional (Titans "surprise")
**Why**: Titans only updates memory on "surprising" events
**Risk**: Medium - needs testing
**Expected gain**: Skip slow state update ~80% of chunks

---

## LET'S DO STEP 1 FIRST: Remove cross-coupling

