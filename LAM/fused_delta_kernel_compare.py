import torch
import time
import sys
import os

# Import the fused delta kernel
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fused_delta_kernel import fused_delta_update

def test_numerical_correctness():
    """Test that fused kernel matches reference PyTorch implementation"""
    print("🧪 TESTING: Numerical Correctness")
    print("=" * 60)
    
    # Config
    BATCH = 2
    HEADS = 4
    DIM = 32        # Head Dim
    SEQ_LEN = 128   # Short sequence for correctness test
    DEVICE = 'cuda'
    
    if not torch.cuda.is_available():
        print("❌ Error: CUDA GPU required for test")
        return False

    # Create test data with fixed seed for reproducibility
    torch.manual_seed(42)
    k = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, device=DEVICE)
    v = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, device=DEVICE)
    decay = torch.sigmoid(torch.randn(BATCH, HEADS, SEQ_LEN, DIM, device=DEVICE))
    initial_state = torch.zeros(BATCH, HEADS, DIM, DIM, device=DEVICE)

    # --- Reference PyTorch Implementation (matches fused kernel logic) ---
    def reference_delta_rule(k, v, decay, state):
        """Reference implementation matching the fused kernel exactly"""
        b, h, l, d = k.shape
        curr_state = state.clone()
        
        for t in range(l):
            k_t = k[:, :, t]  # [b, h, d]
            v_t = v[:, :, t]  # [b, h, d]
            w_t = decay[:, :, t]  # [b, h, d]
            
            # Delta Rule (matching fused kernel)
            # 1. Readout: r = k @ S
            r = torch.einsum('bhd,bhdd->bhd', k_t, curr_state)
            
            # 2. Delta: u = v - r
            u = v_t - r
            
            # 3. Decay: S = S * w_mean (mean decay like fused kernel)
            w_mean = w_t.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [b, h, 1, 1]
            curr_state = curr_state * w_mean
            
            # 4. Update: S = S + u^T @ k (outer product)
            update = torch.einsum('bhd,bhe->bhde', u, k_t)  # [b, h, d, d]
            curr_state = curr_state + update
            
        return curr_state

    print(f"Input Shape: [Batch={BATCH}, Heads={HEADS}, Len={SEQ_LEN}, Dim={DIM}]")
    
    # Run reference implementation
    print("\n1. Running Reference PyTorch Implementation...")
    ref_state = reference_delta_rule(k, v, decay, initial_state)
    
    # Run fused kernel
    print("2. Running Fused Triton Kernel...")
    fused_state = fused_delta_update(k, v, decay, initial_state)
    
    # Compare results
    print("\n3. Comparing Results...")
    max_diff = (ref_state - fused_state).abs().max().item()
    mean_diff = (ref_state - fused_state).abs().mean().item()
    rel_diff = (ref_state - fused_state).abs() / (ref_state.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    
    print(f"   Max Absolute Difference: {max_diff:.2e}")
    print(f"   Mean Absolute Difference: {mean_diff:.2e}")
    print(f"   Max Relative Difference: {max_rel_diff:.2e}")
    
    # Tolerance check
    TOLERANCE = 1e-4
    passed = max_diff < TOLERANCE
    
    if passed:
        print(f"\n✅ TEST PASSED: Differences within tolerance ({TOLERANCE})")
    else:
        print(f"\n❌ TEST FAILED: Differences exceed tolerance ({TOLERANCE})")
        print(f"   State shapes: Reference {ref_state.shape}, Fused {fused_state.shape}")
        print(f"   Sample values:")
        print(f"   Reference[0,0,0,0] = {ref_state[0,0,0,0].item():.6f}")
        print(f"   Fused[0,0,0,0]     = {fused_state[0,0,0,0].item():.6f}")
    
    return passed

def benchmark_performance():
    """Benchmark performance comparison"""
    print("\n🚀 BENCHMARKING: PyTorch Loop vs. Triton Fused Kernel")
    print("=" * 60)
    
    # Config
    BATCH = 8
    HEADS = 12
    DIM = 64        # Head Dim
    SEQ_LEN = 4096  # Long sequence context
    DEVICE = 'cuda'
    
    if not torch.cuda.is_available():
        print("❌ Error: CUDA GPU required for Triton benchmark")
        return

    # Create dummy data
    torch.manual_seed(42)
    k = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, device=DEVICE)
    v = torch.randn(BATCH, HEADS, SEQ_LEN, DIM, device=DEVICE)
    decay = torch.sigmoid(torch.randn(BATCH, HEADS, SEQ_LEN, DIM, device=DEVICE))
    state = torch.zeros(BATCH, HEADS, DIM, DIM, device=DEVICE)

    # --- PyTorch Implementation (Reference) ---
    def pytorch_loop_impl(k, v, decay, state):
        b, h, l, d = k.shape
        curr_state = state.clone()
        for t in range(l):
            k_t = k[:, :, t]
            v_t = v[:, :, t]
            w_t = decay[:, :, t].mean(-1, keepdim=True).unsqueeze(-1)
            
            # Delta Rule
            r = (k_t.unsqueeze(-2) @ curr_state).squeeze(-2)
            u = v_t - r
            curr_state = curr_state * w_t + (u.unsqueeze(-1) @ k_t.unsqueeze(-2))
        return curr_state

    # Warmup
    print(f"Input Shape: [Batch={BATCH}, Heads={HEADS}, Len={SEQ_LEN}, Dim={DIM}]")
    print("Warming up PyTorch...")
    pytorch_loop_impl(k[:,:,:128], v[:,:,:128], decay[:,:,:128], state)
    
    print("Warming up Triton...")
    fused_delta_update(k[:,:,:128], v[:,:,:128], decay[:,:,:128], state)
    torch.cuda.synchronize()
    
    # Measure PyTorch (Reduced length for sanity, it's slow)
    print("\n1. Running PyTorch Loop (SeqLen=512 only, otherwise too slow)...")
    torch.cuda.synchronize()
    start = time.time()
    pytorch_loop_impl(k[:,:,:512], v[:,:,:512], decay[:,:,:512], state)
    torch.cuda.synchronize()
    py_time = time.time() - start
    print(f"   ⏱️  PyTorch Time (512 tokens): {py_time:.4f}s")

    # Measure Triton (Full Length)
    print(f"\n2. Running Triton Kernel (Full SeqLen={SEQ_LEN})...")
    torch.cuda.synchronize()
    start = time.time()
    fused_delta_update(k, v, decay, state)
    torch.cuda.synchronize()
    triton_time = time.time() - start
    print(f"   ⏱️  Triton Time ({SEQ_LEN} tokens): {triton_time:.4f}s")
    
    # Extrapolate
    py_projected = py_time * (SEQ_LEN / 512)
    speedup = py_projected / triton_time
    
    print("\n" + "=" * 60)
    print(f"⚡ SPEEDUP RESULT: {speedup:.1f}x Faster")
    print("=" * 60)
    print("Note: The Triton kernel scales linearly. You can likely")
    print("double the sequence length with negligible memory increase.")

if __name__ == "__main__":
    # Run correctness test first
    correctness_passed = test_numerical_correctness()
    
    # Run performance benchmark
    if correctness_passed:
        benchmark_performance()
    else:
        print("\n⚠️  Skipping performance benchmark due to correctness test failure")