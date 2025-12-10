import torch
import time

def profile_components():
    B, H, L, D = 1, 12, 16384, 64
    CHUNK = 32
    
    q = torch.randn(B, H, L, D, device='cuda')
    k = torch.randn(B, H, L, D, device='cuda')
    v = torch.randn(B, H, L, D, device='cuda')
    
    print(f"🔍 PROFILING BREAKDOWN (Seq Len: {L})")
    print("-" * 40)
    
    # 1. Test Chunk Math Overhead
    # Doing the (I + K^T K) math in Python
    torch.cuda.synchronize()
    start = time.time()
    num_chunks = L // CHUNK
    for _ in range(10): # Just 10 chunks to estimate
        k_c = k[:, :, :CHUNK]
        kt = k_c.transpose(-1, -2)
        M = -torch.matmul(k_c, kt)
        # Force sync to measure calc time
        _ = M + M 
    torch.cuda.synchronize()
    chunk_math_time = (time.time() - start) / 10 * num_chunks
    print(f"1. Python Chunk Math (Est):  {chunk_math_time*1000:.2f} ms")
    
    # 2. Test Recurrence Overhead
    # Moving S between steps
    S = torch.zeros(B, H, D, D, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_chunks):
        S = S + 1.0 # Dummy update
    torch.cuda.synchronize()
    recurrence_time = (time.time() - start)
    print(f"2. Python Loop Overhead:     {recurrence_time*1000:.2f} ms")
    
    print("-" * 40)
    print(f"🛑 TOTAL Python Latency:     ~{(chunk_math_time + recurrence_time)*1000:.2f} ms")
    print(f"🚀 TARGET Titan v4 Latency:  ~15.00 ms")

if __name__ == "__main__":
    profile_components()