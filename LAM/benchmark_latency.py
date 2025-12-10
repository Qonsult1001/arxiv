import torch
import time
from final_solution_formula_final import EnhancedHierarchicalDeltaNet
from latent_brain_poc import LatentBrainDeltaNet

def synchronize_device(device):
    """Synchronize device (GPU only, CPU doesn't need synchronization)"""
    if device.type == 'cuda':
        torch.cuda.synchronize()

def benchmark():
    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Benchmarking on: {device} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"💻 Benchmarking on: {device} (CPU)")
        print("   Note: CPU benchmarks will be slower than GPU")

    config = {
        "d_model": 384,
        "num_heads": 12
    }

    # Initialize models
    print("\nInitializing Baseline (Enhanced Hierarchical DeltaNet)...")
    try:
        baseline = EnhancedHierarchicalDeltaNet(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            use_short_conv=True,
            use_hierarchical_decay=True,
            use_enhanced_flux=True,
        ).to(device)
        baseline.eval()  # Set to eval mode for consistent benchmarking
        print("   ✅ Baseline model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load Baseline: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Initializing Latent Brain (POC)...")
    try:
        latent_brain = LatentBrainDeltaNet(
            d_model=config["d_model"],
            num_heads=config["num_heads"]
        ).to(device)
        latent_brain.eval()  # Set to eval mode for consistent benchmarking
        print("   ✅ Latent Brain model loaded")
    except Exception as e:
        print(f"   ❌ Failed to load Latent Brain: {e}")
        import traceback
        traceback.print_exc()
        return

    # Adjust parameters based on device
    if device.type == 'cpu':
        # CPU: smaller batch size and fewer iterations for reasonable runtime
        seq_lengths = [128, 256, 512]
        batch_size = 1
        iterations = 3
        print("\n   ⚠️  Using reduced parameters for CPU benchmarking")
    else:
        # GPU: can handle larger workloads
        seq_lengths = [128, 512, 1024]
        batch_size = 2
        iterations = 5

    print("\n" + "="*70)
    print(f"{'Seq Len':<10} | {'Baseline (ms)':<15} | {'Latent Brain (ms)':<18} | {'Speedup':<10}")
    print("-" * 70)

    for seq_len in seq_lengths:
        # Create dummy input
        x = torch.randn(batch_size, seq_len, config["d_model"], device=device)
        mask = torch.ones(batch_size, seq_len, device=device)

        # Warmup runs (more warmup for GPU to ensure stable measurements)
        warmup_runs = 3 if device.type == 'cuda' else 1
        print(f"\n   Warming up with seq_len={seq_len}...", end=" ", flush=True)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = baseline(x, attention_mask=mask)
                _ = latent_brain(x, attention_mask=mask)
        synchronize_device(device)
        print("✅")

        # Measure Baseline
        synchronize_device(device)
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = baseline(x, attention_mask=mask)
        synchronize_device(device)
        baseline_time = (time.time() - start) / iterations * 1000

        # Measure Latent Brain
        synchronize_device(device)
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = latent_brain(x, attention_mask=mask)
        synchronize_device(device)
        brain_time = (time.time() - start) / iterations * 1000

        speedup = baseline_time / brain_time if brain_time > 0 else 0.0

        print(f"{seq_len:<10} | {baseline_time:<15.2f} | {brain_time:<18.2f} | {speedup:<10.2f}x")
    
    print("="*70)
    print(f"\n✅ Benchmark complete!")
    if device.type == 'cpu':
        print("   💡 For faster benchmarks, consider using GPU if available")

if __name__ == "__main__":
    benchmark()