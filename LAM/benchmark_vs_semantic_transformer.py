import torch
import time
from sentence_transformers import SentenceTransformer
from train_6layer_deltanet_2 import DeltaNetPure6Layer

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
        "teacher_model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_linear_layers": 6,
        "d_model": 384,
        "num_heads": 12,
        "fast_decay_init": 0.9,
        "slow_decay_init": 0.999,
        "use_kernel_blending": False,
    }

    # Initialize Semantic Transformer (Baseline)
    print("\n🔵 Initializing Semantic Transformer (all-MiniLM-L6-v2)...")
    try:
        semantic_transformer = SentenceTransformer(config["teacher_model"], device=str(device))
        semantic_transformer.eval()
        print("   ✅ Semantic Transformer loaded")
    except Exception as e:
        print(f"   ❌ Failed to load Semantic Transformer: {e}")
        import traceback
        traceback.print_exc()
        return

    # Initialize Your DeltaNet
    print("🟢 Initializing Your DeltaNet (6-layer)...")
    try:
        # Load the teacher model path
        teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"
        
        deltanet = DeltaNetPure6Layer(
            teacher_model_name=teacher_model_path,
            num_linear_layers=config["num_linear_layers"],
            config=config
        ).to(device)
        deltanet.eval()
        print("   ✅ DeltaNet loaded")
    except Exception as e:
        print(f"   ❌ Failed to load DeltaNet: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test sentences
    test_sentences = [
        "This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand human language.",
    ]
    
    # Test different sequence lengths (max 512 for Semantic Transformer)
    seq_lengths = [128, 256, 512]
    batch_size = 4  # Fixed batch size
    iterations = 10

    print("\n" + "="*80)
    print(f"{'Seq Length':<12} | {'Semantic Transformer (ms)':<25} | {'Your DeltaNet (ms)':<20} | {'Speedup':<10}")
    print("-" * 80)

    for seq_len in seq_lengths:
        # Create test sentence of approximately the target length
        # Average word is ~5 chars, so we need seq_len * 5 / 4 words (accounting for spaces)
        num_words = (seq_len * 5) // 6
        test_sentence = " ".join(["word"] * num_words)
        sentences = [test_sentence] * batch_size
        
        # Warmup runs
        warmup_runs = 3 if device.type == 'cuda' else 1
        print(f"\n   Warming up with seq_len={seq_len}...", end=" ", flush=True)
        try:
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = semantic_transformer.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
                    # For DeltaNet, use inference-only encode() method (pure latent space)
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])
                    tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=seq_len).to(device)
                    _ = deltanet.encode(tokens['input_ids'], tokens['attention_mask'])
            synchronize_device(device)
            print("✅")
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue

        # Measure Semantic Transformer
        try:
            synchronize_device(device)
            start = time.time()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = semantic_transformer.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
            synchronize_device(device)
            transformer_time = (time.time() - start) / iterations * 1000
        except Exception as e:
            print(f"{seq_len:<12} | ERROR: {str(e)[:20]:<25} | -                    | -")
            continue

        # Measure Your DeltaNet
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config["teacher_model"])
            tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=seq_len).to(device)
            
            synchronize_device(device)
            start = time.time()
            with torch.no_grad():
                for _ in range(iterations):
                    _ = deltanet.encode(tokens['input_ids'], tokens['attention_mask'])
            synchronize_device(device)
            deltanet_time = (time.time() - start) / iterations * 1000

            speedup = transformer_time / deltanet_time if deltanet_time > 0 else 0.0

            print(f"{seq_len:<12} | {transformer_time:<25.2f} | {deltanet_time:<20.2f} | {speedup:<10.2f}x")
        except Exception as e:
            print(f"{seq_len:<12} | {transformer_time:<25.2f} | ERROR: {str(e)[:15]:<20} | -")
    
    
    print("="*80)
    print(f"\n✅ Benchmark complete!")
    if device.type == 'cpu':
        print("   💡 For faster benchmarks, consider using GPU if available")

if __name__ == "__main__":
    benchmark()
