import torch
from train_6layer_deltanet_2 import DeltaNetPure6Layer

# Quick test for long sequence support
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Testing on: {device}")

config = {
    "teacher_model": "sentence-transformers/all-MiniLM-L6-v2",
    "num_linear_layers": 6,
    "d_model": 384,
    "num_heads": 12,
    "fast_decay_init": 0.9,
    "slow_decay_init": 0.999,
    "use_kernel_blending": False,
}

print("\n🟢 Initializing DeltaNet...")
teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"
deltanet = DeltaNetPure6Layer(
    teacher_model_name=teacher_model_path,
    num_linear_layers=config["num_linear_layers"],
    config=config
).to(device)
deltanet.eval()
print("   ✅ DeltaNet loaded\n")

# Test different sequence lengths
test_lengths = [128, 512, 1024, 2048, 4096]

print("="*60)
print(f"{'Seq Length':<12} | {'Status':<20} | {'Output Shape':<25}")
print("-"*60)

for seq_len in test_lengths:
    try:
        # Create dummy input
        batch_size = 2
        input_ids = torch.randint(0, 30522, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Forward pass
        with torch.no_grad():
            student_emb, teacher_emb, _, _, _ = deltanet(input_ids, attention_mask)
        
        print(f"{seq_len:<12} | ✅ SUCCESS{'':<12} | {str(student_emb.shape):<25}")
    except Exception as e:
        print(f"{seq_len:<12} | ❌ FAILED{'':<12} | {str(e)[:25]:<25}")

print("="*60)
print("\n✅ Long sequence test complete!")
