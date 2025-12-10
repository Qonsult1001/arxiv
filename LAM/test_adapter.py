"""
🧪 TEST HOLOGRAPHIC ADAPTER
===========================
Tests the trained adapter that converts:
Raw Memory State (S_slow) -> Semantic Embedding (MiniLM Space)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys

# Import your stack
sys.path.insert(0, str(Path(__file__).parent))
from train_6layer_deltanet_2 import DeltaNetPure6Layer, config
from infinite_streamer_async import AsyncInfiniteStreamer

class HolographicAdapter(nn.Module):
    def __init__(self, input_dim=384, output_dim=384):
        super().__init__()
        # Enhanced adapter with 2 layers for better capacity
        self.proj1 = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.proj2 = nn.Linear(output_dim, output_dim)
        # Initialize close to identity
        nn.init.eye_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.eye_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, x):
        x = self.proj1(x)
        x = self.activation(x)
        x = self.proj2(x)
        return x

def test_adapter():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧪 Testing Holographic Adapter on {device}...")
    print("="*80)
    
    # 1. Load Model
    print("\n1️⃣  Loading model...")
    model_path = str(Path(__file__).parent / "all-MiniLM-L6-v2")
    model = DeltaNetPure6Layer(model_path, 6, config).to(device)
    
    # Load weights
    weights = torch.load("/workspace/LAM/best/deltanet_shockwave_result.pt", map_location=device)
    model_state = model.state_dict()
    compatible_weights = {}
    for k, v in weights.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                compatible_weights[k] = v
            elif 'W_bilinear' in k and len(v.shape) == 2 and len(model_state[k].shape) == 3:
                num_heads = model_state[k].shape[0]
                compatible_weights[k] = v.unsqueeze(0).expand(num_heads, -1, -1).clone()
    
    model.load_state_dict(compatible_weights, strict=False)
    model.eval()
    print("   ✅ Model loaded")
    
    # 2. Load Adapter
    print("\n2️⃣  Loading adapter...")
    adapter = HolographicAdapter().to(device)
    adapter_path = Path(__file__).parent / "holographic_adapter.pt"
    
    if not adapter_path.exists():
        print(f"   ❌ Adapter not found at {adapter_path}")
        print("   💡 Run train_retrieval_finetune.py first to train the adapter")
        return
    
    adapter.load_state_dict(torch.load(adapter_path, map_location=device))
    adapter.eval()
    print("   ✅ Adapter loaded")
    
    # 3. Initialize Streamer
    print("\n3️⃣  Initializing streamer...")
    streamer = AsyncInfiniteStreamer(model, chunk_size=512)
    print("   ✅ Streamer ready")
    
    # 4. Test on sample texts
    print("\n4️⃣  Testing on sample texts...")
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language.",
        "The weather today is sunny and warm.",
        "Deep learning models require large amounts of data."
    ]
    
    print(f"   Testing {len(test_texts)} texts...")
    
    # Get embeddings using both methods
    adapter_embeddings = []
    teacher_embeddings = []
    streamer_embeddings = []
    
    with torch.no_grad():
        for text in test_texts:
            # Method 1: Teacher embeddings (target)
            encoded = model.tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(device)
            teacher_emb, _ = model.forward_teacher(encoded['input_ids'], encoded['attention_mask'])
            teacher_embeddings.append(teacher_emb)
            
            # Method 2: Streamer -> Adapter (what we trained)
            enc = model.tokenizer.encode(text, add_special_tokens=True)
            ids = torch.tensor([enc], device=device)
            streamer_emb = streamer.stream_embedding(ids, verbose=False)
            streamer_embeddings.append(streamer_emb)
            
            # Apply adapter
            adapter_emb = adapter(streamer_emb)
            adapter_embeddings.append(adapter_emb)
    
    # Stack all embeddings
    teacher_emb = torch.cat(teacher_embeddings, dim=0)
    adapter_emb = torch.cat(adapter_embeddings, dim=0)
    streamer_emb = torch.cat(streamer_embeddings, dim=0)
    
    # 5. Compute metrics
    print("\n5️⃣  Computing alignment metrics...")
    
    # Cosine similarity between adapter output and teacher
    adapter_cosine = F.cosine_similarity(adapter_emb, teacher_emb, dim=1)
    adapter_avg_cosine = adapter_cosine.mean().item()
    
    # MSE between adapter output and teacher
    adapter_mse = F.mse_loss(adapter_emb, teacher_emb).item()
    
    # Cosine similarity between raw streamer and teacher (baseline)
    streamer_cosine = F.cosine_similarity(streamer_emb, teacher_emb, dim=1)
    streamer_avg_cosine = streamer_cosine.mean().item()
    
    # Improvement
    improvement = adapter_avg_cosine - streamer_avg_cosine
    
    print(f"\n📊 Results:")
    print(f"   Raw Streamer → Teacher Cosine: {streamer_avg_cosine:.4f}")
    print(f"   Adapter → Teacher Cosine:      {adapter_avg_cosine:.4f}")
    print(f"   Improvement:                    {improvement:+.4f}")
    print(f"   MSE Loss:                      {adapter_mse:.6f}")
    
    # 6. Test similarity preservation
    print("\n6️⃣  Testing similarity preservation...")
    
    # Compute pairwise similarities
    teacher_sim = torch.mm(teacher_emb, teacher_emb.t())
    adapter_sim = torch.mm(adapter_emb, adapter_emb.t())
    
    # Correlation between similarity matrices
    teacher_sim_flat = teacher_sim[torch.triu(torch.ones_like(teacher_sim), diagonal=1) == 1]
    adapter_sim_flat = adapter_sim[torch.triu(torch.ones_like(adapter_sim), diagonal=1) == 1]
    
    similarity_correlation = torch.corrcoef(torch.stack([teacher_sim_flat, adapter_sim_flat]))[0, 1].item()
    
    print(f"   Similarity matrix correlation: {similarity_correlation:.4f}")
    print(f"   (Higher is better - measures if adapter preserves relative similarities)")
    
    # 7. Summary
    print("\n" + "="*80)
    print("✅ TEST SUMMARY")
    print("="*80)
    
    if adapter_avg_cosine > 0.95:
        print("   🎉 EXCELLENT: Adapter aligns very well with teacher embeddings!")
    elif adapter_avg_cosine > 0.90:
        print("   ✅ GOOD: Adapter aligns well with teacher embeddings")
    elif adapter_avg_cosine > 0.80:
        print("   ⚠️  FAIR: Adapter alignment could be improved")
    else:
        print("   ❌ POOR: Adapter needs more training")
    
    if improvement > 0.05:
        print(f"   🚀 Great improvement over raw streamer ({improvement:.4f})")
    elif improvement > 0:
        print(f"   ✅ Some improvement over raw streamer ({improvement:.4f})")
    else:
        print(f"   ⚠️  Adapter performs worse than raw streamer")
    
    print(f"\n💡 The adapter is ready to use in your streamer!")
    print(f"   Use: adapter(streamer_embedding) to get semantic embeddings")

if __name__ == "__main__":
    test_adapter()

