"""
🚀 HYDRA-LAM Speed Test + STS-B Score Retention
================================================
Goal: Keep 0.8190 STS-B score with MASSIVE speed increase!

Strategy:
1. Load your trained 6-layer model (Teacher - Score 0.8190)
2. Create Hydra 1-layer model (Student - Fast)
3. Distill Teacher → Student
4. Verify score + measure speedup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from pathlib import Path

# Import your components
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

# Import fused kernel
try:
    from fused_delta_kernel import fused_delta_forward
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("⚠️ Triton not found - using PyTorch fallback")


# =============================================================================
# HYDRA-LAM: Wide & Parallel (1 Layer = 6x Faster)
# =============================================================================

class HydraLAM(nn.Module):
    """
    HYDRA-LAM: 1 Wide Layer ≈ 6 Sequential Layers
    
    Speed secret: 1 kernel launch instead of 6
    """
    
    def __init__(self, dim=384, num_heads=12):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_triton = True  # Can be toggled for training
        
        # 1. HYDRA CONV (6 parallel streams in one conv)
        self.hydra_conv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=6)
        
        # 2. HYDRA MEMORY (Q, K, V projections)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        
        # Learnable decay
        self.decay = nn.Parameter(torch.ones(1, num_heads, 1, self.head_dim) * 0.85)
        
        # 3. GLOBAL MIXER (recovers accuracy)
        self.mixer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )
        
        self.out_norm = nn.LayerNorm(dim)
    
    def forward(self, x, attention_mask=None):
        b, s, d = x.shape
        residual = x
        
        # 1. Parallel Conv
        x_conv = self.hydra_conv(x.transpose(1, 2)).transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # 2. Memory with Triton kernel
        q = self.q_proj(x_conv).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_conv).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_conv).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        g = torch.sigmoid(self.gate(x_conv)).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        
        v_gated = v * g
        w = torch.sigmoid(self.decay).expand(b, -1, s, -1)
        
        # Use Triton for inference (fast), PyTorch for training (gradients work)
        if TRITON_AVAILABLE and x.is_cuda and self.use_triton and not self.training:
            x_mem, _ = fused_delta_forward(q, k, v_gated, w)
        else:
            x_mem = self._fallback(q, k, v_gated, w)
        
        x_mem = x_mem.transpose(1, 2).contiguous().view(b, s, d)
        
        # 3. Mix + residual
        out = self.mixer(x_mem)
        return self.out_norm(out + residual)
    
    def _fallback(self, q, k, v, w):
        """PyTorch fallback (supports gradients)"""
        b, h, s, d = q.shape
        out = torch.zeros_like(q)
        state = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
        
        for t in range(s):
            state = state * w[:, :, t:t+1, :].unsqueeze(-1).mean()
            state = state + v[:, :, t, :].unsqueeze(-1) * k[:, :, t, :].unsqueeze(-2)
            out[:, :, t, :] = q[:, :, t, :] * torch.diagonal(state, dim1=-2, dim2=-1)
        return out


# =============================================================================
# TEACHER: Your Trained 6-Layer Model
# =============================================================================

class TeacherModel(nn.Module):
    """Your trained 6-layer model (Score 0.8190)"""
    
    def __init__(self, base_model, d_model=384):
        super().__init__()
        self.embeddings = base_model.embeddings
        self.deltanet_layers = nn.ModuleList([
            EnhancedHierarchicalDeltaNet(d_model=d_model, num_heads=12,
                                         use_hierarchical_decay=True, use_enhanced_flux=True)
            for _ in range(6)
        ])
        self.norms = nn.ModuleList([
            base_model.encoder.layer[i].attention.output.LayerNorm for i in range(6)
        ])
        self.ffns = nn.ModuleList([
            base_model.encoder.layer[i].intermediate for i in range(6)
        ])
        self.ffn_norms = nn.ModuleList([
            base_model.encoder.layer[i].output.LayerNorm for i in range(6)
        ])
        self.output_denses = nn.ModuleList([
            base_model.encoder.layer[i].output.dense for i in range(6)
        ])
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        for i in range(6):
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.norms[i](residual + x_attn)
            residual = x
            x_ffn = F.gelu(self.ffns[i](x))
            x_ffn = self.output_denses[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        return x
    
    def get_embeddings(self, input_ids, attention_mask):
        x = self.forward(input_ids, attention_mask)
        mask_exp = attention_mask.unsqueeze(-1).float()
        emb = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=-1)


# =============================================================================
# STUDENT: Hydra Wrapper for STS-B
# =============================================================================

class HydraStudent(nn.Module):
    """Hydra model wrapped for STS-B"""
    
    def __init__(self, embeddings, d_model=384):
        super().__init__()
        self.embeddings = embeddings
        self.hydra = HydraLAM(dim=d_model, num_heads=12)
    
    def forward(self, input_ids, attention_mask=None):
        x = self.embeddings(input_ids)
        return self.hydra(x, attention_mask)
    
    def get_embeddings(self, input_ids, attention_mask):
        x = self.forward(input_ids, attention_mask)
        mask_exp = attention_mask.unsqueeze(-1).float()
        emb = (x * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        return F.normalize(emb, p=2, dim=-1)


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def benchmark_speed(teacher, student, tokenizer, device, n_samples=100):
    """Benchmark encoding speed"""
    print("\n" + "="*60)
    print("⚡ SPEED BENCHMARK")
    print("="*60)
    
    # Generate test sentences
    sentences = [f"This is test sentence number {i} for benchmarking." for i in range(n_samples)]
    
    # Tokenize
    tokens = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    # Warmup
    with torch.no_grad():
        _ = teacher.get_embeddings(input_ids[:8], attention_mask[:8])
        _ = student.get_embeddings(input_ids[:8], attention_mask[:8])
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark Teacher (6 layers)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(3):
            _ = teacher.get_embeddings(input_ids, attention_mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    teacher_time = (time.time() - start) / 3 * 1000
    
    # Benchmark Student (Hydra)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(3):
            _ = student.get_embeddings(input_ids, attention_mask)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    student_time = (time.time() - start) / 3 * 1000
    
    speedup = teacher_time / student_time
    
    print(f"  Teacher (6-layer): {teacher_time:.2f} ms")
    print(f"  Hydra (1-layer):   {student_time:.2f} ms")
    print(f"  SPEEDUP:           {speedup:.1f}x 🚀")
    
    return teacher_time, student_time, speedup


def evaluate_stsb(model, tokenizer, device, split='validation'):
    """Evaluate STS-B score"""
    ds = load_dataset("sentence-transformers/stsb", split=split)
    s1, s2 = ds["sentence1"], ds["sentence2"]
    labels = np.array(ds["score"] if "score" in ds.column_names else ds["label"], dtype=float)
    
    model.eval()
    all_sims = []
    
    batch_size = 32
    for i in range(0, len(s1), batch_size):
        batch_s1 = s1[i:i+batch_size]
        batch_s2 = s2[i:i+batch_size]
        
        tok1 = tokenizer(batch_s1, padding=True, truncation=True, max_length=128, return_tensors='pt')
        tok2 = tokenizer(batch_s2, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
        with torch.no_grad():
            emb1 = model.get_embeddings(tok1['input_ids'].to(device), tok1['attention_mask'].to(device))
            emb2 = model.get_embeddings(tok2['input_ids'].to(device), tok2['attention_mask'].to(device))
        
        sims = F.cosine_similarity(emb1, emb2).cpu().numpy()
        all_sims.extend(sims)
    
    spearman = spearmanr(all_sims, labels)[0]
    return spearman


def distill_fast(teacher, student, tokenizer, device, steps=500, batch_size=32):
    """Quick distillation to transfer knowledge (using PyTorch fallback for gradients)"""
    print("\n" + "="*60)
    print("📚 QUICK DISTILLATION (Teacher → Student)")
    print("="*60)
    
    # Load some training data
    try:
        ds = load_dataset("sentence-transformers/stsb", split="train")
        sentences = list(ds["sentence1"]) + list(ds["sentence2"])
    except:
        sentences = [f"Sample sentence {i}" for i in range(1000)]
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)
    teacher.eval()
    student.train()  # This triggers PyTorch fallback (supports gradients)
    
    for step in range(steps):
        # Sample batch
        idx = np.random.choice(len(sentences), batch_size)
        batch = [sentences[i] for i in idx]
        
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].to(device)
        
        # Get teacher embeddings (frozen)
        with torch.no_grad():
            teacher_emb = teacher.get_embeddings(input_ids, attention_mask)
        
        # Get student embeddings (uses PyTorch fallback in train mode)
        student_emb = student.get_embeddings(input_ids, attention_mask)
        
        # MSE loss
        loss = F.mse_loss(student_emb, teacher_emb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{steps} | Loss: {loss.item():.6f}")
    
    print("  ✅ Distillation complete!")
    return student


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("🔥 HYDRA-LAM: Speed + Accuracy Test")
    print("   Goal: Keep 0.8190 score, massive speed boost!")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n📍 Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load base model and tokenizer
    print("\n📥 Loading base model...")
    base_path = "/workspace/LAM/all-MiniLM-L6-v2"
    base_model = AutoModel.from_pretrained(base_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    d_model = base_model.config.hidden_size
    
    # Create teacher (your trained 6-layer model)
    print("📥 Creating Teacher (6-layer)...")
    teacher = TeacherModel(base_model, d_model).to(device)
    
    # Load your final trained weights (0.8190 score)
    ckpt_path = "/workspace/LAM/best/pytorch_model.bin"
    print(f"   Loading: {ckpt_path}")
    
    loaded = False
    try:
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Extract deltanet_layers from the state dict
        layer_dict = {}
        for k, v in state_dict.items():
            if 'deltanet_layers.' in k:
                # Remove prefix: "deltanet_layers.0.xxx" -> "0.xxx"
                new_key = k.replace('deltanet_layers.', '')
                layer_dict[new_key] = v
        
        if layer_dict:
            teacher.deltanet_layers.load_state_dict(layer_dict, strict=False)
            print(f"   ✅ Loaded {len(layer_dict)} parameters!")
            loaded = True
        else:
            # Try loading entire state dict directly
            missing, unexpected = teacher.load_state_dict(state_dict, strict=False)
            print(f"   ✅ Loaded weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
            loaded = True
    except Exception as e:
        print(f"   ⚠️ Could not load: {e}")
        import traceback
        traceback.print_exc()
    
    if not loaded:
        print("   ⚠️ No checkpoint loaded - using random weights")
    
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Create student (Hydra - fast)
    print("📥 Creating Student (Hydra - 1 layer)...")
    student = HydraStudent(base_model.embeddings, d_model).to(device)
    
    # Evaluate teacher first
    print("\n" + "="*60)
    print("📊 EVALUATING TEACHER (Your trained model)")
    print("="*60)
    teacher_score = evaluate_stsb(teacher, tokenizer, device)
    print(f"   STS-B Spearman: {teacher_score:.4f}")
    
    # Evaluate untrained student
    print("\n" + "="*60)
    print("📊 EVALUATING STUDENT (Before distillation)")
    print("="*60)
    student_score_before = evaluate_stsb(student, tokenizer, device)
    print(f"   STS-B Spearman: {student_score_before:.4f}")
    
    # Speed benchmark (before distillation)
    teacher_time, student_time, speedup = benchmark_speed(teacher, student, tokenizer, device)
    
    # Quick distillation
    student = distill_fast(teacher, student, tokenizer, device, steps=500)
    
    # Evaluate after distillation
    print("\n" + "="*60)
    print("📊 EVALUATING STUDENT (After distillation)")
    print("="*60)
    student.eval()
    student_score_after = evaluate_stsb(student, tokenizer, device)
    print(f"   STS-B Spearman: {student_score_after:.4f}")
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 FINAL RESULTS")
    print("="*60)
    print(f"  Teacher (6-layer) Score: {teacher_score:.4f}")
    print(f"  Student (Hydra) Score:   {student_score_after:.4f}")
    print(f"  Score Retention:         {(student_score_after/teacher_score)*100:.1f}%")
    print(f"  Speed Improvement:       {speedup:.1f}x 🚀")
    print("="*60)
    
    if student_score_after >= 0.80:
        print("\n✅ SUCCESS! Score ≥ 0.80 with massive speed boost!")
    else:
        print("\n💡 TIP: Train longer (more distillation steps) for higher score")
        print("   Try: distill_fast(teacher, student, tokenizer, device, steps=2000)")
    
    # Save the student model
    save_path = "/workspace/LAM/hydra_student.pt"
    torch.save({
        'model_state_dict': student.state_dict(),
        'score': student_score_after,
        'speedup': speedup
    }, save_path)
    print(f"\n💾 Saved student model to: {save_path}")


if __name__ == "__main__":
    main()

