
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ... (Imports and Helper Classes from final_solution_formula.py) ...
# I will copy the helper classes (RMSNorm, ShortConvolution, etc.) here for standalone testing
# or import them if possible. For now, I'll define the minimal set needed.

def l2norm(x, dim=-1):
    return F.normalize(x, p=2, dim=dim)

class EnhancedResonanceFlux(nn.Module):
    def __init__(self, d_k, d_v, num_heads):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.W_bilinear = nn.Parameter(torch.randn(d_k, d_v) / math.sqrt(d_k * d_v))
        self.temp = nn.Parameter(torch.ones(num_heads) * math.sqrt(d_k))
        self.flux_net = nn.Sequential(
            nn.Linear(d_k // 2, 1),
            nn.Sigmoid()
        )
        self.token_flux_proj = nn.Sequential(
            nn.Linear(d_k, d_k // 2),
            nn.SiLU(),
            nn.Linear(d_k // 2, 1),
            nn.Sigmoid()
        )

    def compute_token_flux(self, k):
        return self.token_flux_proj(k).clamp(0.01, 0.99)
        
    def modulate_keys(self, k):
        # Placeholder for modulation logic
        return k

def parallel_delta_rule(
    q, k, v, beta, fast_decay, slow_decay, 
    fast_gate, slow_gate, resonance_flux,
    inter_block_decay_fast, inter_block_decay_slow,
    brain_state_prev=None
):
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    
    q = l2norm(q)
    k = l2norm(k)
    
    beta_expanded = beta.unsqueeze(-1)
    v_scaled = v * beta_expanded
    k_beta = k * beta_expanded
    
    # Token Flux
    token_flux = resonance_flux.compute_token_flux(k_beta) # [b, h, l, 1]
    
    # Modulate keys (Simplified for debug)
    k_modulated = k_beta
    
    # Geometric Decay (CumSum)
    log_fast = torch.log(fast_decay + 1e-6)
    log_slow = torch.log(slow_decay + 1e-6)
    
    fast_cumsum = torch.cumsum(log_fast, dim=-1)
    slow_cumsum = torch.cumsum(log_slow, dim=-1)
    
    fast_total = fast_cumsum[:, :, -1].unsqueeze(-1).unsqueeze(-1)
    slow_total = slow_cumsum[:, :, -1].unsqueeze(-1).unsqueeze(-1)
    
    fast_cumsum_expanded = fast_cumsum.unsqueeze(-1)
    slow_cumsum_expanded = slow_cumsum.unsqueeze(-1)
    
    fast_weights = torch.exp(fast_total - fast_cumsum_expanded)
    slow_weights = torch.exp(slow_total - slow_cumsum_expanded)
    
    # Apply weights
    k_fast = k_modulated * fast_weights
    k_slow = k_modulated * slow_weights
    
    # Aggregation (Hebbian)
    M_fast_curr = torch.matmul(k_fast.transpose(-1, -2), v_scaled)
    M_slow_curr = torch.matmul(k_slow.transpose(-1, -2), v_scaled)
    
    # Normalization (The potential fix!)
    # Hebbian aggregation scales with Sequence Length. We must normalize.
    # Option 1: Divide by sum of weights
    fast_weight_sum = fast_weights.sum(dim=2).unsqueeze(-1) + 1e-6
    slow_weight_sum = slow_weights.sum(dim=2).unsqueeze(-1) + 1e-6
    
    M_fast_curr = M_fast_curr / fast_weight_sum
    M_slow_curr = M_slow_curr / slow_weight_sum
    
    # Continuous Growth
    if brain_state_prev is not None:
        M_fast_prev, M_slow_prev = brain_state_prev
        M_fast = M_fast_prev * inter_block_decay_fast + M_fast_curr
        M_slow = M_slow_prev * inter_block_decay_slow + M_slow_curr
    else:
        M_fast = M_fast_curr
        M_slow = M_slow_curr
        
    # Readout
    o_fast = torch.matmul(q, M_fast)
    o_slow = torch.matmul(q, M_slow)
    
    # Blending
    alpha = 0.5 + 0.3 * token_flux
    o_blend = alpha * o_fast + (1 - alpha) * o_slow
    
    return o_blend, (M_fast, M_slow)

    return o_blend, (M_fast, M_slow)

# --- Training Loop (Copied from verify_from_scratch.py) ---

import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
import numpy as np

# Configuration
BATCH_SIZE = 16
SEQ_LEN = 64
HIDDEN_SIZE = 384
NUM_HEADS = 12
STEPS = 100
EVAL_INTERVAL = 10
LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running Parallel Debug on {device}")

# Load Teacher
TEACHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
teacher_model = AutoModel.from_pretrained(TEACHER_MODEL).to(device)
teacher_model.eval()
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

# Validation Sentences
val_sentences = [
    ("A man is playing a guitar.", "A man is playing music."),
    ("A woman is slicing a potato.", "A woman is cutting a potato."),
    ("The sun is shining.", "It is a sunny day."),
    ("A cat is sleeping.", "A dog is barking."), 
    ("The car is fast.", "The vehicle is speeding."),
    ("I love pizza.", "Pizza is my favorite food."),
    ("The sky is blue.", "The grass is green."), 
    ("He is running.", "He is walking."), 
    ("The computer is broken.", "My laptop is not working."),
    ("She is reading a book.", "She is looking at a novel."),
    ("The movie was great.", "I enjoyed the film."),
    ("It is raining hard.", "There is a storm."),
    ("A bird is flying.", "A fish is swimming."), 
    ("The coffee is hot.", "The tea is cold."), 
    ("I am happy.", "I am sad."), 
    ("The door is open.", "The door is closed."), 
]
val_sentences = val_sentences * 4 

class ParallelDeltaNet(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.fast_decay_proj = nn.Linear(hidden_size, num_heads)
        self.slow_decay_proj = nn.Linear(hidden_size, num_heads)
        self.fast_gate_proj = nn.Linear(hidden_size, num_heads)
        self.slow_gate_proj = nn.Linear(hidden_size, num_heads)
        
        self.resonance_flux = EnhancedResonanceFlux(self.head_dim, self.head_dim, num_heads)
        
        self.inter_block_decay_fast = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1)
        self.inter_block_decay_slow = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.9)
        
    def forward(self, x):
        b, l, d = x.shape
        q = self.q_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)
        fast_decay = torch.sigmoid(self.fast_decay_proj(x)).transpose(1, 2)
        slow_decay = torch.sigmoid(self.slow_decay_proj(x)).transpose(1, 2)
        fast_gate = torch.sigmoid(self.fast_gate_proj(x)).transpose(1, 2).unsqueeze(-1)
        slow_gate = torch.sigmoid(self.slow_gate_proj(x)).transpose(1, 2).unsqueeze(-1)
        
        o, _ = parallel_delta_rule(
            q, k, v, beta, fast_decay, slow_decay, 
            fast_gate, slow_gate, self.resonance_flux,
            self.inter_block_decay_fast, self.inter_block_decay_slow
        )
        
        o = o.transpose(1, 2).reshape(b, l, d)
        return self.o_proj(o), None, None, None

def get_embeddings(model, sentences):
    model.eval()
    with torch.no_grad():
        flat_sentences = [s for pair in sentences for s in pair]
        tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        emb_layer = teacher_model.embeddings
        x = emb_layer(tokens['input_ids'])
        output, _, _, _ = model(x)
        mask = tokens['attention_mask'].unsqueeze(-1).expand(output.size()).float()
        embeddings = torch.sum(output * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(embeddings, p=2, dim=1)

def evaluate_spearman(model, sentences):
    embeddings = get_embeddings(model, sentences)
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    sims = F.cosine_similarity(emb1, emb2).cpu().numpy()
    
    with torch.no_grad():
        flat_sentences = [s for pair in sentences for s in pair]
        tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        teacher_out = teacher_model(**tokens)
        mask = tokens['attention_mask'].unsqueeze(-1).expand(teacher_out.last_hidden_state.size()).float()
        teacher_emb = torch.sum(teacher_out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        teacher_emb = F.normalize(teacher_emb, p=2, dim=1)
        t_emb1 = teacher_emb[0::2]
        t_emb2 = teacher_emb[1::2]
        teacher_sims = F.cosine_similarity(t_emb1, t_emb2).cpu().numpy()
        
    return spearmanr(sims, teacher_sims)[0]

print(f"\n📉 Training Parallel Formula (No Loop) FROM SCRATCH...")
model = ParallelDeltaNet(HIDDEN_SIZE, NUM_HEADS).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR)

initial_spearman = evaluate_spearman(model, val_sentences)
print(f"   Step 0: Spearman = {initial_spearman:.4f}")

model.train()
flat_sentences = [s for pair in val_sentences for s in pair]

for step in range(1, STEPS + 1):
    with torch.no_grad():
        tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        emb_layer = teacher_model.embeddings
        x_in = emb_layer(tokens['input_ids'])
        teacher_out = teacher_model(**tokens)
        target = teacher_out.last_hidden_state
        
    output, _, _, _ = model(x_in)
    loss = nn.MSELoss()(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    if step % EVAL_INTERVAL == 0:
        s = evaluate_spearman(model, val_sentences)
        print(f"   Step {step}: Spearman = {s:.4f} (Loss: {loss.item():.4f})")

print("✅ Done!")
