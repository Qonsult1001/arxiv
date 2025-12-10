"""
Quick test: Verify checkpoint loads correctly and maintains 0.77 score
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
import numpy as np
from datasets import load_dataset
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DeltaNet8KExtended(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embeddings = nn.Embedding(512, config['d_model'])
        self.token_type_embeddings = nn.Embedding(2, config['d_model'])
        self.embedding_norm = nn.LayerNorm(config['d_model'])
        self.embedding_dropout = nn.Dropout(0.1)
        
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for _ in range(6):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=config['d_model'],
                    num_heads=config['num_heads'],
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=0.30,
                    slow_decay_init=0.85,
                    use_rope=False
                )
            )
            self.deltanet_norms.append(nn.LayerNorm(config['d_model']))
            self.deltanet_ffns.append(nn.Linear(config['d_model'], config['d_model'] * 4))
            self.ffn_norms.append(nn.LayerNorm(config['d_model']))
        
        self.ffn_outputs = nn.ModuleList([
            nn.Linear(config['d_model'] * 4, config['d_model']) 
            for _ in range(6)
        ])
    
    def mean_pooling(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_emb = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_emb / sum_mask
    
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        
        word_emb = self.embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(input_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        
        hidden = word_emb + pos_emb + token_type_emb
        hidden = self.embedding_norm(hidden)
        hidden = self.embedding_dropout(hidden)
        
        for i in range(6):
            residual = hidden
            hidden_attn, _, _, _ = self.deltanet_layers[i](hidden, attention_mask)
            hidden = self.deltanet_norms[i](residual + hidden_attn)
            
            residual = hidden
            hidden_ffn = self.deltanet_ffns[i](hidden)
            hidden_ffn = F.gelu(hidden_ffn)
            hidden_ffn = self.ffn_outputs[i](hidden_ffn)
            hidden = self.ffn_norms[i](residual + hidden_ffn)
        
        pooled = self.mean_pooling(hidden, attention_mask)
        pooled = F.normalize(pooled, p=2, dim=1)
        
        return pooled

print("🔍 Testing Checkpoint Loading...")

config = {
    'd_model': 384,
    'num_heads': 12,
    'num_layers': 6,
    'vocab_size': 30522,
}

# Initialize model
model = DeltaNet8KExtended(config).to(device)

# Load base components
print("\n1️⃣  Loading base MiniLM components...")
base_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

with torch.no_grad():
    model.embeddings.weight.copy_(base_model.embeddings.word_embeddings.weight)
    model.position_embeddings.weight.copy_(base_model.embeddings.position_embeddings.weight)
    model.token_type_embeddings.weight.copy_(base_model.embeddings.token_type_embeddings.weight)
    model.embedding_norm.weight.copy_(base_model.embeddings.LayerNorm.weight)
    model.embedding_norm.bias.copy_(base_model.embeddings.LayerNorm.bias)
    
    for i in range(6):
        layer = base_model.encoder.layer[i]
        model.deltanet_norms[i].load_state_dict(layer.attention.output.LayerNorm.state_dict())
        model.ffn_norms[i].load_state_dict(layer.output.LayerNorm.state_dict())
        
        # 🔧 CRITICAL: Copy FFN intermediate weights (not random!)
        model.deltanet_ffns[i].weight.copy_(layer.intermediate.dense.weight)
        model.deltanet_ffns[i].bias.copy_(layer.intermediate.dense.bias)
        
        model.ffn_outputs[i].load_state_dict(layer.output.dense.state_dict())

print("   ✅ Base components loaded")

# Load checkpoint DeltaNet layers
print("\n2️⃣  Loading checkpoint DeltaNet layers...")
ckpt_path = "/workspace/LAM/deltanet_8k_extended/checkpoint_167000.pt"
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.deltanet_layers.load_state_dict(ckpt['deltanet_layers'], strict=False)
print(f"   ✅ Checkpoint loaded (reported score: {ckpt.get('test_spearman', 'N/A')})")

# Evaluate on STS-B
print("\n3️⃣  Evaluating on STS-B test set...")
sts_test = load_dataset("sentence-transformers/stsb", split="test")
s1 = sts_test["sentence1"]
s2 = sts_test["sentence2"]
labels = np.array(sts_test["score"], dtype=float)

model.eval()
all_sims = []

with torch.no_grad():
    batch_size = 32
    for i in range(0, len(s1), batch_size):
        batch_s1 = s1[i:min(i+batch_size, len(s1))]
        batch_s2 = s2[i:min(i+batch_size, len(s2))]
        
        tokens1 = tokenizer(batch_s1, padding=True, max_length=128, truncation=True, return_tensors='pt').to(device)
        tokens2 = tokenizer(batch_s2, padding=True, max_length=128, truncation=True, return_tensors='pt').to(device)
        
        emb1 = model(tokens1['input_ids'], tokens1['attention_mask'])
        emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])
        
        sim = F.cosine_similarity(emb1, emb2, dim=1)
        all_sims.extend(sim.cpu().numpy().tolist())

spearman = spearmanr(all_sims, labels)[0]

print(f"\n📊 RESULT:")
print(f"   Test Spearman: {spearman:.4f}")
print(f"   Expected: ~0.77")
print(f"   Status: {'✅ PASS' if spearman > 0.75 else '❌ FAIL - FFN weights not loading correctly!'}")