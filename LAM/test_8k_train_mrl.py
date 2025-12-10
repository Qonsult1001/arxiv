import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses, evaluation
from sentence_transformers.datasets import NoDuplicatesDataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import sys
from pathlib import Path

# --- IMPORT YOUR MODEL CLASSES ---
# (Assumes your DeltaNet and MatryoshkaProjection classes are in 'final_solution_formula_final.py')
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

# 1. CONFIG
BATCH_SIZE = 64  # Higher is better for MRL
EPOCHS = 1
MAX_SEQ_LENGTH = 128 # We train on short text to learn the "compression" logic
LR = 2e-5

# 2. MATRYOSHKA PROJECTION CLASS
class MatryoshkaProjection(nn.Module):
    """Projects embeddings to support multiple granularities."""
    def __init__(self, d_model=384, dims=[64, 128, 256, 384]):
        super().__init__()
        self.d_model = d_model
        self.dims = sorted(dims)
        self.norms = nn.ModuleDict({
            str(d): nn.LayerNorm(d) for d in dims
        })
    
    def forward(self, embeddings):
        outputs = {}
        for dim in self.dims:
            truncated = embeddings[:, :dim]
            normalized = F.normalize(self.norms[str(dim)](truncated), p=2, dim=-1)
            outputs[dim] = normalized
        return outputs

# 3. DELTANET MODEL WITH MATRYOSHKA HEAD
class DeltaNetWithMRL(nn.Module):
    def __init__(self, teacher_model_path, checkpoint_path, config=None):
        super().__init__()
        # Load teacher model
        self.teacher_model = AutoModel.from_pretrained(teacher_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        d_model = self.teacher_model.config.hidden_size
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Build DeltaNet structure
        self.embeddings = self.teacher_model.embeddings
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.output_denses = nn.ModuleList()
        
        config = config or {}
        for i in range(6):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=d_model,
                    num_heads=config.get('num_heads', 12),
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=config.get('fast_decay_init', 0.3),
                    slow_decay_init=config.get('slow_decay_init', 0.832),
                )
            )
            self.deltanet_norms.append(self.teacher_model.encoder.layer[i].attention.output.LayerNorm)
            self.deltanet_ffns.append(self.teacher_model.encoder.layer[i].intermediate)
            self.ffn_norms.append(self.teacher_model.encoder.layer[i].output.LayerNorm)
            self.output_denses.append(self.teacher_model.encoder.layer[i].output.dense)
        
        self.pooler = self.teacher_model.pooler
        
        # Add Matryoshka Head
        self.projection = MatryoshkaProjection(d_model=384, dims=[64, 128, 256, 384])

        # Load checkpoint weights
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"📦 Loading checkpoint from {checkpoint_path}...")
            loaded_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            is_raw_state_dict = not any(k in loaded_data for k in ['model_state_dict', 'deltanet_layers', 'config', 'step']) and any('deltanet_layers.' in str(k) for k in loaded_data.keys())
            checkpoint = {'model_state_dict': loaded_data, 'config': {}, 'step': 0} if is_raw_state_dict else loaded_data
            
            model_state_dict = checkpoint.get('model_state_dict', {})
            if model_state_dict:
                deltanet_layers_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith('deltanet_layers.'):
                        new_key = key.replace('deltanet_layers.', '')
                        deltanet_layers_dict[new_key] = value
                
                if deltanet_layers_dict:
                    for i in range(6):
                        layer_state = {k[2:]: v for k, v in deltanet_layers_dict.items() if k.startswith(f'{i}.')}
                        if layer_state:
                            self.deltanet_layers[i].load_state_dict(layer_state, strict=False)
                    print("   ✅ Loaded deltanet_layers from checkpoint")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Get embeddings
        x = self.embeddings(input_ids)
        
        # Process through DeltaNet layers
        for i in range(6):
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.deltanet_norms[i](residual + x_attn)
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_denses[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        # Pool and normalize
        embeddings = self.mean_pooling(x, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def encode(self, input_ids, attention_mask=None, return_dict=False):
        """Encode with Matryoshka projection"""
        raw_emb = self.forward(input_ids, attention_mask)
        mrl_outputs = self.projection(raw_emb)
        
        if return_dict:
            return mrl_outputs
        else:
            return mrl_outputs[384]

# --- MAIN EXECUTION ---
print("🚀 Starting MRL Fine-Tuning (Phase 2)...")

# 1. Load Data (NLI is standard for teaching "meaning")
# Fix: all-nli requires a config name - use 'triplet' for anchor/positive/negative format
dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="train")
train_examples = []
for row in dataset:
    train_examples.append(InputExample(texts=[row['anchor'], row['positive'], row['negative']]))

# Shrink dataset for speed (optional - remove [:10000] for full training)
train_dataloader = NoDuplicatesDataLoader(train_examples[:50000], batch_size=BATCH_SIZE)

# 2. Initialize your LAM Model
print("\n📦 Initializing model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = "/workspace/LAM/LAM-base-v1/pytorch_model.bin"
teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"

model = DeltaNetWithMRL(
    teacher_model_path=teacher_model_path,
    checkpoint_path=checkpoint_path,
    config={'num_heads': 12, 'fast_decay_init': 0.3, 'slow_decay_init': 0.832}
).to(device)

tokenizer = model.tokenizer
print("✅ Model initialized")

# 3. Setup Loss - Simplified contrastive loss for MRL
def compute_mrl_contrastive_loss(anchor_dict, positive_dict, negative_dict, dims=[64, 128, 256, 384], temperature=0.05):
    """Compute InfoNCE loss at each dimension level"""
    total_loss = 0.0
    for dim in dims:
        anchor = anchor_dict[dim]  # [batch, dim]
        positive = positive_dict[dim]  # [batch, dim]
        negative = negative_dict[dim]  # [batch, dim]
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / temperature  # [batch]
        
        # Negative similarities
        neg_sim = torch.matmul(anchor, negative.t()) / temperature  # [batch, batch]
        
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch, 1+batch]
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        
        loss = F.cross_entropy(logits, labels)
        total_loss += loss
    
    return total_loss / len(dims)

# 4. Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

model.train()
print(f"🔥 Training on {len(train_dataloader)} batches...")

for epoch in range(EPOCHS):
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        # Prepare inputs
        # NoDuplicatesDataLoader returns a list of InputExample objects
        # Each InputExample has a .texts attribute with [anchor, positive, negative]
        texts = [example.texts for example in batch]  # [[anchor, pos, neg], [anchor, pos, neg]...]
        batch_size = len(texts)
        
        # Flatten and tokenize
        flat_texts = [t for group in texts for t in group]  # [batch*3]
        inputs = tokenizer(flat_texts, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt").to(device)
        
        # Forward pass (Get all MRL dims)
        embeddings_dict = model.encode(inputs['input_ids'], inputs['attention_mask'], return_dict=True)
        
        # Split into anchor, positive, negative
        anchor_dict = {dim: embeddings_dict[dim][:batch_size] for dim in [64, 128, 256, 384]}
        positive_dict = {dim: embeddings_dict[dim][batch_size:2*batch_size] for dim in [64, 128, 256, 384]}
        negative_dict = {dim: embeddings_dict[dim][2*batch_size:] for dim in [64, 128, 256, 384]}
        
        # Compute MRL contrastive loss
        loss = compute_mrl_contrastive_loss(anchor_dict, positive_dict, negative_dict)

        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Epoch {epoch} Step {i}: Loss {loss.item():.4f}")

# 5. Save
torch.save(model.state_dict(), "LAM_384_MRL_Trained.bin")
print("✅ Saved MRL-Trained Model!")