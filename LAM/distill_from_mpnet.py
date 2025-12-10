"""
6-Layer PURE LINEAR DeltaNet with MPNET Distillation - BREAKTHROUGH VERSION

🔥 DISTILLING FROM STRONGER TEACHER:
- Teacher: all-mpnet-base-v2 (768-dim, 12 layers, 0.87-0.88 Pearson)
- Student: DeltaNet (384-dim, 6 layers)
- Projection: 768 → 384 learned mapping

Expected: 0.857-0.867 Pearson (vs 0.8470 with MiniLM teacher)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import os
import json
import gzip

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# CONFIGURATION - MPNET DISTILLATION
# ============================================================================
config = {
    # STRONGER TEACHER: all-mpnet-base-v2 (768-dim, 12 layers)
    # Trained on TPU v3-8 with: 100k steps, batch_size=1024, lr=2e-5, warmup=500, seq_len=128
    "teacher_model": "sentence-transformers/all-mpnet-base-v2",
    
    # Student architecture - 384-dim, 6 layers
    "d_model": 384,
    "num_heads": 12,
    "num_linear_layers": 6,
    "total_layers": 6,
    
    # DeltaNet parameters
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    
    # Training parameters - MATCH all-mpnet-base-v2 training config
    # TPU training: batch_size=1024 (128 per core), we use GPU: simulate with accumulation
    "peak_learning_rate": 2e-5,      # SAME as original (2e-5)
    "weight_decay": 0.1,
    "dropout": 0.1,                   # Standard transformer dropout
    "gradient_clip": 1.0,
    "batch_size": 256,                # GPU batch size (with accumulation → effective ~1024)
    "gradient_accumulation_steps": 4,  # 256 * 4 = 1024 effective batch
    "max_length": 128,                # SAME as original (128 tokens)
    
    # Distillation parameters - ADJUSTED FOR MPNET
    "distillation_weight": 1.5,       # Higher weight for stronger teacher
    "layer_distill_weight": 1.0,      # Layer-wise distillation (12 → 6 mapping)
    "identity_reg_weight": 0.01,
    "ortho_state_reg_weight": 0.002,
    "state_dropout_rate": 0.10,
    "label_smoothing": 0.1,
    
    # Projection learning rate (higher for faster convergence)
    "projection_lr_multiplier": 5.0,  # Projection layers learn 5x faster
    
    # Training schedule - MATCH original (100k steps on TPU)
    # On GPU with smaller batch: use ~150k steps
    "warmup_steps": 500,              # SAME as original (500)
    "total_steps": 150000,            # ~100k adjusted for smaller effective batch on GPU
    "log_interval": 50,
    "save_interval": 1000,
    
    # Output directory
    "output_dir": "/workspace/LAM/deltanet_minilm_MPNET_DISTILL",
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# PROJECTION-BASED DISTILLATION MODEL
# ============================================================================
class DeltaNetMPNetDistill(nn.Module):
    """
    DeltaNet student (384-dim, 6 layers) distilled from MPNet teacher (768-dim, 12 layers)
    Uses learned projections to handle dimension mismatch
    """
    
    def __init__(self, teacher_model_name, num_linear_layers, config):
        super().__init__()
        
        print(f"Loading MPNET teacher model: {teacher_model_name}")
        self.teacher_model = AutoModel.from_pretrained(teacher_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        
        # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Get dimensions
        self.teacher_dim = self.teacher_model.config.hidden_size  # 768
        self.student_dim = config['d_model']  # 384
        self.num_teacher_layers = self.teacher_model.config.num_hidden_layers  # 12
        self.num_student_layers = num_linear_layers  # 6
        
        print(f"Teacher: {self.num_teacher_layers} layers, {self.teacher_dim}-dim")
        print(f"Student: {self.num_student_layers} layers, {self.student_dim}-dim")
        
        # ⭐ PROJECTION LAYERS for dimension matching (768 → 384)
        # Final embedding projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(self.teacher_dim, self.student_dim),
            nn.LayerNorm(self.student_dim),
        )
        
        # Layer-wise projections (12 teacher layers → 6 student layers)
        # We'll map teacher layers [0,1 → 0], [2,3 → 1], ..., [10,11 → 5]
        self.layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.teacher_dim, self.student_dim),
                nn.LayerNorm(self.student_dim),
            )
            for _ in range(self.num_student_layers)
        ])
        
        # Initialize projections
        for proj in [self.embedding_projection] + list(self.layer_projections):
            for module in proj:
                if isinstance(module, nn.Linear):
                    # Initialize to approximate identity (scaled down)
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        
        # ⭐ Use MPNet embeddings directly (768-dim → project to 384-dim)
        # This allows student to learn from stronger MPNet embeddings
        print(f"Loading embeddings from: all-mpnet-base-v2 (768-dim) → project to 384-dim")
        
        # Try loading from local folder first, fallback to HuggingFace
        try:
            student_base = AutoModel.from_pretrained("/workspace/LAM/all-mpnet-base-v2", local_files_only=False)
        except:
            student_base = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # 1. Student embeddings from MPNet (768-dim)
        self.embeddings = student_base.embeddings
        
        # 2. ADD embedding projection: 768 → 384
        self.embedding_projection_layer = nn.Linear(768, 384)
        nn.init.xavier_uniform_(self.embedding_projection_layer.weight)
        nn.init.zeros_(self.embedding_projection_layer.bias)
        
        # 3. ALL 6 LAYERS ARE DELTANET (trainable)
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.deltanet_ffn_outs = nn.ModuleList()  # Output projections
        self.ffn_norms = nn.ModuleList()
        
        for i in range(6):
            # Enhanced DeltaNet attention
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.student_dim,
                    num_heads=config['num_heads'],
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=config['fast_decay_init'],
                    slow_decay_init=config['slow_decay_init'],
                )
            )
            # Create fresh LayerNorms for this student architecture
            self.deltanet_norms.append(nn.LayerNorm(self.student_dim))
            
            # Create fresh FFN layers (384 → 1536)
            self.deltanet_ffns.append(nn.Linear(self.student_dim, self.student_dim * 4))
            
            # FFN output projection (1536 → 384)
            self.deltanet_ffn_outs.append(nn.Linear(self.student_dim * 4, self.student_dim))
            
            # Output norm after FFN
            self.ffn_norms.append(nn.LayerNorm(self.student_dim))
        
        # 3. Keep pooler from student base
        self.pooler = student_base.pooler
        
        # Initialize DeltaNet with orthogonal weights
        self._init_orthogonal_weights()
        
        print(f"✅ MPNET-distilled 6-layer DeltaNet with projection layers")
        print(f"   Teacher → Student: 768-dim → 384-dim via learned projections")
    
    def _init_orthogonal_weights(self):
        """Initialize DeltaNet projections with orthogonal matrices"""
        for layer in self.deltanet_layers:
            for name, param in layer.named_parameters():
                if 'proj.weight' in name and param.dim() == 2:
                    if param.size(0) == param.size(1):
                        nn.init.orthogonal_(param, gain=1.0)
                    else:
                        nn.init.xavier_uniform_(param, gain=0.1)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward_student(self, input_ids, attention_mask):
        """Forward with intermediate hidden states collection"""
        x = self.embeddings(input_ids=input_ids)
        # Project from 768 → 384
        x = self.embedding_projection_layer(x)
        
        # Collect intermediate hidden states
        student_hidden_states = []
        
        # ALL 6 layers: DeltaNet
        ortho_losses = []
        for i in range(6):
            # Attention block
            residual = x
            x_attn, _, _, ortho_loss = self.deltanet_layers[i](x, attention_mask)
            if ortho_loss is not None:
                ortho_losses.append(ortho_loss)
            x = self.deltanet_norms[i](residual + x_attn)
            
            # FFN block (384 → 1536 → 384)
            residual = x
            x_ffn = self.deltanet_ffns[i](x)  # 384 → 1536
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.deltanet_ffn_outs[i](x_ffn)  # 1536 → 384
            x = self.ffn_norms[i](residual + x_ffn)
            
            # Store layer output
            student_hidden_states.append(x)
        
        # Final embedding
        embeddings = self.mean_pooling(x, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Sum orthogonal losses
        total_ortho_loss = sum(ortho_losses) if ortho_losses else torch.tensor(0.0, device=embeddings.device)
        
        return embeddings, student_hidden_states, total_ortho_loss
    
    def forward_teacher(self, input_ids, attention_mask):
        """Forward teacher with hidden states"""
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # Teacher embeddings (768-dim)
            embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Teacher has 12 layers, we'll map: [0,1]→0, [2,3]→1, ..., [10,11]→5
            # Extract layers 1-12 (skip embedding layer at index 0)
            all_teacher_layers = outputs.hidden_states[1:13]  # 12 layers
            
            # Average pairs of teacher layers to get 6 targets
            teacher_hidden_states = []
            for i in range(6):
                # Average layers 2i and 2i+1
                layer_avg = (all_teacher_layers[2*i] + all_teacher_layers[2*i + 1]) / 2
                teacher_hidden_states.append(layer_avg)
            
        return embeddings, teacher_hidden_states
    
    def forward(self, input_ids, attention_mask):
        """Return embeddings and hidden states for both student and teacher"""
        student_emb, student_hidden, ortho_loss = self.forward_student(input_ids, attention_mask)
        teacher_emb, teacher_hidden = self.forward_teacher(input_ids, attention_mask)
        
        return student_emb, teacher_emb, student_hidden, teacher_hidden, ortho_loss
    
    def get_projection_params(self):
        """Get projection layer parameters for higher learning rate"""
        proj_params = []
        proj_params.extend(self.embedding_projection.parameters())
        for layer_proj in self.layer_projections:
            proj_params.extend(layer_proj.parameters())
        return proj_params

# ============================================================================
# LOSS FUNCTIONS WITH PROJECTION-BASED DISTILLATION
# ============================================================================
def compute_loss(student_emb_a, student_emb_b, teacher_emb_a, teacher_emb_b,
                 student_hidden_a, student_hidden_b, teacher_hidden_a, teacher_hidden_b,
                 labels, model, config, ortho_loss_a=None, ortho_loss_b=None):
    """
    Combined loss with dimension-aware distillation
    Teacher: 768-dim, Student: 384-dim (projected)
    """
    
    # 1. Contrastive loss with label smoothing
    scores = torch.mm(student_emb_a, student_emb_b.transpose(0, 1)) * 20.0
    label_smoothing = config.get('label_smoothing', 0.1)
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    contrastive_loss = (cross_entropy(scores, labels) + 
                       cross_entropy(scores.transpose(0, 1), labels)) / 2
    
    # 2. Embedding distillation with PROJECTION
    # Project teacher embeddings to student space (768 → 384)
    teacher_emb_a_proj = model.embedding_projection(teacher_emb_a)
    teacher_emb_b_proj = model.embedding_projection(teacher_emb_b)
    
    distill_loss_a = F.mse_loss(student_emb_a, teacher_emb_a_proj)
    distill_loss_b = F.mse_loss(student_emb_b, teacher_emb_b_proj)
    distill_loss = (distill_loss_a + distill_loss_b) / 2
    
    # 3. TRUE Orthogonal regularization
    identity_reg = 0.0
    num_matrices = 0
    for layer in model.deltanet_layers:
        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
            if hasattr(layer, proj_name):
                W = getattr(layer, proj_name).weight
                gram = torch.mm(W, W.t())
                identity_target = torch.eye(gram.size(0), device=W.device)
                identity_reg += torch.norm(gram - identity_target, p='fro') ** 2
                num_matrices += 1
    
    if num_matrices > 0:
        identity_reg = identity_reg / num_matrices
    
    # 4. Layer-wise hidden state matching with PROJECTION
    layer_distill_loss = torch.tensor(0.0, device=student_emb_a.device)
    
    # Match 6 student layers to 6 projected teacher layers
    for i, (s_out_a, t_out_a, s_out_b, t_out_b) in enumerate(
        zip(student_hidden_a, teacher_hidden_a, student_hidden_b, teacher_hidden_b)
    ):
        # Project teacher hidden states to student dimension (768 → 384)
        t_out_a_proj = model.layer_projections[i](t_out_a)
        t_out_b_proj = model.layer_projections[i](t_out_b)
        
        # MSE loss in student space
        layer_distill_loss += F.mse_loss(s_out_a, t_out_a_proj)
        layer_distill_loss += F.mse_loss(s_out_b, t_out_b_proj)
    
    # Average over 2 sentences * 6 layers = 12 terms
    layer_distill_loss /= (2.0 * len(student_hidden_a))
    
    # 5. Orthogonal state regularization
    ortho_state_reg = torch.tensor(0.0, device=student_emb_a.device)
    if ortho_loss_a is not None:
        ortho_state_reg += ortho_loss_a
    if ortho_loss_b is not None:
        ortho_state_reg += ortho_loss_b
    ortho_state_reg = ortho_state_reg / 2.0
    
    # Combine all losses
    ortho_state_weight = config.get('ortho_state_reg_weight', 0.001)
    total_loss = (contrastive_loss + 
                  config['distillation_weight'] * distill_loss +
                  config['layer_distill_weight'] * layer_distill_loss +
                  config['identity_reg_weight'] * identity_reg +
                  ortho_state_weight * ortho_state_reg)
    
    return total_loss, contrastive_loss, distill_loss, identity_reg, layer_distill_loss, ortho_state_reg

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load STS-B training data"""
    print("Loading training data...")
    
    # Load STS-B train
    stsb_train = load_dataset("glue", "stsb", split="train")
    train_data = [
        {
            'sentence1': ex['sentence1'],
            'sentence2': ex['sentence2'],
            'label': float(ex['label']) / 5.0  # Normalize to 0-1
        }
        for ex in stsb_train
        if len(ex['sentence1']) > 10 and len(ex['sentence2']) > 10
    ]
    
    print(f"   ✅ Loaded {len(train_data):,} training pairs")
    
    return Dataset.from_list(train_data)

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train():
    print("="*80)
    print("6-LAYER DELTANET - MPNET DISTILLATION")
    print("="*80)
    print(f"\n🔥 STRONGER TEACHER:")
    print(f"  Teacher: all-mpnet-base-v2 (768-dim, 12 layers, 0.87-0.88 Pearson)")
    print(f"  Student: DeltaNet (384-dim, 6 layers)")
    print(f"  Method: Learned projections (768 → 384)")
    print(f"\n🎯 Expected: 0.857-0.867 Pearson (vs 0.8470 with MiniLM)")
    print(f"\n⚡ Training features:")
    print(f"  ✅ Projection layers for dimension matching")
    print(f"  ✅ Layer-wise distillation (12 → 6 mapping)")
    print(f"  ✅ Orthogonal regularization")
    print(f"  ✅ State dropout + label smoothing")
    print("="*80)
    
    # Load data
    dataset = load_data()
    
    # Initialize model
    print("\nInitializing MPNET-distilled model...")
    model = DeltaNetMPNetDistill(
        config['teacher_model'],
        config['num_linear_layers'],
        config
    ).to(device)
    
    # Separate parameter groups with different learning rates
    # Projection layers learn faster (need to adapt quickly)
    projection_params = model.get_projection_params()
    projection_param_ids = set(id(p) for p in projection_params)
    
    # DeltaNet parameters (main learning rate)
    deltanet_params = [
        p for p in model.parameters() 
        if p.requires_grad and id(p) not in projection_param_ids
    ]
    
    print(f"\nParameter groups:")
    print(f"  DeltaNet params: {sum(p.numel() for p in deltanet_params):,}")
    print(f"  Projection params: {sum(p.numel() for p in projection_params):,}")
    
    # Optimizer with separate learning rates
    optimizer = AdamW([
        {
            'params': deltanet_params,
            'lr': config['peak_learning_rate'],
            'weight_decay': config['weight_decay']
        },
        {
            'params': projection_params,
            'lr': config['peak_learning_rate'] * config['projection_lr_multiplier'],
            'weight_decay': config['weight_decay'] * 0.5  # Less regularization for projections
        }
    ])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['total_steps']
    )
    
    # Training loop (same structure as before)
    model.train()
    global_step = 0
    running_loss = 0.0
    running_contrastive = 0.0
    running_distill = 0.0
    running_layer_distill = 0.0
    
    pbar = tqdm(total=config['total_steps'], desc="MPNET Distillation")
    
    while global_step < config['total_steps']:
        # Sample batch
        indices = np.random.randint(0, len(dataset), size=config['batch_size'])
        batch_data = [dataset[int(i)] for i in indices]
        
        sentences_a = [item['sentence1'] for item in batch_data]
        sentences_b = [item['sentence2'] for item in batch_data]
        
        # Tokenize
        tokens_a = model.tokenizer(
            sentences_a,
            padding='max_length',
            max_length=config['max_length'],
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        tokens_b = model.tokenizer(
            sentences_b,
            padding='max_length',
            max_length=config['max_length'],
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # Forward
        student_emb_a, teacher_emb_a, student_hidden_a, teacher_hidden_a, ortho_loss_a = model(
            tokens_a['input_ids'], tokens_a['attention_mask']
        )
        student_emb_b, teacher_emb_b, student_hidden_b, teacher_hidden_b, ortho_loss_b = model(
            tokens_b['input_ids'], tokens_b['attention_mask']
        )
        
        labels = torch.arange(len(student_emb_a), device=device)
        
        # Compute loss
        loss, contrastive_loss, distill_loss, identity_reg, layer_distill_loss, ortho_state_reg = compute_loss(
            student_emb_a, student_emb_b,
            teacher_emb_a, teacher_emb_b,
            student_hidden_a, student_hidden_b,
            teacher_hidden_a, teacher_hidden_b,
            labels, model, config,
            ortho_loss_a=ortho_loss_a, ortho_loss_b=ortho_loss_b
        )
        
        # Backward
        loss = loss / config.get('gradient_accumulation_steps', 1)
        loss.backward()
        
        if (global_step + 1) % config.get('gradient_accumulation_steps', 1) == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        running_loss += loss.item()
        running_contrastive += contrastive_loss.item()
        running_distill += distill_loss.item()
        running_layer_distill += layer_distill_loss.item()
        
        if (global_step + 1) % config['log_interval'] == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/config["log_interval"]:.4f}',
                'contr': f'{running_contrastive/config["log_interval"]:.4f}',
                'emb': f'{running_distill/config["log_interval"]:.4f}',
                'layer': f'{running_layer_distill/config["log_interval"]:.4f}',
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            running_loss = 0.0
            running_contrastive = 0.0
            running_distill = 0.0
            running_layer_distill = 0.0
        
        # Save checkpoint
        if (global_step + 1) % config['save_interval'] == 0:
            output_dir = Path(config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            student_state = {
                'deltanet_layers': model.deltanet_layers.state_dict(),
                'embedding_projection': model.embedding_projection.state_dict(),
                'layer_projections': model.layer_projections.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': global_step,
                'config': config
            }
            torch.save(student_state, output_dir / f"checkpoint_{global_step+1}.pt")
            model.tokenizer.save_pretrained(output_dir)
            print(f"\n💾 Saved checkpoint at step {global_step + 1}")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    model.tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training complete!")
    print(f"   Trained to step {config['total_steps']:,}")
    print(f"   Saved to: {output_dir}/")
    print(f"\n📊 Expected Performance:")
    print(f"   Current (MiniLM teacher): 0.8470")
    print(f"   With MPNet teacher: 0.857-0.867")
    print(f"   Potential gain: +0.010 to +0.020")

if __name__ == "__main__":
    train()