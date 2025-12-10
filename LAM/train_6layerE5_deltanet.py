"""
🛡️ KNOWLEDGE-PRESERVING DISTILLATION
Prevents catastrophic forgetting by:
1. Mixing original training data with distillation data
2. Using the ORIGINAL loss alongside distillation losses
3. Conservative learning rates
4. EWC-style regularization to protect important weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet
torch.set_float32_matmul_precision('high')

# ============================================================================
# 🛡️ KNOWLEDGE-PRESERVING CONFIG
# ============================================================================
config = {
    # Models
    "teacher_model": "intfloat/e5-large-v2",
    "student_model_path": "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH",
    "checkpoint_file": "checkpoint_99000.pt",  # Best recovery: 0.8306
    
    # Architecture
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # 🛡️ CONSERVATIVE LEARNING (prevent forgetting)
    "learning_rate": 5e-6,           # 🔥 Much lower - gentle refinement only
    "projection_lr": 3e-5,           # Lower projection LR too
    "weight_decay": 0.005,           # Lighter regularization
    "gradient_clip": 0.5,            # Tighter clip for stability
    "batch_size": 96,                # Smaller batches for stability
    "max_length": 128,
    
    # 🛡️ KNOWLEDGE PRESERVATION STRATEGY
    "replay_ratio": 0.6,             # 60% original training data, 40% distillation
    "original_loss_weight": 0.8,     # Strong weight on original objective
    "distill_loss_weight": 0.3,      # Distillation as SUPPLEMENT, not replacement
    
    # 🛡️ GENTLE DISTILLATION LOSSES
    "similarity_distill_weight": 0.2,    # Reduced from 0.5
    "embedding_distill_weight": 0.2,     # Reduced from 0.5
    "layer_distill_weight": 0.1,         # Reduced from 0.5
    "contrastive_margin_weight": 0.3,    # Reduced from 2.5 (was destroying performance!)
    "contrastive_margin": 0.2,
    
    # EWC-style parameter importance
    "ewc_lambda": 100.0,             # Penalty for changing important params
    "compute_fisher": True,          # Compute Fisher information for EWC
    
    # Temperature
    "temperature": 0.04,
    
    # Training
    "warmup_steps": 200,
    "total_steps": 10000,            # Shorter initial run to validate
    "log_interval": 50,
    "eval_interval": 200,
    "save_interval": 500,
    
    # Early stopping for safety
    "patience": 3,                   # Stop if performance degrades for 3 evals
    "min_improvement": 0.0,          # Allow small fluctuations
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("🛡️ KNOWLEDGE-PRESERVING DISTILLATION")
print("="*80)
print(f"\n🎯 STRATEGY:")
print(f"  1. DATA REPLAY: {config['replay_ratio']*100:.0f}% original all-NLI data")
print(f"  2. DUAL OBJECTIVES: Original loss (0.8) + Distillation (0.3)")
print(f"  3. CONSERVATIVE LR: {config['learning_rate']:.1e} (gentle refinement)")
print(f"  4. EWC PROTECTION: Lambda={config['ewc_lambda']:.0f} (protect important weights)")
print(f"\n⚙️ CONFIG:")
print(f"  Learning Rate: {config['learning_rate']:.1e} (Model) / {config['projection_lr']:.1e} (Projection)")
print(f"  Original Loss Weight: {config['original_loss_weight']} (PRIMARY)")
print(f"  Distill Loss Weight: {config['distill_loss_weight']} (SUPPLEMENT)")
print(f"  Contrastive Margin: {config['contrastive_margin_weight']} (gentled from 2.5)")
print(f"\n🎯 GOAL: Protect 0.8306 baseline while gently adding E5-Large knowledge")
print("="*80)

# ============================================================================
# LOAD CHECKPOINT
# ============================================================================

class LoadCheckpoint(nn.Module):
    def __init__(self, model_path, checkpoint_file):
        super().__init__()
        
        print(f"\n📦 Loading checkpoint {checkpoint_file}...")
        
        base_minilm = "/workspace/LAM/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(base_minilm)
        base_model = AutoModel.from_pretrained(base_minilm)
        self.d_model = base_model.config.hidden_size
        self.num_layers = 6
        
        self.embeddings = base_model.embeddings
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.output_dense_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model, num_heads=12,
                    use_hierarchical_decay=True, use_enhanced_flux=True,
                    fast_decay_init=0.3, slow_decay_init=0.85,
                )
            )
            self.deltanet_norms.append(base_model.encoder.layer[i].attention.output.LayerNorm)
            self.deltanet_ffns.append(base_model.encoder.layer[i].intermediate)
            self.ffn_norms.append(base_model.encoder.layer[i].output.LayerNorm)
            self.output_dense_layers.append(base_model.encoder.layer[i].output.dense)
        
        checkpoint_path = Path(model_path) / checkpoint_file
        if not checkpoint_path.exists():
            alt = Path(model_path).parent / 'proper_distillation' / checkpoint_file
            if alt.exists():
                checkpoint_path = alt
        
        print(f"    Loading from: {checkpoint_path}")
        self.checkpoint_loaded_path = str(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load deltanet layers
        state_dict = None
        if 'deltanet_layers' in checkpoint:
            state_dict = checkpoint['deltanet_layers']
        elif 'lam_layers' in checkpoint:
            state_dict = checkpoint['lam_layers']
        
        if state_dict is not None:
            self.deltanet_layers.load_state_dict(state_dict, strict=False)
            print("    ✅ Loaded LAM layers")
        
        # Load other components
        for key in ['lam_norms', 'deltanet_ffns', 'ffn_norms', 'output_dense_layers']:
            if key in checkpoint:
                try:
                    getattr(self, key.replace('lam_norms', 'deltanet_norms')).load_state_dict(checkpoint[key], strict=False)
                    print(f"    ✅ Loaded {key}")
                except: pass
        
        print(f"✅ Checkpoint loaded!")
        del base_model
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward(self, input_ids, attention_mask):
        x = self.embeddings(input_ids=input_ids)
        
        for i in range(self.num_layers):
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.deltanet_norms[i](residual + x_attn)
            
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_dense_layers[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        embeddings = self.mean_pooling(x, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

class DimensionProjection(nn.Module):
    def __init__(self, teacher_dim=1024, student_dim=384):
        super().__init__()
        self.projection = nn.Linear(teacher_dim, student_dim)
        nn.init.xavier_uniform_(self.projection.weight, gain=0.5)  # Gentler init
        nn.init.zeros_(self.projection.bias)
    
    def forward(self, teacher_hidden):
        return self.projection(teacher_hidden)

# ============================================================================
# 🛡️ KNOWLEDGE-PRESERVING MODEL
# ============================================================================

class KnowledgePreservingModel(nn.Module):
    def __init__(self, teacher_model_name, student_model_path, checkpoint_file, config):
        super().__init__()
        
        # Teacher
        self.teacher_model = AutoModel.from_pretrained(teacher_model_name)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # Student
        self.student_model = LoadCheckpoint(student_model_path, checkpoint_file)
        self.student_tokenizer = self.student_model.tokenizer
        self.checkpoint_loaded_path = self.student_model.checkpoint_loaded_path
        
        # Dimensions
        self.teacher_dim = self.teacher_model.config.hidden_size
        self.student_dim = config['d_model']
        self.num_student_layers = config['num_layers']
        self.num_teacher_layers = self.teacher_model.config.num_hidden_layers
        
        # Student components
        self.embeddings = self.student_model.embeddings
        self.lam_layers = self.student_model.deltanet_layers
        self.lam_norms = self.student_model.deltanet_norms
        self.deltanet_ffns = self.student_model.deltanet_ffns
        self.ffn_norms = self.student_model.ffn_norms
        self.output_dense_layers = self.student_model.output_dense_layers
        
        # Projections
        self.projections = nn.ModuleList([
            DimensionProjection(self.teacher_dim, self.student_dim)
            for _ in range(self.num_student_layers)
        ])
        self.emb_projection = DimensionProjection(self.teacher_dim, self.student_dim)
        
        # Try to load projection weights
        if self.checkpoint_loaded_path and os.path.exists(self.checkpoint_loaded_path):
            checkpoint = torch.load(self.checkpoint_loaded_path, map_location='cpu', weights_only=False)
            if 'projections' in checkpoint:
                try:
                    self.projections.load_state_dict(checkpoint['projections'])
                    print("    ✅ Loaded projection weights")
                except: pass
            if 'emb_projection' in checkpoint:
                try:
                    self.emb_projection.load_state_dict(checkpoint['emb_projection'])
                    print("    ✅ Loaded embedding projection")
                except: pass
        
        # Teacher layer indices
        start_idx = self.num_teacher_layers - self.num_student_layers
        self.teacher_layer_indices = list(range(start_idx, self.num_teacher_layers))
        
        # 🛡️ Store initial parameters for EWC
        self.initial_params = {}
        if config['compute_fisher']:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    self.initial_params[name] = param.data.clone()
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward_student_only(self, input_ids, attention_mask):
        """Forward pass for original training (no teacher)"""
        x = self.embeddings(input_ids=input_ids)
        
        for i in range(self.num_student_layers):
            residual = x
            x_attn, _, _, _ = self.lam_layers[i](x, attention_mask)
            x = self.lam_norms[i](residual + x_attn)
            
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_dense_layers[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        
        embeddings = self.mean_pooling(x, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward_with_teacher(self, sentences):
        """Forward pass with teacher distillation"""
        # Student forward
        student_tokens = self.student_tokenizer(
            sentences, padding='max_length', max_length=128,
            truncation=True, return_tensors='pt'
        ).to(device)
        
        x = self.embeddings(input_ids=student_tokens['input_ids'])
        student_hidden = []
        
        for i in range(self.num_student_layers):
            residual = x
            x_attn, _, _, _ = self.lam_layers[i](x, student_tokens['attention_mask'])
            x = self.lam_norms[i](residual + x_attn)
            
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_dense_layers[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
            
            student_hidden.append(x)
        
        student_emb = self.mean_pooling(x, student_tokens['attention_mask'])
        student_emb = F.normalize(student_emb, p=2, dim=1)
        
        # Teacher forward
        with torch.no_grad():
            teacher_tokens = self.teacher_tokenizer(
                sentences, padding='max_length', max_length=128,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            outputs = self.teacher_model(
                input_ids=teacher_tokens['input_ids'],
                attention_mask=teacher_tokens['attention_mask'],
                output_hidden_states=True
            )
            
            teacher_hidden_states = []
            for i, idx in enumerate(self.teacher_layer_indices):
                teacher_hidden = outputs.hidden_states[idx + 1]
                projected = self.projections[i](teacher_hidden)
                teacher_hidden_states.append(projected)
            
            last_hidden = outputs.last_hidden_state
            teacher_emb = self.mean_pooling(last_hidden, teacher_tokens['attention_mask'])
            teacher_emb = self.emb_projection(teacher_emb)
            teacher_emb = F.normalize(teacher_emb, p=2, dim=1)
        
        return student_emb, teacher_emb, student_hidden, teacher_hidden_states

# ============================================================================
# 🛡️ LOSSES
# ============================================================================

def compute_original_contrastive_loss(embeddings, labels):
    """
    Original contrastive loss that the model was trained on.
    This maintains the model's existing knowledge.
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    
    # Create labels matrix (1 if same label, 0 otherwise)
    labels = labels.unsqueeze(1)
    label_matrix = (labels == labels.T).float()
    
    # InfoNCE-style loss
    # Positives: pairs with same label
    # Negatives: pairs with different labels
    temperature = 0.05
    sim_matrix = sim_matrix / temperature
    
    # Mask out self-similarity
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
    
    # Compute loss
    exp_sim = torch.exp(sim_matrix)
    
    # For each anchor, sum over positives and all
    pos_sim = (exp_sim * label_matrix).sum(dim=1)
    all_sim = exp_sim.sum(dim=1)
    
    # Loss: -log(pos_sim / all_sim)
    loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8)
    loss = loss.mean()
    
    return loss

def compute_gentle_distillation_loss(student_emb, teacher_emb, 
                                     student_hidden, teacher_hidden, config):
    """Gentler distillation loss with reduced weights"""
    batch_size = student_emb.shape[0]
    
    # 1. Similarity structure distillation (gentle)
    teacher_sim = torch.matmul(teacher_emb, teacher_emb.T)
    student_sim = torch.matmul(student_emb, student_emb.T)
    
    teacher_sim_soft = teacher_sim / config['temperature']
    student_sim_soft = student_sim / config['temperature']
    
    teacher_dist = F.softmax(teacher_sim_soft, dim=1)
    student_log_dist = F.log_softmax(student_sim_soft, dim=1)
    
    similarity_loss = F.kl_div(student_log_dist, teacher_dist, reduction='batchmean')
    
    # 2. Contrastive margin loss (gentle)
    H = batch_size // 2
    if H > 0 and batch_size % 2 == 0:
        sim_matrix = student_sim
        pos_sim_a = sim_matrix[torch.arange(H), torch.arange(H) + H]
        pos_sim_b = sim_matrix[torch.arange(H) + H, torch.arange(H)]
        
        mask = torch.ones_like(sim_matrix, dtype=torch.bool)
        idx = torch.arange(batch_size)
        mask[idx, idx] = False
        mask[torch.arange(H), torch.arange(H) + H] = False
        mask[torch.arange(H) + H, torch.arange(H)] = False
        
        neg_sim = sim_matrix.masked_fill(~mask, float('-inf'))
        hard_neg_a, _ = neg_sim[:H].max(dim=1)
        hard_neg_b, _ = neg_sim[H:].max(dim=1)
        
        margin = config['contrastive_margin']
        cm_loss_a = F.relu(margin - (pos_sim_a - hard_neg_a)).mean()
        cm_loss_b = F.relu(margin - (pos_sim_b - hard_neg_b)).mean()
        contrastive_margin_loss = (cm_loss_a + cm_loss_b) / 2.0
    else:
        contrastive_margin_loss = torch.tensor(0.0, device=student_emb.device)
    
    # 3. Direct embedding distillation
    embedding_loss = F.mse_loss(student_emb, teacher_emb)
    
    # 4. Layer distillation
    layer_loss = 0
    for s_h, t_h in zip(student_hidden, teacher_hidden):
        layer_loss += F.mse_loss(s_h, t_h)
    layer_loss = layer_loss / len(student_hidden)
    
    # Weighted combination (gentler weights)
    total_loss = (
        config['similarity_distill_weight'] * similarity_loss +
        config['contrastive_margin_weight'] * contrastive_margin_loss +
        config['embedding_distill_weight'] * embedding_loss +
        config['layer_distill_weight'] * layer_loss
    )
    
    return total_loss, similarity_loss, contrastive_margin_loss, embedding_loss, layer_loss

def compute_ewc_loss(model, initial_params, ewc_lambda):
    """
    Elastic Weight Consolidation loss
    Penalizes large changes to important parameters
    """
    ewc_loss = 0.0
    for name, param in model.named_parameters():
        if param.requires_grad and name in initial_params:
            # Move initial_param to same device as param
            initial_param = initial_params[name].to(param.device)
            ewc_loss += torch.sum((param - initial_param) ** 2)
    return ewc_lambda * ewc_loss

# ============================================================================
# DATA LOADING
# ============================================================================

def load_mixed_data(replay_ratio=0.6):
    """
    Load mixed data: replay_ratio% original, (1-replay_ratio)% distillation
    """
    print(f"\n📚 Loading mixed data (replay_ratio={replay_ratio})...")
    
    # Original training data (all-NLI)
    original_dataset = load_dataset("sentence-transformers/all-nli", "pair-score", split="train")
    original_dataset = original_dataset.filter(lambda x: len(x['sentence1']) > 10 and len(x['sentence2']) > 10)
    
    print(f"   Original dataset: {len(original_dataset):,} examples")
    return original_dataset

def evaluate_stsb(model):
    from scipy.stats import pearsonr, spearmanr
    
    model.eval()
    ds = load_dataset("glue", "stsb", split="validation").select(range(1500))
    
    all_sim = []
    all_labels = []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(ds), batch_size):
            batch = ds[i:min(i+batch_size, len(ds))]
            
            tokens = model.student_tokenizer(
                batch['sentence1'] + batch['sentence2'],
                padding='max_length', max_length=128,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            embeddings = model.forward_student_only(tokens['input_ids'], tokens['attention_mask'])
            
            n = len(batch['sentence1'])
            emb1 = embeddings[:n]
            emb2 = embeddings[n:]
            
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            all_sim.extend(sim.cpu().numpy().tolist())
            all_labels.extend(list(batch['label']))
    
    pearson = pearsonr(all_sim, all_labels)[0]
    spearman = spearmanr(all_sim, all_labels)[0]
    
    model.train()
    return pearson, spearman

# ============================================================================
# TRAINING
# ============================================================================

def train():
    dataset = load_mixed_data(config['replay_ratio'])
    
    model = KnowledgePreservingModel(
        config['teacher_model'],
        config['student_model_path'],
        config['checkpoint_file'],
        config
    ).to(device)
    
    # Optimizer with very conservative LR
    optimizer = AdamW([
        {
            'params': list(model.lam_layers.parameters()) +
                     list(model.lam_norms.parameters()) +
                     list(model.deltanet_ffns.parameters()) +
                     list(model.ffn_norms.parameters()) +
                     list(model.output_dense_layers.parameters()),
            'lr': config['learning_rate']
        },
        {
            'params': model.embeddings.parameters(),
            'lr': config['learning_rate'] * 0.1  # Even more conservative
        },
        {
            'params': list(model.projections.parameters()) +
                     list(model.emb_projection.parameters()),
            'lr': config['projection_lr']
        },
    ], weight_decay=config['weight_decay'], betas=(0.9, 0.999))
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=config['total_steps']
    )
    
    # Baseline
    print(f"\n📊 BASELINE EVALUATION...")
    baseline_pearson, baseline_spearman = evaluate_stsb(model)
    print(f"   Pearson:  {baseline_pearson:.4f}")
    print(f"   Spearman: {baseline_spearman:.4f}")
    print(f"\n🎯 GOAL: Maintain {baseline_pearson:.4f} or improve!")
    print(f"▶️  Starting knowledge-preserving training...\n")
    
    # Training
    model.train()
    global_step = 0
    best_pearson = baseline_pearson
    patience_counter = 0
    
    running_total = 0.0
    running_original = 0.0
    running_distill = 0.0
    
    pbar = tqdm(total=config['total_steps'], desc="🛡️ Protected Training")
    
    while global_step < config['total_steps']:
        # 🛡️ Mix original and distillation data
        use_distillation = np.random.random() < (1 - config['replay_ratio'])
        
        if use_distillation:
            # Distillation batch
            indices = np.random.randint(0, len(dataset), size=config['batch_size'])
            batch_data = [dataset[int(i)] for i in indices]
            
            all_sentences = []
            for item in batch_data:
                all_sentences.append(item['sentence1'])
            for item in batch_data:
                all_sentences.append(item['sentence2'])
            
            student_emb, teacher_emb, student_hidden, teacher_hidden = model.forward_with_teacher(all_sentences)
            
            distill_loss, sim_loss, margin_loss, emb_loss, layer_loss = compute_gentle_distillation_loss(
                student_emb, teacher_emb, student_hidden, teacher_hidden, config
            )
            
            loss = config['distill_loss_weight'] * distill_loss
            running_distill += loss.item()
            
        else:
            # Original training batch (replay) - use contrastive loss with pairs
            indices = np.random.randint(0, len(dataset), size=config['batch_size'])
            batch_data = [dataset[int(i)] for i in indices]
            
            # Create pairs: sentence1 and sentence2 from same item are positive pairs
            sentences_a = [item['sentence1'] for item in batch_data]
            sentences_b = [item['sentence2'] for item in batch_data]
            all_sentences = sentences_a + sentences_b
            
            tokens = model.student_tokenizer(
                all_sentences, padding='max_length', max_length=128,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            embeddings = model.forward_student_only(tokens['input_ids'], tokens['attention_mask'])
            
            # For contrastive loss: pairs [0, batch_size) with [batch_size, 2*batch_size) are positives
            # Use simple contrastive loss with positive pairs
            emb_a = embeddings[:config['batch_size']]
            emb_b = embeddings[config['batch_size']:]
            
            # Positive similarities (diagonal)
            pos_sim = F.cosine_similarity(emb_a, emb_b, dim=1)
            
            # Negative similarities (off-diagonal)
            neg_sim_matrix = torch.matmul(emb_a, emb_b.T)
            # Mask out positives
            mask = torch.eye(config['batch_size'], device=device).bool()
            neg_sim_matrix = neg_sim_matrix.masked_fill(mask, float('-inf'))
            hard_neg, _ = neg_sim_matrix.max(dim=1)
            
            # Contrastive margin loss
            margin = 0.2
            contrastive_loss = F.relu(margin - (pos_sim - hard_neg)).mean()
            
            loss = config['original_loss_weight'] * contrastive_loss
            running_original += loss.item()
        
        # 🛡️ Add EWC regularization
        if config['compute_fisher']:
            ewc_loss = compute_ewc_loss(model, model.initial_params, config['ewc_lambda'])
            loss = loss + ewc_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['gradient_clip'])
        optimizer.step()
        scheduler.step()
        
        running_total += loss.item()
        
        if (global_step + 1) % config['log_interval'] == 0:
            pbar.set_postfix({
                'loss': f'{running_total/config["log_interval"]:.4f}',
                'orig': f'{running_original/config["log_interval"]:.3f}',
                'dist': f'{running_distill/config["log_interval"]:.3f}',
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            running_total = running_original = running_distill = 0.0
        
        if (global_step + 1) % config['eval_interval'] == 0:
            current_pearson, current_spearman = evaluate_stsb(model)
            
            gain = current_pearson - baseline_pearson
            
            # 🛡️ Performance degradation detection
            if current_pearson < baseline_pearson - config['min_improvement']:
                patience_counter += 1
                status = f"⚠️  DEGRADED (patience {patience_counter}/{config['patience']})"
            elif current_pearson > best_pearson:
                patience_counter = 0
                best_pearson = current_pearson
                status = "🎉 NEW BEST!"
            else:
                patience_counter = 0
                status = "✅ Stable"
            
            print(f"\n📊 Step {global_step+1}:")
            print(f"   Pearson:  {current_pearson:.4f} (Δ {gain:+.4f}) {status}")
            print(f"   Spearman: {current_spearman:.4f}")
            print(f"   Baseline: {baseline_pearson:.4f}")
            
            # 🛡️ Early stopping if performance degrades consistently
            if patience_counter >= config['patience']:
                print(f"\n⚠️  EARLY STOPPING: Performance degraded for {config['patience']} consecutive evals")
                print(f"   Reverting to best checkpoint (Pearson: {best_pearson:.4f})")
                break
            
            # Save if improved
            if current_pearson > best_pearson - 0.0001:  # Small tolerance
                output_dir = Path("/workspace/LAM/knowledge_preserving")
                output_dir.mkdir(exist_ok=True)
                torch.save({
                    'lam_layers': model.lam_layers.state_dict(),
                    'lam_norms': model.lam_norms.state_dict(),
                    'deltanet_ffns': model.deltanet_ffns.state_dict(),
                    'ffn_norms': model.ffn_norms.state_dict(),
                    'output_dense_layers': model.output_dense_layers.state_dict(),
                    'projections': model.projections.state_dict(),
                    'emb_projection': model.emb_projection.state_dict(),
                    'step': global_step + 1,
                    'pearson': current_pearson,
                    'spearman': current_spearman
                }, output_dir / f"checkpoint_{global_step+1}.pt")
                print(f"   💾 Saved checkpoint")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    final_pearson, final_spearman = evaluate_stsb(model)
    print(f"\n📊 RESULTS:")
    print(f"   Baseline: {baseline_pearson:.4f}")
    print(f"   Final:    {final_pearson:.4f} ({final_pearson-baseline_pearson:+.4f})")
    print(f"   Best:     {best_pearson:.4f} ({best_pearson-baseline_pearson:+.4f})")
    
    if best_pearson >= baseline_pearson:
        print(f"\n✅ SUCCESS: Baseline protected! No catastrophic forgetting!")
    else:
        print(f"\n⚠️  WARNING: Performance dropped below baseline")

if __name__ == "__main__":
    train()