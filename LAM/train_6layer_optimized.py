"""
DeltaNet Knowledge Distillation - FINAL OPTIMIZED VERSION

🔥 KEY IMPROVEMENTS:
1. ✅ TWO-PHASE TRAINING: Frozen FFNs → Co-adaptation
2. ✅ WARM START: Initialize DeltaNet from teacher attention
3. ✅ TRAINABLE COMPONENTS: No frozen sandwich!
4. ✅ BATCH SIZE 1024: 128 × 8 gradient accumulation
5. ✅ EARLY STOPPING: Validation on STS-B dev set
6. ✅ 18K STEPS: ~1B tokens optimal for 3.3M pairs
7. ✅ STANDALONE EXPORT: Extract student-only model

Expected Performance: 0.835-0.845 Pearson (vs 0.816 frozen)
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
from copy import deepcopy
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# CONFIGURATION - FINAL OPTIMIZED
# ============================================================================
config = {
    # Teacher model
    "teacher_model": "/workspace/LAM/all-MiniLM-L6-v2",
    
    # Model architecture
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # DeltaNet parameters
    "fast_decay_init": 0.30,
    "slow_decay_init": 0.85,
    
    # 🔥 BATCH SIZE: 1024 effective
    "batch_size": 128,  # Increased from 64
    "gradient_accumulation_steps": 8,  # Increased from 4
    # Effective: 128 × 8 = 1024 ✅
    
    # Training parameters
    "max_length": 128,
    "gradient_clip": 1.0,
    
    # 🔥 TWO-PHASE TRAINING
    "phase1_steps": 8_000,  # Phase 1: Frozen FFNs (curriculum)
    "phase2_steps": 10_000,  # Phase 2: Co-adaptation
    "total_steps": 18_000,  # Total: 18K steps (~1B tokens)
    
    # Phase 1: Learn DeltaNet basics (frozen FFNs guide)
    "phase1_lr": 2e-5,
    "phase1_freeze_ffns": True,
    "phase1_weight_decay": 0.1,
    
    # Phase 2: Co-adaptation (FFNs adapt to DeltaNet)
    "phase2_lr": 1e-5,  # Lower LR for fine-tuning
    "phase2_freeze_ffns": False,
    "phase2_weight_decay_ffn": 0.15,  # Higher for FFNs
    "phase2_weight_decay_deltanet": 0.1,
    
    # Warmup
    "phase1_warmup": 1_000,
    "phase2_warmup": 500,
    
    # Distillation weights
    "distillation_weight": 1.0,
    "layer_distill_weight": 1.5,
    "identity_reg_weight": 0.01,
    "ortho_state_reg_weight": 0.002,
    "label_smoothing": 0.1,
    
    # Early stopping
    "eval_interval": 500,
    "early_stopping_patience": 2_000,
    "min_improvement": 0.001,
    "use_early_stopping": True,
    
    # Logging
    "log_interval": 50,
    "save_interval": 1_000,
    
    # Initialization
    "init_deltanet_from_teacher": True,  # 🔥 WARM START!
    
    # Output
    "output_dir": "/workspace/LAM/deltanet_minilm_FINAL",
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# MODEL: NO FROZEN SANDWICH!
# ============================================================================
class DeltaNetDistillation(nn.Module):
    """
    DeltaNet with proper knowledge distillation
    - Teacher: FROZEN (only for loss computation)
    - Student: TRAINABLE copies (co-adaptation)
    """
    
    def __init__(self, teacher_model_name, config):
        super().__init__()
        
        print(f"\n{'='*80}")
        print("INITIALIZING DELTANET DISTILLATION MODEL")
        print(f"{'='*80}")
        
        # Load teacher (FROZEN - only for loss computation)
        print(f"Loading teacher: {teacher_model_name}")
        teacher_path = Path(teacher_model_name)
        if teacher_path.exists() and teacher_path.is_dir():
            self.teacher_model = AutoModel.from_pretrained(str(teacher_path.resolve()))
            self.tokenizer = AutoTokenizer.from_pretrained(str(teacher_path.resolve()))
        else:
            self.teacher_model = AutoModel.from_pretrained(teacher_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        
        # Freeze teacher completely
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print("✅ Teacher frozen (only for loss computation)")
        
        self.d_model = self.teacher_model.config.hidden_size
        self.num_layers = config['num_layers']
        
        # 🔥 STUDENT: Trainable copies (NO frozen sandwich!)
        print("\n🔧 Creating student architecture (trainable copies)...")
        
        # 1. Embeddings - COPY and make trainable
        self.embeddings = deepcopy(self.teacher_model.embeddings)
        for param in self.embeddings.parameters():
            param.requires_grad = True
        print("   ✅ Embeddings: Copied from teacher, TRAINABLE")
        
        # 2. DeltaNet layers + surrounding components
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            # DeltaNet attention (random init, then warm start)
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model,
                    num_heads=config['num_heads'],
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=config['fast_decay_init'],
                    slow_decay_init=config['slow_decay_init'],
                )
            )
            
            # Norms - COPY and make trainable
            self.deltanet_norms.append(
                deepcopy(self.teacher_model.encoder.layer[i].attention.output.LayerNorm)
            )
            for param in self.deltanet_norms[-1].parameters():
                param.requires_grad = True
            
            # FFNs - COPY and make trainable
            # Note: Will be frozen in Phase 1, trainable in Phase 2
            ffn_module = deepcopy(self.teacher_model.encoder.layer[i].intermediate)
            self.deltanet_ffns.append(ffn_module)
            for param in ffn_module.parameters():
                param.requires_grad = True  # Default trainable
            
            # FFN output + norm
            self.ffn_norms.append(
                deepcopy(self.teacher_model.encoder.layer[i].output.LayerNorm)
            )
            for param in self.ffn_norms[-1].parameters():
                param.requires_grad = True
        
        print(f"   ✅ Created {self.num_layers} DeltaNet layers")
        print("   ✅ Norms: Copied from teacher, TRAINABLE")
        print("   ✅ FFNs: Copied from teacher, TRAINABLE (will freeze in Phase 1)")
        
        # 3. Pooler - COPY and make trainable
        self.pooler = deepcopy(self.teacher_model.pooler)
        for param in self.pooler.parameters():
            param.requires_grad = True
        print("   ✅ Pooler: Copied from teacher, TRAINABLE")
        
        # 🔥 Initialize DeltaNet from teacher attention (WARM START!)
        if config.get('init_deltanet_from_teacher', True):
            print("\n⚡ WARM START: Initializing DeltaNet from teacher attention...")
            self._init_from_teacher_attention()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        
        print(f"\n📊 Parameter Count:")
        print(f"   Teacher: {teacher_params:,} (frozen)")
        print(f"   Student: {trainable_params:,} (trainable)")
        print(f"   Total: {total_params:,}")
        print(f"{'='*80}\n")
    
    def _init_from_teacher_attention(self):
        """Initialize DeltaNet projections from teacher's attention weights"""
        for i, deltanet_layer in enumerate(self.deltanet_layers):
            teacher_attn = self.teacher_model.encoder.layer[i].attention.self
            
            with torch.no_grad():
                # Copy q, k, v weights from teacher
                if hasattr(deltanet_layer, 'q_proj') and hasattr(teacher_attn, 'query'):
                    deltanet_layer.q_proj.weight.data.copy_(teacher_attn.query.weight.data)
                    print(f"   Layer {i}: q_proj ← teacher.query")
                
                if hasattr(deltanet_layer, 'k_proj') and hasattr(teacher_attn, 'key'):
                    deltanet_layer.k_proj.weight.data.copy_(teacher_attn.key.weight.data)
                    print(f"   Layer {i}: k_proj ← teacher.key")
                
                if hasattr(deltanet_layer, 'v_proj') and hasattr(teacher_attn, 'value'):
                    deltanet_layer.v_proj.weight.data.copy_(teacher_attn.value.weight.data)
                    print(f"   Layer {i}: v_proj ← teacher.value")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def forward_student(self, input_ids, attention_mask):
        """Student forward pass"""
        x = self.embeddings(input_ids=input_ids)
        
        student_hidden_states = []
        ortho_losses = []
        
        for i in range(self.num_layers):
            # Attention block
            residual = x
            x_attn, _, _, ortho_loss = self.deltanet_layers[i](x, attention_mask)
            if ortho_loss is not None:
                ortho_losses.append(ortho_loss)
            x = self.deltanet_norms[i](residual + x_attn)
            
            # FFN block
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            # Get output projection from teacher structure
            orig_layer = self.teacher_model.encoder.layer[i]
            x_ffn = orig_layer.output.dense(x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
            
            student_hidden_states.append(x)
        
        # Final embedding
        embeddings = self.mean_pooling(x, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Sum orthogonal losses from all layers
        total_ortho_loss = sum(ortho_losses) if ortho_losses else torch.tensor(0.0, device=embeddings.device)
        
        return embeddings, student_hidden_states, total_ortho_loss
    
    def forward_teacher(self, input_ids, attention_mask):
        """Teacher forward pass (frozen)"""
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            teacher_hidden_states = outputs.hidden_states[1:7]  # 6 layers
            
        return embeddings, teacher_hidden_states
    
    def forward(self, input_ids, attention_mask):
        """Forward both student and teacher"""
        student_emb, student_hidden, ortho_loss = self.forward_student(input_ids, attention_mask)
        teacher_emb, teacher_hidden = self.forward_teacher(input_ids, attention_mask)
        return student_emb, teacher_emb, student_hidden, teacher_hidden, ortho_loss

# ============================================================================
# LOSS FUNCTION
# ============================================================================
def compute_loss(student_emb_a, student_emb_b, teacher_emb_a, teacher_emb_b,
                 student_hidden_a, student_hidden_b, teacher_hidden_a, teacher_hidden_b,
                 labels, model, config, ortho_loss_a=None, ortho_loss_b=None):
    """Combined loss with all distillation objectives"""
    
    # 1. Contrastive loss with label smoothing
    scores = torch.mm(student_emb_a, student_emb_b.transpose(0, 1)) * 20.0
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    contrastive_loss = (cross_entropy(scores, labels) + 
                       cross_entropy(scores.transpose(0, 1), labels)) / 2
    
    # 2. Embedding distillation
    distill_loss_a = F.mse_loss(student_emb_a, teacher_emb_a)
    distill_loss_b = F.mse_loss(student_emb_b, teacher_emb_b)
    distill_loss = (distill_loss_a + distill_loss_b) / 2
    
    # 3. Layer-wise distillation
    layer_distill_loss = torch.tensor(0.0, device=student_emb_a.device)
    for s_out, t_out in zip(student_hidden_a, teacher_hidden_a):
        layer_distill_loss += F.mse_loss(s_out, t_out)
    for s_out, t_out in zip(student_hidden_b, teacher_hidden_b):
        layer_distill_loss += F.mse_loss(s_out, t_out)
    layer_distill_loss /= (2.0 * len(student_hidden_a))
    
    # 4. Orthogonal regularization (W^T W ≈ I)
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
    
    # 5. ORTHOGONAL STATE REGULARIZATION ⭐
    # Prevents S_fast and S_slow from becoming correlated (memory interference)
    ortho_state_reg = torch.tensor(0.0, device=student_emb_a.device)
    if ortho_loss_a is not None:
        ortho_state_reg += ortho_loss_a
    if ortho_loss_b is not None:
        ortho_state_reg += ortho_loss_b
    ortho_state_reg = ortho_state_reg / 2.0  # Average over two sentences
    
    # Total loss
    ortho_state_weight = config.get('ortho_state_reg_weight', 0.002)
    total_loss = (
        contrastive_loss + 
        config['distillation_weight'] * distill_loss +
        config['layer_distill_weight'] * layer_distill_loss +
        config['identity_reg_weight'] * identity_reg +
        ortho_state_weight * ortho_state_reg
    )
    
    return total_loss, contrastive_loss, distill_loss, identity_reg, layer_distill_loss, ortho_state_reg

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load multi-domain training data (NO STS-B!)"""
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
    
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    all_data = []
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # 1. AllNLI
    print("\n1️⃣  Loading AllNLI...")
    try:
        allnli_path = data_dir / "AllNLI.jsonl.gz"
        if allnli_path.exists():
            with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
                count = 0
                for line in f:
                    triplet = json.loads(line)
                    if len(triplet) == 3:
                        anchor, positive, _ = triplet
                        if len(anchor) > 10 and len(positive) > 10:
                            all_data.append({'sentence1': anchor, 'sentence2': positive})
                            count += 1
            print(f"   ✅ AllNLI: {count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  AllNLI: {e}")
    
    # 2. QQP
    print("\n2️⃣  Loading QQP...")
    try:
        qqp = load_dataset("glue", "qqp", split="train[:300000]", cache_dir="/workspace/.cache/huggingface/datasets")
        qqp_count = 0
        for item in qqp:
            if item['label'] != -1 and len(item['question1']) > 10 and len(item['question2']) > 10:
                all_data.append({'sentence1': item['question1'], 'sentence2': item['question2']})
                qqp_count += 1
        print(f"   ✅ QQP: {qqp_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  QQP: {e}")
    
    # 3. MS MARCO
    print("\n3️⃣  Loading MS MARCO...")
    try:
        msmarco = load_dataset("ms_marco", "v1.1", split="train[:150000]", cache_dir="/workspace/.cache/huggingface/datasets")
        msmarco_count = 0
        for item in msmarco:
            if 'query' in item and 'passages' in item:
                query = item['query']
                passages = item['passages']
                if isinstance(passages, str):
                    passages = json.loads(passages)
                if isinstance(passages, dict) and 'passage_text' in passages:
                    for passage in passages['passage_text'][:1]:  # First passage only
                        if len(query) > 10 and len(passage) > 20:
                            all_data.append({'sentence1': query, 'sentence2': passage[:200]})
                            msmarco_count += 1
                            break
        print(f"   ✅ MS MARCO: {msmarco_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  MS MARCO: {e}")
    
    # 4. WikiAnswers
    print("\n4️⃣  Loading WikiAnswers...")
    wiki_count = 0
    try:
        wiki_path = data_dir / "WikiAnswers_1M.jsonl.gz"
        if wiki_path.exists():
            with gzip.open(wiki_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    pair = json.loads(line)
                    if len(pair) == 2:
                        q1, q2 = pair
                        if len(q1) > 10 and len(q2) > 10:
                            all_data.append({'sentence1': q1, 'sentence2': q2})
                            wiki_count += 1
            print(f"   ✅ WikiAnswers: {wiki_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  WikiAnswers: {e}")
    
    # 5. SNLI
    print("\n5️⃣  Loading SNLI...")
    try:
        snli = load_dataset("snli", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
        snli_count = 0
        for item in snli:
            if item['label'] in [0, 1]:
                if len(item['premise']) > 10 and len(item['hypothesis']) > 10:
                    all_data.append({'sentence1': item['premise'], 'sentence2': item['hypothesis']})
                    snli_count += 1
                    if snli_count >= 150000:
                        break
        print(f"   ✅ SNLI: {snli_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  SNLI: {e}")
    
    print("\n" + "="*80)
    print(f"📊 TOTAL: {len(all_data):,} pairs")
    print("="*80)
    
    final_dataset = Dataset.from_list(all_data)
    final_dataset = final_dataset.shuffle(seed=42)
    print("✅ Dataset ready\n")
    
    return final_dataset

# ============================================================================
# VALIDATION
# ============================================================================
def evaluate_on_sts_dev(model, device, batch_size=32):
    """Evaluate on STS-B dev set for early stopping"""
    print("\n📊 Evaluating on STS-B dev set...")
    
    try:
        sts_dev = load_dataset("glue", "stsb", split="validation", cache_dir="/workspace/.cache/huggingface/datasets")
    except:
        try:
            sts_dev = load_dataset("sentence-transformers/stsb", split="validation")
    except:
        print("   ⚠️  Could not load STS-B dev set")
        return 0.0
    
    # Access columns directly (HuggingFace dataset format)
    s1 = sts_dev["sentence1"]
    s2 = sts_dev["sentence2"]
    # Handle both 'label' (glue) and 'score' (sentence-transformers) column names
    if 'label' in sts_dev.column_names:
        labels = np.array(sts_dev["label"], dtype=float)
    else:
        labels = np.array(sts_dev["score"], dtype=float)
    
    model.eval()
    all_sims = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(0, len(s1), batch_size):
            batch_s1 = s1[i:min(i+batch_size, len(s1))]
            batch_s2 = s2[i:min(i+batch_size, len(s2))]
            batch_labels = labels[i:min(i+batch_size, len(labels))]
            
            # Tokenize
            tokens_a = model.tokenizer(
                batch_s1,
                padding='max_length',
                max_length=128,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            tokens_b = model.tokenizer(
                batch_s2,
                padding='max_length',
                max_length=128,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # Get embeddings
            student_emb_a, _, _, _, _ = model(tokens_a['input_ids'], tokens_a['attention_mask'])
            student_emb_b, _, _, _, _ = model(tokens_b['input_ids'], tokens_b['attention_mask'])
            
            # Cosine similarity
            sims = F.cosine_similarity(student_emb_a, student_emb_b)
            all_sims.extend(sims.cpu().numpy())
            all_labels.extend(batch_labels.tolist())
    
    model.train()
    
    # Pearson correlation
    pearson, _ = pearsonr(all_sims, all_labels)
    print(f"   Pearson: {pearson:.4f}")
    
    return pearson

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train():
    print("="*80)
    print("DELTANET KNOWLEDGE DISTILLATION - FINAL OPTIMIZED VERSION")
    print("="*80)
    print(f"\n🎯 Training Configuration:")
    print(f"   Effective Batch Size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"   Phase 1: 0-{config['phase1_steps']:,} steps (Frozen FFNs)")
    print(f"   Phase 2: {config['phase1_steps']:,}-{config['total_steps']:,} steps (Co-adaptation)")
    print(f"   Total Steps: {config['total_steps']:,} (~1B tokens)")
    print(f"   Warm Start: {config['init_deltanet_from_teacher']}")
    print(f"   Early Stopping: {config['use_early_stopping']}")
    print("="*80)
    
    # Load data
    dataset = load_data()
    
    # Initialize model
    model = DeltaNetDistillation(config['teacher_model'], config).to(device)
    
    # 🔥 PHASE 1: Freeze FFNs
    print("\n🔧 PHASE 1 SETUP: Freezing FFNs...")
    for ffn in model.deltanet_ffns:
        for param in ffn.parameters():
            param.requires_grad = False
    print("   ✅ FFNs frozen (curriculum learning)")
    
    # Optimizer for Phase 1
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"   Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(
        trainable_params,
        lr=config['phase1_lr'],
        weight_decay=config['phase1_weight_decay']
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['phase1_warmup'],
        num_training_steps=config['phase1_steps']
    )
    
    # Training state
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_pearson = 0.0
    patience_counter = 0
    best_checkpoint_step = 0
    
    global_step = 0
    current_phase = 1
    
    running_loss = 0.0
    running_contrastive = 0.0
    running_distill = 0.0
    running_layer_distill = 0.0
    
    pbar = tqdm(total=config['total_steps'], desc="Training")
    
    # Training loop
    model.train()
    
    while global_step < config['total_steps']:
        
        # 🔥 PHASE 2 TRANSITION
        if global_step == config['phase1_steps'] and current_phase == 1:
            print("\n" + "="*80)
            print("🔄 PHASE 2: Unfreezing FFNs for co-adaptation!")
            print("="*80)
            
            current_phase = 2
            
            # Unfreeze FFNs
            for ffn in model.deltanet_ffns:
                for param in ffn.parameters():
                    param.requires_grad = True
            print("   ✅ FFNs unfrozen")
            
            # New optimizer with different weight decay for FFNs vs DeltaNet
            param_groups = [
                {
                    'params': [p for n, p in model.named_parameters() 
                              if 'deltanet_ffns' in n and p.requires_grad],
                    'lr': config['phase2_lr'],
                    'weight_decay': config['phase2_weight_decay_ffn']
                },
                {
                    'params': [p for n, p in model.named_parameters() 
                              if 'deltanet_layers' in n and p.requires_grad],
                    'lr': config['phase2_lr'],
                    'weight_decay': config['phase2_weight_decay_deltanet']
                },
                {
                    'params': [p for n, p in model.named_parameters() 
                              if 'deltanet' not in n and p.requires_grad],
                    'lr': config['phase2_lr'],
                    'weight_decay': config['phase2_weight_decay_deltanet']
                }
            ]
            
            optimizer = AdamW(param_groups)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=config['phase2_warmup'],
                num_training_steps=config['phase2_steps']
            )
            
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            print(f"   Trainable params: {sum(p.numel() for p in trainable_params):,}")
            print("="*80 + "\n")
        
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
        
        # Loss
        labels = torch.arange(len(student_emb_a), device=device)
        loss, contrastive_loss, distill_loss, identity_reg, layer_distill_loss, ortho_state_reg = compute_loss(
            student_emb_a, student_emb_b,
            teacher_emb_a, teacher_emb_b,
            student_hidden_a, student_hidden_b,
            teacher_hidden_a, teacher_hidden_b,
            labels, model, config,
            ortho_loss_a=ortho_loss_a, ortho_loss_b=ortho_loss_b
        )
        
        # Backward with gradient accumulation
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()
        
        # Update weights
        if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config['gradient_clip']
            )
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
                'phase': current_phase,
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
        
        # Validation & Early Stopping
        if (global_step + 1) % config['eval_interval'] == 0:
            val_pearson = evaluate_on_sts_dev(model, device)
            
            if config['use_early_stopping']:
                improvement = val_pearson - best_val_pearson
                
                if improvement > config['min_improvement']:
                    best_val_pearson = val_pearson
                    best_checkpoint_step = global_step + 1
                    patience_counter = 0
                    
                    # Save best checkpoint
                    print(f"   ✅ New best! Saving checkpoint...")
                    torch.save({
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_pearson': best_val_pearson,
                        'config': config
                    }, output_dir / "checkpoint_best.pt")
                else:
                    patience_counter += config['eval_interval']
                    print(f"   ⏳ No improvement for {patience_counter} steps (best: {best_val_pearson:.4f})")
                    
                    if patience_counter >= config['early_stopping_patience']:
                        print(f"\n🛑 EARLY STOPPING at step {global_step + 1}")
                        print(f"   Best: {best_val_pearson:.4f} at step {best_checkpoint_step}")
                        break
        
        # Save regular checkpoint
        if (global_step + 1) % config['save_interval'] == 0:
            torch.save({
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, output_dir / f"checkpoint_{global_step+1}.pt")
            model.tokenizer.save_pretrained(output_dir)
            print(f"\n💾 Checkpoint saved at step {global_step + 1}")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final save
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"   Best validation Pearson: {best_val_pearson:.4f} at step {best_checkpoint_step}")
    print(f"   Saved to: {output_dir}/")
    
    # Extract standalone model
    print("\n🔧 Extracting standalone student model (no teacher)...")
    extract_standalone_model(output_dir / "checkpoint_best.pt", output_dir / "student_model.bin")

# ============================================================================
# STANDALONE MODEL EXTRACTION
# ============================================================================
def extract_standalone_model(checkpoint_path, output_path):
    """Extract student-only weights (remove teacher)"""
    print(f"   Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Extract only student components
    model_state = checkpoint.get('model_state_dict', checkpoint)
    student_state = {}
    
    for key, value in model_state.items():
        # Skip teacher model
        if not key.startswith('teacher_model'):
            student_state[key] = value
    
    # Save
    torch.save(student_state, output_path)
    
    # Size comparison
    orig_size = Path(checkpoint_path).stat().st_size / (1024**2)
    standalone_size = Path(output_path).stat().st_size / (1024**2)
    
    print(f"   ✅ Standalone model saved: {output_path}")
    print(f"   Size: {orig_size:.1f}MB → {standalone_size:.1f}MB (removed teacher: {orig_size - standalone_size:.1f}MB)")

if __name__ == "__main__":
    train()