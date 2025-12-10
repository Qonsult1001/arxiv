"""
DeltaNet Knowledge Distillation - 100K RESEARCH-SCALE TRAINING
🔥 ADAPTIVE PHASE TRANSITIONS: Plateaus trigger early phase advancement (not stopping!)

IMPROVEMENT: If any phase plateaus for 5K steps:
- Phase 1 plateau → Move to Phase 2 early
- Phase 2 plateau → Move to Phase 3 early  
- Phase 3 plateau → Stop (final phase complete)

Expected Performance: 0.87-0.89 Pearson (SOTA for linear attention)
Training Time: ~42 hours @ 2 it/s (RTX 5000 Ada)
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
import re
from copy import deepcopy
from scipy.stats import pearsonr
import time

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# CONFIGURATION - 100K RESEARCH-SCALE WITH ADAPTIVE PHASES
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
    "batch_size": 128,
    "gradient_accumulation_steps": 8,
    
    # Training parameters
    "max_length": 128,
    "gradient_clip": 0.01,
    
    # 🔥 100K THREE-PHASE TRAINING (with adaptive transitions)
    "phase1_max_steps": 20_000,   # Phase 1: Max 20K (may end early if plateau)
    "phase2_max_steps": 60_000,   # Phase 2: Max 60K (may end early if plateau)
    "phase3_max_steps": 20_000,   # Phase 3: Max 20K (stop if plateau)
    "total_steps": 100_000,
    
    # Target tokens per update
    "target_tokens_per_update": 524_288,
    
    # Phase 1: Foundation (frozen FFNs, build basics)
    "phase1_lr": 1.5e-4,
    "phase1_freeze_ffns": True,
    "phase1_weight_decay": 0.1,
    "phase1_warmup": 2_000,
    
    # Phase 2: Co-adaptation (trainable FFNs, main learning)
    "phase2_lr": 1e-5,
    "phase2_freeze_ffns": False,
    "phase2_weight_decay_ffn": 0.15,
    "phase2_weight_decay_deltanet": 0.1,
    "phase2_warmup": 2_000,
    
    # Phase 3: Fine-tuning (very low LR, polish)
    "phase3_lr": 3e-5,
    "phase3_weight_decay": 0.15,
    "phase3_warmup": 1_000,
    
    # Distillation weights
    "distillation_weight": 1.0,
    "layer_distill_weight": 1.5,
    "identity_reg_weight": 0.01,
    "ortho_state_reg_weight": 0.002,
    "label_smoothing": 0.1,
    
    # 🔥 ADAPTIVE PHASE TRANSITIONS (not full stopping!)
    "eval_interval": 500,
    "phase_patience": 5_000,  # If no improvement for 5K steps → advance phase
    "min_improvement": 0.0005,
    "use_adaptive_phases": True,  # Enable adaptive phase transitions
    
    # Logging
    "log_interval": 100,
    "save_interval": 3_000,
    
    # Initialization
    "init_deltanet_from_teacher": True,
    
    # Output
    "output_dir": "/workspace/LAM/deltanet_100k_SOTA",
}

# Calculate total tokens
config['total_tokens'] = config['total_steps'] * config['target_tokens_per_update']

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ============================================================================
# MODEL: [Same as before - keeping it concise]
# ============================================================================
class DeltaNetDistillation(nn.Module):
    """DeltaNet with proper knowledge distillation"""
    
    def __init__(self, teacher_model_name, config):
        super().__init__()
        
        print(f"\n{'='*80}")
        print("INITIALIZING DELTANET DISTILLATION MODEL")
        print(f"{'='*80}")
        
        # Load teacher
        print(f"Loading teacher: {teacher_model_name}")
        teacher_path = Path(teacher_model_name)
        if teacher_path.exists() and teacher_path.is_dir():
            self.teacher_model = AutoModel.from_pretrained(str(teacher_path.resolve()))
            self.tokenizer = AutoTokenizer.from_pretrained(str(teacher_path.resolve()))
        else:
            self.teacher_model = AutoModel.from_pretrained(teacher_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        print("✅ Teacher frozen")
        
        self.d_model = self.teacher_model.config.hidden_size
        self.num_layers = config['num_layers']
        
        print("\n🔧 Creating student architecture...")
        
        # Student components
        self.embeddings = deepcopy(self.teacher_model.embeddings)
        for param in self.embeddings.parameters():
            param.requires_grad = True
        print("   ✅ Embeddings: TRAINABLE")
        
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
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
            
            self.deltanet_norms.append(
                deepcopy(self.teacher_model.encoder.layer[i].attention.output.LayerNorm)
            )
            for param in self.deltanet_norms[-1].parameters():
                param.requires_grad = True
            
            ffn_module = deepcopy(self.teacher_model.encoder.layer[i].intermediate)
            self.deltanet_ffns.append(ffn_module)
            for param in ffn_module.parameters():
                param.requires_grad = True
            
            self.ffn_norms.append(
                deepcopy(self.teacher_model.encoder.layer[i].output.LayerNorm)
            )
            for param in self.ffn_norms[-1].parameters():
                param.requires_grad = True
        
        print(f"   ✅ Created {self.num_layers} DeltaNet layers")
        
        self.pooler = deepcopy(self.teacher_model.pooler)
        for param in self.pooler.parameters():
            param.requires_grad = True
        print("   ✅ Pooler: TRAINABLE")
        
        if config.get('init_deltanet_from_teacher', True):
            print("\n⚡ WARM START: Initializing DeltaNet from teacher...")
            self._init_from_teacher_attention()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        
        print(f"\n📊 Parameter Count:")
        print(f"   Teacher: {teacher_params:,} (frozen)")
        print(f"   Student: {trainable_params:,} (trainable)")
        print(f"   Total: {total_params:,}")
        print(f"{'='*80}\n")
    
    def _init_from_teacher_attention(self):
        for i, deltanet_layer in enumerate(self.deltanet_layers):
            teacher_attn = self.teacher_model.encoder.layer[i].attention.self
            
            with torch.no_grad():
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
        x = self.embeddings(input_ids=input_ids)
        
        student_hidden_states = []
        ortho_losses = []
        
        for i in range(self.num_layers):
            residual = x
            x_attn, _, _, ortho_loss = self.deltanet_layers[i](x, attention_mask)
            if ortho_loss is not None:
                ortho_losses.append(ortho_loss)
            x = self.deltanet_norms[i](residual + x_attn)
            
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            orig_layer = self.teacher_model.encoder.layer[i]
            x_ffn = orig_layer.output.dense(x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
            
            student_hidden_states.append(x)
        
        embeddings = self.mean_pooling(x, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Sum orthogonal losses from all layers
        total_ortho_loss = sum(ortho_losses) if ortho_losses else torch.tensor(0.0, device=embeddings.device)
        
        return embeddings, student_hidden_states, total_ortho_loss
    
    def forward_teacher(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            teacher_hidden_states = outputs.hidden_states[1:7]
            
        return embeddings, teacher_hidden_states
    
    def forward(self, input_ids, attention_mask):
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
    
    # 1. Contrastive loss
    scores = torch.mm(student_emb_a, student_emb_b.transpose(0, 1)) * 20.0
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    contrastive_loss = (cross_entropy(scores, labels) + 
                       cross_entropy(scores.transpose(0, 1), labels)) / 2
    
    # 2. Embedding distillation
    distill_loss = (F.mse_loss(student_emb_a, teacher_emb_a) + 
                   F.mse_loss(student_emb_b, teacher_emb_b)) / 2
    
    # 3. Layer-wise distillation
    layer_distill_loss = torch.tensor(0.0, device=student_emb_a.device)
    for s_out, t_out in zip(student_hidden_a, teacher_hidden_a):
        layer_distill_loss += F.mse_loss(s_out, t_out)
    for s_out, t_out in zip(student_hidden_b, teacher_hidden_b):
        layer_distill_loss += F.mse_loss(s_out, t_out)
    layer_distill_loss /= (2.0 * len(student_hidden_a))
    
    # 4. Orthogonal regularization
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
    """Load multi-domain training data - EXPANDED VERSION"""
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
    
    print("\n" + "="*80)
    print("LOADING TRAINING DATA - EXPANDED")
    print("="*80)
    all_data = []
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # 1. AllNLI (keep as is)
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
    
    # 2. QQP - EXPANDED (300K → 450K to reach 3M target)
    print("\n2️⃣  Loading QQP (expanded to 450K)...")
    try:
        # Just change the split - data already cached!
        qqp = load_dataset("glue", "qqp", split="train[:450000]", cache_dir="/workspace/.cache/huggingface/datasets")
        qqp_count = 0
        for item in qqp:
            if item['label'] != -1 and len(item['question1']) > 10 and len(item['question2']) > 10:
                all_data.append({'sentence1': item['question1'], 'sentence2': item['question2']})
                qqp_count += 1
        print(f"   ✅ QQP: {qqp_count:,} pairs (+150K more!)")
    except Exception as e:
        print(f"   ⚠️  QQP: {e}")
    
    # 3. MS MARCO - EXPANDED (150K → 500K to reach 3M target)
    print("\n3️⃣  Loading MS MARCO (expanded to 500K)...")
    try:
        # Increase limit - fast since already cached
        msmarco = load_dataset("ms_marco", "v1.1", split="train[:500000]", cache_dir="/workspace/.cache/huggingface/datasets")
        msmarco_count = 0
        for item in msmarco:
            if 'query' in item and 'passages' in item:
                query = item['query']
                passages = item['passages']
                if isinstance(passages, str):
                    passages = json.loads(passages)
                if isinstance(passages, dict) and 'passage_text' in passages:
                    for passage in passages['passage_text'][:1]:
                        if len(query) > 10 and len(passage) > 20:
                            all_data.append({'sentence1': query, 'sentence2': passage[:200]})
                            msmarco_count += 1
                            break
        print(f"   ✅ MS MARCO: {msmarco_count:,} pairs (+350K more!)")
    except Exception as e:
        print(f"   ⚠️  MS MARCO: {e}")
    
    # 4. WikiAnswers - EXPANDED (1M → 2M to reach 3M target)
    print("\n4️⃣  Loading WikiAnswers (expanded to 2M)...")
    wiki_count = 0
    try:
        wiki_path = data_dir / "WikiAnswers_1M.jsonl.gz"
        if wiki_path.exists():
            # First, load from pre-generated file
            with gzip.open(wiki_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    pair = json.loads(line)
                    if len(pair) == 2:
                        q1, q2 = pair
                        if len(q1) > 10 and len(q2) > 10:
                            all_data.append({'sentence1': q1, 'sentence2': q2})
                            wiki_count += 1
            
            # If we need more to reach 2M, stream from dataset
            if wiki_count < 2000000:
                print(f"   📥 Loaded {wiki_count:,} from file, streaming more to reach 2M...")
                wiki = load_dataset("sentence-transformers/wikianswers-duplicates", split="train", 
                                  cache_dir="/workspace/.cache/huggingface/datasets", streaming=True)
                for item in wiki:
                    q1, q2 = None, None
                    if 'anchor' in item and 'positive' in item:
                        q1, q2 = item['anchor'], item['positive']
                    elif 'question1' in item and 'question2' in item:
                        q1, q2 = item['question1'], item['question2']
                    
                    if q1 and q2 and len(q1) > 10 and len(q2) > 10:
                        all_data.append({'sentence1': q1, 'sentence2': q2})
                        wiki_count += 1
                        if wiki_count >= 2000000:  # Target 2M pairs
                            break
            print(f"   ✅ WikiAnswers: {wiki_count:,} pairs (target: 2M)")
        else:
            # No pre-generated file, stream directly from dataset
            print("   📥 Streaming WikiAnswers from dataset (target: 2M pairs)...")
            wiki = load_dataset("sentence-transformers/wikianswers-duplicates", split="train", 
                              cache_dir="/workspace/.cache/huggingface/datasets", streaming=True)
            for item in wiki:
                q1, q2 = None, None
                if 'anchor' in item and 'positive' in item:
                    q1, q2 = item['anchor'], item['positive']
                elif 'question1' in item and 'question2' in item:
                    q1, q2 = item['question1'], item['question2']
                
                if q1 and q2 and len(q1) > 10 and len(q2) > 10:
                    all_data.append({'sentence1': q1, 'sentence2': q2})
                    wiki_count += 1
                    if wiki_count >= 2000000:  # Target 2M pairs
                        break
            print(f"   ✅ WikiAnswers: {wiki_count:,} pairs (from streaming)")
    except Exception as e:
        print(f"   ⚠️  WikiAnswers: {e}")
    
    # 5. SNLI - EXPANDED (150K → 300K to reach 3M target)
    print("\n5️⃣  Loading SNLI (expanded to 300K)...")
    try:
        snli = load_dataset("snli", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
        snli_count = 0
        for item in snli:
            if item['label'] in [0, 1]:  # entailment or neutral
                if len(item['premise']) > 10 and len(item['hypothesis']) > 10:
                    all_data.append({'sentence1': item['premise'], 'sentence2': item['hypothesis']})
                    snli_count += 1
                    if snli_count >= 300000:  # Increased from 250K to reach 3M
                        break
        print(f"   ✅ SNLI: {snli_count:,} pairs (+150K more!)")
    except Exception as e:
        print(f"   ⚠️  SNLI: {e}")
    
    # 6. SQuAD (OPTIONAL - small, fast, useful!)
    print("\n6️⃣  Loading SQuAD v1.1...")
    try:
        squad = load_dataset("squad", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
        squad_count = 0
        for item in squad:
            if 'question' in item and 'context' in item:
                question = item['question']
                # Use first 200 chars of context as "answer"
                context = item['context'][:200]
                if len(question) > 10 and len(context) > 20:
                    all_data.append({'sentence1': question, 'sentence2': context})
                    squad_count += 1
                    if squad_count >= 87000:  # Full SQuAD train set
                        break
        print(f"   ✅ SQuAD: {squad_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  SQuAD: {e}")
    
    print("\n" + "="*80)
    print(f"📊 TOTAL: {len(all_data):,} pairs")
    print("="*80)
    
    # Calculate expected epochs for 100K steps with batch_size 1024
    if len(all_data) > 0:
        steps_per_epoch = len(all_data) / 1024  # Assuming batch_size 1024
        total_epochs = 100000 / steps_per_epoch
        print(f"📈 Expected epochs for 100K steps: {total_epochs:.1f} epochs")
        print(f"   (Safe zone: 30-40 epochs for best generalization)")
    
    final_dataset = Dataset.from_list(all_data)
    final_dataset = final_dataset.shuffle(seed=42)
    print("✅ Dataset ready\n")
    
    return final_dataset

# ============================================================================
# VALIDATION
# ============================================================================
def evaluate_on_sts_dev(model, device, batch_size=32):
    """Evaluate on STS-B dev set - aligned with test_checkpoints.py calculation"""
    print("\n📊 Evaluating on STS-B dev set...")
    
    # Use same dataset source as stsb_evaluation.py
    try:
        sts_dev = load_dataset("sentence-transformers/stsb", split="validation", cache_dir="/workspace/.cache/huggingface/datasets")
    except:
        try:
            sts_dev = load_dataset("glue", "stsb", split="validation", cache_dir="/workspace/.cache/huggingface/datasets")
        except:
            print("   ⚠️  Could not load STS-B dev set")
            return 0.0
    
    # Access columns directly (same as stsb_evaluation.py)
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
            
            tokens_a = model.tokenizer(
                batch_s1, padding='max_length', max_length=128,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            tokens_b = model.tokenizer(
                batch_s2, padding='max_length', max_length=128,
                truncation=True, return_tensors='pt'
            ).to(device)
            
            student_emb_a, _, _, _, _ = model(tokens_a['input_ids'], tokens_a['attention_mask'])
            student_emb_b, _, _, _, _ = model(tokens_b['input_ids'], tokens_b['attention_mask'])
            
            # Use same calculation as stsb_evaluation.py: cosine similarity
            # F.cosine_similarity is equivalent to compute_pairwise_sims (1 - cosine_distance)
            sims = F.cosine_similarity(student_emb_a, student_emb_b, dim=1)
            all_sims.extend(sims.cpu().numpy())
            all_labels.extend(batch_labels.tolist())
    
    model.train()
    
    # Compute Pearson correlation (same as stsb_evaluation.py pearson_cosine)
    # Use all_labels (the list we built) not labels (the numpy array)
    try:
        pearson = pearsonr(all_sims, all_labels)[0]
        if np.isnan(pearson):
            pearson = 0.0
    except (ValueError, RuntimeError):
        pearson = 0.0
    
    print(f"   Pearson: {pearson:.4f}")
    
    return pearson

# ============================================================================
# 🔥 ADAPTIVE PHASE TRANSITION LOGIC
# ============================================================================
def transition_to_next_phase(model, current_phase, config, global_step):
    """Transition to next phase when current phase plateaus"""
    
    if current_phase == 1:
        # Phase 1 → Phase 2: Unfreeze FFNs
        print("\n" + "="*80)
        print(f"🔄 EARLY TRANSITION: Phase 1 → Phase 2 at step {global_step}")
        print("   Phase 1 plateaued - moving to co-adaptation phase!")
        print("="*80)
        
        # Unfreeze FFNs
        for ffn in model.deltanet_ffns:
            for param in ffn.parameters():
                param.requires_grad = True
        print("   ✅ FFNs unfrozen")
        
        # New optimizer with different weight decay
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
        
        # Calculate remaining steps for Phase 2
        phase1_actual_steps = global_step
        remaining_total = config['total_steps'] - phase1_actual_steps
        phase2_remaining = min(config['phase2_max_steps'], remaining_total - config['phase3_max_steps'])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['phase2_warmup'],
            num_training_steps=phase2_remaining
        )
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"   Trainable params: {sum(p.numel() for p in trainable_params):,}")
        print(f"   Phase 2 will run for up to {phase2_remaining:,} more steps")
        print("="*80 + "\n")
        
        return 2, optimizer, scheduler, phase1_actual_steps + phase2_remaining
    
    elif current_phase == 2:
        # Phase 2 → Phase 3: Lower LR for fine-tuning
        print("\n" + "="*80)
        print(f"🔄 EARLY TRANSITION: Phase 2 → Phase 3 at step {global_step}")
        print("   Phase 2 plateaued - moving to fine-tuning phase!")
        print("="*80)
        
        # Lower LR optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=config['phase3_lr'],
            weight_decay=config['phase3_weight_decay']
        )
        
        # Calculate remaining steps for Phase 3
        remaining_total = config['total_steps'] - global_step
        phase3_remaining = min(config['phase3_max_steps'], remaining_total)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['phase3_warmup'],
            num_training_steps=phase3_remaining
        )
        
        print(f"   ✅ LR lowered to {config['phase3_lr']:.0e} for fine-tuning")
        print(f"   Phase 3 will run for up to {phase3_remaining:,} more steps")
        print("="*80 + "\n")
        
        return 3, optimizer, scheduler, global_step + phase3_remaining
    
    else:
        # Phase 3 plateaued - this is the final phase, so stop training
        return current_phase, None, None, None

# ============================================================================
# TRAINING LOOP WITH ADAPTIVE PHASE TRANSITIONS
# ============================================================================
def train():
    print("="*80)
    print("DELTANET 100K TRAINING - ADAPTIVE PHASE TRANSITIONS")
    print("="*80)
    print(f"\n🎯 Training Configuration:")
    print(f"   Effective Batch Size: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print(f"   Total Steps: {config['total_steps']:,}")
    print(f"   Total Tokens: {config['total_tokens']/1e9:.1f}B")
    print(f"\n🔄 ADAPTIVE THREE-PHASE TRAINING:")
    print(f"   Phase 1: Max {config['phase1_max_steps']:,} steps (frozen FFNs, LR={config['phase1_lr']:.0e})")
    print(f"      → Plateaus after 5K patience? Advance to Phase 2 early ✅")
    print(f"   Phase 2: Max {config['phase2_max_steps']:,} steps (trainable FFNs, LR={config['phase2_lr']:.0e})")
    print(f"      → Plateaus after 5K patience? Advance to Phase 3 early ✅")
    print(f"   Phase 3: Max {config['phase3_max_steps']:,} steps (fine-tune, LR={config['phase3_lr']:.0e})")
    print(f"      → Plateaus after 5K patience? Stop training ⏹️")
    print(f"\n🎯 TARGET MILESTONES:")
    print(f"   Step ~15-20K: 0.79-0.82 (Phase 1 → 2)")
    print(f"   Step ~60-70K: 0.84-0.86 (Phase 2 → 3)")
    print(f"   Step ~90-100K: 0.87-0.89 (SOTA!)")
    print("="*80)
    
    # Load data
    dataset = load_data()
    
    # Initialize model
    model = DeltaNetDistillation(config['teacher_model'], config).to(device)
    
    # 🔥 PHASE 1 SETUP
    print("\n🔧 PHASE 1 SETUP: Freezing FFNs...")
    for ffn in model.deltanet_ffns:
        for param in ffn.parameters():
            param.requires_grad = False
    print("   ✅ FFNs frozen")
    
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
        num_training_steps=config['phase1_max_steps']
    )
    
    # Training state
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_pearson = 0.0
    patience_counter = 0
    best_checkpoint_step = 0
    
    global_step = 0
    current_phase = 1
    phase_end_step = config['phase1_max_steps']  # When to force phase transition
    
    running_loss = 0.0
    running_contrastive = 0.0
    running_distill = 0.0
    running_layer_distill = 0.0
    
    start_time = time.time()
    pbar = tqdm(total=config['total_steps'], desc="Training")
    
    model.train()
    
    while global_step < config['total_steps']:
        
        # 🔥 FORCED PHASE TRANSITION (if max steps reached)
        if global_step >= phase_end_step and current_phase < 3:
            print(f"\n⏰ Max steps reached for Phase {current_phase}")
            new_phase, new_optimizer, new_scheduler, new_phase_end = transition_to_next_phase(
                model, current_phase, config, global_step
            )
            
            if new_phase != current_phase:
                current_phase = new_phase
                optimizer = new_optimizer
                scheduler = new_scheduler
                phase_end_step = new_phase_end
                patience_counter = 0  # Reset patience for new phase
                print(f"   🔄 Now in Phase {current_phase}")
        
        # Sample batch
        indices = np.random.randint(0, len(dataset), size=config['batch_size'])
        batch_data = [dataset[int(i)] for i in indices]
        
        sentences_a = [item['sentence1'] for item in batch_data]
        sentences_b = [item['sentence2'] for item in batch_data]
        
        # Tokenize
        tokens_a = model.tokenizer(
            sentences_a, padding='max_length', max_length=config['max_length'],
            truncation=True, return_tensors='pt'
        ).to(device)
        
        tokens_b = model.tokenizer(
            sentences_b, padding='max_length', max_length=config['max_length'],
            truncation=True, return_tensors='pt'
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
        
        # Backward
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()
        
        # Update
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
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0
            eta_hours = (config['total_steps'] - global_step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
            
            pbar.set_postfix({
                'phase': current_phase,
                'loss': f'{running_loss/config["log_interval"]:.4f}',
                'contr': f'{running_contrastive/config["log_interval"]:.4f}',
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                'patience': patience_counter,
                'eta': f"{eta_hours:.1f}h"
            })
            
            running_loss = 0.0
            running_contrastive = 0.0
            running_distill = 0.0
            running_layer_distill = 0.0
        
        # 🔥 VALIDATION & ADAPTIVE PHASE TRANSITION
        if (global_step + 1) % config['eval_interval'] == 0:
            val_pearson = evaluate_on_sts_dev(model, device)
            
            improvement = val_pearson - best_val_pearson
            
            if improvement > config['min_improvement']:
                best_val_pearson = val_pearson
                best_checkpoint_step = global_step + 1
                patience_counter = 0
                
                # Save best
                print(f"   ✅ New best! Saving checkpoint...")
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_pearson': best_val_pearson,
                    'config': config,
                    'phase': current_phase
                }, output_dir / "checkpoint_best.pt")
                
                # Save phase-specific checkpoint
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'best_val_pearson': best_val_pearson,
                    'config': config,
                    'phase': current_phase
                }, output_dir / f"step{global_step+1:05d}_p{val_pearson:.4f}.pt")
            else:
                patience_counter += config['eval_interval']
                print(f"   ⏳ No improvement for {patience_counter} steps (best: {best_val_pearson:.4f})")
                
                # 🔥 ADAPTIVE PHASE TRANSITION
                if config['use_adaptive_phases'] and patience_counter >= config['phase_patience']:
                    
                    if current_phase < 3:
                        # Transition to next phase
                        print(f"\n⚡ PLATEAU DETECTED: {patience_counter} steps without improvement")
                        
                        new_phase, new_optimizer, new_scheduler, new_phase_end = transition_to_next_phase(
                            model, current_phase, config, global_step
                        )
                        
                        if new_phase != current_phase:
                            current_phase = new_phase
                            optimizer = new_optimizer
                            scheduler = new_scheduler
                            phase_end_step = new_phase_end
                            patience_counter = 0  # Reset patience
                            print(f"   🔄 Continuing training in Phase {current_phase}")
                    else:
                        # Phase 3 plateaued - stop training
                        print(f"\n🛑 TRAINING COMPLETE: Phase 3 plateaued after {patience_counter} steps")
                        print(f"   Best: {best_val_pearson:.4f} at step {best_checkpoint_step}")
                        break
        
        # Save regular checkpoint
        if (global_step + 1) % config['save_interval'] == 0:
            torch.save({
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'phase': current_phase
            }, output_dir / f"checkpoint_{global_step+1}.pt")
            model.tokenizer.save_pretrained(output_dir)
            print(f"\n💾 Checkpoint saved at step {global_step + 1}")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"   Best validation Pearson: {best_val_pearson:.4f} at step {best_checkpoint_step}")
    print(f"   Final phase reached: Phase {current_phase}")
    print(f"   Total time: {(time.time() - start_time)/3600:.1f} hours")
    print(f"   Saved to: {output_dir}/")
    
    # Extract standalone model
    print("\n🔧 Extracting standalone student model...")
    extract_standalone_model(output_dir / "checkpoint_best.pt", output_dir / "student_model.bin")

# ============================================================================
# STANDALONE MODEL EXTRACTION
# ============================================================================
def extract_standalone_model(checkpoint_path, output_path):
    """Extract student-only weights (remove teacher)"""
    print(f"   Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model_state = checkpoint.get('model_state_dict', checkpoint)
    student_state = {}
    
    for key, value in model_state.items():
        if not key.startswith('teacher_model'):
            student_state[key] = value
    
    torch.save(student_state, output_path)
    
    orig_size = Path(checkpoint_path).stat().st_size / (1024**2)
    standalone_size = Path(output_path).stat().st_size / (1024**2)
    
    print(f"   ✅ Standalone model saved: {output_path}")
    print(f"   Size: {orig_size:.1f}MB → {standalone_size:.1f}MB")

if __name__ == "__main__":
    train()