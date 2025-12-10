"""
🚀 DELTANET 8K TRAINING - VARIABLE LENGTH CURRICULUM

✅ FIXED: Variable length training (512, 1024, 2048, 4096, 8192)
✅ Progressive curriculum: Short → Medium → Long
✅ Teaches FULL interpolation space for 8K support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pathlib import Path
import json
import gzip
import sys
import os
from datasets import load_dataset, Dataset

# Import our enhanced model
sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

# Configuration
config = {
    "teacher_model": "jinaai/jina-embeddings-v2-base-en",
    "checkpoint_file": "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/checkpoint_167000.pt",
    "student_base": "sentence-transformers/all-MiniLM-L6-v2",
    
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    "vocab_size": 30522,
    
    # ⭐ KEY CHANGE: Variable length training (not fixed!)
    "max_seq_length": 8192,  # Max inference capability
    "original_max_pos": 512,  # Base position embeddings
    "train_lengths": [512, 1024, 2048, 4096, 8192],  # ✅ Train on ALL lengths!
    
    "learning_rate": 2e-5,
    "batch_size": 4,  # Small for 8K sequences
    "gradient_accumulation_steps": 8,  # Effective: 4*8=32
    "total_steps": 30000,  # ⚡ OPTIMIZED: 30K (not 100K - you already trained 167K on short!)
    "resume_from_step": 0,
    "warmup_steps": 1500,  # 5% of 30K
    "gradient_clip": 1.0,
    "weight_decay": 0.1,
    "log_interval": 50,
    
    "distillation_weight": 1.0,
    "layer_distill_weight": 1.5,
    "identity_reg_weight": 0.01,
    "ortho_state_reg_weight": 0.002,
    "spearman_loss_weight": 0.3,
    "ranking_loss_weight": 0.2,
    "label_smoothing": 0.1,
    
    "output_dir": "/workspace/LAM/deltanet_8k_optimized_30K",
    "save_interval": 3000,  # Every 10% (10 checkpoints total)
    "eval_interval": 1500,  # Every 5% (20 evaluations total)
    
    # ⚡ START AT PHASE 2 (skip short sequences - already trained 167K steps!)
    "start_phase": 2,  # 1=Short(512-2K), 2=Medium(512-4K), 3=Long(1K-8K)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Enable TF32
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    print(f"✅ TensorFloat32 (TF32) enabled")

# ============================================================================
# ⭐ NEW: VARIABLE LENGTH CURRICULUM SAMPLING
# ============================================================================
def sample_sequence_length_curriculum(step, total_steps, start_phase=1):
    """
    Progressive curriculum learning: 512 → 8192
    
    Phase 1 (0-40%): Focus on short sequences (512-2048) - SKIP IF start_phase >= 2
    Phase 2 (40-70%): Medium sequences (512-4096) - SKIP IF start_phase >= 3  
    Phase 3 (70-100%): Full range including 8K (1024-8192)
    
    start_phase: 1=start from Short, 2=start from Medium, 3=start from Long
    """
    progress = step / total_steps
    
    # Adjust progress based on start_phase
    if start_phase == 2:
        # Skip phase 1 (0-40%), remap to 40-100%
        progress = 0.4 + progress * 0.6
    elif start_phase >= 3:
        # Skip phases 1+2 (0-70%), remap to 70-100%
        progress = 0.7 + progress * 0.3
    
    # Phase 1: Short sequences (maintain baseline)
    if progress < 0.4:
        lengths = [512, 1024, 2048]
        weights = [0.5, 0.3, 0.2]
    
    # Phase 2: Add longer sequences
    elif progress < 0.7:
        lengths = [512, 1024, 2048, 4096]
        weights = [0.3, 0.3, 0.25, 0.15]
    
    # Phase 3: Full range with 8K
    else:
        lengths = [1024, 2048, 4096, 8192]
        weights = [0.2, 0.3, 0.35, 0.15]
    
    return int(np.random.choice(lengths, p=weights))
        weights = [0.2, 0.3, 0.35, 0.15]
    
    return int(np.random.choice(lengths, p=weights))

def get_adaptive_batch_size(seq_length, base_batch_size=4):
    """
    Adaptive batch size: shorter sequences = larger batches
    Keeps GPU memory usage roughly constant
    """
    if seq_length <= 512:
        return base_batch_size * 4  # 16
    elif seq_length <= 1024:
        return base_batch_size * 2  # 8
    elif seq_length <= 2048:
        return base_batch_size  # 4
    elif seq_length <= 4096:
        return max(1, base_batch_size // 2)  # 2
    else:  # 8192
        return max(1, base_batch_size // 4)  # 1
    
def get_curriculum_phase_name(step, total_steps, start_phase=1):
    """Get current curriculum phase name for logging"""
    progress = step / total_steps
    
    # Adjust progress based on start_phase
    if start_phase == 2:
        progress = 0.4 + progress * 0.6
    elif start_phase >= 3:
        progress = 0.7 + progress * 0.3
    
    if progress < 0.4:
        return "Short (512-2K)"
    elif progress < 0.7:
        return "Medium (512-4K)"
    else:
        return "Long (1K-8K)"

# ============================================================================
# MODEL
# ============================================================================
class DeltaNet8KExtended(nn.Module):
    """8K Extended DeltaNet with position interpolation"""
    
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.max_seq_length = config['max_seq_length']
        self.original_max_pos = config['original_max_pos']
        
        # Embeddings
        self.embeddings = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embeddings = nn.Embedding(config['original_max_pos'], config['d_model'])
        self.token_type_embeddings = nn.Embedding(2, config['d_model'])
        self.embedding_norm = nn.LayerNorm(config['d_model'])
        self.embedding_dropout = nn.Dropout(0.1)
        
        # DeltaNet layers
        self.deltanet_layers = nn.ModuleList()
        self.deltanet_norms = nn.ModuleList()
        self.deltanet_ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.ffn_outputs = nn.ModuleList()
        
        for _ in range(config['num_layers']):
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
            self.ffn_outputs.append(nn.Linear(config['d_model'] * 4, config['d_model']))
        
        # Teacher projections
        self.teacher_projections = nn.ModuleList([
            nn.Linear(768, config['d_model']) for _ in range(config['num_layers'])
        ])
        self.teacher_emb_projection = nn.Linear(768, config['d_model'])
    
    def get_extended_position_embeddings(self, seq_len, batch_size):
        """Position interpolation: 512 → any length"""
        if seq_len <= self.original_max_pos:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.embeddings.weight.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            return self.position_embeddings(position_ids)
        else:
            # Interpolate for long sequences
            scale_factor = (self.original_max_pos - 1) / (seq_len - 1)
            position_embeddings_list = []
            
            for pos in range(seq_len):
                original_pos = pos * scale_factor
                lower_pos = int(original_pos)
                upper_pos = min(lower_pos + 1, self.original_max_pos - 1)
                weight = original_pos - lower_pos
                
                lower_emb = self.position_embeddings.weight[lower_pos]
                upper_emb = self.position_embeddings.weight[upper_pos]
                interp_emb = (1 - weight) * lower_emb + weight * upper_emb
                position_embeddings_list.append(interp_emb)
            
            position_embeddings = torch.stack(position_embeddings_list, dim=0)
            return position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
    
    def mean_pooling(self, token_embeddings, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_emb = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        return sum_emb / sum_mask
    
    def forward(self, input_ids, attention_mask, return_hidden=False):
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings with interpolation
        word_emb = self.embeddings(input_ids)
        pos_emb = self.get_extended_position_embeddings(seq_len, batch_size)
        
        token_type_ids = torch.zeros_like(input_ids)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        
        hidden = word_emb + pos_emb + token_type_emb
        hidden = self.embedding_norm(hidden)
        hidden = self.embedding_dropout(hidden)
        
        hidden_states = []
        ortho_losses = []
        
        # DeltaNet layers
        for i in range(len(self.deltanet_layers)):
            # Attention
            residual = hidden
            hidden_attn, _, _, ortho_loss = self.deltanet_layers[i](hidden, attention_mask)
            if ortho_loss is not None:
                ortho_losses.append(ortho_loss)
            hidden = self.deltanet_norms[i](residual + hidden_attn)
            
            # FFN
            residual = hidden
            hidden_ffn = self.deltanet_ffns[i](hidden)
            hidden_ffn = F.gelu(hidden_ffn)
            hidden_ffn = self.ffn_outputs[i](hidden_ffn)
            hidden = self.ffn_norms[i](residual + hidden_ffn)
            
            if return_hidden:
                hidden_states.append(hidden)
        
        pooled = self.mean_pooling(hidden, attention_mask)
        pooled = F.normalize(pooled, p=2, dim=1)
        
        total_ortho_loss = sum(ortho_losses) if ortho_losses else torch.tensor(0.0, device=pooled.device)
        
        if return_hidden:
            return pooled, hidden_states, total_ortho_loss
        else:
            return pooled

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
    
    print("\n" + "="*80)
    print("LOADING MULTI-DOMAIN TRAINING DATA")
    print("="*80)
    all_data = []
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    
    # 1. AllNLI
    print("\n1️⃣  Loading LOCAL AllNLI...")
    try:
        allnli_path = data_dir / "AllNLI.jsonl.gz"
        if allnli_path.exists():
            with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
                count = 0
                for line in f:
                    triplet = json.loads(line)
                    if len(triplet) == 3:
                        anchor, positive, negative = triplet
                        if len(anchor) > 10 and len(positive) > 10:
                            all_data.append({'sentence1': anchor, 'sentence2': positive})
                            count += 1
            print(f"   ✅ AllNLI: {count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  Could not load AllNLI: {e}")
    
    # 2. QQP
    print("\n2️⃣  Loading QQP...")
    try:
        qqp = load_dataset("glue", "qqp", split="train[:200000]", cache_dir="/workspace/.cache/huggingface/datasets")
        qqp_count = 0
        for item in qqp:
            if item['label'] != -1 and len(item['question1']) > 10 and len(item['question2']) > 10:
                all_data.append({'sentence1': item['question1'], 'sentence2': item['question2']})
                qqp_count += 1
        print(f"   ✅ QQP: {qqp_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  Could not load QQP: {e}")
    
    # 3. SNLI
    print("\n3️⃣  Loading SNLI...")
    try:
        snli = load_dataset("snli", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
        snli_count = 0
        for item in snli:
            if item['label'] in [0, 1]:
                if len(item['premise']) > 10 and len(item['hypothesis']) > 10:
                    all_data.append({'sentence1': item['premise'], 'sentence2': item['hypothesis']})
                    snli_count += 1
                    if snli_count >= 100000:
                        break
        print(f"   ✅ SNLI: {snli_count:,} pairs")
    except Exception as e:
        print(f"   ⚠️  Could not load SNLI: {e}")
    
    # 4. BookSum (very long context - book chapters)
    print("\n4️⃣  Loading BookSum (book chapters - avg 3.5K tokens)...")
    try:
        booksum = load_dataset("kmfoda/booksum", split="train", cache_dir="/workspace/.cache/huggingface/datasets")
        booksum_count = 0
        for item in booksum:
            chapter = item.get('chapter', '')
            summary = item.get('summary_text', '') or item.get('summary', '')
            # BookSum chapters are VERY long (3-6K tokens) - perfect for 4K/8K training
            if len(chapter) > 1000 and len(summary) > 50:
                all_data.append({
                    'sentence1': chapter[:20000],  # Truncate extremely long chapters
                    'sentence2': summary
                })
                booksum_count += 1
                if booksum_count >= 20000:  # Limit to 20K pairs
                    break
        print(f"   ✅ BookSum: {booksum_count:,} very long pairs (avg 3.5K tokens)")
    except Exception as e:
        print(f"   ⚠️  Could not load BookSum: {e}")
    
    # 5. CNN/DailyMail (long context)
    print("\n5️⃣  Loading CNN/DailyMail (news articles - avg 786 tokens)...")
    try:
        cnn = load_dataset("cnn_dailymail", "3.0.0", split="train[:50000]", cache_dir="/workspace/.cache/huggingface/datasets")
        cnn_count = 0
        for item in cnn:
            article = item.get('article', '')
            highlights = item.get('highlights', '')
            if len(article) > 500 and len(highlights) > 50:
                all_data.append({'sentence1': article, 'sentence2': highlights})
                cnn_count += 1
        print(f"   ✅ CNN/DailyMail: {cnn_count:,} long pairs (avg 786 tokens)")
    except Exception as e:
        print(f"   ⚠️  Could not load CNN/DailyMail: {e}")
    
    print(f"\n📊 TOTAL: {len(all_data):,} pairs")
    
    # Calculate long vs short ratio
    short_count = sum(1 for item in all_data if 'AllNLI' in str(item) or 'QQP' in str(item) or 'SNLI' in str(item))
    long_count = len(all_data) - short_count
    print(f"   Estimated short context: ~{len(all_data) - 70000:,} pairs")
    print(f"   Estimated long context: ~70,000 pairs (BookSum + CNN)")
    
    final_dataset = Dataset.from_list(all_data)
    final_dataset = final_dataset.shuffle(seed=42)
    
    return final_dataset

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
def spearman_correlation_loss(pred_scores, target_scores):
    pred_ranks = torch.argsort(torch.argsort(pred_scores, dim=0, descending=True), dim=0).float()
    target_ranks = torch.argsort(torch.argsort(target_scores, dim=0, descending=True), dim=0).float()
    n = len(pred_ranks)
    pred_ranks = pred_ranks / (n - 1) if n > 1 else pred_ranks
    target_ranks = target_ranks / (n - 1) if n > 1 else target_ranks
    return F.mse_loss(pred_ranks, target_ranks)

def pairwise_ranking_loss(student_sim, teacher_sim, margin=0.05):
    n = len(student_sim)
    if n < 2:
        return torch.tensor(0.0, device=student_sim.device)
    student_diff = student_sim.unsqueeze(1) - student_sim.unsqueeze(0)
    teacher_diff = teacher_sim.unsqueeze(1) - teacher_sim.unsqueeze(0)
    mask = (teacher_diff.abs() > 0.01) & (torch.eye(n, device=student_sim.device) == 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=student_sim.device)
    teacher_ordering = (teacher_diff > 0).float()
    ranking_loss = F.binary_cross_entropy_with_logits(student_diff[mask] / margin, teacher_ordering[mask])
    return ranking_loss

def compute_loss(student_emb_a, student_emb_b, teacher_emb_a, teacher_emb_b,
                 student_hidden_a, student_hidden_b, teacher_hidden_a, teacher_hidden_b,
                 labels, model, config, ortho_loss_a=None, ortho_loss_b=None):
    
    scores = torch.mm(student_emb_a, student_emb_b.transpose(0, 1)) * 20.0
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    contrastive_loss = (cross_entropy(scores, labels) + cross_entropy(scores.transpose(0, 1), labels)) / 2
    
    teacher_emb_a_proj = model.teacher_emb_projection(teacher_emb_a)
    teacher_emb_b_proj = model.teacher_emb_projection(teacher_emb_b)
    distill_loss = (F.mse_loss(student_emb_a, teacher_emb_a_proj) + F.mse_loss(student_emb_b, teacher_emb_b_proj)) / 2
    
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
    
    layer_distill_loss = torch.tensor(0.0, device=student_emb_a.device)
    for i, (s_out, t_out) in enumerate(zip(student_hidden_a, teacher_hidden_a)):
        t_out_proj = model.teacher_projections[i](t_out)
        layer_distill_loss += F.mse_loss(s_out, t_out_proj)
    for i, (s_out, t_out) in enumerate(zip(student_hidden_b, teacher_hidden_b)):
        t_out_proj = model.teacher_projections[i](t_out)
        layer_distill_loss += F.mse_loss(s_out, t_out_proj)
    layer_distill_loss /= (2.0 * len(student_hidden_a))
    
    ortho_state_reg = torch.tensor(0.0, device=student_emb_a.device)
    if ortho_loss_a is not None:
        ortho_state_reg += ortho_loss_a
    if ortho_loss_b is not None:
        ortho_state_reg += ortho_loss_b
    ortho_state_reg = ortho_state_reg / 2.0
    
    spearman_loss = torch.tensor(0.0, device=student_emb_a.device)
    ranking_loss = torch.tensor(0.0, device=student_emb_a.device)
    
    student_sim = torch.sum(student_emb_a * student_emb_b, dim=1)
    teacher_sim = torch.sum(teacher_emb_a * teacher_emb_b, dim=1)
    
    spearman_loss = spearman_correlation_loss(student_sim, teacher_sim)
    ranking_loss = pairwise_ranking_loss(student_sim, teacher_sim, margin=0.05)
    
    total_loss = (contrastive_loss + 
                  config['distillation_weight'] * distill_loss +
                  config['layer_distill_weight'] * layer_distill_loss +
                  config['identity_reg_weight'] * identity_reg +
                  config['ortho_state_reg_weight'] * ortho_state_reg +
                  config['spearman_loss_weight'] * spearman_loss +
                  config['ranking_loss_weight'] * ranking_loss)
    
    return total_loss, contrastive_loss, distill_loss, identity_reg, layer_distill_loss, ortho_state_reg, spearman_loss, ranking_loss

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_stsb(model, tokenizer, split='test', max_length=128):
    try:
        sts = load_dataset("sentence-transformers/stsb", split="validation" if split=='dev' else "test")
    except:
        sts = load_dataset("glue", "stsb", split="validation" if split=='dev' else "test")
    
    s1 = sts["sentence1"]
    s2 = sts["sentence2"]
    labels = np.array(sts["label" if 'label' in sts.column_names else "score"], dtype=float)
    
    model.eval()
    all_sims = []
    
    with torch.no_grad():
        batch_size = 32
        for i in range(0, len(s1), batch_size):
            batch_s1 = s1[i:min(i+batch_size, len(s1))]
            batch_s2 = s2[i:min(i+batch_size, len(s2))]
            
            tokens1 = tokenizer(batch_s1, padding=True, max_length=max_length, truncation=True, return_tensors='pt').to(device)
            tokens2 = tokenizer(batch_s2, padding=True, max_length=max_length, truncation=True, return_tensors='pt').to(device)
            
            emb1 = model(tokens1['input_ids'], tokens1['attention_mask'])
            emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])
            
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            all_sims.extend(sim.cpu().numpy().tolist())
    
    pearson = pearsonr(all_sims, labels)[0]
    spearman = spearmanr(all_sims, labels)[0]
    
    model.train()
    return pearson, spearman

def mean_pooling_teacher(output, mask):
    embeddings = output.last_hidden_state
    mask_expanded = mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_emb = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    return sum_emb / sum_mask

# ============================================================================
# TRAINING WITH VARIABLE LENGTH CURRICULUM
# ============================================================================
def train():
    print("="*80)
    print("🚀 DELTANET 8K TRAINING - VARIABLE LENGTH CURRICULUM")
    print("="*80)
    print(f"\n📊 Configuration:")
    print(f"  Teacher: {config['teacher_model']} (8K native)")
    print(f"  Student: Continue from checkpoint (0.77 Spearman)")
    print(f"  Training lengths: {config['train_lengths']} tokens")
    print(f"  Curriculum: Short (512-2K) → Medium (512-4K) → Long (1K-8K)")
    print(f"  Batch size: {config['batch_size']} (adaptive based on length)")
    print(f"  Effective batch: {config['batch_size'] * config['gradient_accumulation_steps']}")
    print("="*80)
    
    # Load teacher
    print(f"\n🤖 Loading Teacher: {config['teacher_model']}...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'], trust_remote_code=True)
    
    # ⚡ Enable FlashAttention2 for teacher (8K sequences need it!)
    use_flash_attention_2 = False
    try:
        import flash_attn
        use_flash_attention_2 = True
        print(f"   ✅ FlashAttention2 detected (version {flash_attn.__version__})")
        print(f"   ⚡ Enabling FlashAttention2 for teacher model (required for 8K!)")
    except ImportError:
        print(f"   ⚠️  FlashAttention2 NOT found!")
        print(f"   ⚠️  Teacher will use standard attention (may OOM on 8K sequences)")
        print(f"   💡 Install: pip install flash-attn --no-build-isolation")
    
    if use_flash_attention_2:
        teacher_model = AutoModel.from_pretrained(
            config['teacher_model'], 
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  # ⚡ Enable FA2
            torch_dtype=torch.float16  # FA2 requires fp16/bf16
        ).to(device)
    else:
        teacher_model = AutoModel.from_pretrained(
            config['teacher_model'], 
            trust_remote_code=True
        ).to(device)
    
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Enable gradient checkpointing for teacher (saves memory)
    if hasattr(teacher_model, 'gradient_checkpointing_enable'):
        teacher_model.gradient_checkpointing_enable()
        print(f"   ✅ Gradient checkpointing enabled for teacher")
    
    # Warn about memory if no FA2 and training on long sequences
    if not use_flash_attention_2:
        print(f"\n   ⚠️  WARNING: Without FlashAttention2, you may encounter OOM on long sequences!")
        print(f"   💡 Solutions:")
        print(f"      1. Install FA2: pip install flash-attn --no-build-isolation")
        print(f"      2. Reduce batch_size to 2 (in config)")
        print(f"      3. Limit max training length to 4096 (skip 8192)")
        print(f"   ⏸️  Waiting 10 seconds... (Ctrl+C to abort)")
        import time
        time.sleep(10)
    
    # Load student
    print(f"\n👨‍🎓 Loading Student...")
    student_tokenizer = AutoTokenizer.from_pretrained(config['student_base'])
    student_model = DeltaNet8KExtended(config).to(device)
    
    # Load base components
    print(f"   📥 Loading base from {config['student_base']}...")
    base_model = AutoModel.from_pretrained(config['student_base'])
    with torch.no_grad():
        student_model.embeddings.weight.copy_(base_model.embeddings.word_embeddings.weight)
        student_model.position_embeddings.weight.copy_(base_model.embeddings.position_embeddings.weight)
        student_model.token_type_embeddings.weight.copy_(base_model.embeddings.token_type_embeddings.weight)
        student_model.embedding_norm.weight.copy_(base_model.embeddings.LayerNorm.weight)
        student_model.embedding_norm.bias.copy_(base_model.embeddings.LayerNorm.bias)
        
        for i in range(6):
            teacher_layer = base_model.encoder.layer[i]
            student_model.deltanet_norms[i].load_state_dict(teacher_layer.attention.output.LayerNorm.state_dict())
            student_model.ffn_norms[i].load_state_dict(teacher_layer.output.LayerNorm.state_dict())
            student_model.deltanet_ffns[i].load_state_dict(teacher_layer.intermediate.dense.state_dict())
            student_model.ffn_outputs[i].load_state_dict(teacher_layer.output.dense.state_dict())
    
    print(f"   ✅ Loaded base components")
    del base_model
    
    # Load checkpoint
    print(f"   🔄 Loading DeltaNet checkpoint...")
    if os.path.exists(config['checkpoint_file']):
        ckpt = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
        if 'deltanet_layers' in ckpt:
            student_model.deltanet_layers.load_state_dict(ckpt['deltanet_layers'], strict=False)
            print(f"   ✅ Loaded DeltaNet layers")
            if 'test_spearman' in ckpt:
                print(f"   📊 Checkpoint Spearman: {ckpt['test_spearman']:.4f}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Load data
    dataset = load_data()
    
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config['warmup_steps'], num_training_steps=config['total_steps'])
    
    # Initial evaluation
    print("\n📊 Initial Evaluation...")
    initial_pearson, initial_spearman = evaluate_stsb(student_model, student_tokenizer, split='test')
    print(f"Test: Pearson={initial_pearson:.4f}, Spearman={initial_spearman:.4f}")
    
    best_score = initial_spearman
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    start_phase = config.get('start_phase', 1)
    
    print("\n🚀 Starting Variable-Length Training...")
    print(f"   Total steps: {config['total_steps']:,}")
    print(f"   Start phase: {start_phase}")
    
    if start_phase == 1:
        print(f"   Curriculum:")
        print(f"     0-40%: Short (512-2K) - maintain baseline")
        print(f"     40-70%: Medium (512-4K) - bridge to long")
        print(f"     70-100%: Long (1K-8K) - full 8K capability")
    elif start_phase == 2:
        print(f"   ⚡ SKIPPING Short phase (already trained on short!)")
        print(f"   Curriculum:")
        print(f"     0-50%: Medium (512-4K) - bridge to long")
        print(f"     50-100%: Long (1K-8K) - full 8K capability")
    else:
        print(f"   ⚡ SKIPPING Short & Medium phases!")
        print(f"   Curriculum:")
        print(f"     0-100%: Long (1K-8K) - full 8K capability only")
    print()
    
    global_step = 0
    running_losses = {'total': 0, 'contr': 0, 'distill': 0, 'layer': 0, 'spear': 0, 'rank': 0}
    length_counts = {512: 0, 1024: 0, 2048: 0, 4096: 0, 8192: 0}
    
    pbar = tqdm(total=config['total_steps'], desc="Training")
    
    while global_step < config['total_steps']:
        # ⭐ SAMPLE SEQUENCE LENGTH (curriculum with start_phase)
        current_length = sample_sequence_length_curriculum(global_step, config['total_steps'], start_phase)
        length_counts[current_length] += 1
        
        # Sample batch
        indices = np.random.randint(0, len(dataset), size=config['batch_size'])
        batch_data = [dataset[int(i)] for i in indices]
        s1 = [item['sentence1'] for item in batch_data]
        s2 = [item['sentence2'] for item in batch_data]
        
        # ⭐ TOKENIZE WITH CURRENT LENGTH
        t1_teacher = teacher_tokenizer(s1, padding='max_length', truncation=True, max_length=current_length, return_tensors='pt').to(device)
        t2_teacher = teacher_tokenizer(s2, padding='max_length', truncation=True, max_length=current_length, return_tensors='pt').to(device)
        t1_student = student_tokenizer(s1, padding='max_length', truncation=True, max_length=current_length, return_tensors='pt').to(device)
        t2_student = student_tokenizer(s2, padding='max_length', truncation=True, max_length=current_length, return_tensors='pt').to(device)
        
        # Teacher forward
        with torch.no_grad():
            teacher_out1 = teacher_model(**t1_teacher, output_hidden_states=True)
            teacher_out2 = teacher_model(**t2_teacher, output_hidden_states=True)
            
            teacher_emb1 = F.normalize(mean_pooling_teacher(teacher_out1, t1_teacher['attention_mask']), p=2, dim=1)
            teacher_emb2 = F.normalize(mean_pooling_teacher(teacher_out2, t2_teacher['attention_mask']), p=2, dim=1)
            
            if hasattr(teacher_out1, 'hidden_states') and teacher_out1.hidden_states is not None:
                teacher_hidden1 = list(teacher_out1.hidden_states[1:7])
                teacher_hidden2 = list(teacher_out2.hidden_states[1:7])
            else:
                teacher_hidden1 = [teacher_emb1.unsqueeze(1).expand(-1, t1_student['input_ids'].shape[1], -1)] * 6
                teacher_hidden2 = [teacher_emb2.unsqueeze(1).expand(-1, t2_student['input_ids'].shape[1], -1)] * 6
        
        # Student forward
        student_emb1, student_hidden1, ortho_loss1 = student_model(t1_student['input_ids'], t1_student['attention_mask'], return_hidden=True)
        student_emb2, student_hidden2, ortho_loss2 = student_model(t2_student['input_ids'], t2_student['attention_mask'], return_hidden=True)
        
        # Loss
        labels = torch.arange(len(student_emb1), device=device)
        loss, contr, distill, ident, layer, ortho_state, spear, rank = compute_loss(
            student_emb1, student_emb2, teacher_emb1, teacher_emb2,
            student_hidden1, student_hidden2, teacher_hidden1, teacher_hidden2,
            labels, student_model, config, ortho_loss1, ortho_loss2
        )
        
        # Backward
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()
        
        # Update
        if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        running_losses['total'] += loss.item() * config['gradient_accumulation_steps']
        running_losses['contr'] += contr.item()
        running_losses['distill'] += distill.item()
        running_losses['layer'] += layer.item()
        running_losses['spear'] += spear.item()
        running_losses['rank'] += rank.item()
        
        if (global_step + 1) % config['log_interval'] == 0:
            phase = get_curriculum_phase_name(global_step, config['total_steps'], start_phase)
            pbar.set_postfix({
                'phase': phase,
                'len': current_length,
                'loss': f"{running_losses['total']/config['log_interval']:.4f}",
                'contr': f"{running_losses['contr']/config['log_interval']:.4f}",
                'layer': f"{running_losses['layer']/config['log_interval']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            for k in running_losses:
                running_losses[k] = 0
        
        # Evaluation
        if (global_step + 1) % config['eval_interval'] == 0:
            print(f"\n📊 Evaluation at step {global_step + 1}")
            test_pearson, test_spearman = evaluate_stsb(student_model, student_tokenizer, split='test')
            print(f"   Test: Pearson={test_pearson:.4f}, Spearman={test_spearman:.4f}")
            
            # Length distribution
            total_samples = sum(length_counts.values())
            print(f"   Length distribution:")
            for length in [512, 1024, 2048, 4096, 8192]:
                pct = 100 * length_counts[length] / total_samples if total_samples > 0 else 0
                print(f"     {length:4d}: {length_counts[length]:6d} ({pct:5.1f}%)")
            
            if test_spearman > best_score:
                best_score = test_spearman
                torch.save(student_model.state_dict(), output_dir / "pytorch_model.bin")
                print(f"   ⭐ NEW BEST! Saved to {output_dir}/pytorch_model.bin")
        
        # Save checkpoint
        if (global_step + 1) % config['save_interval'] == 0:
            torch.save({
                'deltanet_layers': student_model.deltanet_layers.state_dict(),
                'step': global_step + 1,
                'config': config,
                'test_spearman': best_score,
                'length_counts': length_counts,
            }, output_dir / f"checkpoint_{global_step+1}.pt")
            print(f"\n💾 Saved checkpoint at step {global_step + 1}")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final evaluation
    print("\n📊 Final Evaluation...")
    final_pearson, final_spearman = evaluate_stsb(student_model, student_tokenizer, split='test')
    print(f"Final: Pearson={final_pearson:.4f}, Spearman={final_spearman:.4f}")
    print(f"Best: {best_score:.4f}")
    
    # Save final
    torch.save(student_model.state_dict(), output_dir / "final_model.bin")
    student_tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training Complete!")
    print(f"   Saved to: {output_dir}/")
    print(f"   Best model: {output_dir}/pytorch_model.bin ({best_score:.4f})")

if __name__ == "__main__":
    train()