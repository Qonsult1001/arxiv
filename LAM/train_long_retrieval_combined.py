"""
🚀 COMBINED LONG CONTEXT + RETRIEVAL TRAINING
==============================================
Trains LAM for:
1. Long context understanding (512 → 8K tokens)
2. Retrieval (query-document matching)
3. Preserves STS-B score (0.81+)

Key: Variable length curriculum + Contrastive retrieval loss
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

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

# ============================================================================
# CONFIGURATION - GOOGLE-SCALE 1M+ TOKEN CONTEXT
# ============================================================================
config = {
    "teacher_model": "jinaai/jina-embeddings-v2-base-en",  # 8K native (for short)
    "checkpoint_file": "/workspace/LAM/best/deltanet_shockwave_result.pt",
    "student_base": "sentence-transformers/all-MiniLM-L6-v2",
    
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    "vocab_size": 30522,
    
    # 🚀 GOOGLE-SCALE CONTEXT: Progressive up to 1M tokens!
    "max_seq_length": 1048576,  # 1M tokens (like Gemini)
    "original_max_pos": 512,
    
    # Progressive context curriculum (tokens):
    # Phase 1: Short (0-30%)   → 512-4K tokens (with teacher)
    # Phase 2: Medium (30-60%) → 4K-32K tokens (self-supervised)
    # Phase 3: Long (60-80%)   → 32K-128K tokens (self-supervised)
    # Phase 4: Ultra (80-100%) → 128K-1M tokens (self-supervised)
    "context_phases": {
        "short": {"range": [512, 4096], "progress": (0.0, 0.3), "use_teacher": True},
        "medium": {"range": [4096, 32768], "progress": (0.3, 0.6), "use_teacher": False},
        "long": {"range": [32768, 131072], "progress": (0.6, 0.8), "use_teacher": False},
        "ultra": {"range": [131072, 1048576], "progress": (0.8, 1.0), "use_teacher": False},
    },
    
    # Training params (longer for 1M context)
    "learning_rate": 5e-6,
    "batch_size": 1,  # Single sample for ultra-long context
    "gradient_accumulation_steps": 16,  # Effective batch = 16
    "total_steps": 50000,  # More steps for 1M context
    "warmup_steps": 2500,
    "gradient_clip": 1.0,
    "weight_decay": 0.01,
    
    # Loss weights
    "contrastive_weight": 1.0,      # Retrieval objective
    "distillation_weight": 0.5,     # Match teacher (short only)
    "preservation_weight": 1.0,     # Preserve STS-B
    "needle_weight": 2.0,           # 🎯 Needle-in-haystack loss (CRITICAL for long context)
    
    "output_dir": "/workspace/LAM/deltanet_1M_context",
    "save_interval": 2500,
    "eval_interval": 1000,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 🚀 GOOGLE-SCALE CURRICULUM: 512 → 1M TOKENS
# ============================================================================
def sample_length_curriculum(step, total_steps, config):
    """
    Progressive curriculum for 1M token context (like Gemini).
    
    Phase 1 (0-30%):   512-4K tokens     - Learn basics with teacher
    Phase 2 (30-60%):  4K-32K tokens     - Extend context, self-supervised
    Phase 3 (60-80%):  32K-128K tokens   - Long context, needle training
    Phase 4 (80-100%): 128K-1M tokens    - Ultra-long, full capability
    """
    progress = step / total_steps
    phases = config['context_phases']
    
    # Determine current phase
    if progress < phases['short']['progress'][1]:
        phase = 'short'
    elif progress < phases['medium']['progress'][1]:
        phase = 'medium'
    elif progress < phases['long']['progress'][1]:
        phase = 'long'
    else:
        phase = 'ultra'
    
    # Sample length within phase range
    min_len, max_len = phases[phase]['range']
    
    # Log-uniform sampling (more variety in long contexts)
    log_min = np.log(min_len)
    log_max = np.log(max_len)
    length = int(np.exp(np.random.uniform(log_min, log_max)))
    
    # Round to nearest 512 for efficiency
    length = ((length + 255) // 512) * 512
    
    use_teacher = phases[phase]['use_teacher']
    
    return length, phase, use_teacher

def get_phase_name(step, total_steps, config):
    """Get current training phase for logging"""
    progress = step / total_steps
    phases = config['context_phases']
    
    if progress < phases['short']['progress'][1]:
        return "Short (512-4K)"
    elif progress < phases['medium']['progress'][1]:
        return "Medium (4K-32K)"
    elif progress < phases['long']['progress'][1]:
        return "Long (32K-128K)"
    else:
        return "Ultra (128K-1M) 🚀"

def tokens_to_chars(tokens):
    """Approximate tokens to characters (4 chars/token average)"""
    return tokens * 4

def chars_to_tokens(chars):
    """Approximate characters to tokens"""
    return chars // 4

# ============================================================================
# MODEL WITH POSITION INTERPOLATION
# ============================================================================
class LongRetrievalModel(nn.Module):
    """384d model with position interpolation for long sequences"""
    
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
        
        # Teacher projection (768 → 384)
        self.teacher_projection = nn.Linear(768, config['d_model'])
    
    def get_position_embeddings(self, seq_len, batch_size):
        """Position interpolation for long sequences"""
        if seq_len <= self.original_max_pos:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.embeddings.weight.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            return self.position_embeddings(position_ids)
        else:
            # Interpolate
            scale = (self.original_max_pos - 1) / (seq_len - 1)
            pos_embs = []
            for pos in range(seq_len):
                orig_pos = pos * scale
                lower = int(orig_pos)
                upper = min(lower + 1, self.original_max_pos - 1)
                weight = orig_pos - lower
                interp = (1 - weight) * self.position_embeddings.weight[lower] + weight * self.position_embeddings.weight[upper]
                pos_embs.append(interp)
            return torch.stack(pos_embs, dim=0).unsqueeze(0).expand(batch_size, -1, -1)
    
    def mean_pooling(self, hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        return torch.sum(hidden * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    
    def forward_chunk(self, input_ids, attention_mask):
        """Process a single chunk - used by streaming"""
        batch_size, seq_len = input_ids.shape
        
        word_emb = self.embeddings(input_ids)
        pos_emb = self.get_position_embeddings(seq_len, batch_size)
        token_type_emb = self.token_type_embeddings(torch.zeros_like(input_ids))
        
        hidden = word_emb + pos_emb + token_type_emb
        hidden = self.embedding_norm(hidden)
        hidden = self.embedding_dropout(hidden)
        
        for i in range(len(self.deltanet_layers)):
            residual = hidden
            hidden_attn, _, _, _ = self.deltanet_layers[i](hidden, attention_mask)
            hidden = self.deltanet_norms[i](residual + hidden_attn)
            
            residual = hidden
            hidden_ffn = F.gelu(self.deltanet_ffns[i](hidden))
            hidden_ffn = self.ffn_outputs[i](hidden_ffn)
            hidden = self.ffn_norms[i](residual + hidden_ffn)
        
        return hidden  # Return per-token embeddings for streaming
    
    def forward(self, input_ids, attention_mask, use_streaming=False):
        """
        Forward pass with optional streaming for long sequences.
        Streaming: Process in 512-token chunks, mean pool across all.
        """
        batch_size, seq_len = input_ids.shape
        
        # Use streaming for sequences > 512 tokens (matches inference!)
        CHUNK_SIZE = 512
        if use_streaming and seq_len > CHUNK_SIZE:
            # Stream in chunks like inference
            running_sum = None
            total_tokens = 0
            
            for start in range(0, seq_len, CHUNK_SIZE):
                end = min(start + CHUNK_SIZE, seq_len)
                chunk_ids = input_ids[:, start:end]
                chunk_mask = attention_mask[:, start:end]
                
                # Get per-token embeddings for this chunk
                chunk_hidden = self.forward_chunk(chunk_ids, chunk_mask)
                
                # Weighted sum for mean pooling
                mask_expanded = chunk_mask.unsqueeze(-1).float()
                chunk_sum = (chunk_hidden * mask_expanded).sum(dim=1)
                chunk_tokens = chunk_mask.sum(dim=1, keepdim=True).float()
                
                if running_sum is None:
                    running_sum = chunk_sum
                    total_tokens = chunk_tokens.sum().item()
                else:
                    running_sum = running_sum + chunk_sum
                    total_tokens = total_tokens + chunk_tokens.sum().item()
            
            # Final mean pooling
            pooled = running_sum / max(total_tokens, 1)
            pooled = F.normalize(pooled, p=2, dim=1)
            return pooled
        
        else:
            # Short sequence - direct processing
            hidden = self.forward_chunk(input_ids, attention_mask)
            pooled = self.mean_pooling(hidden, attention_mask)
            pooled = F.normalize(pooled, p=2, dim=1)
            return pooled

# ============================================================================
# SYNTHETIC LONG-CONTEXT DATA GENERATION
# ============================================================================
def create_needle_in_haystack(query, answer_passage, target_position, total_length, filler_texts):
    """
    Create synthetic long document with answer at specific position.
    This teaches the model to find relevant info ANYWHERE in a long doc.
    
    Args:
        query: The search query
        answer_passage: The relevant passage (needle)
        target_position: Where to place needle (0.0 = start, 1.0 = end)
        total_length: Target total length in chars
        filler_texts: List of irrelevant texts to use as haystack
    
    Returns:
        Long document with needle at target_position
    """
    # Build haystack
    haystack_parts = []
    current_length = 0
    needle_inserted = False
    needle_position = int(target_position * total_length)
    
    filler_idx = 0
    while current_length < total_length:
        # Check if we should insert needle
        if not needle_inserted and current_length >= needle_position:
            haystack_parts.append(f"\n\n{answer_passage}\n\n")
            current_length += len(answer_passage) + 4
            needle_inserted = True
        else:
            # Add filler text
            filler = filler_texts[filler_idx % len(filler_texts)]
            haystack_parts.append(filler + " ")
            current_length += len(filler) + 1
            filler_idx += 1
    
    # If needle wasn't inserted (target was at very end), add it now
    if not needle_inserted:
        haystack_parts.append(f"\n\n{answer_passage}\n\n")
    
    return ''.join(haystack_parts)

def load_filler_texts():
    """Load generic filler texts for haystack generation"""
    fillers = []
    
    # Wikipedia-style fillers
    wiki_fillers = [
        "The history of civilization spans thousands of years across multiple continents.",
        "Scientific discoveries have transformed our understanding of the natural world.",
        "Economic systems vary widely between different nations and cultures.",
        "Technology continues to advance at an unprecedented rate in modern society.",
        "Environmental concerns have become increasingly important in recent decades.",
        "Political structures differ significantly across various forms of government.",
        "Cultural traditions are passed down through generations in communities worldwide.",
        "Educational systems aim to prepare young people for their future roles.",
        "Healthcare advances have significantly increased human life expectancy.",
        "Transportation networks connect cities and enable global commerce.",
    ]
    fillers.extend(wiki_fillers * 100)  # Repeat for variety
    
    # Try to load real Wikipedia text
    try:
        wiki = load_dataset("wikipedia", "20220301.en", split="train[:1000]",
                           cache_dir="/workspace/.cache/huggingface/datasets")
        for article in wiki:
            text = article.get('text', '')
            # Take chunks of 200-500 chars
            for i in range(0, len(text) - 200, 300):
                chunk = text[i:i+300].replace('\n', ' ').strip()
                if len(chunk) > 100:
                    fillers.append(chunk)
                if len(fillers) > 10000:
                    break
            if len(fillers) > 10000:
                break
        print(f"   ✅ Loaded {len(fillers)} filler texts from Wikipedia")
    except Exception as e:
        print(f"   ⚠️  Wikipedia load failed, using generic fillers: {e}")
    
    return fillers

# ============================================================================
# DATA LOADING
# ============================================================================
def load_retrieval_data():
    """Load retrieval triplets (query, positive, negative)"""
    os.environ['HF_HOME'] = '/workspace/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/workspace/.cache/huggingface/datasets'
    
    print("\n📦 Loading retrieval data...")
    triplets = []
    
    # 1. MS MARCO (short queries, medium passages)
    try:
        msmarco = load_dataset(
            "sentence-transformers/embedding-training-data",
            data_files="msmarco-triplets.jsonl.gz",
            split="train[:20000]",
            cache_dir="/workspace/.cache/huggingface/datasets"
        )
        for row in msmarco:
            if 'query' in row and 'pos' in row:
                pos = row['pos'][0] if isinstance(row['pos'], list) else row['pos']
                neg = row['neg'][0] if isinstance(row['neg'], list) and row['neg'] else pos
                triplets.append({'query': row['query'], 'positive': pos, 'negative': neg})
        print(f"   ✅ MS MARCO: {len(triplets)} triplets")
    except Exception as e:
        print(f"   ⚠️  MS MARCO error: {e}")
    
    # 2. NQ - Skip to speed up (MS MARCO is enough for retrieval)
    print(f"   ⏭️  Skipping NQ (MS MARCO sufficient for retrieval training)")
    
    # 3. STS-B (for preservation)
    try:
        stsb = load_dataset("sentence-transformers/stsb", split="train",
                           cache_dir="/workspace/.cache/huggingface/datasets")
        stsb_count = 0
        for row in stsb:
            if row['score'] >= 3.0:
                triplets.append({
                    'query': row['sentence1'],
                    'positive': row['sentence2'],
                    'negative': '',
                    'is_stsb': True
                })
                stsb_count += 1
        print(f"   ✅ STS-B (high-sim): {stsb_count} pairs")
    except Exception as e:
        print(f"   ⚠️  STS-B error: {e}")
    
    print(f"\n📊 Total: {len(triplets)} training samples")
    return triplets

def create_long_context_batch(triplets, filler_texts, target_length_chars, num_samples=4):
    """
    Create synthetic long-context training batch.
    Needle (answer) is placed at RANDOM positions to teach full-context retrieval.
    
    Args:
        triplets: List of (query, positive, negative) triplets
        filler_texts: List of filler texts for haystack
        target_length_chars: Target document length (e.g., 500K chars = ~125K tokens)
        num_samples: Number of samples to generate
    
    Returns:
        List of (query, long_doc) pairs with needle at various positions
    """
    batch = []
    
    for _ in range(num_samples):
        # Random triplet
        triplet = triplets[np.random.randint(0, len(triplets))]
        query = triplet['query']
        answer = triplet['positive']
        
        # Random position for needle (0.0 to 1.0)
        # Bias toward later positions since that's what current model struggles with
        position = np.random.beta(2, 1)  # Biased toward end
        
        # Create long doc with needle at position
        long_doc = create_needle_in_haystack(
            query=query,
            answer_passage=answer,
            target_position=position,
            total_length=target_length_chars,
            filler_texts=filler_texts
        )
        
        batch.append({
            'query': query,
            'positive': long_doc,
            'negative': '',
            'needle_position': position,
            'is_long_context': True
        })
    
    return batch

# ============================================================================
# LOSSES
# ============================================================================
def contrastive_loss(query_emb, pos_emb, neg_embs, temperature=0.05):
    """InfoNCE loss for retrieval"""
    query_emb = F.normalize(query_emb, p=2, dim=1)
    pos_emb = F.normalize(pos_emb, p=2, dim=1)
    
    pos_sim = torch.sum(query_emb * pos_emb, dim=1, keepdim=True) / temperature
    
    if neg_embs is not None and neg_embs.shape[0] > 0:
        neg_embs = F.normalize(neg_embs, p=2, dim=1)
        neg_sim = torch.mm(query_emb, neg_embs.t()) / temperature
        logits = torch.cat([pos_sim, neg_sim], dim=1)
    else:
        logits = pos_sim
    
    labels = torch.zeros(query_emb.size(0), dtype=torch.long, device=query_emb.device)
    return F.cross_entropy(logits, labels)

def distillation_loss(student_emb, teacher_emb):
    """MSE loss to match teacher embeddings"""
    return F.mse_loss(student_emb, teacher_emb)

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_stsb(model, tokenizer, device):
    """Evaluate on STS-B"""
    try:
        sts = load_dataset("sentence-transformers/stsb", split="test",
                          cache_dir="/workspace/.cache/huggingface/datasets")
    except:
        return None, None
    
    model.eval()
    sims = []
    labels = []
    
    with torch.no_grad():
        for i in range(0, min(len(sts), 500), 16):
            batch = sts[i:i+16]
            
            t1 = tokenizer(batch['sentence1'], padding=True, truncation=True, 
                          max_length=128, return_tensors='pt').to(device)
            t2 = tokenizer(batch['sentence2'], padding=True, truncation=True,
                          max_length=128, return_tensors='pt').to(device)
            
            e1 = model(t1['input_ids'], t1['attention_mask'])
            e2 = model(t2['input_ids'], t2['attention_mask'])
            
            sim = F.cosine_similarity(e1, e2, dim=1)
            sims.extend(sim.cpu().tolist())
            labels.extend(batch['score'])
    
    model.train()
    
    if len(sims) > 0:
        return pearsonr(sims, labels)[0], spearmanr(sims, labels)[0]
    return None, None

# ============================================================================
# TRAINING - GOOGLE-SCALE 1M CONTEXT
# ============================================================================
def train():
    print("="*80)
    print("🚀 GOOGLE-SCALE 1M TOKEN CONTEXT TRAINING")
    print("="*80)
    print(f"Model: 384d DeltaNet with streaming + position interpolation")
    print(f"Max context: {config['max_seq_length']:,} tokens (1M)")
    print(f"Phases: Short → Medium → Long → Ultra (1M)")
    print(f"Objective: Retrieval + Needle-in-Haystack + STS-B Preservation")
    print("="*80)
    
    # Load teacher (8K native)
    print("\n🤖 Loading teacher (Jina 8K)...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(config['teacher_model'], trust_remote_code=True)
    teacher_model = AutoModel.from_pretrained(config['teacher_model'], trust_remote_code=True).to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    print("   ✅ Teacher loaded")
    
    # Load student
    print("\n👨‍🎓 Loading student...")
    student_tokenizer = AutoTokenizer.from_pretrained(config['student_base'])
    student = LongRetrievalModel(config).to(device)
    
    # Load base weights
    base = AutoModel.from_pretrained(config['student_base'])
    with torch.no_grad():
        student.embeddings.weight.copy_(base.embeddings.word_embeddings.weight)
        student.position_embeddings.weight.copy_(base.embeddings.position_embeddings.weight)
        student.token_type_embeddings.weight.copy_(base.embeddings.token_type_embeddings.weight)
        student.embedding_norm.weight.copy_(base.embeddings.LayerNorm.weight)
        student.embedding_norm.bias.copy_(base.embeddings.LayerNorm.bias)
        
        for i in range(6):
            layer = base.encoder.layer[i]
            student.deltanet_norms[i].load_state_dict(layer.attention.output.LayerNorm.state_dict())
            student.ffn_norms[i].load_state_dict(layer.output.LayerNorm.state_dict())
            student.deltanet_ffns[i].load_state_dict(layer.intermediate.dense.state_dict())
            student.ffn_outputs[i].load_state_dict(layer.output.dense.state_dict())
    del base
    
    # Load DeltaNet checkpoint
    if os.path.exists(config['checkpoint_file']):
        ckpt = torch.load(config['checkpoint_file'], map_location=device, weights_only=False)
        # Load compatible weights
        student_state = student.state_dict()
        for k, v in ckpt.items():
            if k in student_state and student_state[k].shape == v.shape:
                student_state[k] = v
        student.load_state_dict(student_state, strict=False)
        print("   ✅ Loaded DeltaNet checkpoint")
    
    # Optimizer
    optimizer = torch.optim.AdamW(student.parameters(), lr=config['learning_rate'], 
                                   weight_decay=config['weight_decay'])
    scheduler = get_linear_schedule_with_warmup(optimizer, config['warmup_steps'], config['total_steps'])
    
    # Data
    triplets = load_retrieval_data()
    
    # Initial eval
    print("\n📊 Initial STS-B evaluation...")
    _, initial_spearman = evaluate_stsb(student, student_tokenizer, device)
    print(f"   Initial Spearman: {initial_spearman:.4f}" if initial_spearman else "   Could not evaluate")
    
    best_spearman = initial_spearman or 0.0
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load filler texts for needle-in-haystack training
    print("\n📚 Loading filler texts for long-context training...")
    filler_texts = load_filler_texts()
    
    # Training loop
    print("\n🚀 Starting 1M context training...")
    print(f"   Phases: {list(config['context_phases'].keys())}")
    global_step = 0
    pbar = tqdm(total=config['total_steps'], desc="Training")
    
    while global_step < config['total_steps']:
        # Sample length using new curriculum
        current_length, phase, use_teacher = sample_length_curriculum(global_step, config['total_steps'], config)
        
        # For long phases, create synthetic needle-in-haystack data
        if phase in ['long', 'ultra']:
            # Create synthetic long document with needle at random position
            target_chars = tokens_to_chars(current_length)
            batch = create_long_context_batch(triplets, filler_texts, target_chars, num_samples=config['batch_size'])
        else:
            # Use regular triplets for short/medium phases
            indices = np.random.randint(0, len(triplets), size=config['batch_size'])
            batch = [triplets[int(i)] for i in indices]
        
        queries = [t['query'] for t in batch]
        positives = [t['positive'] for t in batch]
        negatives = [t.get('negative', '') for t in batch if t.get('negative', '')]
        is_stsb = any(t.get('is_stsb', False) for t in batch)
        is_long_context = any(t.get('is_long_context', False) for t in batch)
        
        # Tokenize with current length
        q_tokens = student_tokenizer(queries, padding='max_length', truncation=True,
                                     max_length=min(current_length, 512), return_tensors='pt').to(device)
        p_tokens = student_tokenizer(positives, padding='max_length', truncation=True,
                                     max_length=current_length, return_tensors='pt').to(device)
        
        # Student forward (use streaming for long sequences - matches inference!)
        use_stream = current_length > 512
        q_emb = student(q_tokens['input_ids'], q_tokens['attention_mask'], use_streaming=False)
        p_emb = student(p_tokens['input_ids'], p_tokens['attention_mask'], use_streaming=use_stream)
        
        # Teacher forward (only for short phase where teacher can handle the length)
        loss_distill = torch.tensor(0.0, device=device)
        if use_teacher and current_length <= 8192:  # Jina can only handle 8K
            with torch.no_grad():
                q_teacher_tok = teacher_tokenizer(queries, padding='max_length', truncation=True,
                                              max_length=min(current_length, 512), return_tensors='pt').to(device)
                p_teacher_tok = teacher_tokenizer(positives, padding='max_length', truncation=True,
                                              max_length=min(current_length, 8192), return_tensors='pt').to(device)
                
                q_t_out = teacher_model(**q_teacher_tok)
                p_t_out = teacher_model(**p_teacher_tok)
                
                def mean_pool(output, mask):
                    emb = output.last_hidden_state
                    mask = mask.unsqueeze(-1).expand(emb.size()).float()
                    return torch.sum(emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                
                q_t_emb = F.normalize(mean_pool(q_t_out, q_teacher_tok['attention_mask']), p=2, dim=1)
                p_t_emb = F.normalize(mean_pool(p_t_out, p_teacher_tok['attention_mask']), p=2, dim=1)
                
                q_t_proj = student.teacher_projection(q_t_emb)
                p_t_proj = student.teacher_projection(p_t_emb)
            
            loss_distill = (distillation_loss(q_emb, q_t_proj) + distillation_loss(p_emb, p_t_proj)) / 2
        
        # Negatives
        n_emb = None
        if negatives:
            n_tokens = student_tokenizer(negatives, padding='max_length', truncation=True,
                                        max_length=min(current_length, 4096), return_tensors='pt').to(device)
            n_emb = student(n_tokens['input_ids'], n_tokens['attention_mask'], use_streaming=use_stream)
        
        # Losses
        loss_contrastive = contrastive_loss(q_emb, p_emb, n_emb)
        
        # Extra preservation for STS-B samples
        loss_preserve = torch.tensor(0.0, device=device)
        if is_stsb:
            loss_preserve = F.cosine_embedding_loss(
                q_emb, p_emb, 
                torch.ones(q_emb.size(0), device=device)
            )
        
        # Needle-in-haystack loss: query should have HIGH similarity to doc containing answer
        loss_needle = torch.tensor(0.0, device=device)
        if is_long_context:
            # The doc CONTAINS the answer, so similarity should be high
            target_sim = torch.ones(q_emb.size(0), device=device) * 0.8  # Target 0.8 similarity
            actual_sim = F.cosine_similarity(q_emb, p_emb, dim=1)
            loss_needle = F.mse_loss(actual_sim, target_sim)
        
        # Total loss
        loss = (config['contrastive_weight'] * loss_contrastive +
                config['distillation_weight'] * loss_distill +
                config['preservation_weight'] * loss_preserve +
                config.get('needle_weight', 2.0) * loss_needle)
        
        # Backward
        loss = loss / config['gradient_accumulation_steps']
        loss.backward()
        
        if (global_step + 1) % config['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), config['gradient_clip'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if (global_step + 1) % 50 == 0:
            phase_name = get_phase_name(global_step, config['total_steps'], config)
            pbar.set_postfix({
                'phase': phase_name,
                'len': f'{current_length//1000}K' if current_length >= 1000 else current_length,
                'loss': f'{loss.item() * config["gradient_accumulation_steps"]:.4f}',
                'contr': f'{loss_contrastive.item():.4f}',
            })
        
        # Evaluation
        if (global_step + 1) % config['eval_interval'] == 0:
            print(f"\n📊 Step {global_step + 1}")
            pearson, spearman = evaluate_stsb(student, student_tokenizer, device)
            if spearman:
                print(f"   STS-B: Pearson={pearson:.4f}, Spearman={spearman:.4f}")
                if spearman > best_spearman:
                    best_spearman = spearman
                    torch.save(student.state_dict(), output_dir / "best_model.pt")
                    print(f"   ⭐ New best! Saved.")
        
        # Save checkpoint
        if (global_step + 1) % config['save_interval'] == 0:
            torch.save({
                'model': student.state_dict(),
                'step': global_step + 1,
                'best_spearman': best_spearman,
            }, output_dir / f"checkpoint_{global_step+1}.pt")
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final eval
    print("\n📊 Final evaluation...")
    final_pearson, final_spearman = evaluate_stsb(student, student_tokenizer, device)
    print(f"Final: Pearson={final_pearson:.4f}, Spearman={final_spearman:.4f}")
    print(f"Best: {best_spearman:.4f}")
    
    # Save
    torch.save(student.state_dict(), output_dir / "final_model.pt")
    student_tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Training complete! Saved to {output_dir}/")

if __name__ == "__main__":
    train()

