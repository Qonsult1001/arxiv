"""
🧠 KNOWLEDGE DISTILLATION: Learn from E5-Large (0.876 Pearson)
When all else fails, learn from a stronger teacher

E5-Large achieves 0.876 on STS-B. We'll use it to teach your 0.8358 model.

Strategy:
1. Load E5-Large as frozen teacher
2. Minimize KL divergence between student and teacher embeddings
3. Mix with STS-B regression loss (80/20 split)
4. Ultra-conservative LR to preserve existing knowledge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

config = {
    "student_checkpoint": "/workspace/LAM/final_push_850/step0070_p0.8363.pt",
    "teacher_model": "intfloat/e5-large-v2",  # 0.876 Pearson on STS-B
    "student_base": "/workspace/LAM/all-MiniLM-L6-v2",
    "output_dir": "/workspace/LAM/distill_from_e5_large",
    
    "d_model": 384,
    "num_heads": 12,
    "num_layers": 6,
    
    # 🧠 DISTILLATION CONFIGURATION
    "learning_rate": 1e-6,      # ULTRA low - preserve existing knowledge
    "weight_decay": 0.001,      # Very light
    "gradient_clip": 0.03,
    "batch_size": 24,
    "max_length": 128,
    
    # Loss weights
    "distill_weight": 0.8,      # 80% distillation loss
    "stsb_weight": 0.2,         # 20% STS-B regression loss
    "temperature": 2.0,         # Temperature for soft targets
    
    # Training
    "max_steps": 400,
    "eval_interval": 10,
    "patience": 25,
    
    # Target
    "target_score": 0.8500,
    "min_score": 0.8350,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("="*80)
print("🧠 KNOWLEDGE DISTILLATION FROM E5-LARGE")
print("="*80)
print(f"Teacher: E5-Large (0.876 Pearson)")
print(f"Student: Your model (0.8358 Pearson)")
print(f"Target: 0.850")
print(f"Gap: 0.0142 (1.7%)")
print()
print("🎯 STRATEGY:")
print(f"  80% Distillation loss (learn from teacher)")
print(f"  20% STS-B regression loss (stay grounded)")
print(f"  Temperature: {config['temperature']}")
print(f"  LR: {config['learning_rate']:.2e} (ultra-conservative)")
print("="*80)

# ============================================================================
# MODELS
# ============================================================================
class StudentModel(nn.Module):
    def __init__(self, base_model_path, checkpoint_path, config):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModel.from_pretrained(base_model_path)
        self.d_model = config['d_model']
        self.num_layers = config['num_layers']
        
        self.embeddings = base_model.embeddings
        for param in self.embeddings.parameters():
            param.requires_grad = False
        
        self.deltanet_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.output_denses = nn.ModuleList()
        
        for i in range(self.num_layers):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model, num_heads=config['num_heads'],
                    use_hierarchical_decay=True, use_enhanced_flux=True,
                    fast_decay_init=0.3, slow_decay_init=0.85, layer_idx=i,
                )
            )
            self.norms.append(base_model.encoder.layer[i].attention.output.LayerNorm)
            self.ffns.append(base_model.encoder.layer[i].intermediate)
            self.ffn_norms.append(base_model.encoder.layer[i].output.LayerNorm)
            self.output_denses.append(base_model.encoder.layer[i].output.dense)
        
        for param in self.output_denses.parameters():
            param.requires_grad = False
        
        # Projection layer to map teacher embeddings (1024) to student dimension (384)
        self.teacher_projection = nn.Linear(1024, self.d_model)
        nn.init.xavier_uniform_(self.teacher_projection.weight, gain=0.5)
        nn.init.zeros_(self.teacher_projection.bias)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle new format from train_6layer_SOTA.py (model_state_dict)
        model_state_dict = checkpoint.get('model_state_dict', {})
        if model_state_dict:
            # Extract components from model_state_dict
            deltanet_layers_dict = {}
            deltanet_norms_dict = {}
            deltanet_ffns_dict = {}
            ffn_norms_dict = {}
            
            for key, value in model_state_dict.items():
                if key.startswith('deltanet_layers.'):
                    new_key = key.replace('deltanet_layers.', '')
                    deltanet_layers_dict[new_key] = value
                elif key.startswith('deltanet_norms.'):
                    new_key = key.replace('deltanet_norms.', '')
                    deltanet_norms_dict[new_key] = value
                elif key.startswith('deltanet_ffns.'):
                    new_key = key.replace('deltanet_ffns.', '')
                    deltanet_ffns_dict[new_key] = value
                elif key.startswith('ffn_norms.'):
                    new_key = key.replace('ffn_norms.', '')
                    ffn_norms_dict[new_key] = value
            
            # Load deltanet_layers
            if deltanet_layers_dict:
                for i in range(self.num_layers):
                    layer_state = {k[2:]: v for k, v in deltanet_layers_dict.items() if k.startswith(f'{i}.')}
                    if layer_state:
                        try:
                            self.deltanet_layers[i].load_state_dict(layer_state, strict=False)
                        except: pass
            
            # Load other components
            if deltanet_norms_dict:
                try:
                    self.norms.load_state_dict(deltanet_norms_dict, strict=False)
                except:
                    for i in range(self.num_layers):
                        layer_state = {k[2:]: v for k, v in deltanet_norms_dict.items() if k.startswith(f'{i}.')}
                        if layer_state:
                            self.norms[i].load_state_dict(layer_state, strict=False)
            
            if deltanet_ffns_dict:
                try:
                    self.ffns.load_state_dict(deltanet_ffns_dict, strict=False)
                except:
                    for i in range(self.num_layers):
                        layer_state = {k[2:]: v for k, v in deltanet_ffns_dict.items() if k.startswith(f'{i}.')}
                        if layer_state:
                            self.ffns[i].load_state_dict(layer_state, strict=False)
            
            if ffn_norms_dict:
                try:
                    self.ffn_norms.load_state_dict(ffn_norms_dict, strict=False)
                except:
                    for i in range(self.num_layers):
                        layer_state = {k[2:]: v for k, v in ffn_norms_dict.items() if k.startswith(f'{i}.')}
                        if layer_state:
                            self.ffn_norms[i].load_state_dict(layer_state, strict=False)
        else:
            # Old format
            if 'deltanet_layers' in checkpoint:
                self.deltanet_layers.load_state_dict(checkpoint['deltanet_layers'], strict=False)
            
            for key, attr in [('deltanet_norms', 'norms'), ('deltanet_ffns', 'ffns'), 
                              ('ffn_norms', 'ffn_norms')]:
                if key in checkpoint:
                    try:
                        getattr(self, attr).load_state_dict(checkpoint[key], strict=False)
                    except: pass
        
        print(f"✅ Student loaded")
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
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)  # Returns 4 values: output, None, past_key_values, ortho_loss
            x = self.norms[i](residual + x_attn)
            residual = x
            x_ffn = self.ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_denses[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        embeddings = self.mean_pooling(x, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

class TeacherModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        print(f"📚 Loading teacher: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        print(f"✅ Teacher loaded (frozen)")
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(embeddings, p=2, dim=1)

# ============================================================================
# DISTILLATION LOSS
# ============================================================================
def distillation_loss(student_emb1, student_emb2, teacher_emb1, teacher_emb2, teacher_projection, temperature=2.0):
    """
    KL divergence between student and teacher similarity distributions
    Temperature softens the distributions for better knowledge transfer
    """
    # Project teacher embeddings to student dimension
    teacher_emb1_proj = teacher_projection(teacher_emb1)
    teacher_emb2_proj = teacher_projection(teacher_emb2)
    
    # Student similarities
    student_sim = F.cosine_similarity(student_emb1, student_emb2, dim=1) / temperature
    student_probs = torch.sigmoid(student_sim)
    
    # Teacher similarities (detached)
    teacher_sim = F.cosine_similarity(teacher_emb1, teacher_emb2, dim=1) / temperature
    teacher_probs = torch.sigmoid(teacher_sim).detach()
    
    # KL divergence
    kl_loss = F.kl_div(
        torch.log(student_probs + 1e-8),
        teacher_probs,
        reduction='batchmean'
    )
    
    # Also add L2 loss between embeddings for stability (using projected teacher embeddings)
    l2_loss = F.mse_loss(student_emb1, teacher_emb1_proj) + F.mse_loss(student_emb2, teacher_emb2_proj)
    
    return kl_loss + 0.1 * l2_loss

def evaluate_stsb(model):
    model.eval()
    try:
        ds = load_dataset("sentence-transformers/stsb", split="validation")
    except:
        ds = load_dataset("glue", "stsb", split="validation")
    
    s1 = ds["sentence1"]
    s2 = ds["sentence2"]
    labels = np.array(ds["label"] if "label" in ds.column_names else ds["score"], dtype=float)
    
    all_sim = []
    with torch.no_grad():
        for i in range(0, len(s1), 32):
            batch_s1 = s1[i:min(i+32, len(s1))]
            batch_s2 = s2[i:min(i+32, len(s2))]
            
            tokens = model.tokenizer(batch_s1, padding=True, max_length=128,
                                    truncation=True, return_tensors='pt').to(device)
            tokens2 = model.tokenizer(batch_s2, padding=True, max_length=128,
                                     truncation=True, return_tensors='pt').to(device)
            
            emb1 = model(tokens['input_ids'], tokens['attention_mask'])
            emb2 = model(tokens2['input_ids'], tokens2['attention_mask'])
            
            sim = F.cosine_similarity(emb1, emb2, dim=1)
            all_sim.extend(sim.cpu().numpy().tolist())
    
    pearson = pearsonr(all_sim, labels)[0]
    spearman = spearmanr(all_sim, labels)[0]
    model.train()
    return pearson, spearman

# ============================================================================
# TRAINING
# ============================================================================
def train_with_distillation():
    # Load data
    stsb_train = load_dataset("glue", "stsb", split="train")
    stsb_data = [(ex['sentence1'], ex['sentence2'], ex['label']) 
                 for ex in stsb_train if len(ex['sentence1']) > 10 and len(ex['sentence2']) > 10]
    print(f"\n📚 STS-B: {len(stsb_data):,} pairs")
    
    # Load models
    student = StudentModel(config['student_base'], config['student_checkpoint'], config).to(device)
    teacher = TeacherModel(config['teacher_model']).to(device)
    
    # Baseline
    print(f"\n📊 BASELINE...")
    baseline_pearson, _ = evaluate_stsb(student)
    print(f"   Student: {baseline_pearson:.4f}")
    print(f"   Gap to 0.850: {config['target_score'] - baseline_pearson:.4f}")
    
    # Optimizer (include teacher_projection in trainable params)
    trainable = (list(student.deltanet_layers.parameters()) + list(student.norms.parameters()) +
                list(student.ffns.parameters()) + list(student.ffn_norms.parameters()) +
                list(student.teacher_projection.parameters()))
    
    optimizer = AdamW(trainable, lr=config['learning_rate'], 
                     weight_decay=config['weight_decay'])
    
    print(f"\n▶️  Distillation training with LR {config['learning_rate']:.2e}")
    
    # Training
    student.train()
    best_pearson = baseline_pearson
    best_step = 0
    patience = 0
    
    global_step = 0
    running_distill_loss = 0.0
    running_stsb_loss = 0.0
    log_count = 0
    
    pbar = tqdm(total=config['max_steps'], desc="🧠 Distill")
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    while global_step < config['max_steps']:
        # Sample batch
        batch_data = [stsb_data[np.random.randint(0, len(stsb_data))]
                     for _ in range(config['batch_size'])]
        
        s1 = [item[0] for item in batch_data]
        s2 = [item[1] for item in batch_data]
        scores = torch.tensor([item[2] for item in batch_data], 
                             dtype=torch.float32, device=device)
        
        # Tokenize for student
        tokens1_student = student.tokenizer(s1, padding=True, max_length=128,
                                           truncation=True, return_tensors='pt').to(device)
        tokens2_student = student.tokenizer(s2, padding=True, max_length=128,
                                           truncation=True, return_tensors='pt').to(device)
        
        # Tokenize for teacher
        tokens1_teacher = teacher.tokenizer(s1, padding=True, max_length=128,
                                           truncation=True, return_tensors='pt').to(device)
        tokens2_teacher = teacher.tokenizer(s2, padding=True, max_length=128,
                                           truncation=True, return_tensors='pt').to(device)
        
        # Forward - Student
        student_emb1 = student(tokens1_student['input_ids'], tokens1_student['attention_mask'])
        student_emb2 = student(tokens2_student['input_ids'], tokens2_student['attention_mask'])
        
        # Forward - Teacher (no grad)
        with torch.no_grad():
            teacher_emb1 = teacher(tokens1_teacher['input_ids'], tokens1_teacher['attention_mask'])
            teacher_emb2 = teacher(tokens2_teacher['input_ids'], tokens2_teacher['attention_mask'])
        
        # Distillation loss (pass teacher_projection to project teacher embeddings)
        loss_distill = distillation_loss(
            student_emb1, student_emb2,
            teacher_emb1, teacher_emb2,
            student.teacher_projection,
            temperature=config['temperature']
        )
        
        # STS-B regression loss
        sim = F.cosine_similarity(student_emb1, student_emb2, dim=1)
        pred = (sim + 1) * 2.5
        loss_stsb = F.mse_loss(pred, scores)
        
        # Combined loss
        loss = config['distill_weight'] * loss_distill + config['stsb_weight'] * loss_stsb
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, config['gradient_clip'])
        optimizer.step()
        
        running_distill_loss += loss_distill.item()
        running_stsb_loss += loss_stsb.item()
        log_count += 1
        
        if log_count >= 5:
            pbar.set_postfix({
                'distill': f'{running_distill_loss/log_count:.4f}',
                'stsb': f'{running_stsb_loss/log_count:.4f}',
                'best': f'{best_pearson:.4f}'
            })
            running_distill_loss = 0.0
            running_stsb_loss = 0.0
            log_count = 0
        
        if (global_step + 1) % config['eval_interval'] == 0:
            current_pearson, current_spearman = evaluate_stsb(student)
            
            if current_pearson > best_pearson:
                best_pearson = current_pearson
                best_step = global_step + 1
                patience = 0
                
                torch.save({
                    'deltanet_layers': student.deltanet_layers.state_dict(),
                    'deltanet_norms': student.norms.state_dict(),
                    'deltanet_ffns': student.ffns.state_dict(),
                    'ffn_norms': student.ffn_norms.state_dict(),
                    'step': global_step + 1,
                    'pearson': current_pearson,
                    'spearman': current_spearman
                }, output_dir / f"checkpoint_best_{current_pearson:.4f}.pt")
                
                status = "🎉 NEW BEST!"
            elif current_pearson < config['min_score']:
                patience += 1
                status = f"⚠️  Low (patience {patience}/{config['patience']})"
            else:
                patience += 1
                status = f"📊 (patience {patience}/{config['patience']})"
            
            print(f"\n📊 Step {global_step+1}:")
            print(f"   Pearson:  {current_pearson:.4f} ({current_pearson-baseline_pearson:+.4f}) {status}")
            print(f"   Gap to 0.850: {config['target_score']-current_pearson:.4f}")
            
            if current_pearson >= config['target_score']:
                print(f"\n🎉🎉🎉 TARGET 0.850 REACHED VIA DISTILLATION! 🎉🎉🎉")
                break
            
            if patience >= config['patience']:
                print(f"\n⚠️  Early stopping")
                break
        
        global_step += 1
        pbar.update(1)
    
    pbar.close()
    
    # Final
    print("\n" + "="*80)
    print("✅ DISTILLATION COMPLETE")
    print("="*80)
    final_pearson, _ = evaluate_stsb(student)
    print(f"\n📊 RESULTS:")
    print(f"   Baseline: {baseline_pearson:.4f}")
    print(f"   Final:    {final_pearson:.4f} ({final_pearson-baseline_pearson:+.4f})")
    print(f"   Best:     {best_pearson:.4f} (step {best_step})")
    
    if best_pearson >= config['target_score']:
        print(f"\n🎉🎉🎉 TARGET 0.850 ACHIEVED! 🎉🎉🎉")

if __name__ == "__main__":
    train_with_distillation()