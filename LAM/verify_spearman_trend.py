
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import importlib.util
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer

# Import both versions
def import_class_from_file(file_path, class_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

print("🔄 Importing models...")
try:
    OldDeltaNet = import_class_from_file("/workspace/LAM/final_solution_formula.py", "EnhancedHierarchicalDeltaNet")
    print("✅ Loaded Old DeltaNet (Baseline)")
except Exception as e:
    print(f"❌ Failed to load Old DeltaNet: {e}")
    sys.exit(1)

try:
    NewDeltaNet = import_class_from_file("/workspace/LAM/final_solution_formula_final.py", "EnhancedHierarchicalDeltaNet")
    print("✅ Loaded New DeltaNet (Fixed)")
except Exception as e:
    print(f"❌ Failed to load New DeltaNet: {e}")
    sys.exit(1)

# Configuration
CHECKPOINT_PATH = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/checkpoint_40000.pt"
BATCH_SIZE = 16
SEQ_LEN = 64
HIDDEN_SIZE = 384
NUM_HEADS = 12
STEPS = 100
EVAL_INTERVAL = 10
LR = 1e-5 # Low LR for fine-tuning

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on {device}")

# Load Tokenizer (needed for real text if we used it, but we'll use synthetic embeddings for speed)
# To properly test Spearman, we need semantic similarity.
# Synthetic data with random vectors won't have real semantic structure to learn.
# WE MUST USE REAL DATA or a Teacher-Student setup where the Teacher provides the structure.
# Let's use the Teacher-Student setup from the training script.

from transformers import AutoModel
TEACHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
print(f"🔄 Loading Teacher: {TEACHER_MODEL}")
teacher_model = AutoModel.from_pretrained(TEACHER_MODEL).to(device)
teacher_model.eval()
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

# Create a small validation set of real sentences
val_sentences = [
    ("A man is playing a guitar.", "A man is playing music."),
    ("A woman is slicing a potato.", "A woman is cutting a potato."),
    ("The sun is shining.", "It is a sunny day."),
    ("A cat is sleeping.", "A dog is barking."), # Low sim
    ("The car is fast.", "The vehicle is speeding."),
    ("I love pizza.", "Pizza is my favorite food."),
    ("The sky is blue.", "The grass is green."), # Low sim
    ("He is running.", "He is walking."), # Med sim
    ("The computer is broken.", "My laptop is not working."),
    ("She is reading a book.", "She is looking at a novel."),
    ("The movie was great.", "I enjoyed the film."),
    ("It is raining hard.", "There is a storm."),
    ("A bird is flying.", "A fish is swimming."), # Low sim
    ("The coffee is hot.", "The tea is cold."), # Low sim
    ("I am happy.", "I am sad."), # Low sim (antonym)
    ("The door is open.", "The door is closed."), # Low sim
]
# Duplicate to make a larger batch
val_sentences = val_sentences * 4 

def get_embeddings(model, sentences):
    model.eval()
    with torch.no_grad():
        # Flatten pairs
        flat_sentences = [s for pair in sentences for s in pair]
        
        tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        
        # For DeltaNet, we need to handle the forward pass manually if it's just the component
        # But wait, the DeltaNet class is just the attention mechanism?
        # No, EnhancedHierarchicalDeltaNet is a layer.
        # The full model is DeltaNetPure6Layer.
        # We can't easily instantiate the full model without the training script's class.
        # Let's wrap the DeltaNet layer in a simple embedding model for this test.
        
        # Simple wrapper: Embedding -> DeltaNet -> Mean Pooling
        # We need the embeddings from the teacher first
        
        # Get initial embeddings from teacher's embedding layer (which is frozen in student)
        emb_layer = teacher_model.embeddings
        x = emb_layer(tokens['input_ids'])
        
        # Pass through DeltaNet
        # We'll just use ONE layer of DeltaNet for this test to see if it learns
        output, _, _, _ = model(x)
        
        # Mean pooling
        mask = tokens['attention_mask'].unsqueeze(-1).expand(output.size()).float()
        embeddings = torch.sum(output * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def evaluate_spearman(model, sentences):
    embeddings = get_embeddings(model, sentences)
    
    # Reshape to pairs [N, 2, D]
    n_pairs = len(sentences)
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    
    # Cosine similarity
    sims = F.cosine_similarity(emb1, emb2).cpu().numpy()
    
    # Teacher scores (Ground Truth for this test)
    with torch.no_grad():
        flat_sentences = [s for pair in sentences for s in pair]
        tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        teacher_out = teacher_model(**tokens)
        # Mean pooling
        mask = tokens['attention_mask'].unsqueeze(-1).expand(teacher_out.last_hidden_state.size()).float()
        teacher_emb = torch.sum(teacher_out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        teacher_emb = F.normalize(teacher_emb, p=2, dim=1)
        
        t_emb1 = teacher_emb[0::2]
        t_emb2 = teacher_emb[1::2]
        teacher_sims = F.cosine_similarity(t_emb1, t_emb2).cpu().numpy()
        
    return spearmanr(sims, teacher_sims)[0]

def load_checkpoint_weights(model, checkpoint_path):
    print(f"   Loading weights from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Filter for the first DeltaNet layer
        layer0_prefix = "deltanet_layers.0."
        layer0_state = {}
        for k, v in state_dict.items():
            if k.startswith(layer0_prefix):
                new_key = k[len(layer0_prefix):]
                layer0_state[new_key] = v
                
        model.load_state_dict(layer0_state, strict=False)
        print("   ✅ Weights loaded successfully (Layer 0)")
        return True
    except Exception as e:
        print(f"   ❌ Failed to load weights: {e}")
        return False

def train_and_eval(ModelClass, name):
    print(f"\n📉 Training {name}...")
    
    model = ModelClass(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        use_enhanced_flux=True
    ).to(device)
    
    if not load_checkpoint_weights(model, CHECKPOINT_PATH):
        return []
        
    # EXPERIMENTAL: Zero out decay projections to reset decay dynamics to linear baseline
    # This removes the "noise" from the old weights which were learned for a different formula
    if name == "New Formula":
        print("   🔧 Zeroing decay projections for clean slate...")
        with torch.no_grad():
            model.fast_decay_proj.weight.data.zero_()
            model.slow_decay_proj.weight.data.zero_()
            # We keep the bias as is, which sets the baseline decay rate
            
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    spearmans = []
    
    # Initial eval
    initial_spearman = evaluate_spearman(model, val_sentences)
    print(f"   Step 0: Spearman = {initial_spearman:.4f}")
    spearmans.append(initial_spearman)
    
    # Training loop
    model.train()
    
    # We need some dummy training data that mimics the structure
    # We'll use the same validation sentences but train on them (overfitting test)
    # If it can't overfit/learn these, it's broken.
    
    flat_sentences = [s for pair in val_sentences for s in pair]
    
    for step in range(1, STEPS + 1):
        # Get teacher embeddings as target
        with torch.no_grad():
            tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            emb_layer = teacher_model.embeddings
            x_in = emb_layer(tokens['input_ids'])
            
            # Teacher output for this layer (approximate)
            # We want the student to match the teacher's output
            teacher_out = teacher_model(**tokens)
            target = teacher_out.last_hidden_state
            
            # Project target to match student dimension if needed (here they match 384)
        
        # Student forward
        output, _, _, ortho_loss = model(x_in)
        
        # Loss: MSE with Teacher Output
        mse_loss = nn.MSELoss()(output, target)
        total_loss = mse_loss + (ortho_loss if ortho_loss is not None else 0) * 0.01
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % EVAL_INTERVAL == 0:
            s = evaluate_spearman(model, val_sentences)
            print(f"   Step {step}: Spearman = {s:.4f} (Loss: {total_loss.item():.4f})")
            spearmans.append(s)
            model.train()
            
    return spearmans

# Run comparison
spearmans_old = train_and_eval(OldDeltaNet, "Old Formula")
spearmans_new = train_and_eval(NewDeltaNet, "New Formula")

print("\n📊 Spearman Improvement Results:")
print(f"Old: {spearmans_old[0]:.4f} -> {spearmans_old[-1]:.4f} (Delta: {spearmans_old[-1]-spearmans_old[0]:.4f})")
print(f"New: {spearmans_new[0]:.4f} -> {spearmans_new[-1]:.4f} (Delta: {spearmans_new[-1]-spearmans_new[0]:.4f})")

if (spearmans_new[-1] - spearmans_new[0]) > (spearmans_old[-1] - spearmans_old[0]):
    print("✅ SUCCESS: New formula improves Spearman faster!")
else:
    print("⚠️  WARNING: New formula did not improve Spearman faster.")
