
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import importlib.util
import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel

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
BATCH_SIZE = 16
SEQ_LEN = 64
HIDDEN_SIZE = 384
NUM_HEADS = 12
STEPS = 100
EVAL_INTERVAL = 10
LR = 1e-4 # Standard training LR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on {device}")

# Load Teacher
TEACHER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
print(f"🔄 Loading Teacher: {TEACHER_MODEL}")
teacher_model = AutoModel.from_pretrained(TEACHER_MODEL).to(device)
teacher_model.eval()
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)

# Validation Sentences (Semantic Structure)
val_sentences = [
    ("A man is playing a guitar.", "A man is playing music."),
    ("A woman is slicing a potato.", "A woman is cutting a potato."),
    ("The sun is shining.", "It is a sunny day."),
    ("A cat is sleeping.", "A dog is barking."), 
    ("The car is fast.", "The vehicle is speeding."),
    ("I love pizza.", "Pizza is my favorite food."),
    ("The sky is blue.", "The grass is green."), 
    ("He is running.", "He is walking."), 
    ("The computer is broken.", "My laptop is not working."),
    ("She is reading a book.", "She is looking at a novel."),
    ("The movie was great.", "I enjoyed the film."),
    ("It is raining hard.", "There is a storm."),
    ("A bird is flying.", "A fish is swimming."), 
    ("The coffee is hot.", "The tea is cold."), 
    ("I am happy.", "I am sad."), 
    ("The door is open.", "The door is closed."), 
]
val_sentences = val_sentences * 4 

def get_embeddings(model, sentences):
    model.eval()
    with torch.no_grad():
        flat_sentences = [s for pair in sentences for s in pair]
        tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        
        emb_layer = teacher_model.embeddings
        x = emb_layer(tokens['input_ids'])
        
        output, _, _, _ = model(x)
        
        mask = tokens['attention_mask'].unsqueeze(-1).expand(output.size()).float()
        embeddings = torch.sum(output * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

def evaluate_spearman(model, sentences):
    embeddings = get_embeddings(model, sentences)
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    sims = F.cosine_similarity(emb1, emb2).cpu().numpy()
    
    with torch.no_grad():
        flat_sentences = [s for pair in sentences for s in pair]
        tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
        teacher_out = teacher_model(**tokens)
        mask = tokens['attention_mask'].unsqueeze(-1).expand(teacher_out.last_hidden_state.size()).float()
        teacher_emb = torch.sum(teacher_out.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        teacher_emb = F.normalize(teacher_emb, p=2, dim=1)
        
        t_emb1 = teacher_emb[0::2]
        t_emb2 = teacher_emb[1::2]
        teacher_sims = F.cosine_similarity(t_emb1, t_emb2).cpu().numpy()
        
    return spearmanr(sims, teacher_sims)[0]

def train_and_eval(ModelClass, name):
    print(f"\n📉 Training {name} FROM SCRATCH...")
    
    # Initialize model with random weights
    model = ModelClass(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        use_enhanced_flux=True
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    spearmans = []
    losses = []
    
    # Initial eval
    initial_spearman = evaluate_spearman(model, val_sentences)
    print(f"   Step 0: Spearman = {initial_spearman:.4f}")
    spearmans.append(initial_spearman)
    
    model.train()
    flat_sentences = [s for pair in val_sentences for s in pair]
    
    for step in range(1, STEPS + 1):
        with torch.no_grad():
            tokens = tokenizer(flat_sentences, padding=True, truncation=True, return_tensors="pt").to(device)
            emb_layer = teacher_model.embeddings
            x_in = emb_layer(tokens['input_ids'])
            teacher_out = teacher_model(**tokens)
            target = teacher_out.last_hidden_state
            
        output, _, _, ortho_loss = model(x_in)
        
        mse_loss = nn.MSELoss()(output, target)
        total_loss = mse_loss + (ortho_loss if ortho_loss is not None else 0) * 0.01
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if step % EVAL_INTERVAL == 0:
            s = evaluate_spearman(model, val_sentences)
            print(f"   Step {step}: Spearman = {s:.4f} (Loss: {total_loss.item():.4f})")
            spearmans.append(s)
            model.train()
            
    return spearmans, losses

# Run comparison
spearmans_old, losses_old = train_and_eval(OldDeltaNet, "Old Formula")
spearmans_new, losses_new = train_and_eval(NewDeltaNet, "New Formula")

print("\n📊 FROM SCRATCH Results (100 Steps):")
print(f"Old Formula: Final Loss = {losses_old[-1]:.4f}, Final Spearman = {spearmans_old[-1]:.4f}")
print(f"New Formula: Final Loss = {losses_new[-1]:.4f}, Final Spearman = {spearmans_new[-1]:.4f}")

loss_imp = losses_old[-1] - losses_new[-1]
spearman_imp = spearmans_new[-1] - spearmans_old[-1]

print(f"Loss Improvement: {loss_imp:.4f}")
print(f"Spearman Improvement: {spearman_imp:.4f}")

if losses_new[-1] < losses_old[-1]:
    print("✅ SUCCESS: New formula converges better from scratch!")
else:
    print("⚠️  WARNING: New formula did not outperform old one.")
