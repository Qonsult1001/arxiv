
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os

# Import both versions
# We need to do some trickery to import classes with the same name from different files
import importlib.util

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
BATCH_SIZE = 8
SEQ_LEN = 128
HIDDEN_SIZE = 384
NUM_HEADS = 12
STEPS = 50
LR = 1e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on {device}")

def train_model(ModelClass, name):
    print(f"\n📉 Training {name}...")
    
    # Initialize model
    model = ModelClass(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        use_enhanced_flux=True
    ).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Dummy Data (Teacher-Student setup)
    # Fixed random seed for reproducibility
    torch.manual_seed(42)
    
    losses = []
    
    for step in range(STEPS):
        # Random input
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device)
        
        # Target (simulate teacher output - just random fixed target for this test)
        # In real training, this would be the teacher's output
        target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device)
        
        # Forward
        output, _, _, ortho_loss = model(x)
        
        # Loss (MSE + Ortho)
        mse_loss = nn.MSELoss()(output, target)
        total_loss = mse_loss + (ortho_loss if ortho_loss is not None else 0) * 0.01
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (to prevent explosion if broken)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if step % 10 == 0:
            print(f"   Step {step}: Loss = {total_loss.item():.6f}")
            
    return losses

# Train both
losses_old = train_model(OldDeltaNet, "Old Formula (Baseline)")
losses_new = train_model(NewDeltaNet, "New Formula (Fixed)")

print("\n📊 Results:")
print(f"Old Final Loss: {losses_old[-1]:.6f}")
print(f"New Final Loss: {losses_new[-1]:.6f}")

improvement = losses_old[-1] - losses_new[-1]
print(f"Improvement: {improvement:.6f}")

if losses_new[-1] < losses_old[-1]:
    print("✅ SUCCESS: New formula converges better!")
else:
    print("⚠️  WARNING: New formula did not outperform old one in this short test.")
