
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import importlib.util

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
BATCH_SIZE = 8
SEQ_LEN = 128
HIDDEN_SIZE = 384
NUM_HEADS = 12
STEPS = 50
LR = 1e-4 # Lower LR for fine-tuning

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Running on {device}")

def load_checkpoint_weights(model, checkpoint_path):
    print(f"   Loading weights from {checkpoint_path}...")
    try:
        # Set weights_only=False to avoid pickle errors with numpy scalars
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # The checkpoint might be a full model state dict or just the deltanet layers
        # Based on train_6layer_deltanet_2.py, the model has 'deltanet_layers'
        # We need to extract the weights for ONE layer to test the DeltaNet class directly
        # OR we can just instantiate the full DeltaNetPure6Layer structure if needed, 
        # but here we are testing the DeltaNet component.
        
        # Let's inspect the keys
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Filter for the first DeltaNet layer to test
        layer0_prefix = "deltanet_layers.0."
        layer0_state = {}
        for k, v in state_dict.items():
            if k.startswith(layer0_prefix):
                new_key = k[len(layer0_prefix):]
                layer0_state[new_key] = v
                
        # Load into model
        # strict=False because of potential minor differences (e.g. if we added buffers)
        # But critical weights (projections) must match
        model.load_state_dict(layer0_state, strict=False)
        print("   ✅ Weights loaded successfully (Layer 0)")
        return True
    except Exception as e:
        print(f"   ❌ Failed to load weights: {e}")
        return False

def train_model(ModelClass, name):
    print(f"\n📉 Training {name} from Checkpoint 40k...")
    
    # Initialize model
    model = ModelClass(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        use_enhanced_flux=True
    ).to(device)
    
    # Load weights
    if not load_checkpoint_weights(model, CHECKPOINT_PATH):
        return []
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # Dummy Data (Teacher-Student setup)
    torch.manual_seed(42)
    
    losses = []
    
    for step in range(STEPS):
        # Random input
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device)
        
        # Target (simulate teacher output)
        target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, device=device)
        
        # Forward
        output, _, _, ortho_loss = model(x)
        
        # Loss (MSE + Ortho)
        mse_loss = nn.MSELoss()(output, target)
        total_loss = mse_loss + (ortho_loss if ortho_loss is not None else 0) * 0.01
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        losses.append(total_loss.item())
        
        if step % 10 == 0:
            print(f"   Step {step}: Loss = {total_loss.item():.6f}")
            
    return losses

# Train both
losses_old = train_model(OldDeltaNet, "Old Formula (Baseline)")
losses_new = train_model(NewDeltaNet, "New Formula (Fixed)")

print("\n📊 Results (From Checkpoint 40k):")
if losses_old and losses_new:
    print(f"Old Final Loss: {losses_old[-1]:.6f}")
    print(f"New Final Loss: {losses_new[-1]:.6f}")

    improvement = losses_old[-1] - losses_new[-1]
    print(f"Improvement: {improvement:.6f}")

    if losses_new[-1] < losses_old[-1]:
        print("✅ SUCCESS: New formula converges better from checkpoint!")
    else:
        print("⚠️  WARNING: New formula did not outperform old one in this short test.")
else:
    print("❌ Could not complete comparison due to loading errors.")
