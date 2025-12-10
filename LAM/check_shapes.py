import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

# Load checkpoint
ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_8k/checkpoint_167000.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

if 'deltanet_layers' in ckpt:
    state_dict = ckpt['deltanet_layers']
else:
    state_dict = ckpt

# Create a model layer
layer = EnhancedHierarchicalDeltaNet(
    d_model=384,
    num_heads=12,
    use_hierarchical_decay=True,
    use_enhanced_flux=True,
    fast_decay_init=0.30,
    slow_decay_init=0.85,
    use_rope=False
)

# Get checkpoint keys for layer 0
ckpt_layer0 = {k.replace('0.', ''): v for k, v in state_dict.items() if k.startswith('0.')}

# Get model keys
model_keys = layer.state_dict()

print("Checking shape mismatches:")
for k in ckpt_layer0.keys():
    if k in model_keys:
        if ckpt_layer0[k].shape != model_keys[k].shape:
            print(f"  MISMATCH: {k}")
            print(f"    Checkpoint: {ckpt_layer0[k].shape}")
            print(f"    Model:      {model_keys[k].shape}")
    else:
        print(f"  MISSING in model: {k}")

print("\nKeys in model but not in checkpoint:")
for k in model_keys.keys():
    if k not in ckpt_layer0:
        print(f"  {k}: {model_keys[k].shape}")
