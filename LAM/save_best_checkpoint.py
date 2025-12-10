#!/usr/bin/env python3
"""
Script to save a specific checkpoint as pytorch_model.bin
Useful when the best checkpoint wasn't automatically saved during training
"""
import torch
from pathlib import Path
import sys
import json
from transformers import AutoModel, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from train_6layer_deltanet_2 import DeltaNetPure6Layer

def save_checkpoint_as_pytorch_model(checkpoint_path, output_dir=None):
    """
    Load a checkpoint and save it as pytorch_model.bin (same format as save_pretrained)
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., checkpoint_25000.pt)
        output_dir: Directory to save pytorch_model.bin (default: same as checkpoint parent)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    # Determine output directory
    if output_dir is None:
        output_dir = checkpoint_path.parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("="*80)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    step = checkpoint.get('step', 0)
    
    print(f"Checkpoint step: {step}")
    print(f"Config keys: {list(config.keys())[:5]}...")
    
    # Get teacher model path from config (with defaults)
    teacher_model_path = config.get('teacher_model', '/workspace/LAM/all-MiniLM-L6-v2')
    num_linear_layers = config.get('num_linear_layers', 6)
    
    # Ensure config has required keys with defaults
    if 'num_heads' not in config:
        config['num_heads'] = 12
    if 'fast_decay_init' not in config:
        config['fast_decay_init'] = 0.3
    if 'slow_decay_init' not in config:
        config['slow_decay_init'] = 0.832
    
    print(f"\nBuilding DeltaNet model...")
    print(f"   Teacher model: {teacher_model_path}")
    print(f"   Num layers: {num_linear_layers}")
    
    # Build DeltaNet model (it will load teacher model internally)
    model = DeltaNetPure6Layer(
        teacher_model_name=teacher_model_path,
        num_linear_layers=num_linear_layers,
        config=config
    )
    
    # Load checkpoint weights
    print("\nLoading checkpoint weights...")
    
    # Check if checkpoint is a full state dict (keys like 'deltanet_layers.0.q_proj.weight')
    # or a nested dict (keys like 'deltanet_layers': {...})
    is_full_state_dict = any('deltanet_layers.' in str(k) for k in checkpoint.keys())
    
    if is_full_state_dict:
        # Full state dict format - extract deltanet_layers
        print("   Detected full state dict format...")
        deltanet_layers_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('deltanet_layers.'):
                new_key = key.replace('deltanet_layers.', '')
                deltanet_layers_dict[new_key] = value
        
        if deltanet_layers_dict:
            # Load each layer separately, filtering out mismatched parameters
            total_loaded = 0
            total_skipped = 0
            for i in range(6):
                layer_state = {k[2:]: v for k, v in deltanet_layers_dict.items() if k.startswith(f'{i}.')}
                if layer_state:
                    # Filter out parameters that don't match current model shape
                    # Handle shape conversions for compatible parameters
                    filtered_state = {}
                    model_state = model.deltanet_layers[i].state_dict()
                    for key, value in layer_state.items():
                        if key in model_state:
                            expected_shape = model_state[key].shape
                            if expected_shape == value.shape:
                                filtered_state[key] = value
                                total_loaded += 1
                            else:
                                # Try to convert compatible shapes
                                converted = False
                                
                                # Handle W_bilinear: [32, 32] -> [12, 32, 32] (old shared -> new per-head)
                                if key == 'resonance_flux.W_bilinear':
                                    if value.dim() == 2 and expected_shape[0] > 1:
                                        # Old format: [d_k, d_v] -> New format: [num_heads, d_k, d_v]
                                        # Repeat the same matrix for each head
                                        num_heads = expected_shape[0]
                                        converted_value = value.unsqueeze(0).expand(num_heads, -1, -1).clone()
                                        if converted_value.shape == expected_shape:
                                            filtered_state[key] = converted_value
                                            total_loaded += 1
                                            converted = True
                                            print(f"   ✅ Converted {key} from {value.shape} to {expected_shape} (expanded for {num_heads} heads)")
                                
                                if not converted:
                                    total_skipped += 1
                                    print(f"   ⚠️  Skipping {key} (shape mismatch: {value.shape} vs {expected_shape})")
                        else:
                            total_skipped += 1
                    
                    if filtered_state:
                        model.deltanet_layers[i].load_state_dict(filtered_state, strict=False)
            
            print(f"   ✅ Loaded deltanet_layers ({total_loaded} parameters loaded, {total_skipped} skipped)")
    else:
        # Nested dict format
        if 'deltanet_layers' in checkpoint:
            model.deltanet_layers.load_state_dict(checkpoint['deltanet_layers'], strict=False)
            print("   ✅ Loaded deltanet_layers")
        
        if 'deltanet_norms' in checkpoint:
            model.deltanet_norms.load_state_dict(checkpoint['deltanet_norms'], strict=False)
            print("   ✅ Loaded deltanet_norms")
        
        if 'deltanet_ffns' in checkpoint:
            model.deltanet_ffns.load_state_dict(checkpoint['deltanet_ffns'], strict=False)
            print("   ✅ Loaded deltanet_ffns")
        
        if 'ffn_norms' in checkpoint:
            model.ffn_norms.load_state_dict(checkpoint['ffn_norms'], strict=False)
            print("   ✅ Loaded ffn_norms")
        
        # Handle output_denses if present (for newer checkpoints)
        if 'output_denses' in checkpoint and hasattr(model, 'output_denses'):
            model.output_denses.load_state_dict(checkpoint['output_denses'], strict=False)
            print("   ✅ Loaded output_denses")
        elif 'output_dense_layers' in checkpoint and hasattr(model, 'output_denses'):
            model.output_denses.load_state_dict(checkpoint['output_dense_layers'], strict=False)
            print("   ✅ Loaded output_dense_layers")
    
    # Save as pytorch_model.bin (using save_pretrained format)
    print(f"\n💾 Saving model to {output_dir}/pytorch_model.bin...")
    model.save_pretrained(output_dir)
    
    print("\n" + "="*80)
    print("✅ SUCCESS!")
    print("="*80)
    print(f"Saved best model from checkpoint {checkpoint_path.name}")
    print(f"   Step: {step}")
    if 'test_spearman' in checkpoint:
        print(f"   Test Spearman: {checkpoint['test_spearman']:.4f}")
    if 'val_spearman' in checkpoint:
        print(f"   Val Spearman: {checkpoint['val_spearman']:.4f}")
    print(f"\nOutput: {output_dir}/pytorch_model.bin")
    print("="*80)
    
    return True

if __name__ == "__main__":
    # Default to deltanet_shockwave_result.pt if no arguments provided
    if len(sys.argv) < 2:
        checkpoint_path = "/workspace/LAM/best/deltanet_shockwave_result.pt"
        output_dir = "/workspace/LAM/best"  # Save in same directory
        print(f"Using default checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = save_checkpoint_as_pytorch_model(checkpoint_path, output_dir)
    if not success:
        sys.exit(1)

