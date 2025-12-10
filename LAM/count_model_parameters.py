#!/usr/bin/env python3
"""
Count actual parameters in pytorch_model.bin to verify the model size.
"""
import torch
from pathlib import Path
import sys

def count_parameters_in_checkpoint(checkpoint_path):
    """Count parameters in a PyTorch checkpoint file."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ File not found: {checkpoint_path}")
        return
    
    print("="*80)
    print(f"Loading checkpoint: {checkpoint_path}")
    print("="*80)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    # Check structure
    print(f"\nTop-level keys: {list(checkpoint.keys())[:20]}")
    
    # Determine if it's a state dict or nested structure
    state_dict = None
    if isinstance(checkpoint, dict):
        # Check if it's a direct state dict (has model keys)
        has_model_keys = any('deltanet' in str(k).lower() or 'embeddings' in str(k).lower() or 'layer' in str(k).lower() for k in checkpoint.keys())
        
        if has_model_keys and 'model_state_dict' not in checkpoint:
            # Direct state dict
            state_dict = checkpoint
            print("\n✅ Detected: Direct state dict format")
        elif 'model_state_dict' in checkpoint:
            # Nested with model_state_dict
            state_dict = checkpoint['model_state_dict']
            print("\n✅ Detected: Nested format with 'model_state_dict'")
        elif 'deltanet_layers' in checkpoint:
            # Nested with deltanet_layers
            state_dict = checkpoint
            print("\n✅ Detected: Nested format with 'deltanet_layers'")
        else:
            # Try to use the whole dict
            state_dict = checkpoint
            print("\n✅ Using entire checkpoint as state dict")
    
    if state_dict is None:
        print("❌ Could not determine state dict structure")
        return
    
    # Count parameters
    print("\n" + "="*80)
    print("PARAMETER COUNT ANALYSIS")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    # Categorize parameters
    deltanet_params = 0
    base_params = 0
    embedding_params = 0
    ffn_params = 0
    norm_params = 0
    other_params = 0
    
    param_details = {
        'deltanet': [],
        'base': [],
        'embeddings': [],
        'ffn': [],
        'norm': [],
        'other': []
    }
    
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        
        num_params = param.numel()
        total_params += num_params
        
        # Categorize by name
        name_lower = name.lower()
        if 'deltanet' in name_lower or 'lam' in name_lower:
            deltanet_params += num_params
            param_details['deltanet'].append((name, num_params))
        elif 'embedding' in name_lower:
            embedding_params += num_params
            param_details['embeddings'].append((name, num_params))
        elif 'ffn' in name_lower or 'intermediate' in name_lower or 'output.dense' in name_lower:
            ffn_params += num_params
            param_details['ffn'].append((name, num_params))
        elif 'norm' in name_lower or 'layer_norm' in name_lower:
            norm_params += num_params
            param_details['norm'].append((name, num_params))
        elif 'attention' in name_lower or 'pooler' in name_lower:
            base_params += num_params
            param_details['base'].append((name, num_params))
        else:
            other_params += num_params
            param_details['other'].append((name, num_params))
    
    # Print summary
    print(f"\n📊 TOTAL PARAMETERS: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"\n📦 BREAKDOWN BY COMPONENT:")
    print(f"   DeltaNet layers:     {deltanet_params:>12,} ({deltanet_params/1e6:>6.2f}M) - {deltanet_params/total_params*100:>5.1f}%")
    print(f"   Embeddings:          {embedding_params:>12,} ({embedding_params/1e6:>6.2f}M) - {embedding_params/total_params*100:>5.1f}%")
    print(f"   FFN layers:          {ffn_params:>12,} ({ffn_params/1e6:>6.2f}M) - {ffn_params/total_params*100:>5.1f}%")
    print(f"   Layer Norms:         {norm_params:>12,} ({norm_params/1e6:>6.2f}M) - {norm_params/total_params*100:>5.1f}%")
    print(f"   Base/Attention:      {base_params:>12,} ({base_params/1e6:>6.2f}M) - {base_params/total_params*100:>5.1f}%")
    print(f"   Other:               {other_params:>12,} ({other_params/1e6:>6.2f}M) - {other_params/total_params*100:>5.1f}%")
    
    # Estimate frozen vs trained
    # Typically: embeddings + FFN + norms from base = frozen
    # DeltaNet = trained
    estimated_frozen = embedding_params + ffn_params + norm_params + base_params
    estimated_trained = deltanet_params
    
    print(f"\n🔒 ESTIMATED BREAKDOWN (Frozen vs Trained):")
    print(f"   Frozen (base model): {estimated_frozen:>12,} ({estimated_frozen/1e6:>6.2f}M) - {estimated_frozen/total_params*100:>5.1f}%")
    print(f"   Trained (DeltaNet):  {estimated_trained:>12,} ({estimated_trained/1e6:>6.2f}M) - {estimated_trained/total_params*100:>5.1f}%")
    
    # Show some example parameter names
    print(f"\n📋 SAMPLE PARAMETER NAMES:")
    print(f"   DeltaNet (first 5):")
    for name, count in param_details['deltanet'][:5]:
        print(f"      {name}: {count:,} params")
    if len(param_details['deltanet']) > 5:
        print(f"      ... and {len(param_details['deltanet']) - 5} more")
    
    print(f"\n   Embeddings (first 3):")
    for name, count in param_details['embeddings'][:3]:
        print(f"      {name}: {count:,} params")
    
    print(f"\n   FFN (first 3):")
    for name, count in param_details['ffn'][:3]:
        print(f"      {name}: {count:,} params")
    
    # File size estimate
    file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    params_size_mb = total_params * 4 / (1024 * 1024)  # float32 = 4 bytes
    
    print(f"\n💾 FILE SIZE:")
    print(f"   Actual file size:   {file_size_mb:.2f} MB")
    print(f"   Estimated (float32): {params_size_mb:.2f} MB")
    print(f"   Compression ratio:   {file_size_mb/params_size_mb:.2f}x" if params_size_mb > 0 else "   (couldn't calculate)")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  → Frozen (base): ~{estimated_frozen:,} (~{estimated_frozen/1e6:.2f}M)")
    print(f"  → Trained (DeltaNet): ~{estimated_trained:,} (~{estimated_trained/1e6:.2f}M)")
    print("="*80)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = "/workspace/LAM/best/pytorch_model.bin"
    
    count_parameters_in_checkpoint(checkpoint_path)

