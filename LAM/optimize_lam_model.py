#!/usr/bin/env python3
"""
Optimize LAM model for faster loading and inference
Options:
1. TorchScript (JIT) - Fastest loading, good inference speed
2. torch.compile - Best inference speed (but cache issues)
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import time

# Add LAM directory to path
sys.path.insert(0, str(Path(__file__).parent))
from train_6layer_deltanet_2 import DeltaNetPure6Layer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def optimize_with_torchscript(model_path, output_path):
    """
    Convert model to TorchScript for faster loading
    TorchScript is optimized and loads much faster than regular PyTorch
    """
    print("="*80)
    print("OPTION 1: TorchScript (JIT) Optimization")
    print("="*80)
    print("✅ Fastest model loading")
    print("✅ Good inference speed")
    print("✅ No cache issues")
    print()
    
    # Load model
    print(f"Loading model from: {model_path}")
    config = {
        "teacher_model": "/workspace/LAM/all-MiniLM-L6-v2",
        "num_layers": 6,
        "num_heads": 12,
        "fast_decay_init": 0.30,
        "slow_decay_init": 0.85,
        "use_kernel_blending": False,
    }
    
    model = DeltaNetPure6Layer(
        teacher_model_name=config['teacher_model'],
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)
    
    # Load weights
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded")
    
    model.eval()
    
    # Create example input for tracing
    print("Creating TorchScript model...")
    tokenizer = model.tokenizer
    example_text = "This is a test sentence for optimization."
    example_tokens = tokenizer(
        example_text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)
    
    # Create a wrapper class for tracing
    class EncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            return self.model.encode(input_ids, attention_mask)
    
    wrapper = EncoderWrapper(model).eval()
    
    # Trace the wrapper model
    print("Tracing model...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapper,
            (example_tokens['input_ids'], example_tokens['attention_mask'])
        )
    
    # Save optimized model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))
    print(f"✅ Saved TorchScript model to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Test loading speed
    print("\nTesting loading speed...")
    start = time.time()
    loaded_model = torch.jit.load(str(output_path), map_location=device)
    load_time = time.time() - start
    print(f"✅ TorchScript model loaded in {load_time*1000:.1f}ms")
    
    # Test inference speed
    print("Testing inference speed...")
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = loaded_model(example_tokens['input_ids'], example_tokens['attention_mask'])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = (time.time() - start) / 10 * 1000
    print(f"✅ Average inference: {inference_time:.1f}ms")
    
    return traced_model

def optimize_with_compile(model_path, output_path):
    """
    Use torch.compile for best inference speed
    Note: May have cache issues, but fastest inference
    """
    print("="*80)
    print("OPTION 2: torch.compile Optimization")
    print("="*80)
    print("✅ Best inference speed")
    print("⚠️  May have cache issues")
    print()
    
    # Load model
    print(f"Loading model from: {model_path}")
    config = {
        "teacher_model": "/workspace/LAM/all-MiniLM-L6-v2",
        "num_layers": 6,
        "num_heads": 12,
        "fast_decay_init": 0.30,
        "slow_decay_init": 0.85,
        "use_kernel_blending": False,
    }
    
    model = DeltaNetPure6Layer(
        teacher_model_name=config['teacher_model'],
        num_linear_layers=config['num_layers'],
        config=config
    ).to(device)
    
    # Load weights
    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded")
    
    model.eval()
    
    # Compile the encode method
    print("Compiling model...")
    compiled_encode = torch.compile(model.encode, mode='reduce-overhead')
    model.encode = compiled_encode
    
    # Save model (with compiled function)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"✅ Saved compiled model state to: {output_path}")
    print("⚠️  Note: Compiled function is in memory, not saved")
    
    return model

def main():
    print("🚀 LAM Model Optimizer")
    print("="*80)
    print()
    
    # Find model file
    model_paths = [
        Path("/workspace/LAM/LAM-base-v1/pytorch_model.bin"),
        Path("/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/pytorch_model.bin"),
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if not model_path:
        print("❌ Model file not found!")
        return
    
    print(f"Using model: {model_path}")
    print()
    
    # Option 1: TorchScript (Recommended for testing)
    print("\n" + "="*80)
    print("RECOMMENDED: TorchScript Optimization")
    print("="*80)
    output_script = Path("/workspace/LAM/LAM-base-v1/lam_model_optimized.pt")
    optimize_with_torchscript(model_path, output_script)
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nOptimized model saved to: {output_script}")
    print("\nTo use in quick_proof_30sec.py:")
    print(f"  Load with: torch.jit.load('{output_script}', map_location=device)")
    print("  This will be MUCH faster than loading .bin file!")

if __name__ == "__main__":
    main()

