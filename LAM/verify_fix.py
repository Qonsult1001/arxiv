import torch
import torch.nn as nn
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

def test_fix():
    print("🚀 Testing Enhanced Hierarchical DeltaNet Fixes...")
    
    # Configuration
    batch_size = 2
    seq_len = 64
    hidden_size = 128
    num_heads = 4
    
    # Initialize model
    model = EnhancedHierarchicalDeltaNet(
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_enhanced_flux=True
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model initialized on {device}")
    
    # Dummy input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    
    # Forward pass
    print("🔄 Running forward pass...")
    output, _, _, ortho_loss = model(x)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Orthogonal Loss: {ortho_loss.item() if ortho_loss is not None else 'None'}")
    
    # Check output stats
    print(f"   Output Mean: {output.mean().item():.4f}")
    print(f"   Output Std: {output.std().item():.4f}")
    
    if torch.isnan(output).any():
        print("❌ Output contains NaNs!")
        return
    
    # Backward pass
    print("🔄 Running backward pass...")
    loss = output.mean() + (ortho_loss if ortho_loss is not None else 0)
    loss.backward()
    
    # Check gradients
    print("🔍 Checking gradients...")
    
    # Check decay projection gradients
    if model.fast_decay_proj.weight.grad is not None:
        print(f"   ✅ fast_decay_proj gradient magnitude: {model.fast_decay_proj.weight.grad.norm().item():.4f}")
    else:
        print("   ❌ fast_decay_proj has NO gradient!")
        
    # Check bilinear flux gradients
    if model.resonance_flux.W_bilinear.grad is not None:
        print(f"   ✅ W_bilinear gradient magnitude: {model.resonance_flux.W_bilinear.grad.norm().item():.4f}")
    else:
        print("   ❌ W_bilinear has NO gradient!")
        
    print("\n✅ Verification Complete!")

if __name__ == "__main__":
    test_fix()
