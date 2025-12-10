#!/usr/bin/env python3
"""
Test what happens if we remove duplicate embeddings from the checkpoint.
"""
import torch
from pathlib import Path
import shutil

checkpoint_path = "/workspace/LAM/best/pytorch_model.bin"
test_checkpoint_path = "/workspace/LAM/best/pytorch_model_no_duplicates.bin"

print("="*80)
print("TESTING: Removing Duplicate Embeddings")
print("="*80)

# Load original checkpoint
print("\n1. Loading original checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
original_size = len(checkpoint)
original_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
print(f"   Original: {original_size} keys, {original_params:,} parameters")

# Identify duplicates
duplicates_to_remove = []
for name in list(checkpoint.keys()):
    if name.startswith('embeddings.') and not name.startswith('teacher_model'):
        # Check if there's a teacher_model equivalent
        teacher_name = 'teacher_model.' + name
        if teacher_name in checkpoint:
            teacher_param = checkpoint[teacher_name]
            standalone_param = checkpoint[name]
            if torch.equal(teacher_param, standalone_param):
                duplicates_to_remove.append(name)
                print(f"   Found duplicate: {name} ←→ {teacher_name}")

print(f"\n2. Removing {len(duplicates_to_remove)} duplicate keys...")
for key in duplicates_to_remove:
    del checkpoint[key]

new_size = len(checkpoint)
new_params = sum(p.numel() for p in checkpoint.values() if isinstance(p, torch.Tensor))
print(f"   After removal: {new_size} keys, {new_params:,} parameters")
print(f"   Removed: {original_params - new_params:,} parameters ({original_params - new_params:.2f}M)")

# Save test checkpoint
print(f"\n3. Saving test checkpoint (without duplicates)...")
torch.save(checkpoint, test_checkpoint_path)
test_file_size = Path(test_checkpoint_path).stat().st_size / (1024 * 1024)
original_file_size = Path(checkpoint_path).stat().st_size / (1024 * 1024)
print(f"   Original file: {original_file_size:.2f} MB")
print(f"   Test file: {test_file_size:.2f} MB")
print(f"   Size reduction: {original_file_size - test_file_size:.2f} MB")

# Test if model can load
print(f"\n4. Testing if model can load without duplicates...")
try:
    from test_8k_LAM import LAM
    model = LAM(checkpoint_path=test_checkpoint_path, device='cpu')
    print("   ✅ Model loaded successfully!")
    
    # Test inference
    test_sentences = ["Hello world", "Test sentence"]
    embeddings = model.encode(test_sentences)
    print(f"   ✅ Inference works! Embeddings shape: {embeddings.shape}")
    
    # Check which embeddings are actually used
    print(f"\n5. Checking which embeddings the model uses...")
    model_state = model.model.state_dict()
    has_teacher_emb = any('teacher_model.embeddings' in k for k in model_state.keys())
    has_standalone_emb = any(k.startswith('embeddings.') and 'teacher_model' not in k for k in model_state.keys())
    print(f"   Model has teacher_model.embeddings: {has_teacher_emb}")
    print(f"   Model has standalone embeddings: {has_standalone_emb}")
    
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"Duplicates removed: {len(duplicates_to_remove)} keys")
print(f"Parameters saved: {original_params - new_params:,} ({original_params - new_params:.2f}M)")
print(f"File size reduction: {original_file_size - test_file_size:.2f} MB")
print("="*80)

