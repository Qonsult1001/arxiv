import torch
import sys

ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_8k/checkpoint_167000.pt"
try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    if 'embeddings.word_embeddings.weight' in ckpt:
        print("Found 'embeddings.word_embeddings.weight'")
    if 'embeddings.weight' in ckpt:
        print("Found 'embeddings.weight'")
    
    # Check if it's a nested dict (e.g. state_dict inside)
    if 'state_dict' in ckpt:
        print("Found 'state_dict' key. Keys inside:")
        print(list(ckpt['state_dict'].keys())[:10])
    
    # Print first 20 keys to see structure
    print("First 20 keys in checkpoint:")
    for k in list(ckpt.keys())[:20]:
        print(k)
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
