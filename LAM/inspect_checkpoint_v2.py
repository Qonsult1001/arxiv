import torch
import sys

ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_8k/checkpoint_167000.pt"
try:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    
    if 'model_state_dict' in ckpt:
        print("Found 'model_state_dict'. Keys inside (first 10):")
        print(list(ckpt['model_state_dict'].keys())[:10])
        
        state_dict = ckpt['model_state_dict']
        if 'embeddings.word_embeddings.weight' in state_dict:
            print("✅ Found 'embeddings.word_embeddings.weight' in model_state_dict")
        else:
            print("❌ 'embeddings.word_embeddings.weight' NOT found in model_state_dict")
            
except Exception as e:
    print(f"Error loading checkpoint: {e}")
