import torch

ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/pytorch_model.bin"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("Top-level keys in pytorch_model.bin:")
print(list(ckpt.keys())[:20])

# Check if it's a direct state dict or nested
if 'deltanet_layers' in ckpt:
    print("\nFound 'deltanet_layers' key")
    print(f"First 5 keys in deltanet_layers: {list(ckpt['deltanet_layers'].keys())[:5]}")
else:
    print("\nNo 'deltanet_layers' key - this is a direct state dict")
    print(f"First 10 keys: {list(ckpt.keys())[:10]}")
