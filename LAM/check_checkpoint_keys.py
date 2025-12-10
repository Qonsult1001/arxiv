import torch

ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEWNEWFINAL/pytorch_model.bin"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Find all keys related to norms and FFNs
norm_keys = [k for k in ckpt.keys() if 'norm' in k.lower()]
ffn_keys = [k for k in ckpt.keys() if 'ffn' in k.lower() or 'intermediate' in k.lower() or 'output.dense' in k]

print("Norm-related keys in checkpoint:")
for k in sorted(norm_keys)[:20]:
    print(f"  {k}")

print("\nFFN-related keys in checkpoint:")
for k in sorted(ffn_keys)[:20]:
    print(f"  {k}")
