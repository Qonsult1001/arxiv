import torch

ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_8k/checkpoint_167000.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

print("Checkpoint metadata:")
for k in ckpt.keys():
    if k != 'deltanet_layers':
        print(f"  {k}: {ckpt[k]}")
