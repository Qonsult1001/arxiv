import torch

ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_8k/checkpoint_167000.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Check what's in the checkpoint
if 'deltanet_layers' in ckpt:
    state_dict = ckpt['deltanet_layers']
else:
    state_dict = ckpt

print("All keys in checkpoint:")
for k in sorted(state_dict.keys()):
    print(f"  {k}: {state_dict[k].shape if hasattr(state_dict[k], 'shape') else type(state_dict[k])}")
