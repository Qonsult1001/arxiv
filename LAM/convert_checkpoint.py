import torch

# Load checkpoint
ckpt_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_8k/checkpoint_167000.pt"
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

state_dict = ckpt['deltanet_layers']

# Convert old flux_net format to new Sequential format
new_state_dict = {}
for k, v in state_dict.items():
    if 'flux_net.weight' in k:
        # Old: flux_net.weight -> New: flux_net.0.weight (first layer of Sequential)
        new_k = k.replace('flux_net.weight', 'flux_net.0.weight')
        new_state_dict[new_k] = v
        print(f"Converted: {k} -> {new_k}")
    elif 'flux_net.bias' in k:
        new_k = k.replace('flux_net.bias', 'flux_net.0.bias')
        new_state_dict[new_k] = v
        print(f"Converted: {k} -> {new_k}")
    elif 'token_flux_proj.weight' in k:
        new_k = k.replace('token_flux_proj.weight', 'token_flux_proj.0.weight')
        new_state_dict[new_k] = v
        print(f"Converted: {k} -> {new_k}")
    elif 'token_flux_proj.bias' in k:
        new_k = k.replace('token_flux_proj.bias', 'token_flux_proj.0.bias')
        new_state_dict[new_k] = v
        print(f"Converted: {k} -> {new_k}")
    else:
        new_state_dict[k] = v

# Save converted checkpoint
ckpt['deltanet_layers'] = new_state_dict
output_path = "/workspace/LAM/checkpoint_167000_converted.pt"
torch.save(ckpt, output_path)
print(f"\n✅ Saved converted checkpoint to: {output_path}")
print(f"   Test Spearman: {ckpt.get('test_spearman', 'N/A')}")
print(f"   Val Spearman: {ckpt.get('val_spearman', 'N/A')}")
