"""
Simple script to load your trained model for inference
Same pattern as existing model: Load base all-MiniLM + your trained layers on top

After training, you only need:
1. Base model: all-MiniLM-L6-v2
2. Your checkpoint .pt file (with trained DeltaNet layers)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load your trained model for inference - same as existing model pattern
    
    Args:
        checkpoint_path: Path to your .pt checkpoint file (e.g., "proper_distillation/checkpoint_best_2000.pt")
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model ready for inference
        tokenizer: Tokenizer for the model
    """
    # 1. Load base model (all-MiniLM-L6-v2) - same as existing model
    base_minilm = "/workspace/LAM/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(base_minilm)
    base_model = AutoModel.from_pretrained(base_minilm)
    d_model = base_model.config.hidden_size
    
    # 2. Build model: base model + your trained DeltaNet layers on top
    class TrainedModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            # Use base model's embeddings (same as existing)
            self.embeddings = base_model.embeddings
            
            # Your trained DeltaNet layers (replacing base model's transformer layers)
            self.deltanet_layers = nn.ModuleList()
            self.deltanet_norms = nn.ModuleList()
            self.deltanet_ffns = nn.ModuleList()
            self.ffn_norms = nn.ModuleList()
            self.output_dense_layers = nn.ModuleList()
            
            # Initialize with base model structure, then load your trained weights
            for i in range(6):
                self.deltanet_layers.append(
                    EnhancedHierarchicalDeltaNet(
                        d_model=d_model, num_heads=12,
                        use_hierarchical_decay=True, use_enhanced_flux=True,
                        fast_decay_init=0.3, slow_decay_init=0.85,
                    )
                )
                # Use base model's norms and FFNs (same as existing)
                self.deltanet_norms.append(base_model.encoder.layer[i].attention.output.LayerNorm)
                self.deltanet_ffns.append(base_model.encoder.layer[i].intermediate)
                self.ffn_norms.append(base_model.encoder.layer[i].output.LayerNorm)
                self.output_dense_layers.append(base_model.encoder.layer[i].output.dense)
            
            # Use base model's pooler (same as existing)
            self.pooler = base_model.pooler
        
        def forward(self, input_ids, attention_mask=None):
            # Same forward pass as existing model
            x = self.embeddings(input_ids=input_ids)
            for i in range(6):
                residual = x
                x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
                x = self.deltanet_norms[i](residual + x_attn)
                residual = x
                x_ffn = self.deltanet_ffns[i](x)
                x_ffn = F.gelu(x_ffn)
                x_ffn = self.output_dense_layers[i](x_ffn)
                x = self.ffn_norms[i](residual + x_ffn)
            return self.pooler(x)
    
    # Build model with base model
    model = TrainedModel(base_model)
    
    # 3. Load your trained checkpoint (only the DeltaNet layers you trained)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load your trained layers (same as existing model loading pattern)
    if 'lam_layers' in checkpoint:
        model.deltanet_layers.load_state_dict(checkpoint['lam_layers'], strict=False)
        print("✅ Loaded trained DeltaNet layers (lam_layers)")
    
    if 'lam_norms' in checkpoint:
        model.deltanet_norms.load_state_dict(checkpoint['lam_norms'], strict=False)
        print("✅ Loaded trained norms (lam_norms)")
    
    if 'deltanet_ffns' in checkpoint:
        model.deltanet_ffns.load_state_dict(checkpoint['deltanet_ffns'], strict=False)
        print("✅ Loaded trained FFNs (deltanet_ffns)")
    
    if 'ffn_norms' in checkpoint:
        model.ffn_norms.load_state_dict(checkpoint['ffn_norms'], strict=False)
        print("✅ Loaded trained FFN norms (ffn_norms)")
    
    if 'output_dense_layers' in checkpoint:
        model.output_dense_layers.load_state_dict(checkpoint['output_dense_layers'], strict=False)
        print("✅ Loaded trained output dense layers")
    
    # Note: Projections are NOT needed for inference - they were only for training
    # (to match teacher dimensions during distillation)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    step = checkpoint.get('step', 'unknown')
    pearson = checkpoint.get('pearson', 'unknown')
    print(f"\n✅ Model loaded: Base all-MiniLM + Your trained layers from step {step}")
    print(f"   Performance: Pearson {pearson}")
    
    return model, tokenizer


# Example usage:
if __name__ == "__main__":
    # Load your best checkpoint
    checkpoint_path = "proper_distillation_reaccelerate/checkpoint_best_2000.pt"
    
    model, tokenizer = load_trained_model(checkpoint_path, device='cuda')
    
    # Example: Get embeddings for two sentences
    sentences = [
        "The cat sat on the mat",
        "A feline was sitting on the rug"
    ]
    
    # Tokenize
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=256)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        embeddings = model(**inputs)
    
    # Compute similarity
    similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2])
    print(f"\nSimilarity: {similarity.item():.4f}")

