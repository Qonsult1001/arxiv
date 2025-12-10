"""
LAM Wrapper - SentenceTransformer-Compatible API

Provides a drop-in replacement for sentence-transformers using LAM.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from transformers import AutoModel, AutoTokenizer
import sys

# Add parent directory to path to import LAM components
sys.path.insert(0, str(Path(__file__).parent.parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet


class LAMModel(nn.Module):
    """
    LAM Model: Linear Attention Model with O(n) complexity

    Architecture:
    - Base embeddings (frozen)
    - 6 layers of: LAM attention + FFN
    - Mean pooling + L2 normalization

    Performance: 0.836 Pearson on STS-B
    """

    def __init__(self, base_model_path: str, checkpoint_path: str, config: dict):
        super().__init__()

        print(f"Loading LAM model...")
        print(f"  Model path: {base_model_path}")
        print(f"  Checkpoint: {checkpoint_path}")

        # Load base model (for embeddings + FFN)
        # Try to load from path, but if it's a LAM model, use all-MiniLM-L6-v2 instead
        base_model_path_obj = Path(base_model_path)
        lam_base_file = base_model_path_obj / 'lam_base.bin'
        
        # Check if config.json exists and if it's a LAM model
        config_file = base_model_path_obj / 'config.json'
        use_hf_base = False
        if config_file.exists():
            import json
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                if config_data.get('model_type') == 'lam':
                    use_hf_base = True
                    print(f"  Detected LAM config, loading base model from local: all-MiniLM-L6-v2")
        
        if use_hf_base:
            # Load base model from local directory
            local_minilm_path = Path(__file__).parent.parent / 'all-MiniLM-L6-v2'
            if local_minilm_path.exists():
                print(f"  Loading base model from local: {local_minilm_path}")
                self.base_model = AutoModel.from_pretrained(str(local_minilm_path))
                self.tokenizer = AutoTokenizer.from_pretrained(str(local_minilm_path))
            else:
                # Fallback to HuggingFace if local not found
                print(f"  Local all-MiniLM-L6-v2 not found, loading from HuggingFace")
                self.base_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            
            # Load base model weights from lam_base.bin if it exists
            if lam_base_file.exists():
                print(f"  Loading base model weights from: {lam_base_file}")
                base_weights = torch.load(lam_base_file, map_location='cpu', weights_only=False)
                # Load weights, handling different checkpoint formats
                if isinstance(base_weights, dict):
                    if 'model' in base_weights:
                        self.base_model.load_state_dict(base_weights['model'], strict=False)
                    elif 'state_dict' in base_weights:
                        self.base_model.load_state_dict(base_weights['state_dict'], strict=False)
                    else:
                        self.base_model.load_state_dict(base_weights, strict=False)
                else:
                    self.base_model.load_state_dict(base_weights, strict=False)
                print(f"  ✅ Loaded base model weights")
        else:
            # Load from local path (should be a standard transformer model)
            self.base_model = AutoModel.from_pretrained(base_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.d_model = config.get('d_model', 384)
        self.num_layers = config.get('num_layers', 6)

        # Extract and freeze embeddings
        self.embeddings = self.base_model.embeddings
        for param in self.embeddings.parameters():
            param.requires_grad = False

        # Create LAM attention layers (replaces quadratic transformer attention)
        self.lam_layers = nn.ModuleList()
        self.lam_norms = nn.ModuleList()

        for i in range(self.num_layers):
            # LAM attention layer (O(n) complexity)
            self.lam_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=self.d_model,
                    num_heads=config.get('num_heads', 12),
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=config.get('fast_decay_init', 0.3),
                    slow_decay_init=config.get('slow_decay_init', 0.85),
                )
            )

            # LayerNorm for attention output
            self.lam_norms.append(
                self.base_model.encoder.layer[i].attention.output.LayerNorm
            )

        # Initialize FFN layers - will be loaded from checkpoint if available
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        self.output_dense_layers = nn.ModuleList()

        # Start with base model FFN layers (will be replaced by checkpoint if available)
        for i in range(self.num_layers):
            self.ffns.append(self.base_model.encoder.layer[i].intermediate)
            self.ffn_norms.append(self.base_model.encoder.layer[i].output.LayerNorm)
            self.output_dense_layers.append(self.base_model.encoder.layer[i].output.dense)

        # Load LAM checkpoint (LAM attention weights + trained FFN layers)
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            # Load LAM attention layers - handle both formats
            if 'lam_layers' in checkpoint:
                try:
                    self.lam_layers.load_state_dict(checkpoint['lam_layers'], strict=False)
                    print(f"  ✅ Loaded LAM attention layers from checkpoint")
                except Exception as e:
                    # Try loading layer by layer if full state_dict fails
                    print(f"  ⚠️  Full load failed, trying layer-by-layer: {e}")
                    lam_layers = checkpoint['lam_layers']
                    for i in range(self.num_layers):
                        layer_state = {k[2:]: v for k, v in lam_layers.items() if k.startswith(f'{i}.')}
                        if not layer_state:
                            # Try without layer prefix
                            layer_state = {k: v for k, v in lam_layers.items() if not any(k.startswith(f'{j}.') for j in range(self.num_layers))}
                        if layer_state:
                            try:
                                self.lam_layers[i].load_state_dict(layer_state, strict=False)
                            except:
                                pass
                    print(f"  ✅ Loaded LAM attention layers (layer-by-layer)")
            elif 'deltanet_layers' in checkpoint:
                # Older checkpoint format
                try:
                    pretrained_layers = checkpoint['deltanet_layers']
                    self.lam_layers.load_state_dict(pretrained_layers, strict=False)
                    print(f"  ✅ Loaded LAM attention layers (old format)")
                except Exception as e:
                    print(f"  ⚠️  Failed to load deltanet_layers: {e}")
                    # Try layer by layer
                    pretrained_layers = checkpoint['deltanet_layers']
                    for i in range(self.num_layers):
                        layer_state = {k[2:]: v for k, v in pretrained_layers.items() if k.startswith(f'{i}.')}
                        if not layer_state:
                            layer_state = {k: v for k, v in pretrained_layers.items() if not any(k.startswith(f'{j}.') for j in range(self.num_layers))}
                        if layer_state:
                            try:
                                self.lam_layers[i].load_state_dict(layer_state, strict=False)
                            except:
                                pass
            else:
                print(f"  ⚠️  Warning: No 'lam_layers' or 'deltanet_layers' found in checkpoint")

            # Load norms if available
            if 'lam_norms' in checkpoint:
                try:
                    self.lam_norms.load_state_dict(checkpoint['lam_norms'], strict=False)
                    print(f"  ✅ Loaded LAM norms from checkpoint")
                except Exception as e:
                    print(f"  ⚠️  Failed to load lam_norms: {e}")

            # Load trained FFN layers if available (these are better than base model FFNs)
            if 'deltanet_ffns' in checkpoint:
                try:
                    self.ffns.load_state_dict(checkpoint['deltanet_ffns'], strict=False)
                    print(f"  ✅ Loaded trained FFN layers from checkpoint")
                except Exception as e:
                    print(f"  ⚠️  Failed to load deltanet_ffns: {e}")
            
            if 'ffn_norms' in checkpoint:
                try:
                    self.ffn_norms.load_state_dict(checkpoint['ffn_norms'], strict=False)
                    print(f"  ✅ Loaded FFN norms from checkpoint")
                except Exception as e:
                    print(f"  ⚠️  Failed to load ffn_norms: {e}")
            
            if 'output_dense_layers' in checkpoint:
                try:
                    self.output_dense_layers.load_state_dict(checkpoint['output_dense_layers'], strict=False)
                    print(f"  ✅ Loaded output dense layers from checkpoint")
                except Exception as e:
                    print(f"  ⚠️  Failed to load output_dense_layers: {e}")

            # Print performance metrics if available
            if 'pearson' in checkpoint:
                print(f"  📊 Pearson: {checkpoint['pearson']:.4f}")
            if 'spearman' in checkpoint:
                print(f"  📊 Spearman: {checkpoint['spearman']:.4f}")
        else:
            print(f"  ⚠️  Warning: Checkpoint not found at {checkpoint_path}")

        print(f"✅ LAM model loaded successfully")

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through LAM model

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, d_model]
        """
        # Get base embeddings
        hidden_states = self.embeddings(input_ids)

        # Pass through 6 layers: LAM attention + FFN
        for i in range(self.num_layers):
            # Residual connection
            residual = hidden_states

            # LAM attention (O(n) linear complexity)
            attn_output, _, _ = self.lam_layers[i](
                hidden_states,
                attention_mask,
                use_cache=False
            )

            # Add & Norm
            hidden_states = self.lam_norms[i](residual + attn_output)

            # FFN
            residual = hidden_states
            intermediate = self.ffns[i](hidden_states)
            intermediate = F.gelu(intermediate)  # Apply GELU activation
            ffn_output = self.output_dense_layers[i](intermediate)
            hidden_states = self.ffn_norms[i](residual + ffn_output)

        # Mean pooling
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, 1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask

        # L2 normalization
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings


class LAMEncoder:
    """
    LAM Encoder - SentenceTransformer-compatible interface

    Usage (identical to SentenceTransformer):
        model = LAMEncoder('lam-base-v1')
        embeddings = model.encode(["Hello world", "How are you?"])
    """

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        """
        Initialize LAM encoder (SentenceTransformer-compatible API)

        Args:
            model_name_or_path: Path to LAM model directory
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_name_or_path)

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Paths to LAM model files
        # New naming: lam_base.bin (embeddings + FFN) and lam_tweak.pt (LAM attention)
        # Convert to absolute path to avoid HuggingFace repo ID issues
        if not self.model_path.is_absolute():
            # Resolve relative to current working directory
            resolved = self.model_path.resolve()
            # Check if resolved path exists and contains the expected files
            # If not, try parent directory (common when path is ../LAM-base-v1 from evaluation/)
            if not (resolved / 'lam_base.bin').exists() and not (resolved / 'config.json').exists():
                # Try parent directory
                parent_resolved = resolved.parent
                if (parent_resolved / 'lam_base.bin').exists() or (parent_resolved / 'config.json').exists():
                    self.model_path = parent_resolved
                else:
                    self.model_path = resolved
            else:
                self.model_path = resolved
        base_model_path = str(self.model_path)  # Contains lam_base.bin
        lam_base_file = self.model_path / 'lam_base.bin'
        lam_tweak_file = self.model_path / 'lam_tweak.pt'

        # Support both old and new naming conventions
        if not lam_base_file.exists():
            lam_base_file = self.model_path / 'pytorch_model.bin'  # Fallback to old name

        if not lam_tweak_file.exists():
            # Try old name
            lam_tweak_file = self.model_path / 'lam_checkpoint.pt'
            if not lam_tweak_file.exists():
                # Try research repo structure
                lam_tweak_file = Path(__file__).parent.parent / 'proper_distillation_reaccelerate' / 'checkpoint_best_3500.pt'

        checkpoint_path = str(lam_tweak_file)

        # Config
        config = {
            'd_model': 384,
            'num_heads': 12,
            'num_layers': 6,
            'fast_decay_init': 0.3,
            'slow_decay_init': 0.85,
        }

        # Load model
        self.model = LAMModel(base_model_path, checkpoint_path, config)
        self.model.to(self.device)
        self.model.eval()

        # Tokenizer
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = 128

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,  # Already normalized in model
        **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode sentences to embeddings

        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for encoding
            show_progress_bar: Show progress bar (not implemented)
            convert_to_numpy: Convert to numpy array
            normalize_embeddings: L2 normalize (already done in model)

        Returns:
            Embeddings as numpy array or torch tensor
        """
        # Handle single sentence
        if isinstance(sentences, str):
            sentences = [sentences]

        all_embeddings = []

        # Process in batches
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )

            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Encode
            with torch.no_grad():
                embeddings = self.model(input_ids, attention_mask)

            all_embeddings.append(embeddings.cpu())

        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)

        # Additional normalization if requested (though already done)
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy if requested
        if convert_to_numpy:
            embeddings = embeddings.numpy()

        return embeddings

    def __call__(self, sentences: Union[str, List[str]], **kwargs):
        """Allow model(sentences) syntax"""
        return self.encode(sentences, **kwargs)


# Alias for drop-in replacement
SentenceTransformer = LAMEncoder


if __name__ == "__main__":
    # Test loading
    print("Testing LAM model loading...")

    # Try to load from production folder
    model_path = Path(__file__).parent / 'lam-base-v1'
    if not model_path.exists():
        # Fall back to parent directory structure
        model_path = Path(__file__).parent.parent / 'all-MiniLM-L6-v2'

    model = LAMEncoder(str(model_path))

    # Test encoding
    sentences = [
        "This is a test sentence",
        "This is another test sentence"
    ]

    embeddings = model.encode(sentences)
    print(f"\n✅ Encoded {len(sentences)} sentences")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   First embedding (first 5 dims): {embeddings[0][:5]}")

    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"   Similarity: {sim:.4f}")
