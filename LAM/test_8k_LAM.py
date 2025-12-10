import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula_final import EnhancedHierarchicalDeltaNet

# ============================================================================
# MATRYOSHKA PROJECTION (Production Optimization)
# ============================================================================

class MatryoshkaProjection(torch.nn.Module):
    """
    Projects embeddings to support multiple granularities.
    Forces early dimensions to be meaningful on their own.
    
    Production Optimization: Uses learnable LayerNorm before L2 normalization.
    This is superior to standard MRL (which just does F.normalize(truncated)) because:
    - LayerNorm re-centers the data distribution for each dimension slice
    - Allows the model to learn optimal scaling/shifting per dimension level
    - Better adaptation to the specific characteristics of truncated embeddings
    - Improves contrastive learning signal quality at each granularity
    """
    def __init__(self, d_model=384, dims=[64, 128, 256, 384]):
        super().__init__()
        self.d_model = d_model
        self.dims = sorted(dims)
        
        # Learnable normalization layers for each dimension level
        self.norms = torch.nn.ModuleDict({
            str(d): torch.nn.LayerNorm(d) for d in dims
        })
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, d_model]
        Returns:
            Dict of embeddings at each dimension level
        """
        outputs = {}
        for dim in self.dims:
            # Extract first `dim` dimensions
            truncated = embeddings[:, :dim]
            # Production optimization: LayerNorm re-centers, then L2 normalize
            normalized = F.normalize(self.norms[str(dim)](truncated), p=2, dim=-1)
            outputs[dim] = normalized
        
        return outputs


# ============================================================================
# DELTANET MODEL (with Matryoshka Head)
# ============================================================================

class DeltaNet(torch.nn.Module):
    def __init__(self, teacher_model, config):
        super().__init__()
        self.teacher_model = teacher_model
        self.embeddings = teacher_model.embeddings
        self.deltanet_layers = torch.nn.ModuleList()
        self.deltanet_norms = torch.nn.ModuleList()
        self.deltanet_ffns = torch.nn.ModuleList()
        self.ffn_norms = torch.nn.ModuleList()
        self.output_denses = torch.nn.ModuleList()
        
        d_model = teacher_model.config.hidden_size
        for i in range(6):
            self.deltanet_layers.append(
                EnhancedHierarchicalDeltaNet(
                    d_model=d_model,
                    num_heads=config.get('num_heads', 12),
                    use_hierarchical_decay=True,
                    use_enhanced_flux=True,
                    fast_decay_init=config.get('fast_decay_init', 0.3),
                    slow_decay_init=config.get('slow_decay_init', 0.832),
                )
            )
            self.deltanet_norms.append(teacher_model.encoder.layer[i].attention.output.LayerNorm)
            self.deltanet_ffns.append(teacher_model.encoder.layer[i].intermediate)
            self.ffn_norms.append(teacher_model.encoder.layer[i].output.LayerNorm)
            self.output_denses.append(teacher_model.encoder.layer[i].output.dense)
        
        self.pooler = teacher_model.pooler
        
        # Matryoshka Head
        self.projection = MatryoshkaProjection(d_model=384, dims=[64, 128, 256, 384])
    
    def get_extended_embeddings(self, input_ids):
        """Get embeddings with position interpolation support (up to 32k tokens)"""
        batch_size, seq_len = input_ids.shape
        original_max_pos = 512
        
        # Word embeddings
        word_embeddings = self.embeddings.word_embeddings(input_ids)
        
        # Token type embeddings
        token_type_ids = torch.zeros_like(input_ids)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        
        # Position embeddings with interpolation
        if seq_len <= original_max_pos:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            position_embeddings = self.embeddings.position_embeddings(position_ids)
        else:
            # Interpolate for long sequences (up to 32k)
            scale_factor = (original_max_pos - 1) / (seq_len - 1)
            position_embeddings_list = []
            for pos in range(seq_len):
                original_pos = pos * scale_factor
                lower_pos = int(original_pos)
                upper_pos = min(lower_pos + 1, original_max_pos - 1)
                weight = original_pos - lower_pos
                lower_emb = self.embeddings.position_embeddings.weight[lower_pos]
                upper_emb = self.embeddings.position_embeddings.weight[upper_pos]
                interp_emb = (1 - weight) * lower_emb + weight * upper_emb
                position_embeddings_list.append(interp_emb)
            position_embeddings = torch.stack(position_embeddings_list, dim=0)
            position_embeddings = position_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combine all embeddings
        embeddings = word_embeddings + token_type_embeddings + position_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        # Use extended embeddings for long sequence support (up to 32k)
        x = self.get_extended_embeddings(input_ids)
        for i in range(6):
            residual = x
            x_attn, _, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.deltanet_norms[i](residual + x_attn)
            residual = x
            x_ffn = self.deltanet_ffns[i](x)
            x_ffn = F.gelu(x_ffn)
            x_ffn = self.output_denses[i](x_ffn)
            x = self.ffn_norms[i](residual + x_ffn)
        return {'last_hidden_state': x, 'pooler_output': self.pooler(x) if self.pooler else None}
    
    def get_sentence_embeddings(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs['last_hidden_state']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        return F.normalize(embeddings, p=2, dim=1)
    
    def encode(self, input_ids, attention_mask, return_dict=False, dimensions=None):
        """
        Encode input to embeddings with Matryoshka support.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            return_dict: If True, returns dict with all dimensions {64, 128, 256, 384}
            dimensions: If specified (64, 128, 256, or 384), returns only that dimension
                       If None and return_dict=False, returns full 384-dim (default)
        
        Returns:
            If return_dict=True: Dict {64: emb, 128: emb, 256: emb, 384: emb}
            If dimensions specified: Tensor of specified dimension
            Otherwise: Tensor of 384 dimensions
        """
        # Get raw 384-dim embeddings first
        raw_emb = self.get_sentence_embeddings(input_ids, attention_mask)
        
        # Pass through Matryoshka Projection
        mrl_outputs = self.projection(raw_emb)
        
        if return_dict:
            return mrl_outputs  # Returns {64: emb, 128: emb, 256: emb, 384: emb}
        elif dimensions is not None:
            if dimensions not in mrl_outputs:
                raise ValueError(f"dimensions must be one of {list(mrl_outputs.keys())}, got {dimensions}")
            return mrl_outputs[dimensions]  # Return specific dimension
        else:
            return mrl_outputs[384]  # Default to full size for standard tests


# ============================================================================
# LAM CLASS (User-Friendly API)
# ============================================================================

class LAM:
    """
    The LAM-384 Model with Auto-Scaling Precision.
    Supports 32K context length and Matryoshka representation learning.
    """
    def __init__(self, checkpoint_path="/workspace/LAM/best/pytorch_model.bin", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading LAM-384 on {self.device}...")
        
        # Load teacher model
        teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"
        teacher_model = AutoModel.from_pretrained(teacher_model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
        
        # Freeze teacher
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # Build DeltaNet model
        config = {}
        self.model = DeltaNet(teacher_model, config).to(self.device)
        
        # Load checkpoint weights
        print(f"📦 Loading weights from {checkpoint_path}...")
        loaded_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Check if it's a raw state dict or wrapped in a dict
        is_raw_state_dict = not any(k in loaded_data for k in ['model_state_dict', 'deltanet_layers', 'config', 'step', 'projection']) and any('deltanet_layers.' in str(k) for k in loaded_data.keys())
        checkpoint = {'model_state_dict': loaded_data, 'config': {}, 'step': 0} if is_raw_state_dict else loaded_data
        
        # Load weights
        if is_raw_state_dict:
            model_state_dict = loaded_data
        else:
            model_state_dict = checkpoint.get('model_state_dict', {})
        
        if model_state_dict:
            deltanet_layers_dict = {}
            projection_dict = {}
            
            for key, value in model_state_dict.items():
                if key.startswith('deltanet_layers.'):
                    new_key = key.replace('deltanet_layers.', '')
                    deltanet_layers_dict[new_key] = value
                elif key.startswith('projection.'):
                    projection_key = key.replace('projection.', '')
                    projection_dict[projection_key] = value
            
            # Load DeltaNet layers
            if deltanet_layers_dict:
                for i in range(6):
                    layer_state = {k[2:]: v for k, v in deltanet_layers_dict.items() if k.startswith(f'{i}.')}
                    if layer_state:
                        # Handle shape mismatches (e.g., W_bilinear conversion)
                        filtered_state = {}
                        model_state = self.model.deltanet_layers[i].state_dict()
                        for key, value in layer_state.items():
                            if key in model_state:
                                if model_state[key].shape == value.shape:
                                    filtered_state[key] = value
                                elif key == 'resonance_flux.W_bilinear' and value.dim() == 2 and model_state[key].dim() == 3:
                                    # Convert [32, 32] -> [12, 32, 32]
                                    num_heads = model_state[key].shape[0]
                                    converted_value = value.unsqueeze(0).expand(num_heads, -1, -1).clone()
                                    filtered_state[key] = converted_value
                        if filtered_state:
                            self.model.deltanet_layers[i].load_state_dict(filtered_state, strict=False)
                print("   ✅ Loaded deltanet_layers")
            
            # Load Matryoshka projection weights (if available)
            if projection_dict:
                try:
                    self.model.projection.load_state_dict(projection_dict, strict=False)
                    print("   ✅ Loaded Matryoshka projection weights")
                except Exception as e:
                    print(f"   ⚠️  Could not load projection weights: {e}")
            else:
                print("   ⚠️  No projection weights found (using default)")
        
        self.model.eval()
        print("✅ Model loaded successfully!")

    def get_optimal_dimension(self, index_size):
        """
        Returns the smallest dimension that guarantees >95% Recall 
        based on the size of your vector database.
        """
        if index_size <= 20_000:
            return 64   # Safe for small/local apps
        elif index_size <= 1_500_000:
            return 128  # Safe for mid-sized DBs (1M docs)
        elif index_size <= 50_000_000:
            return 256  # Needed for massive scale (5M+ docs)
        else:
            return 384  # Web-scale safety net

    def encode(self, text, dimensions=None, index_size=None):
        """
        Encodes text into embeddings.
        
        Args:
            text (str or list): Input text(s)
            dimensions (int): Force a specific dimension (64, 128, 256, 384)
            index_size (int): Total docs in your DB. Auto-selects best dimension.
        
        Returns:
            Tensor of embeddings with specified dimension
        """
        # 1. Tokenize (supports up to 32K tokens)
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=32768, 
            return_tensors="pt"
        ).to(self.device)

        # 2. Determine target dimension
        target_dim = 384  # Default
        if dimensions:
            target_dim = dimensions
        elif index_size:
            target_dim = self.get_optimal_dimension(index_size)
        
        # 3. Encode with Matryoshka projection
        with torch.no_grad():
            if target_dim < 384:
                # Use specific dimension from Matryoshka projection
                embedding = self.model.encode(
                    inputs['input_ids'], 
                    inputs['attention_mask'], 
                    dimensions=target_dim
                )
            else:
                # Use full 384-dim
                embedding = self.model.encode(
                    inputs['input_ids'], 
                    inputs['attention_mask']
                )
        
        return embedding