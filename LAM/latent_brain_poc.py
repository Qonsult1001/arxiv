import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
from einops import rearrange

class LatentBrainCore(nn.Module):
    """
    The 'Latent Brain' Engine (Hierarchical + Resonance + Continuous Growth).

    Features:
    1. Parallel Geometry: O(N) processing within blocks.
    2. Continuous Growth: Accepts previous brain state (M_prev) and evolves it.
    3. Hierarchical Memory: Fast/Slow geometric decay.
    4. Enhanced Resonance: Token-level bilinear interaction.

    This enables "Streaming Consciousness": processing 1M+ tokens by updating
    the fixed-size brain state continuously.
    """
    def __init__(self, dim: int, num_heads: int, expand_k: float = 1.0, expand_v: float = 1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.key_dim = int(dim * expand_k)
        self.value_dim = int(dim * expand_v)
        self.head_k = self.key_dim // num_heads
        self.head_v = self.value_dim // num_heads

        # Semantic Projections
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.value_dim, bias=False)

        # Resonance Flux (Bilinear)
        self.W_bilinear = nn.Parameter(torch.randn(num_heads, self.head_k, self.head_k) * 0.02)

        # Geometric Decay Slopes (Learnable)
        self.fast_decay_slope = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.5)
        self.slow_decay_slope = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.01)

        # State Decay Factors (For carrying state between blocks)
        # Determines how much of the OLD brain we keep when moving to new block
        # exp(-slope * block_length) roughly
        self.inter_block_decay_fast = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.1) # Fast forget
        self.inter_block_decay_slow = nn.Parameter(torch.ones(num_heads, 1, 1) * 0.9) # Slow forget

        # Static Knowledge Kernels
        self.S_fast = nn.Parameter(torch.randn(num_heads, self.head_k, self.head_k) * 0.05)
        self.S_slow = nn.Parameter(torch.randn(num_heads, self.head_k, self.head_k) * 0.02)

        # Flux Gating
        self.flux_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, num_heads),
            nn.Sigmoid()
        )

        # Output
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, brain_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Forward pass with State Passing.
        brain_state: (M_fast_prev, M_slow_prev) from previous block.
        """
        b, n, d = x.shape
        device = x.device

        # 1. Project
        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v = self.v_proj(x)

        q = rearrange(q_raw, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k_raw, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        # 2. Resonance Flux (Token-Level)
        resonance = torch.matmul(k, self.W_bilinear)
        resonance_score = torch.tanh(resonance)
        k_modulated = k * (1.0 + 0.5 * resonance_score)
        k_modulated = k_modulated / math.sqrt(self.head_k)

        # 3. Geometric Decay (Intra-Block)
        indices = torch.arange(n, device=device).float().view(1, 1, n, 1)
        fast_weights = torch.exp(self.fast_decay_slope * (indices - (n - 1)))
        slow_weights = torch.exp(self.slow_decay_slope * (indices - (n - 1)))

        k_fast = k_modulated * fast_weights
        k_slow = k_modulated * slow_weights

        # 4. Aggregate Current Block
        # M_current: [b, h, dk, dv]
        M_fast_curr = torch.matmul(k_fast.transpose(-1, -2), v)
        M_slow_curr = torch.matmul(k_slow.transpose(-1, -2), v)

        # 5. Evolve Brain State (Continuous Growth)
        # NewState = OldState * Decay + CurrentState
        if brain_state is not None:
            M_fast_prev, M_slow_prev = brain_state

            # Apply inter-block decay to previous state
            # This represents the time passed during the current block
            # (Simplified: Constant decay for the whole block transition)
            M_fast = M_fast_prev * self.inter_block_decay_fast + M_fast_curr
            M_slow = M_slow_prev * self.inter_block_decay_slow + M_slow_curr
        else:
            M_fast = M_fast_curr
            M_slow = M_slow_curr

        # 6. Apply Reasoning Kernels (Space Thinking)
        I = torch.eye(self.head_k, device=device).view(1, 1, self.head_k, self.head_k)
        M_fast_refined = torch.matmul(I + self.S_fast.unsqueeze(0), M_fast)
        M_slow_refined = torch.matmul(I + self.S_slow.unsqueeze(0), M_slow)

        # 7. Readout (Querying the Evolved Brain)
        o_fast = torch.matmul(q, M_fast_refined)
        o_slow = torch.matmul(q, M_slow_refined)

        # 8. Flux Gating
        x_pooled = x.mean(dim=1)
        psi = self.flux_net(x_pooled)
        psi_expanded = psi.view(b, self.num_heads, 1, 1)

        o_blend = (1.0 - psi_expanded) * o_fast + psi_expanded * o_slow

        # 9. Output
        o_blend = rearrange(o_blend, 'b h n d -> b n (h d)')
        out = self.o_proj(o_blend)

        # Return new brain state for next block
        new_brain_state = (M_fast.detach(), M_slow.detach())

        return out, new_brain_state, {"psi": psi, "M_fast_norm": M_fast.norm(), "M_slow_norm": M_slow.norm()}

class LatentBrainDeltaNet(nn.Module):
    """Wrapper to match DeltaNet interface."""
    def __init__(self, d_model: int = 384, num_heads: int = 12):
        super().__init__()
        self.core = LatentBrainCore(dim=d_model, num_heads=num_heads)

    def forward(self, hidden_states, attention_mask=None, past_key_values=None, **kwargs):
        """
        Supports passing 'past_key_values' which now holds the Brain State (M matrices).
        """
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1) # [b, n, 1]
            hidden_states = hidden_states * mask

        # Extract brain state if present
        brain_state = None
        if past_key_values is not None:
            # Assuming past_key_values is simply the brain state tuple for now
            # In HF format, this might be wrapped in a list/dict.
            # For direct usage, we assume simple tuple.
            if isinstance(past_key_values, tuple) and len(past_key_values) == 2:
                brain_state = past_key_values

        out, new_brain_state, info = self.core(hidden_states, brain_state=brain_state)

        # Return new_brain_state in past_key_values position
        return out, None, new_brain_state, info