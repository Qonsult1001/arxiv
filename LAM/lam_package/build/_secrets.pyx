# cython: language_level=3

import torch

cdef int FREE_TIER_LIMIT = 0x2000

def interpolate_positions(
    position_embedding_weight,
    int seq_length,
    int original_max_pos=512,
    device="cpu",
    int license_limit=0x2000
):
    """
    Interpolate position embeddings.
    """
    
    if seq_length > license_limit:
        raise ValueError("CONTEXT_LIMIT_EXCEEDED")
    
    if seq_length <= original_max_pos:
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        return position_embedding_weight[position_ids]
    
    cdef float scale_factor = (original_max_pos - 1) / (seq_length - 1)
    position_embeddings_list = []
    
    cdef int pos
    cdef float original_pos, weight
    cdef int lower_pos, upper_pos
    
    for pos in range(seq_length):
        original_pos = pos * scale_factor
        lower_pos = int(original_pos)
        upper_pos = min(lower_pos + 1, original_max_pos - 1)
        weight = original_pos - lower_pos
        
        lower_emb = position_embedding_weight[lower_pos]
        upper_emb = position_embedding_weight[upper_pos]
        interp_emb = (1 - weight) * lower_emb + weight * upper_emb
        position_embeddings_list.append(interp_emb)
    
    return torch.stack(position_embeddings_list, dim=0)

def truncate_embeddings(
    embeddings,
    int target_dim
):
    """
    Truncate embeddings to target dimension with normalization.
    Compiled to hide parameters and logic.
    """
    cdef int full_dim = 384
    cdef int batch_size = embeddings.shape[0]
    
    if target_dim >= full_dim:
        return embeddings
    
    if target_dim not in [64, 128, 256]:
        raise ValueError("INVALID_DIMENSION")
    
    truncated = embeddings[:, :target_dim]
    normalized = torch.nn.functional.normalize(truncated, p=2, dim=1)
    
    return normalized

