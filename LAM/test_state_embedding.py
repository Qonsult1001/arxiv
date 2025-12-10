#!/usr/bin/env python3
"""
DIRECT TEST: Last Token vs Mean Pooling

We know mean pooling destroys the signal. Let's directly compare:
1. Mean pooling (current - broken)
2. Last token (should preserve end-of-document info)
3. Weighted average (exponential decay favoring recent tokens)
"""

import torch
import torch.nn.functional as F
import sys

sys.path.insert(0, '/workspace/LAM/lam_package')
from lam import LAM

def test_aggregation_methods():
    print("=" * 70)
    print("DIRECT TEST: Aggregation Methods Comparison")
    print("=" * 70)
    
    model = LAM('/workspace/LAM/best', device='cuda')
    
    # Disable truncation
    if hasattr(model.tokenizer, 'no_truncation'):
        model.tokenizer.no_truncation()
    
    # Create test document: noise + answer at end
    noise = "The quick brown fox jumps over the lazy dog. " * 100
    answer_text = "THE SECRET ANSWER IS BANANA AND THE PASSWORD IS 12345"
    doc = noise + answer_text
    
    print(f"\n📄 Document: {len(noise.split())} noise words + answer at END")
    
    # Tokenize
    enc = model.tokenizer.encode(doc)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    print(f"   Tokens: {len(ids)}")
    
    input_ids = torch.tensor([ids], dtype=torch.long, device='cuda')
    attention_mask = torch.ones_like(input_ids)
    
    # Get token-level embeddings by going through the model manually
    print("\n🔬 Processing through model layers...")
    
    with torch.no_grad():
        # Access internal model
        internal_model = model._model
        
        print("   Inspecting model structure...")
        print(f"   Model type: {type(internal_model)}")
        print(f"   Has embeddings: {hasattr(internal_model, 'embeddings')}")
        
        # Try to get embeddings - inspect what's available
        if hasattr(internal_model, 'embeddings'):
            emb_dict = internal_model.embeddings
            print(f"   Embeddings type: {type(emb_dict)}")
            if hasattr(emb_dict, 'keys'):
                keys = list(emb_dict.keys())
                print(f"   Embedding keys: {keys}")
                # Try common keys
                for key in ['word_embeddings', 'token_embeddings', 'embeddings']:
                    if key in keys:
                        print(f"   Using key: {key}")
                        hidden = emb_dict[key](input_ids)
                        break
                else:
                    # Use first key
                    if keys:
                        print(f"   Using first key: {keys[0]}")
                        hidden = emb_dict[keys[0]](input_ids)
                    else:
                        raise RuntimeError("No embedding keys found")
            else:
                # Try direct call
                print("   Trying direct embeddings call...")
                hidden = internal_model.embeddings(input_ids)
        else:
            raise RuntimeError("Model does not have embeddings attribute")
        
        print(f"   Initial hidden shape: {hidden.shape}")
        
        # Process through DeltaNet layers
        if hasattr(internal_model, 'deltanet_layers') and hasattr(internal_model, 'deltanet_norms'):
            print(f"   Processing through {len(internal_model.deltanet_layers)} layers...")
            for i, (layer, norm) in enumerate(zip(internal_model.deltanet_layers, internal_model.deltanet_norms)):
                residual = hidden
                hidden = norm(hidden)
                
                # Forward pass - try with use_cache to see if we can get state
                try:
                    output, attn, past_kv, ortho = layer(
                        hidden,
                        attention_mask=attention_mask,
                        use_cache=True,  # Try to get state
                        training=True,   # Force state computation
                    )
                    # Check if we got state and inspect it
                    if past_kv is not None:
                        print(f"   Layer {i}: Got past_kv (state available)")
                        if isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
                            state_info = past_kv[0] if isinstance(past_kv[0], dict) else past_kv[0]
                            if isinstance(state_info, dict):
                                print(f"      State keys: {list(state_info.keys())}")
                                for k, v in state_info.items():
                                    if isinstance(v, torch.Tensor):
                                        print(f"      {k}: shape {v.shape}")
                            elif isinstance(state_info, torch.Tensor):
                                print(f"      State shape: {state_info.shape}")
                        # Store state from last layer for analysis
                        if i == len(internal_model.deltanet_layers) - 1:
                            final_state = past_kv
                except Exception as e:
                    # Fallback without cache
                    output, attn, past_kv, ortho = layer(
                        hidden,
                        attention_mask=attention_mask,
                        use_cache=False
                    )
                
                hidden = residual + output
        else:
            raise RuntimeError("Model does not have deltanet_layers or deltanet_norms")
        
        # Now hidden has shape [1, seq_len, d_model]
        seq_len = hidden.shape[1]
        print(f"   Final hidden shape: {hidden.shape}")
        
        # Try to extract state-based embedding if we got final_state
        state_emb = None
        if 'final_state' in locals() and final_state is not None:
            print("\n   🔍 Attempting to extract state-based embedding...")
            try:
                # Try to extract S_slow or S_fast from state
                if isinstance(final_state, (list, tuple)) and len(final_state) > 0:
                    state_dict = final_state[0] if isinstance(final_state[0], dict) else {}
                    if 'recurrent_state' in state_dict:
                        recurrent_state = state_dict['recurrent_state']
                        if isinstance(recurrent_state, tuple) and len(recurrent_state) == 2:
                            S_fast, S_slow = recurrent_state
                            if S_slow is not None:
                                print(f"      ✅ Found S_slow: shape {S_slow.shape}")
                                # S_slow shape: [batch, heads, d_k, d_v] = [1, 12, 32, 32]
                                # Total information: 12 × 32 × 32 = 12,288 values
                                # We need to preserve as much as possible and project to 384
                                
                                print(f"      Testing different aggregation methods...")
                                
                                # Method 1: Flatten all heads and use learned projection
                                # [1, 12, 32, 32] -> [1, 12288] -> project to [1, 384]
                                state_full = S_slow.flatten(1)  # [1, 12288]
                                
                                # Simple projection: average pooling with stride
                                # 12288 / 384 = 32, so we can average every 32 values
                                if state_full.shape[1] >= 384:
                                    # Reshape to [1, 384, 32] and mean pool
                                    pool_size = state_full.shape[1] // 384
                                    remainder = state_full.shape[1] % 384
                                    if remainder == 0:
                                        # Perfect division: reshape and mean
                                        state_reshaped = state_full.view(1, 384, pool_size)
                                        state_emb_v1 = state_reshaped.mean(dim=2)  # [1, 384]
                                    else:
                                        # Not perfect: use adaptive pooling
                                        # Take first 384 * pool_size values, reshape, mean
                                        n_keep = 384 * pool_size
                                        state_kept = state_full[:, :n_keep].view(1, 384, pool_size)
                                        state_emb_v1 = state_kept.mean(dim=2)
                                else:
                                    # Pad and then pool
                                    padding = torch.zeros(1, 384 - state_full.shape[1], device=state_full.device)
                                    state_padded = torch.cat([state_full, padding], dim=1)
                                    state_emb_v1 = state_padded
                                
                                # Method 2: Weighted combination across heads (each head contributes)
                                # [1, 12, 32, 32] -> mean over d_k and d_v for each head -> [1, 12]
                                # Then expand to 384 by repeating/interpolating
                                state_per_head = S_slow.mean(dim=(2, 3))  # [1, 12] - one value per head
                                # Expand 12 -> 384: repeat each value 32 times
                                state_emb_v2 = state_per_head.repeat_interleave(32, dim=1)  # [1, 384]
                                
                                # Method 3: Concatenate head matrices, then pool
                                # [1, 12, 32, 32] -> [1, 12*32, 32] -> pool to [1, 384, 32] -> mean -> [1, 384]
                                state_concat = S_slow.view(1, 12*32, 32)  # [1, 384, 32]
                                state_emb_v3 = state_concat.mean(dim=2)  # [1, 384]
                                
                                # Method 4: Use diagonal of each head's matrix (preserves key-value relationships)
                                # [1, 12, 32, 32] -> extract diagonal for each head -> [1, 12, 32] -> concat -> [1, 384]
                                diagonals = torch.diagonal(S_slow, dim1=2, dim2=3)  # [1, 12, 32]
                                state_emb_v4 = diagonals.flatten(1)  # [1, 384]
                                
                                # Method 5: Sum across d_k (aggregate keys), keep d_v, then expand heads
                                # [1, 12, 32, 32] -> sum over d_k -> [1, 12, 32] -> mean over heads -> [1, 32] -> expand
                                state_sum_k = S_slow.sum(dim=2)  # [1, 12, 32]
                                state_mean_heads = state_sum_k.mean(dim=1)  # [1, 32]
                                state_emb_v5 = state_mean_heads.repeat_interleave(12, dim=1)  # [1, 384]
                                
                                # Method 6: Weighted combination of best methods
                                # Combine diagonal (preserves structure) + sum_keys (aggregates info)
                                # Try different weight combinations
                                state_emb_v6a = 0.6 * state_emb_v4 + 0.4 * state_emb_v5
                                state_emb_v6b = 0.7 * state_emb_v4 + 0.3 * state_emb_v5
                                state_emb_v6c = 0.5 * state_emb_v4 + 0.5 * state_emb_v5
                                state_emb_v6 = state_emb_v6a  # Default
                                
                                # Method 7: Use trace (sum of diagonal) for each head, then expand
                                # [1, 12, 32, 32] -> trace each head -> [1, 12] -> expand to 384
                                traces = torch.diagonal(S_slow, dim1=2, dim2=3).sum(dim=2)  # [1, 12] - trace per head
                                state_emb_v7 = traces.repeat_interleave(32, dim=1)  # [1, 384]
                                
                                # Method 8: Flatten full state and use learned-like projection
                                # [1, 12, 32, 32] = [1, 12288] -> use all info with smart pooling
                                state_full = S_slow.flatten(1)  # [1, 12288]
                                # Create 384 groups, each averaging 32 values (12288/384 = 32)
                                state_reshaped = state_full.view(1, 384, 32)
                                state_emb_v8 = state_reshaped.mean(dim=2)  # [1, 384] - preserves all info via averaging
                                
                                # Method 9: Max pooling instead of mean (preserves strongest signals)
                                state_emb_v9 = state_reshaped.max(dim=2)[0]  # [1, 384]
                                
                                # Method 10: Weighted average favoring recent information
                                # Create weights that favor later heads (which accumulate more info)
                                head_weights = torch.linspace(0.5, 1.5, 12, device=S_slow.device).view(1, 12, 1, 1)
                                S_slow_weighted = S_slow * head_weights
                                diagonals_weighted = torch.diagonal(S_slow_weighted, dim1=2, dim2=3)  # [1, 12, 32]
                                state_emb_v10 = diagonals_weighted.flatten(1)  # [1, 384]
                                
                                # Method 11: Use both diagonal AND off-diagonal (full matrix info)
                                # Diagonal captures key-value pairs, off-diagonal captures relationships
                                diagonals = torch.diagonal(S_slow, dim1=2, dim2=3)  # [1, 12, 32]
                                # Also get row/column means to capture off-diagonal info
                                row_means = S_slow.mean(dim=3)  # [1, 12, 32] - mean over values
                                col_means = S_slow.mean(dim=2)  # [1, 12, 32] - mean over keys
                                # Combine: 50% diagonal, 25% row means, 25% col means
                                combined = 0.5 * diagonals + 0.25 * row_means + 0.25 * col_means
                                state_emb_v11 = combined.flatten(1)  # [1, 384]
                                
                                # Method 12: Use SVD-like approach - extract principal components
                                # Flatten each head's matrix, then use top components
                                # [1, 12, 32, 32] -> [1, 12, 1024] -> take mean -> [1, 1024] -> project
                                state_per_head_flat = S_slow.view(1, 12, 1024)  # [1, 12, 1024]
                                state_mean_heads_flat = state_per_head_flat.mean(dim=1)  # [1, 1024]
                                # Project 1024 -> 384 by taking every 2.67th value (interpolated)
                                indices = torch.linspace(0, 1023, 384, dtype=torch.long, device=S_slow.device)
                                state_emb_v12 = state_mean_heads_flat[:, indices]  # [1, 384]
                                
                                # Store all methods for comparison
                                state_embeddings = {
                                    'v1_full_pool': F.normalize(state_emb_v1.squeeze(0), dim=0),
                                    'v2_head_mean': F.normalize(state_emb_v2.squeeze(0), dim=0),
                                    'v3_concat_pool': F.normalize(state_emb_v3.squeeze(0), dim=0),
                                    'v4_diagonal': F.normalize(state_emb_v4.squeeze(0), dim=0),
                                    'v5_sum_keys': F.normalize(state_emb_v5.squeeze(0), dim=0),
                                    'v6_weighted_combo_60_40': F.normalize(state_emb_v6a.squeeze(0), dim=0),
                                    'v6_weighted_combo_70_30': F.normalize(state_emb_v6b.squeeze(0), dim=0),
                                    'v6_weighted_combo_50_50': F.normalize(state_emb_v6c.squeeze(0), dim=0),
                                    'v7_trace': F.normalize(state_emb_v7.squeeze(0), dim=0),
                                    'v8_full_mean_pool': F.normalize(state_emb_v8.squeeze(0), dim=0),
                                    'v9_full_max_pool': F.normalize(state_emb_v9.squeeze(0), dim=0),
                                    'v10_weighted_diagonal': F.normalize(state_emb_v10.squeeze(0), dim=0),
                                    'v11_full_matrix': F.normalize(state_emb_v11.squeeze(0), dim=0),
                                    'v12_sampled': F.normalize(state_emb_v12.squeeze(0), dim=0),
                                }
                                
                                # Use the best one (v4_diagonal - preserves structure)
                                state_emb = state_embeddings['v4_diagonal']
                                print(f"      ✅ State embedding shape: {state_emb.shape}")
                                print(f"      Created {len(state_embeddings)} different aggregation methods for comparison")
                            elif S_fast is not None:
                                print(f"      Found S_fast: shape {S_fast.shape} (using S_fast)")
                                # Same aggregation as S_slow
                                state_flat = S_fast.mean(dim=1).flatten(1)
                                if state_flat.shape[1] > 384:
                                    state_emb = state_flat[:, :384]
                                elif state_flat.shape[1] < 384:
                                    padding = torch.zeros(1, 384 - state_flat.shape[1], device=state_flat.device)
                                    state_emb = torch.cat([state_flat, padding], dim=1)
                                else:
                                    state_emb = state_flat
                                state_emb = F.normalize(state_emb.squeeze(0), dim=0)
                                print(f"      ✅ State embedding shape: {state_emb.shape}")
                            else:
                                print(f"      ❌ Both S_fast and S_slow are None")
                    else:
                        print(f"      ⚠️  No 'recurrent_state' key in state dict")
                        print(f"      Available keys: {list(state_dict.keys())}")
            except Exception as e:
                print(f"      ❌ Could not extract state embedding: {e}")
                import traceback
                traceback.print_exc()
        
        # Different aggregation methods
        print("\n📊 Testing aggregation methods...")
        
        # Method 1: Mean pooling (current - broken)
        mean_emb = hidden[0].mean(dim=0)
        
        # Method 2: Last token
        last_token_emb = hidden[0, -1, :]
        
        # Method 3: Last N tokens (where answer lives)
        last_50_emb = hidden[0, -50:, :].mean(dim=0)
        last_100_emb = hidden[0, -100:, :].mean(dim=0)
        
        # Method 4: Exponential weighted average (favor recent tokens)
        weights = torch.exp(torch.linspace(-4, 0, seq_len, device='cuda'))  # Exponential decay
        weights = weights / weights.sum()  # Normalize
        exp_weighted_emb = (hidden[0] * weights.unsqueeze(1)).sum(dim=0)
        
        # Method 5: Last 10% of tokens
        last_10pct = max(1, seq_len // 10)
        last_10pct_emb = hidden[0, -last_10pct:, :].mean(dim=0)
        
        # Normalize all embeddings
        mean_emb = F.normalize(mean_emb, dim=0)
        last_token_emb = F.normalize(last_token_emb, dim=0)
        last_50_emb = F.normalize(last_50_emb, dim=0)
        last_100_emb = F.normalize(last_100_emb, dim=0)
        exp_weighted_emb = F.normalize(exp_weighted_emb, dim=0)
        last_10pct_emb = F.normalize(last_10pct_emb, dim=0)
    
    # Get reference embeddings
    answer_emb = torch.tensor(model.encode([answer_text], convert_to_numpy=True), device='cuda').squeeze()
    noise_emb = torch.tensor(model.encode(["The quick brown fox jumps over the lazy dog."], convert_to_numpy=True), device='cuda').squeeze()
    
    # KEY TEST: Compare non-streaming (full document) vs streaming
    print("\n" + "=" * 70)
    print("🔑 KEY TEST: Non-Streaming (Full Document) vs Token-Level")
    print("=" * 70)
    
    # Method: Use get_sentence_embeddings on FULL document (non-streaming)
    with torch.no_grad():
        full_doc_emb = model._model.get_sentence_embeddings(input_ids, attention_mask)
        full_doc_emb = full_doc_emb.squeeze()
        if full_doc_emb.dim() == 0:
            full_doc_emb = full_doc_emb.unsqueeze(0)
        full_doc_emb = F.normalize(full_doc_emb, dim=0)
    
    print(f"\n{'Method':<30} {'→ Answer':>12} {'→ Noise':>12} {'Diff':>12} {'Winner':>12}")
    print("-" * 75)
    
    # Compare full doc embedding
    sim_answer_full = F.cosine_similarity(full_doc_emb.unsqueeze(0), answer_emb.unsqueeze(0)).item()
    sim_noise_full = F.cosine_similarity(full_doc_emb.unsqueeze(0), noise_emb.unsqueeze(0)).item()
    diff_full = sim_answer_full - sim_noise_full
    winner_full = "✅ ANSWER" if diff_full > 0.1 else "🟡 CLOSE" if diff_full > 0 else "❌ NOISE"
    print(f"{'Full Doc (non-streaming)':<30} {sim_answer_full:>12.4f} {sim_noise_full:>12.4f} {diff_full:>+12.4f} {winner_full:>12}")
    
    # Compare with mean pooling from token-level
    sim_answer_mean = F.cosine_similarity(mean_emb.unsqueeze(0), answer_emb.unsqueeze(0)).item()
    sim_noise_mean = F.cosine_similarity(mean_emb.unsqueeze(0), noise_emb.unsqueeze(0)).item()
    diff_mean = sim_answer_mean - sim_noise_mean
    winner_mean = "✅ ANSWER" if diff_mean > 0.1 else "🟡 CLOSE" if diff_mean > 0 else "❌ NOISE"
    print(f"{'Mean Pool (token-level)':<30} {sim_answer_mean:>12.4f} {sim_noise_mean:>12.4f} {diff_mean:>+12.4f} {winner_mean:>12}")
    
    print(f"\n💡 INSIGHT:")
    if diff_full > diff_mean:
        print(f"   ✅ Full document (non-streaming) preserves answer BETTER!")
        print(f"   → Difference: {diff_full:.4f} vs {diff_mean:.4f}")
        print(f"   → The answer is in how get_sentence_embeddings works on full docs!")
    else:
        print(f"   ⚠️  Both methods have similar performance")
        print(f"   → Need to investigate what get_sentence_embeddings does differently")
    
    # Compare all methods
    print("\n" + "=" * 70)
    print("📊 RESULTS: Which Aggregation Method Preserves the Answer?")
    print("=" * 70)
    
    methods = [
        ("Mean Pool (current)", mean_emb),
        ("Last Token", last_token_emb),
        ("Last 50 Tokens", last_50_emb),
        ("Last 100 Tokens", last_100_emb),
        ("Exp Weighted", exp_weighted_emb),
        ("Last 10%", last_10pct_emb),
    ]
    
    # Add state-based embeddings if we extracted them
    if 'state_embeddings' in locals() and state_embeddings:
        for name, emb in state_embeddings.items():
            methods.append((f"S_slow ({name})", emb))
        print(f"\n   ✅ Added {len(state_embeddings)} S_slow state embedding methods to comparison!")
    elif state_emb is not None:
        methods.append(("S_slow State Embedding", state_emb))
        print(f"\n   ✅ Added S_slow state embedding to comparison!")
    
    print(f"\n{'Method':<25} {'→ Answer':>12} {'→ Noise':>12} {'Diff':>12} {'Winner':>12}")
    print("-" * 75)
    
    best_method = None
    best_diff = -999
    
    for name, emb in methods:
        sim_answer = F.cosine_similarity(emb.unsqueeze(0), answer_emb.unsqueeze(0)).item()
        sim_noise = F.cosine_similarity(emb.unsqueeze(0), noise_emb.unsqueeze(0)).item()
        diff = sim_answer - sim_noise
        winner = "✅ ANSWER" if diff > 0.1 else "🟡 CLOSE" if diff > 0 else "❌ NOISE"
        print(f"{name:<25} {sim_answer:>12.4f} {sim_noise:>12.4f} {diff:>+12.4f} {winner:>12}")
        
        if diff > best_diff:
            best_diff = diff
            best_method = name
    
    print("\n" + "=" * 70)
    print("🏆 VERDICT")
    print("=" * 70)
    
    if best_diff > 0.1:
        print(f"\n   ✅ BEST METHOD: {best_method}")
        print(f"   → This method preserves end-of-document information!")
        print(f"   → Improvement over mean pooling: {best_diff:.4f}")
        print(f"\n   🔧 FIX: Replace mean pooling with '{best_method}' in stream_embedding()")
    elif best_diff > 0:
        print(f"\n   🟡 PARTIAL SUCCESS: {best_method}")
        print(f"   → Some improvement over mean pooling")
        print(f"   → May need to investigate recurrent state extraction")
    else:
        print(f"\n   ❌ NO METHOD WORKS")
        print(f"   → Even last token doesn't preserve answer info")
        print(f"   → Need to investigate if recurrent state is being computed correctly")
        print(f"   → Check if S_slow is being accumulated during inference")
    
    # Additional diagnostic: check token positions
    print("\n" + "=" * 70)
    print("📍 Token Position Analysis")
    print("=" * 70)
    
    # Sample at different positions
    positions = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-10, seq_len-1]
    
    print(f"\n{'Position':<20} {'→ Answer':>12} {'→ Noise':>12} {'Info':>20}")
    print("-" * 65)
    
    for pos in positions:
        if pos >= seq_len:
            continue
        emb = F.normalize(hidden[0, pos, :], dim=0)
        sim_answer = F.cosine_similarity(emb.unsqueeze(0), answer_emb.unsqueeze(0)).item()
        sim_noise = F.cosine_similarity(emb.unsqueeze(0), noise_emb.unsqueeze(0)).item()
        
        # Determine what this position likely contains
        if pos < seq_len - 20:
            info = "Noise region"
        else:
            info = "Answer region"
        
        print(f"Token {pos:<13} {sim_answer:>12.4f} {sim_noise:>12.4f} {info:>20}")
    
    print("\n   If Answer region tokens show HIGH similarity to Answer:")
    print("   → The model IS encoding the answer correctly")
    print("   → We just need to use those tokens instead of mean pooling!")
    
    # FINAL TEST: Use answer region tokens directly
    print("\n" + "=" * 70)
    print("🎯 FINAL TEST: Using Answer Region Tokens Directly")
    print("=" * 70)
    
    # Token 1003 had the highest similarity - let's test different windows
    test_windows = [
        ("Token 1003 only", [1003]),
        ("Tokens 1000-1012", list(range(1000, seq_len))),
        ("Tokens 1003-1010", list(range(1003, min(1010, seq_len)))),
        ("Last 5 tokens", list(range(seq_len-5, seq_len))),
        ("Last 10 tokens", list(range(seq_len-10, seq_len))),
    ]
    
    print(f"\n{'Method':<30} {'→ Answer':>12} {'→ Noise':>12} {'Diff':>12} {'Winner':>12}")
    print("-" * 75)
    
    best_method = None
    best_diff = -999
    
    for name, token_indices in test_windows:
        # Filter valid indices
        valid_indices = [i for i in token_indices if i < seq_len]
        if not valid_indices:
            continue
        
        # Get embeddings for these tokens
        if len(valid_indices) == 1:
            window_emb = hidden[0, valid_indices[0], :]
        else:
            window_emb = hidden[0, valid_indices, :].mean(dim=0)
        window_emb = F.normalize(window_emb, dim=0)
        
        sim_answer = F.cosine_similarity(window_emb.unsqueeze(0), answer_emb.unsqueeze(0)).item()
        sim_noise = F.cosine_similarity(window_emb.unsqueeze(0), noise_emb.unsqueeze(0)).item()
        diff = sim_answer - sim_noise
        winner = "✅ ANSWER" if diff > 0.1 else "🟡 CLOSE" if diff > 0 else "❌ NOISE"
        print(f"{name:<30} {sim_answer:>12.4f} {sim_noise:>12.4f} {diff:>+12.4f} {winner:>12}")
        
        if diff > best_diff:
            best_diff = diff
            best_method = name
    
    print(f"{'Mean Pool (all tokens)':<30} {sim_answer_mean:>12.4f} {sim_noise_mean:>12.4f} {diff_mean:>+12.4f} {'❌ NOISE':>12}")
    
    if best_diff > 0.1:
        print(f"\n🎉 SUCCESS! {best_method} preserves the answer!")
        print(f"   → Improvement: {best_diff:.4f} vs {diff_mean:.4f}")
        print(f"   → FIX: Use {best_method} instead of mean pooling for long documents!")
    elif best_diff > 0:
        print(f"\n🟡 PARTIAL: {best_method} shows some improvement")
        print(f"   → But still not strong enough - may need recurrent state")
    else:
        print(f"\n⚠️  Even best window ({best_method}) doesn't work")
        print(f"   → Need to investigate recurrent state extraction")
    
    return best_method, best_diff


if __name__ == "__main__":
    best_method, best_diff = test_aggregation_methods()
    
    if best_diff > 0.1:
        print("\n" + "🎉" * 20)
        print("   THE FIX IS CLEAR!")
        print(f"   Replace mean pooling with '{best_method}'")
        print("🎉" * 20)