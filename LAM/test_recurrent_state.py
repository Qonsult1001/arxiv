#!/usr/bin/env python3

"""
CRITICAL TEST: Is recurrent_state actually computed during inference?

We know the state dict has 'recurrent_state' key, but is it (None, None)?
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/workspace/LAM/lam_package')
from lam import LAM

def check_recurrent_state():
    print("=" * 70)
    print("CRITICAL TEST: Recurrent State Content During Inference")
    print("=" * 70)
    
    model = LAM('/workspace/LAM/best', device='cuda')
    
    # Simple test input
    test_text = "The secret answer is BANANA. " * 10
    
    enc = model.tokenizer.encode(test_text)
    ids = enc.ids if hasattr(enc, 'ids') else enc
    input_ids = torch.tensor([ids], dtype=torch.long, device='cuda')
    attention_mask = torch.ones_like(input_ids)
    
    print(f"\n📄 Test input: {len(ids)} tokens")
    
    # Process through model and check state
    with torch.no_grad():
        # Access internal model
        internal_model = model._model
        
        # Get embeddings
        if hasattr(internal_model, 'embeddings'):
            emb_dict = internal_model.embeddings
            if hasattr(emb_dict, 'keys') and 'word_embeddings' in emb_dict:
                hidden = emb_dict['word_embeddings'](input_ids)
            else:
                raise AttributeError("Cannot find word_embeddings")
        else:
            raise AttributeError("Cannot find embeddings")
        
        # Process through layers
        if hasattr(internal_model, 'deltanet_layers'):
            layers = internal_model.deltanet_layers
            norms = internal_model.deltanet_norms if hasattr(internal_model, 'deltanet_norms') else [None] * len(layers)
        else:
            raise AttributeError("Cannot find deltanet_layers")
        
        all_states = []
        
        for i, layer in enumerate(layers):
            residual = hidden
            
            if norms[i] is not None:
                hidden = norms[i](hidden)
            
            # Get output with state - try with training=True to force state computation
            try:
                output, attn, past_kv, ortho = layer(
                    hidden,
                    attention_mask=attention_mask,
                    use_cache=True,
                    training=True  # Force state computation!
                )
            except TypeError:
                # Fallback without training parameter
                output, attn, past_kv, ortho = layer(
                    hidden,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            
            hidden = residual + output
            
            # Check the state
            print(f"\n🔍 Layer {i}:")
            print(f"   past_kv type: {type(past_kv)}")
            
            if past_kv is not None:
                if isinstance(past_kv, (list, tuple)):
                    state_dict = past_kv[0] if len(past_kv) > 0 else None
                elif isinstance(past_kv, dict):
                    state_dict = past_kv
                else:
                    state_dict = None
                
                if state_dict and 'recurrent_state' in state_dict:
                    recurrent_state = state_dict['recurrent_state']
                    print(f"   recurrent_state type: {type(recurrent_state)}")
                    
                    if recurrent_state is None:
                        print(f"   ❌ recurrent_state is None!")
                    elif isinstance(recurrent_state, tuple):
                        S_fast, S_slow = recurrent_state
                        print(f"   S_fast: {type(S_fast)}, S_slow: {type(S_slow)}")
                        
                        if S_fast is None:
                            print(f"   ❌ S_fast is None!")
                        else:
                            print(f"   ✅ S_fast shape: {S_fast.shape}")
                            print(f"      S_fast stats: mean={S_fast.mean():.6f}, std={S_fast.std():.6f}, norm={S_fast.norm():.6f}")
                        
                        if S_slow is None:
                            print(f"   ❌ S_slow is None!")
                        else:
                            print(f"   ✅ S_slow shape: {S_slow.shape}")
                            print(f"      S_slow stats: mean={S_slow.mean():.6f}, std={S_slow.std():.6f}, norm={S_slow.norm():.6f}")
                        
                        all_states.append((S_fast, S_slow))
                    else:
                        print(f"   recurrent_state is: {recurrent_state}")
            else:
                print(f"   ❌ past_kv is None!")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    has_valid_state = False
    for i, state in enumerate(all_states):
        S_fast, S_slow = state
        if S_fast is not None or S_slow is not None:
            has_valid_state = True
            break
    
    if has_valid_state:
        print("""
   ✅ RECURRENT STATE IS COMPUTED!
   
   The state exists but we're not using it for embeddings.
   
   FIX: Extract embedding from S_slow instead of mean pooling
   
   S_slow shape: [batch, heads, d_k, d_v]
   Can be aggregated to: [batch, d_model] for embedding
        """)
        
        # Try to create embedding from state
        print("\n🔧 Attempting to create embedding from S_slow...")
        
        last_S_fast, last_S_slow = all_states[-1]
        
        if last_S_slow is not None:
            # Option 1: Mean across heads, then flatten
            # S_slow: [1, num_heads, d_k, d_v]
            state_emb_v1 = last_S_slow.mean(dim=1).flatten()  # [d_k * d_v]
            print(f"   State embedding v1 shape: {state_emb_v1.shape}")
            
            # Option 2: Diagonal of each head's matrix, then concat
            # S_slow[h] is a d_k x d_v matrix - diagonal gives d_k values
            # Not applicable if d_k != d_v
            
            # Option 3: Sum across heads and both matrix dims
            state_emb_v3 = last_S_slow.sum(dim=(1, 2, 3))  # scalar per batch
            print(f"   State embedding v3 (scalar): {state_emb_v3}")
            
            # Compare to reference
            answer_emb = torch.tensor(model.encode(["The secret answer is BANANA"], convert_to_numpy=True), device='cuda').squeeze()
            
            # Pad state_emb_v1 to match answer_emb size if needed
            if state_emb_v1.shape[0] != answer_emb.shape[0]:
                print(f"   ⚠️  State embedding size ({state_emb_v1.shape[0]}) != answer embedding size ({answer_emb.shape[0]})")
                # Truncate or pad
                min_size = min(state_emb_v1.shape[0], answer_emb.shape[0])
                state_emb_v1 = state_emb_v1[:min_size]
                answer_emb_truncated = answer_emb[:min_size]
                
                state_emb_v1 = F.normalize(state_emb_v1, dim=0)
                answer_emb_truncated = F.normalize(answer_emb_truncated, dim=0)
                
                sim = F.cosine_similarity(state_emb_v1.unsqueeze(0), answer_emb_truncated.unsqueeze(0)).item()
                print(f"   State → Answer similarity (truncated): {sim:.4f}")
        
    else:
        print("""
   ❌ RECURRENT STATE IS NOT COMPUTED DURING INFERENCE!
   
   The bug is confirmed:
   - State computation is skipped when training=False
   - S_fast and S_slow are set to None
   
   FIX: Modify _enhanced_hierarchical_delta_rule_impl to ALWAYS compute state
   OR: Force training=True during streaming inference
        """)
    
    return has_valid_state, all_states

if __name__ == "__main__":
    has_state, states = check_recurrent_state()
    
    if not has_state:
        print("\n" + "🔧" * 20)
        print("NEXT STEP: Modify the forward pass to compute state during inference")
        print("🔧" * 20)

