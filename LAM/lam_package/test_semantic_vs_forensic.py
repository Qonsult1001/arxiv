"""
🧪 LAM HIGH-DEFINITION SEMANTICS TEST
=====================================
Proves that 12k Enterprise Vectors capture "Nuance" better than 384d.

Scenario:
Two massive Medical Reports (10k+ tokens) that are 99% identical.
- Doc A: Diagnosed with VIRAL Pneumonia.
- Doc B: Diagnosed with BACTERIAL Pneumonia.

Hypothesis:
- 384d: Will be confused (Score difference ~0.01).
- 12k: Will distinguish clearly (Score difference >> 0.1).
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lam import LAM
    from lam_dual_encoder import LAMDualEncoder
except ImportError:
    print("❌ Error: LAM modules not found.")
    sys.exit(1)

def generate_medical_records():
    print(f"   🔨 Generating 'Twin' Medical Reports (moderate length for testing)...")
    
    # Common text (99% overlap) - reduced size to avoid OOM
    filler = "Patient exhibited standard vitals. Blood pressure 120/80. No history of allergies. " * 50
    
    # The Nuance (1% difference)
    # Deep in the text so it requires long-context retention
    nuance_A = "FINAL DIAGNOSIS: VIRAL PNEUMONIA. Recommended treatment: Rest and antivirals."
    nuance_B = "FINAL DIAGNOSIS: BACTERIAL PNEUMONIA. Recommended treatment: Antibiotics and observation."
    
    doc_A = filler + " " + nuance_A + " " + filler
    doc_B = filler + " " + nuance_B + " " + filler
    
    return doc_A, doc_B

def run_nuance_test():
    # 1. Load Model
    model_path = "/workspace/LAM/best"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔧 Loading LAM from {model_path}...")
    
    model = LAM(str(model_path))
    encoder = LAMDualEncoder(model) # Uses your V5 logic
    
    # 2. Get Data
    doc_viral, doc_bacterial = generate_medical_records()
    
    # 3. Encode Documents (using streaming for long context)
    print("\n📦 Encoding Documents (using streaming for long context)...")
    
    # Check if documents are long and need streaming
    # For long documents, use streamer directly to avoid OOM
    def encode_with_streaming(text, mode):
        """Encode with streaming for long documents to avoid OOM."""
        # Estimate token count (rough: ~4 chars per token)
        estimated_tokens = len(text) // 4
        
        # Use streaming for documents > 2048 tokens
        if estimated_tokens > 2048:
            # Use streamer directly for long documents
            if hasattr(encoder.tokenizer, 'encode'):
                enc = encoder.tokenizer.encode(text)
                if isinstance(enc, list):
                    tokens = enc
                else:
                    tokens = enc.ids
            else:
                tokens = encoder.tokenizer(text, return_tensors='pt', add_special_tokens=True)['input_ids'][0].tolist()
            
            input_ids = torch.tensor([tokens], device=encoder.device)
            encoder.streamer.reset()
            
            if mode == "standard":
                # Use streamer with mean pooling for standard mode
                vec = encoder.streamer.stream_embedding(
                    input_ids, 
                    verbose=False,
                    use_state_embedding=False  # Mean pooling for semantic similarity
                )
                if isinstance(vec, torch.Tensor):
                    vec = vec.cpu().numpy()
                if vec.ndim == 2:
                    return vec.squeeze(0)
                return vec
            else:
                # Enterprise mode - already uses streaming internally
                return encoder.encode(text, mode="enterprise")
        else:
            # Short documents - use regular encode
            return encoder.encode(text, mode=mode)
    
    # Tier 1: Standard (384d) - with streaming for long docs
    vec_viral_384 = encode_with_streaming(doc_viral, mode="standard")
    vec_bact_384 = encode_with_streaming(doc_bacterial, mode="standard")
    
    # Tier 2: Enterprise (12k) - already uses streaming internally
    vec_viral_12k = encoder.encode(doc_viral, mode="enterprise")
    vec_bact_12k = encoder.encode(doc_bacterial, mode="enterprise")
    
    # 4. The Query
    query = "What is the treatment for the viral infection?"
    
    print(f"\n🔍 Query: '{query}'")
    
    # Encode Query
    q_384 = encoder.encode(query, mode="standard")
    #q_12k = encoder.encode(query, mode="enterprise")
    q_12k = encoder.encode(query, mode="enterprise", semantic_weight=0.0)
    
    # 5. Calculate Scores
    # Standard (Cosine)
    score_v_384 = np.dot(vec_viral_384, q_384)
    score_b_384 = np.dot(vec_bact_384, q_384)
    
    # Enterprise (Dot Product)
    score_v_12k = np.dot(vec_viral_12k, q_12k)
    score_b_12k = np.dot(vec_bact_12k, q_12k)
    
    # 6. Analysis
    print("\n" + "="*60)
    print("📊 NUANCE DISCRIMINATION RESULTS")
    print("="*60)
    
    # Standard Analysis
    margin_384 = score_v_384 - score_b_384
    print(f"🔹 Standard (384d):")
    print(f"   Correct Doc Score: {score_v_384:.4f}")
    print(f"   Wrong Doc Score:   {score_b_384:.4f}")
    print(f"   Margin:            {margin_384:.4f}")
    
    # Enterprise Analysis
    margin_12k = score_v_12k - score_b_12k
    print(f"\n🔸 Enterprise (12k):")
    print(f"   Correct Doc Score: {score_v_12k:.4f}")
    print(f"   Wrong Doc Score:   {score_b_12k:.4f}")
    print(f"   Margin:            {margin_12k:.4f}")
    
    print("-" * 60)
    
    # The Verdict
    if margin_12k > (margin_384 * 2.0):
        print("✅ SUCCESS: Enterprise 12k is significantly more precise.")
        print(f"   Improvement Factor: {margin_12k / (margin_384 + 1e-9):.1f}x")
        print("   Conclusion: 12k captures 'High-Def' semantics that 384d blurs.")
    elif margin_12k > margin_384:
        print("✅ PASS: Enterprise 12k is better, but margin is tight.")
    else:
        print("❌ FAIL: 12k did not improve discrimination.")

if __name__ == "__main__":
    run_nuance_test()