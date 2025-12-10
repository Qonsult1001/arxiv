import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
import gzip
from tqdm import tqdm
import numpy as np
import random
from typing import List, Union
import sys

# Import DeltaNet model from train_6layer_deltanet_1.py
sys.path.insert(0, str(Path(__file__).parent))
try:
    from train_6layer_deltanet_1 import DeltaNetPure6Layer
    DELTANET_AVAILABLE = True
except ImportError:
    DELTANET_AVAILABLE = False
    print("⚠️  Could not import DeltaNetPure6Layer. Will use SentenceTransformer instead.")

# ============================================================================
# 1. SETUP AND UTILITIES
# ============================================================================

# --- Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 🔑 CONFIGURATION OPTIONS
# Option 1: Cross-dimensional KD (768D teacher → 384D student) - REQUIRES PROJECTION HEAD
# Option 2: Same-dimensional KD (384D teacher → 384D student) - NO PROJECTION HEAD (TEST THIS!)

# ⚠️ TEST MODE: Use all-MiniLM-L12-v2 as teacher to eliminate projection head issues
USE_TEST_MODE = True  # Set to True to test without projection head

if USE_TEST_MODE:
    # Same dimensions - no projection head needed
    TEACHER_MODEL_NAME = 'sentence-transformers/all-MiniLM-L12-v2' # 384 dimensions
    BLEND_ALPHA = 0.70 # Optimal alpha for all-MiniLM-L12-v2
    print("🧪 TEST MODE: Using all-MiniLM-L12-v2 teacher (384D) - NO projection head needed")
else:
    # Cross-dimensional - requires projection head
    TEACHER_MODEL_NAME = 'stsb-roberta-base-v2' # 768 dimensions
    BLEND_ALPHA = 0.40 # Optimal alpha for stsb-roberta-base-v2
    print("🔧 PRODUCTION MODE: Using stsb-roberta-base-v2 teacher (768D) - projection head required")

# ⭐ UPDATED: Use trained DeltaNet model from train_6layer_deltanet_1.py
# Path to the trained DeltaNet model (should contain pytorch_model.bin)
STUDENT_MODEL_PATH = '/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEW'
# Fallback to SentenceTransformer if DeltaNet not available
STUDENT_MODEL_NAME_FALLBACK = 'sentence-transformers/all-MiniLM-L12-v2' # 384 dimensions

BATCH_SIZE = 128
# ⚠️ IMPORTANT: Linear architectures may need more epochs to learn from transformer teachers
# Increased to 3 epochs for better convergence
# Monitor cosine similarity during training - should approach 1.0
DISTILLATION_EPOCHS = 3 if USE_TEST_MODE else 2  # Test mode: 3 epochs, Production: 2 epochs
MAX_DISTILLATION_SENTENCES = 500000
LEARNING_RATE = 5e-5  # Increased from 2e-5 for faster convergence (robust for linear architectures)

script_dir = Path(__file__).parent
data_dir = script_dir / "data"
data_dir.mkdir(exist_ok=True)

# Dynamic paths based on the model names
teacher_suffix = TEACHER_MODEL_NAME.split('/')[-1]
student_suffix = Path(STUDENT_MODEL_PATH).name if DELTANET_AVAILABLE else STUDENT_MODEL_NAME_FALLBACK.split('/')[-1]

kernel_path = data_dir / f"kernel_{teacher_suffix}.pt"
# New output path to reflect cross-dimensional distillation
student_output_path = data_dir / f"distilled_deltanet_student_from_{teacher_suffix}"

# ==========================================
# 2. BLENDED SEMANTIC KERNEL MODEL (THE TEACHER)
# ==========================================
class BlendedSemanticKernelModel:
    """
    Inference class acting as the high-quality Teacher Model for KD.
    The Teacher produces the refined, blended embeddings (768-dim).
    """
    
    def __init__(self, model_name: str, kernel_path: str, alpha: float = 0.10):
        print(f"Loading Teacher Model: {model_name}...")
        self.raw_model = SentenceTransformer(model_name).to(device)
        
        print(f"Loading Kernel from: {kernel_path.name}...")
        kernel_state = torch.load(str(kernel_path), map_location=device, weights_only=False)
        self.kernel = kernel_state['kernel'].to(device) 
        self.alpha = alpha

    def encode_blended(self, sentences: List[str]) -> torch.Tensor:
        """
        Encodes sentences and returns the normalized, blended tensor.
        This is the 768-dim target embedding for the Student.
        """
        with torch.no_grad():
            # Get raw embeddings from the base model (V_raw) - normalized
            V_raw = self.raw_model.encode(
                sentences, 
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE
            ).to(device)
            
            # Apply the kernel transformation (V_kernel)
            V_kernel = torch.matmul(V_raw, self.kernel)
            V_kernel = F.normalize(V_kernel, p=2, dim=1)
            
            # Blend the vectors: V_final = (1-alpha) * V_raw + alpha * V_kernel
            V_final = (1.0 - self.alpha) * V_raw + self.alpha * V_kernel
            
            # Normalize and return (768-dim)
            return F.normalize(V_final, p=2, dim=1)

# ==========================================
# 3. DATASET AND DISTILLATION LOSS
# ==========================================
class SimpleSentenceDataset(Dataset):
    """Dataset for distilling a large corpus of sentences."""
    def __init__(self, sentences: List[str]):
        self.sentences = sentences
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return self.sentences[idx]

def load_distillation_corpus(max_sentences: int) -> List[str]:
    """Load individual sentences from the AllNLI corpus for distillation."""
    print(f"\n📚 Loading {max_sentences:,} sentences for distillation...")
    
    data_dir = script_dir / "data"
    allnli_path = data_dir / "AllNLI.jsonl.gz"
    
    if not allnli_path.exists():
        print("⚠️ AllNLI.jsonl.gz not found. Please ensure it's in the data folder.")
        return []
    
    sentences = set()
    
    with gzip.open(allnli_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading AllNLI for sentences"):
            if len(sentences) >= max_sentences:
                break
            try:
                data = json.loads(line)
                if isinstance(data, list) and len(data) == 3:
                    anchor, positive, negative = data
                    sentences.add(str(anchor).strip())
                    sentences.add(str(positive).strip())
                    sentences.add(str(negative).strip())
            except:
                continue
    
    sentence_list = list(sentences)[:max_sentences]
    print(f"✅ Loaded and unique-ified {len(sentence_list):,} sentences.")
    return sentence_list

# ==========================================
# 4. DISTILLATION TRAINING LOOP
# ==========================================

def distill_kernel_model():
    """Main function for training the student model."""
    print("="*80)
    # Get teacher dimension for display (will get actual from model later)
    temp_teacher = SentenceTransformer(TEACHER_MODEL_NAME)
    teacher_dim_display = temp_teacher.get_sentence_embedding_dimension()
    del temp_teacher  # Free memory
    student_dim = 384  # DeltaNet is 384D
    print(f"🧠 KNOWLEDGE DISTILLATION: BLENDED TEACHER ({teacher_dim_display}D) → STUDENT ({student_dim}D)")
    if DELTANET_AVAILABLE and Path(STUDENT_MODEL_PATH).exists():
        print(f"   Using trained DeltaNet model from: {STUDENT_MODEL_PATH}")
    else:
        print(f"   Using SentenceTransformer: {STUDENT_MODEL_NAME_FALLBACK}")
    if teacher_dim_display == student_dim:
        print(f"   ✅ Same dimensions - NO projection head needed!")
    else:
        print(f"   ⚠️  Different dimensions - projection head will be created")
    print("="*80)
    
    # Check if kernel exists
    if not kernel_path.exists():
        print(f"\n❌ Kernel not found at {kernel_path}!")
        print(f"   Please run 'semantic_kernel_pipeline.py' first to train the kernel.")
        print(f"   Make sure BASE_MODEL_NAME is set to: {TEACHER_MODEL_NAME}")
        return
    
    # 1. Initialize Teacher (Blended Model)
    teacher_model = BlendedSemanticKernelModel(TEACHER_MODEL_NAME, kernel_path, BLEND_ALPHA)
    teacher_dim = teacher_model.raw_model.get_sentence_embedding_dimension()  # Get actual dimension
    
    # Initialize projection_head variable (will be set later if needed)
    projection_head = None
    
    # 2. Initialize Student Model (DeltaNet from train_6layer_deltanet_1.py)
    print(f"\n👨‍🎓 Initializing Student Model...")
    
    if DELTANET_AVAILABLE and Path(STUDENT_MODEL_PATH).exists():
        print(f"   Loading trained DeltaNet model from: {STUDENT_MODEL_PATH}")
        # Load the trained DeltaNet model
        # We need the teacher model path for initialization (from train_6layer_deltanet_1.py config)
        teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"
        
        # Create a dummy config for DeltaNet initialization
        dummy_config = {
            'num_heads': 12,
            'fast_decay_init': 0.30,
            'slow_decay_init': 0.85,
        }
        
        # Initialize DeltaNet model
        student_model = DeltaNetPure6Layer(teacher_model_path, 6, dummy_config).to(device)
        
        # Load the trained weights
        model_path = Path(STUDENT_MODEL_PATH) / "pytorch_model.bin"
        if model_path.exists():
            print(f"   Loading weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            # Load only the deltanet_layers (the trainable parts)
            if 'deltanet_layers' in state_dict:
                student_model.deltanet_layers.load_state_dict(state_dict['deltanet_layers'], strict=False)
            else:
                # Try loading directly if it's the full state dict
                student_model.load_state_dict(state_dict, strict=False)
            print("   ✅ Loaded DeltaNet weights")
        else:
            print(f"   ⚠️  Model file not found: {model_path}")
            print(f"   Using randomly initialized DeltaNet (will train from scratch)")
        
        student_model.train() # Set to training mode
        student_dim = student_model.d_model  # DeltaNet uses d_model attribute
        use_deltanet = True
    else:
        print(f"   Using SentenceTransformer fallback: {STUDENT_MODEL_NAME_FALLBACK}")
        student_model = SentenceTransformer(STUDENT_MODEL_NAME_FALLBACK).to(device)
    student_model.train() # Set to training mode
        student_dim = student_model.get_sentence_embedding_dimension()
        use_deltanet = False
    
    # *** PROJECTION LAYER for Cross-Dimensional KD ***
    print(f"\n📐 Dimension Check: Student={student_dim}D, Teacher={teacher_dim}D")
    if student_dim != teacher_dim:
        print(f"⚠️ Initializing Projection Head: {student_dim}-dim -> {teacher_dim}-dim")
        # This layer projects the student's 384-dim embedding up to 768-dim for the loss calculation
        projection_head = nn.Linear(student_dim, teacher_dim).to(device)
        projection_head.train()
        print(f"   ✅ Projection head created and will be trained")
        
        # Combine parameters for optimization
        if use_deltanet:
            # For DeltaNet, only train deltanet_layers (not frozen teacher parts)
            trainable_params = list(student_model.deltanet_layers.parameters()) + list(projection_head.parameters())
        else:
            trainable_params = list(student_model.parameters()) + list(projection_head.parameters())
        optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)
        print(f"   ✅ Optimizer initialized with {sum(p.numel() for p in projection_head.parameters()):,} projection head parameters")
    else:
        # Standard KD if dimensions match
        projection_head = None
        print(f"   ✅ Dimensions match - no projection head needed")
        if use_deltanet:
            # For DeltaNet, only train deltanet_layers
            trainable_params = list(student_model.deltanet_layers.parameters())
        else:
            trainable_params = list(student_model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    # 3. Prepare Data
    sentence_list = load_distillation_corpus(MAX_DISTILLATION_SENTENCES)
    if not sentence_list:
        return
        
    dataset = SimpleSentenceDataset(sentence_list)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # 4. Loss - Negative Cosine Similarity Loss
    # This directly minimizes the angle between Student and Teacher embeddings
    # Works correctly for normalized embeddings (focuses on direction, not magnitude)
    def negative_cosine_similarity_loss(student_emb, teacher_emb):
        """
        Negative Cosine Similarity Loss: -mean(cosine_similarity)
        
        Since both embeddings are normalized:
        - cosine_similarity = dot_product (since ||emb|| = 1)
        - We want to maximize cosine similarity = minimize negative cosine similarity
        - Returns: -mean(cosine_similarity) so minimizing this maximizes alignment
        
        This loss focuses on the ANGLE between embeddings, not their magnitude,
        which is correct for semantic similarity learning.
        """
        # Both embeddings are normalized, so cosine similarity = dot product
        cosine_sim = (student_emb * teacher_emb).sum(dim=1)  # [batch_size]
        # Return negative mean (minimizing this maximizes cosine similarity)
        return -cosine_sim.mean()
    
    # Keep MSE for monitoring only (not used for training)
    mse_loss_fn = nn.MSELoss()
    
    # 5. Training Loop
    print(f"\n🚀 Starting Distillation for {DISTILLATION_EPOCHS} epoch(s)...")
    print(f"   Using Negative Cosine Similarity Loss (focuses on angle, not magnitude)")
    print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Target: Maximize cosine similarity (watch 'Cos Sim' → should approach 1.0)")
    
    for epoch in range(DISTILLATION_EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{DISTILLATION_EPOCHS}", unit="batch")
        
        for sentences in pbar:
            # Step A: Teacher produces the refined target embeddings (768-dim)
            # Teacher is frozen, so use no_grad
            with torch.no_grad():
            V_teacher = teacher_model.encode_blended(sentences)
            
            # Step B: Student produces its embeddings (384-dim)
            optimizer.zero_grad()
            
            # Tokenize
            tokenizer = student_model.tokenizer
            encoded = tokenizer(
                sentences, 
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            ).to(device)
            
            if use_deltanet:
                # DeltaNet forward pass (preserves gradients)
                # forward_student returns (embeddings, hidden_states, ortho_loss)
                # embeddings are already normalized
                V_student_emb, _, _ = student_model.forward_student(
                    encoded['input_ids'], 
                    encoded['attention_mask']
                )
                # DeltaNet embeddings are already normalized, but we need unnormalized for projection
                # So we'll use them as-is (they're normalized, which is fine for the loss)
                V_student_raw = V_student_emb
            else:
                # SentenceTransformer forward pass (preserves gradients)
                features = {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']}
                
                # Pass through all modules (this preserves gradients)
                for module in student_model:
                    features = module(features)
                
                # Extract sentence embeddings from the features
                if 'sentence_embedding' in features:
                    V_student_raw = features['sentence_embedding']
                else:
                    # Fallback: use token embeddings with mean pooling
                    token_embeddings = features['token_embeddings']
                    attention_mask = encoded['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    V_student_raw = sum_embeddings / sum_mask
            
            # Note: V_student_raw is normalized for DeltaNet, may be normalized for SentenceTransformer
            
            # Step C: Project and Normalize (if needed)
            if projection_head:
                # Project V_student (384) up to Teacher's dimension (768)
                V_student_loss = F.normalize(projection_head(V_student_raw), p=2, dim=1)
            else:
                V_student_loss = F.normalize(V_student_raw, p=2, dim=1)
                
            # Step D: Calculate Loss - Negative Cosine Similarity Loss
            # This directly minimizes the angle between Student and Teacher embeddings
            # Detach teacher to prevent gradients from flowing back
            loss = negative_cosine_similarity_loss(V_student_loss, V_teacher.detach())
            
            # Also compute MSE for monitoring (but don't use it for training)
            mse_loss_val = mse_loss_fn(V_student_loss, V_teacher.detach())
            
            # Step E: Backpropagate
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute average cosine similarity for diagnostics
            with torch.no_grad():
                avg_cosine_sim = (V_student_loss * V_teacher.detach()).sum(dim=1).mean().item()
            
            # Show both losses and similarity
            # Note: loss is negative cosine, so more negative = better alignment
            pbar.set_postfix({
                'NegCos Loss': f'{total_loss/(pbar.n+1):.4f}',
                'MSE': f'{mse_loss_val.item():.6f}',
                'Cos Sim': f'{avg_cosine_sim:.4f}'  # Should approach 1.0
            })
        
        # End of epoch summary
        epoch_avg_loss = total_loss / len(dataloader)
        # Compute final epoch cosine similarity
        with torch.no_grad():
            # Quick check: get a sample batch to compute final Cos Sim
            sample_batch = next(iter(dataloader))
            with torch.no_grad():
                V_teacher_sample = teacher_model.encode_blended(sample_batch)
            tokenizer = student_model.tokenizer
            encoded = tokenizer(sample_batch, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            if use_deltanet:
                V_student_sample, _, _ = student_model.forward_student(encoded['input_ids'], encoded['attention_mask'])
            else:
                features = {'input_ids': encoded['input_ids'], 'attention_mask': encoded['attention_mask']}
                for module in student_model:
                    features = module(features)
                V_student_sample = features.get('sentence_embedding', 
                    torch.sum(features['token_embeddings'] * encoded['attention_mask'].unsqueeze(-1).float(), 1) / 
                    torch.clamp(encoded['attention_mask'].sum(1, keepdim=True).float(), min=1e-9))
            if projection_head:
                V_student_sample = F.normalize(projection_head(V_student_sample), p=2, dim=1)
            else:
                V_student_sample = F.normalize(V_student_sample, p=2, dim=1)
            final_cos_sim = (V_student_sample * V_teacher_sample).sum(dim=1).mean().item()
        
        print(f"\n📊 Epoch {epoch+1}/{DISTILLATION_EPOCHS} Complete:")
        print(f"   NegCos Loss: {epoch_avg_loss:.6f}")
        print(f"   Cos Sim: {final_cos_sim:.4f} (target: >0.90)")
        if epoch == 0:
            print(f"   💡 Progress: Cos Sim improving → target 0.90+ for good STS-B performance")
        elif final_cos_sim < 0.85:
            print(f"   ⚠️  Cos Sim still below 0.85 - may need more epochs after evaluation")
        elif final_cos_sim >= 0.90:
            print(f"   ✅ Excellent alignment! Should see good STS-B improvement")

    avg_loss = total_loss / len(dataloader)
    print(f"\n✅ Distillation Finished. Final Avg Negative Cosine Loss: {avg_loss:.6f}")
    print(f"   (More negative is better - target: < -0.9 means cosine similarity > 0.9)")
    print(f"   This indicates good alignment between student and teacher embeddings")
    print(f"\n💡 Next Step: Run 'evaluate_distilled_student.py' to see STS-B performance")
    print(f"   If score < 0.82, consider increasing DISTILLATION_EPOCHS to 4-5")
    
    # 6. Save the Distilled Student Model (ONLY the 384-dim model, NOT the head)
    print(f"\n💾 Saving distilled student model to {student_output_path}...")
    
    # Ensure output directory exists
    Path(student_output_path).mkdir(parents=True, exist_ok=True)
    
    if use_deltanet:
        # DeltaNet uses save_pretrained method
        student_model.save_pretrained(str(student_output_path))
    else:
        # SentenceTransformer uses save method
    student_model.save(str(student_output_path))
    
    # Save the projection head if it exists (for both DeltaNet and SentenceTransformer)
    if projection_head is not None:
        projection_path = Path(student_output_path) / "projection_head.pt"
        print(f"   💾 Saving projection head to: {projection_path}")
        torch.save(
            {'projection_head': projection_head.state_dict()},
            str(projection_path)
        )
        print(f"   ✅ Projection head saved successfully!")
    else:
        print(f"   ⚠️  No projection head to save (dimensions match or not created)")
    
    print(f"✅ Student Model saved. Ready for final evaluation.")
    
    return student_model

if __name__ == "__main__":
    # Ensure data directory exists
    Path(script_dir / "data").mkdir(exist_ok=True)
    
    # Execute distillation
    distilled_student = distill_kernel_model()

    # --- Next: Evaluate the Distilled Student ---
    # We will now use this student model in a new evaluation script
    # to see its raw score and its blended score.

    print("\n\n" + "="*80)
    print("NEXT STEP: Evaluating the Distilled Student Model.")
    print(f"Student model is saved at: {student_output_path}")
    print("Run the evaluation script to compare:")
    print("1. Raw Student Score")
    print("2. Blended Student Score (Student + Original Kernel)")
    print("="*80)