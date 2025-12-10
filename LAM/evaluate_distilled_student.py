import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pathlib import Path
import gzip
import csv
import os
from scipy.stats import spearmanr
import numpy as np
from typing import List, Union
from tqdm import tqdm
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_dir = Path(__file__).parent

# 🔑 UPDATED CONFIGURATION - Using distilled student from stsb-roberta-base-v2 teacher
TEACHER_MODEL_NAME = "stsb-roberta-base-v2"
STUDENT_MODEL_NAME = "all-MiniLM-L12-v2"
BLEND_ALPHA = 0.40 # Alpha used for teacher and blended student evaluation (matches distill script)
TEACHER_TARGET_SCORE = 0.8315 # The Blended Score of the Teacher (stsb-roberta-base-v2 @ alpha 0.40)
ORIGINAL_RAW_SCORE = 0.8309 # The baseline score of the raw L12-v2 model

# Paths to the necessary resources
# Kernel paths: Check both test mode (all-MiniLM-L12-v2) and production mode (stsb-roberta-base-v2)
teacher_suffix = TEACHER_MODEL_NAME.split('/')[-1]
kernel_path_production = script_dir / "data" / f"kernel_{teacher_suffix}.pt"
kernel_path_test = script_dir / "data" / f"kernel_all-MiniLM-L12-v2.pt"
kernel_path = kernel_path_production  # Default, will be updated based on which model is found

# Distilled student paths: matches the format from distill_semantic_kernel.py
# Production mode: distilled_deltanet_student_from_stsb-roberta-base-v2
# Test mode: distilled_deltanet_student_from_all-MiniLM-L12-v2
student_output_path = script_dir / "data" / f"distilled_student_model_{STUDENT_MODEL_NAME}_from_{teacher_suffix}"
# DeltaNet path format (production mode - 768D teacher)
deltanet_student_path = script_dir / "data" / f"distilled_deltanet_student_from_{teacher_suffix}"
# DeltaNet path format (test mode - 384D teacher, no projection head)
deltanet_test_path = script_dir / "data" / f"distilled_deltanet_student_from_all-MiniLM-L12-v2"

stsb_path = script_dir / "data" / "stsbenchmark.tsv.gz"


# ==========================================
# 2. BLENDED MODEL CLASS (for evaluation purposes)
# We use this class to blend the student's output with the kernel
# ==========================================
class BlendedInferenceModel:
    """
    Inference class to apply blending to the *Distilled Student* model.
    """
    
    def __init__(self, student_path: str, kernel_path: str, alpha: float):
        print(f"Loading Student Model from: {student_path}...")
        
        # Check if this is a DeltaNet model
        student_path_obj = Path(student_path)
        is_deltanet = (student_path_obj / "pytorch_model.bin").exists() and DELTANET_AVAILABLE
        
        if is_deltanet:
            print(f"   Detected DeltaNet model - loading...")
            # Load DeltaNet model
            teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"
            dummy_config = {
                'num_heads': 12,
                'fast_decay_init': 0.30,
                'slow_decay_init': 0.85,
            }
            self.student_model = DeltaNetPure6Layer(teacher_model_path, 6, dummy_config).to(device)
            # Load the trained weights
            state_dict = torch.load(str(student_path_obj / "pytorch_model.bin"), map_location=device, weights_only=False)
            self.student_model.load_state_dict(state_dict, strict=False)
            self.student_model.eval()
            student_dim = self.student_model.d_model
            self.is_deltanet = True
            print(f"   ✅ Loaded DeltaNet model ({student_dim}D)")
        else:
            # Load as SentenceTransformer
        self.student_model = SentenceTransformer(str(student_path)).to(device)
        self.student_model.eval()
            student_dim = self.student_model.get_sentence_embedding_dimension()
            self.is_deltanet = False
            print(f"   ✅ Loaded SentenceTransformer model ({student_dim}D)")
        
        print(f"Loading Kernel from: {kernel_path}...")
        kernel_state = torch.load(str(kernel_path), map_location=device, weights_only=False)
        self.kernel = kernel_state['kernel'].to(device)
        kernel_dim = self.kernel.shape[0]  # Kernel is square matrix
        
        self.alpha = alpha
        
        # Handle dimension mismatch: student (384) vs kernel (768)
        self.projection_head = None
        self.use_kernel = True
        if student_dim != kernel_dim:
            print(f"⚠️  Dimension mismatch: Student={student_dim}D, Kernel={kernel_dim}D")
            # Try to load saved projection head from distillation
            # Ensure student_path is a Path object
            student_path_obj = Path(student_path) if not isinstance(student_path, Path) else student_path
            projection_path = student_path_obj / "projection_head.pt"
            
            print(f"   Checking for projection head at: {projection_path}")
            if projection_path.exists():
                print(f"   ✅ Found projection head! Loading from: {projection_path}")
                proj_state = torch.load(str(projection_path), map_location=device, weights_only=False)
                self.projection_head = nn.Linear(student_dim, kernel_dim).to(device)
                self.projection_head.load_state_dict(proj_state['projection_head'])
                self.projection_head.eval()
                print(f"   ✅ Loaded projection head: {student_dim}D -> {kernel_dim}D")
            else:
                print(f"   ⚠️  No saved projection head found at: {projection_path}")
                print(f"   ⚠️  Kernel blending will be skipped (using raw student embeddings only).")
                print(f"   💡 To enable kernel blending, re-run 'distill_semantic_kernel.py' to save the projection head.")
                self.use_kernel = False
        
        print(f"✅ Blended Inference Model Ready (Alpha = {self.alpha:.2f})")

    def encode(self, sentences: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encodes sentences using the blended approach."""
        if isinstance(sentences, str):
            sentences = [sentences]
            
        with torch.no_grad():
            # Get raw student embeddings (V_raw_student) - normalized
            if self.is_deltanet:
                # DeltaNet uses tokenizer and forward_student
                batch_size = kwargs.get('batch_size', 64)
                all_embeddings = []
                for i in range(0, len(sentences), batch_size):
                    batch_sentences = sentences[i:i+batch_size]
                    # Tokenize
                    encoded = self.student_model.tokenizer(
                        batch_sentences,
                        padding=True,
                        truncation=True,
                        return_tensors='pt',
                        max_length=512
                    ).to(device)
                    # Forward pass
                    embeddings, _, _ = self.student_model.forward_student(
                        encoded['input_ids'],
                        encoded['attention_mask']
                    )
                    all_embeddings.append(embeddings)
                V_raw = torch.cat(all_embeddings, dim=0).to(device)
            else:
                # SentenceTransformer uses encode method
            V_raw = self.student_model.encode(
                sentences, 
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=kwargs.get('batch_size', 64)
            ).to(device)
            
            # If kernel blending is disabled (no projection head), just return raw embeddings
            if not self.use_kernel:
                return V_raw.cpu().numpy()
            
            # Project to kernel dimension if needed
            if self.projection_head is not None:
                # Project student embeddings to match kernel dimension
                V_raw_projected = self.projection_head(V_raw)
                V_raw_projected = F.normalize(V_raw_projected, p=2, dim=1)
            else:
                V_raw_projected = V_raw
            
            # Apply the kernel transformation (V_kernel_student)
            V_kernel = torch.matmul(V_raw_projected, self.kernel)
            V_kernel = F.normalize(V_kernel, p=2, dim=1)
            
            # Blend the vectors: V_final = (1-alpha) * V_raw_projected + alpha * V_kernel
            V_final = (1.0 - self.alpha) * V_raw_projected + self.alpha * V_kernel
            
            # Normalize and return as a numpy array
            V_final = F.normalize(V_final, p=2, dim=1)
            return V_final.cpu().numpy()

# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================

def load_stsb_test_data():
    """Reads the STS-B Test Set."""
    print("📚 Reading STS-B Test Set...")
    test_samples = []
    if not stsb_path.exists():
        print("❌ STS-B dataset not found. Please run the previous script once to download it.")
        return []

    with gzip.open(str(stsb_path), 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            if row['split'] == 'test':
                score = float(row['score']) / 5.0  # Normalize 0-5 to 0-1
                test_samples.append({
                    's1': row['sentence1'],
                    's2': row['sentence2'],
                    'score': score
                })
    print(f"✅ Loaded {len(test_samples)} test pairs.")
    return test_samples

def evaluate_original_deltanet():
    """Evaluate the original DeltaNet model (before distillation) for comparison."""
    if not DELTANET_AVAILABLE:
        print("⚠️  DeltaNet not available - cannot evaluate original model")
        return None
    
    original_model_path = "/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH_NEW"
    model_file = Path(original_model_path) / "pytorch_model.bin"
    
    if not model_file.exists():
        print(f"⚠️  Original DeltaNet model not found at {model_file}")
        return None
    
    print("\n" + "="*80)
    print("🔍 EVALUATING ORIGINAL DELTANET MODEL (Before Distillation)")
    print("="*80)
    
    # Load test data
    test_samples = load_stsb_test_data()
    if not test_samples:
        return None
    
    # Load original model
    print(f"\nLoading original DeltaNet model from: {original_model_path}")
    teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"
    dummy_config = {
        'num_heads': 12,
        'fast_decay_init': 0.30,
        'slow_decay_init': 0.85,
    }
    original_model = DeltaNetPure6Layer(teacher_model_path, 6, dummy_config).to(device)
    state_dict = torch.load(str(model_file), map_location=device, weights_only=False)
    original_model.load_state_dict(state_dict, strict=False)
    original_model.eval()
    
    # Evaluate
    scores = []
    human_scores = []
    batch_size = 64
    
    print("\n⚡ Running Inference on Original DeltaNet...")
    with torch.no_grad():
        for i in tqdm(range(0, len(test_samples), batch_size), desc="Evaluating Original"):
            batch = test_samples[i:i+batch_size]
            s1_batch = [x['s1'] for x in batch]
            s2_batch = [x['s2'] for x in batch]
            labels = [x['score'] for x in batch]
            
            # Tokenize and encode
            tokens1 = original_model.tokenizer(
                s1_batch, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(device)
            tokens2 = original_model.tokenizer(
                s2_batch, padding=True, truncation=True, return_tensors='pt', max_length=512
            ).to(device)
            
            emb1, _, _ = original_model.forward_student(tokens1['input_ids'], tokens1['attention_mask'])
            emb2, _, _ = original_model.forward_student(tokens2['input_ids'], tokens2['attention_mask'])
            
            sims = F.cosine_similarity(emb1, emb2, dim=1).cpu().numpy()
            scores.extend(sims)
            human_scores.extend(labels)
    
    # Calculate Spearman
    spearman, _ = spearmanr(scores, human_scores)
    
    print(f"\n✅ Original DeltaNet Score: {spearman:.4f}")
    print("="*80)
    
    return spearman

def evaluate_student():
    """Evaluates Raw Student and Blended Student performance."""
    # Check for student model in all possible locations
    # Priority: Check DeltaNet paths first (test mode, then production mode)
    actual_student_path = None
    actual_kernel_path = None
    use_test_mode = False
    
    if deltanet_test_path.exists() and (deltanet_test_path / "pytorch_model.bin").exists():
        actual_student_path = deltanet_test_path
        actual_kernel_path = kernel_path_test
        use_test_mode = True
        print(f"✅ Found DeltaNet student model (TEST MODE) at: {deltanet_test_path}")
    elif deltanet_student_path.exists() and (deltanet_student_path / "pytorch_model.bin").exists():
        actual_student_path = deltanet_student_path
        actual_kernel_path = kernel_path_production
        use_test_mode = False
        print(f"✅ Found DeltaNet student model (PRODUCTION MODE) at: {deltanet_student_path}")
    elif student_output_path.exists():
        actual_student_path = student_output_path
        actual_kernel_path = kernel_path_production
        use_test_mode = False
        print(f"✅ Found student model at: {student_output_path}")
    else:
        print(f"❌ Distilled student model not found!")
        print(f"   Checked: {deltanet_test_path} (DeltaNet - Test Mode)")
        print(f"   Checked: {deltanet_student_path} (DeltaNet - Production Mode)")
        print(f"   Checked: {student_output_path} (SentenceTransformer)")
        print(f"   Please run 'distill_semantic_kernel.py' first.")
        return
    
    # Check for kernel path (based on which model was found)
    if not actual_kernel_path.exists():
        print(f"❌ Kernel not found at {actual_kernel_path}!")
        if use_test_mode:
            print(f"   For TEST MODE, please run 'semantic_kernel_pipeline.py' with BASE_MODEL_NAME='sentence-transformers/all-MiniLM-L12-v2'")
        else:
            print(f"   For PRODUCTION MODE, please run 'semantic_kernel_pipeline.py' with BASE_MODEL_NAME='stsb-roberta-base-v2'")
        return
    
    # Update kernel_path and BLEND_ALPHA for use in BlendedInferenceModel
    kernel_path = actual_kernel_path
    # Update alpha based on mode: test mode uses 0.70, production uses 0.40
    actual_blend_alpha = 0.70 if use_test_mode else BLEND_ALPHA
    if use_test_mode:
        print(f"   Using TEST MODE alpha: {actual_blend_alpha:.2f}")
    else:
        print(f"   Using PRODUCTION MODE alpha: {actual_blend_alpha:.2f}")

    test_samples = load_stsb_test_data()
    if not test_samples:
        return

    # Initialize models
    print("\nInitializing Distilled Student Models...")
    
    # Check if this is a DeltaNet model
    student_path_obj = Path(actual_student_path)
    is_deltanet = (student_path_obj / "pytorch_model.bin").exists() and DELTANET_AVAILABLE
    
    if is_deltanet:
        print(f"   Loading DeltaNet model for raw evaluation...")
        teacher_model_path = "/workspace/LAM/all-MiniLM-L6-v2"
        dummy_config = {
            'num_heads': 12,
            'fast_decay_init': 0.30,
            'slow_decay_init': 0.85,
        }
        student_raw_model = DeltaNetPure6Layer(teacher_model_path, 6, dummy_config).to(device)
        state_dict = torch.load(str(student_path_obj / "pytorch_model.bin"), map_location=device, weights_only=False)
        student_raw_model.load_state_dict(state_dict, strict=False)
        student_raw_model.eval()
        use_deltanet_for_raw = True
    else:
        student_raw_model = SentenceTransformer(str(actual_student_path)).to(device)
    student_raw_model.eval()
        use_deltanet_for_raw = False
    
    # Convert Path to string for BlendedInferenceModel (it handles both, but be explicit)
    blended_student_model = BlendedInferenceModel(str(actual_student_path), str(kernel_path), actual_blend_alpha)
    
    raw_student_scores = []
    blended_student_scores = []
    human_scores = []

    print("\n⚡ Running Inference on Distilled Student...")

    batch_size = 64
    for i in tqdm(range(0, len(test_samples), batch_size), desc="Evaluating Student"):
        batch = test_samples[i:i+batch_size]
        s1_batch = [x['s1'] for x in batch]
        s2_batch = [x['s2'] for x in batch]
        labels = [x['score'] for x in batch]
        
        # 1. RAW Student Score
        with torch.no_grad():
            if use_deltanet_for_raw:
                # DeltaNet encoding
                tokens1 = student_raw_model.tokenizer(
                    s1_batch, padding=True, truncation=True, return_tensors='pt', max_length=512
                ).to(device)
                tokens2 = student_raw_model.tokenizer(
                    s2_batch, padding=True, truncation=True, return_tensors='pt', max_length=512
                ).to(device)
                r1, _, _ = student_raw_model.forward_student(tokens1['input_ids'], tokens1['attention_mask'])
                r2, _, _ = student_raw_model.forward_student(tokens2['input_ids'], tokens2['attention_mask'])
            else:
                # SentenceTransformer encoding
            r1 = student_raw_model.encode(s1_batch, convert_to_tensor=True, normalize_embeddings=True).to(device)
            r2 = student_raw_model.encode(s2_batch, convert_to_tensor=True, normalize_embeddings=True).to(device)
            curr_raw_sims = F.cosine_similarity(r1, r2, dim=1).cpu().numpy()
            raw_student_scores.extend(curr_raw_sims)
            
        # 2. BLENDED Student Score
        b1 = blended_student_model.encode(s1_batch, batch_size=batch_size)
        b2 = blended_student_model.encode(s2_batch, batch_size=batch_size)
        curr_blended_sims = np.array([np.dot(b1[j], b2[j]) for j in range(len(b1))])
        blended_student_scores.extend(curr_blended_sims)
        
        human_scores.extend(labels)

    # Calculate metrics
    raw_spearman, _ = spearmanr(raw_student_scores, human_scores)
    blended_spearman, _ = spearmanr(blended_student_scores, human_scores)

    # Evaluate original DeltaNet model for comparison
    original_score = evaluate_original_deltanet()

    # Print Results
    print("\n" + "="*80)
    print("🎉 DISTILLED STUDENT FINAL EVALUATION")
    print(f"🎯 Teacher Blended Score (Original Target): {TEACHER_TARGET_SCORE:.4f}")
    print("="*80)
    if original_score is not None:
        print(f"Original DeltaNet Score (Before KD):      {original_score:.4f}")
    print(f"Student Raw Score (After KD):             {raw_spearman:.4f}")
    print(f"Student Blended Score (Student + Kernel): {blended_spearman:.4f} (Alpha={actual_blend_alpha:.2f})")
    print("-" * 80)
    
    # Calculate improvements
    if original_score is not None:
        kd_improvement_vs_original = raw_spearman - original_score
        print(f"📈 KD Improvement (vs. Original DeltaNet): {kd_improvement_vs_original:+.4f}")
    
    kd_improvement = raw_spearman - ORIGINAL_RAW_SCORE
    blended_improvement = blended_spearman - raw_spearman

    print(f"📈 KD Improvement (vs. Original Raw L12):  {kd_improvement:+.4f}")
    print(f"📈 Blending Improvement (on Student):    {blended_improvement:+.4f}")

    # Final conclusion
    if original_score is not None:
        if raw_spearman > original_score:
            print(f"\n✅ KD improved the model! (Original: {original_score:.4f} → Distilled: {raw_spearman:.4f})")
        elif raw_spearman < original_score:
            print(f"\n⚠️  KD decreased performance (Original: {original_score:.4f} → Distilled: {raw_spearman:.4f})")
            print(f"   💡 Consider: More training epochs, different learning rate, or check distillation loss")
        else:
            print(f"\n➡️  KD maintained performance (Original: {original_score:.4f} = Distilled: {raw_spearman:.4f})")

    if raw_spearman > TEACHER_TARGET_SCORE:
        print("\n✅ KD successfully improved the Student model BEYOND the Teacher's knowledge!")
        print(f"   Final model of choice: The Raw Student (Score: {raw_spearman:.4f})")
    elif blended_spearman > raw_spearman:
        print("\n✅ The Blending technique successfully improved the Distilled Student!")
        print(f"   Final model of choice: The Blended Student (Score: {blended_spearman:.4f})")
    elif raw_spearman > ORIGINAL_RAW_SCORE:
        print("\n✅ KD successfully matched the Teacher's knowledge without needing blending.")
        print(f"   Final model of choice: The Raw Student (Score: {raw_spearman:.4f})")
    else:
        print("\n⚠️ Distillation was not fully effective. The original blending method is still the best.")
        
    print("="*80)

if __name__ == "__main__":
    evaluate_student()