import torch
import sys
import numpy as np
from pathlib import Path
from mteb import MTEB
from typing import List, Union

# Import LAM pip model
sys.path.insert(0, str(Path(__file__).parent))
try:
    from lam import LAM, InfiniteContextStreamer
    LAM_AVAILABLE = True
except ImportError:
    # Fallback: try parent directory
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from lam import LAM, InfiniteContextStreamer
    LAM_AVAILABLE = True

class LAMWrapper:
    """
    Wrapper to make LAM compatible with MTEB standard.
    Acts like a SentenceTransformer.
    Uses LAM pip model with Sync-512 streaming (production setting).
    """
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize LAM wrapper.
        
        Args:
            model_path: Path to LAM model directory (e.g., '/workspace/LAM/best' or 'LAM-base-v1')
            device: Device to use ('cuda' or 'cpu'). Defaults to auto-detect.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resolve model path
        if model_path is None:
            # Try to find LAM-base-v1 in common locations
            possible_paths = [
                Path(__file__).parent.parent / "LAM-base-v1",
                Path("/workspace/LAM/best"),
                Path("/workspace/LAM/LAM-base-v1"),
                "LAM-base-v1"
            ]
            for path in possible_paths:
                if isinstance(path, str):
                    model_path = path
                    break
                elif path.exists():
                    model_path = str(path)
                    break
            else:
                model_path = "LAM-base-v1"  # Default
        
        print(f"📦 Loading LAM from: {model_path}")
        print(f"   Device: {self.device}")
        
        # Load LAM model
        self.model = LAM(model_path, device=self.device)
        
        # Use Sync-512 streamer (production setting)
        self.streamer = InfiniteContextStreamer(self.model, chunk_size=512)

    def encode(self, sentences: Union[str, List[str]], batch_size: int = 32, 
               convert_to_numpy: bool = True, **kwargs):
        """
        MTEB calls this method to get embeddings.
        
        Args:
            sentences: Single string or list of strings
            batch_size: Batch size for processing
            convert_to_numpy: Whether to return numpy array (MTEB expects this)
            **kwargs: Additional arguments (e.g., dimensions for Matryoshka)
        """
        # Handle single string
        if isinstance(sentences, str):
            sentences = [sentences]
        
        all_embeddings = []
        
        # MTEB passes a list of strings
        for i in range(0, len(sentences), batch_size):
            batch_texts = sentences[i : i + batch_size]
            
            # Use LAM's encode method (handles tokenization internally)
            # For short sequences, use standard encoding
            # For long sequences, use streaming
            
            # Check if any sentence is long (rough estimate: >512 tokens)
            max_len = max(len(text.split()) * 1.3 for text in batch_texts)  # Rough token estimate
            
            if max_len > 512:
                # Use streaming for long sequences
                batch_embs = []
                for text in batch_texts:
                    # Tokenize
                    tokens = self.model.tokenizer.encode(text)
                    input_ids = torch.tensor([tokens.ids if hasattr(tokens, 'ids') else tokens], 
                                            dtype=torch.long)
                    attention_mask = torch.ones_like(input_ids)
                    
                    # Stream embedding
                    emb = self.streamer.stream_embedding(
                        input_ids.cpu(), 
                        attention_mask.cpu(), 
                        verbose=False
                    )
                    batch_embs.append(emb)
                
                embeddings = torch.cat(batch_embs, dim=0)
            else:
                # Use standard encoding for short sequences (faster)
                try:
                    embeddings = self.model.encode(
                        batch_texts, 
                        convert_to_tensor=True,
                        **kwargs
                    )
                except TypeError:
                    # Fallback if convert_to_tensor not supported
                    embeddings = self.model.encode(batch_texts, **kwargs)
                    if isinstance(embeddings, np.ndarray):
                        embeddings = torch.tensor(embeddings)
            
            # Ensure correct shape
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            
            all_embeddings.append(embeddings.cpu())
        
        result = torch.cat(all_embeddings, dim=0)
        
        # Convert to numpy if requested (MTEB expects this)
        if convert_to_numpy:
            result = result.numpy()
        
        return result

if __name__ == "__main__":
    print("🚀 RUNNING OFFICIAL MTEB BENCHMARK FOR LAM")
    
    # 1. Initialize your model
    model = LAMWrapper()
    
    # 2. Select Tasks (Start with STS to confirm quality, then Retrieval)
    # "STS17" is a standard semantic similarity test
    # "SCIDOCS" is a citation retrieval test (good for long context)
    tasks = ["STS17", "SciDocs", "NFCorpus"] 
    
    evaluation = MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder="mteb_results")
    
    print("\n✅ MTEB Complete. Check 'mteb_results' folder.")