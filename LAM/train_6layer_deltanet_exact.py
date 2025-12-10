"""
DeltaNet Training - EXACT REPLICA of all-MiniLM-L6-v2 methodology
Train script matching the original all-MiniLM-L6-v2 approach 1-to-1

ARCHITECTURE DIFFERENCE:
- Original: 6 transformer attention layers
- This: 6 DeltaNet linear attention layers

TRAINING: IDENTICAL
- Same contrastive loss (scale=20)
- Same symmetric loss
- Same in-batch negatives
- Same optimizer/schedule
- Same data format
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import threading
import time
import random
import sys
import argparse
import gzip
import json
import logging
import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import os
from shutil import copyfile

from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).parent))
from final_solution_formula import EnhancedHierarchicalDeltaNet

# ============================================================================
# DELTANET MODEL FOR SENTENCE EMBEDDING
# Replaces AutoModelForSentenceEmbedding from original
# ============================================================================
class DeltaNetForSentenceEmbedding(nn.Module):
    """
    DeltaNet sentence encoder - EXACT structure as original but with DeltaNet layers
    
    Original had: embeddings → 6 transformer layers → pooling → normalize
    This has:     embeddings → 6 DeltaNet layers → pooling → normalize
    """
    def __init__(self, model_name, tokenizer, normalize=True, 
                 d_model=384, num_heads=12, num_layers=6):
        super(DeltaNetForSentenceEmbedding, self).__init__()
        
        self.tokenizer = tokenizer
        self.normalize = normalize
        
        # Load embeddings from base model (for warm start)
        print(f"Loading embeddings from: {model_name}")
        base_model = AutoModel.from_pretrained(model_name)
        
        # Extract embeddings (these are pre-trained, like original)
        self.embeddings = base_model.embeddings
        
        # Get embedding dimension from base model
        embed_dim = base_model.config.hidden_size
        
        # If embedding dim != d_model, add projection
        if embed_dim != d_model:
            self.embedding_projection = nn.Linear(embed_dim, d_model)
            nn.init.xavier_uniform_(self.embedding_projection.weight)
            nn.init.zeros_(self.embedding_projection.bias)
        else:
            self.embedding_projection = None
        
        # REPLACEMENT: DeltaNet layers instead of transformer layers
        self.deltanet_layers = nn.ModuleList([
            EnhancedHierarchicalDeltaNet(
                d_model=d_model,
                num_heads=num_heads,
                use_hierarchical_decay=True,
                use_enhanced_flux=True,
                fast_decay_init=0.30,
                slow_decay_init=0.85,
            )
            for _ in range(num_layers)
        ])
        
        # Layer norms (standard)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Optional: FFN layers (like transformer)
        self.use_ffn = True
        if self.use_ffn:
            self.ffn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
                for _ in range(num_layers)
            ])
            self.ffn_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_layers)
            ])
    
    def forward(self, **kwargs):
        """Forward pass - returns normalized embeddings"""
        input_ids = kwargs['input_ids']
        attention_mask = kwargs['attention_mask']
        
        # Embeddings
        x = self.embeddings(input_ids=input_ids)
        
        # Project if needed
        if self.embedding_projection is not None:
            x = self.embedding_projection(x)
        
        # DeltaNet layers (replacing transformer layers)
        for i in range(len(self.deltanet_layers)):
            # Attention block
            residual = x
            x_attn, _, _ = self.deltanet_layers[i](x, attention_mask)
            x = self.layer_norms[i](residual + x_attn)
            
            # FFN block (optional, like transformer)
            if self.use_ffn:
                residual = x
                x_ffn = self.ffn_layers[i](x)
                x = self.ffn_norms[i](residual + x_ffn)
        
        # Mean pooling
        embeddings = self.mean_pooling(x, attention_mask)
        
        # Normalize
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling - EXACT same as original"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    
    def save_pretrained(self, output_path):
        """Save model - same signature as original"""
        os.makedirs(output_path, exist_ok=True)
        self.tokenizer.save_pretrained(output_path)
        
        # Save config
        config_dict = {
            'model_type': 'DeltaNetForSentenceEmbedding',
            'd_model': self.deltanet_layers[0].hidden_size,
            'num_heads': self.deltanet_layers[0].num_heads,
            'num_layers': len(self.deltanet_layers),
            'normalize': self.normalize,
            'use_ffn': self.use_ffn,
        }
        with open(os.path.join(output_path, 'deltanet_config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save weights
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

# ============================================================================
# TRAINING FUNCTION - EXACT COPY FROM ORIGINAL
# Only model instantiation changes
# ============================================================================
def train_function(index, args, queue):
    """Training function - MATCHES original exactly"""
    
    # Get device (GPU instead of TPU)
    device = torch.device(f'cuda:{index % torch.cuda.device_count()}')
    
    # Instantiate model - THIS IS THE ONLY CHANGE
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = DeltaNetForSentenceEmbedding(
        model_name=args.model,
        tokenizer=tokenizer,
        normalize=True,
        d_model=384,  # Match MiniLM dimension
        num_heads=12,
        num_layers=6
    )
    model = model.to(device)
    
    # Optimizer - EXACT same as original
    optimizer = AdamW(params=model.parameters(), lr=2e-5, correct_bias=True)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.steps,
    )
    
    # Loss - EXACT same as original
    cross_entropy_loss = nn.CrossEntropyLoss()
    max_grad_norm = 1
    
    model.train()
    
    for global_step in tqdm.trange(args.steps, disable=(index != 0)):
        #### Get the batch data
        batch = queue.get()
        
        if len(batch[0]) == 2:  # (anchor, positive)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", 
                            max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", 
                            max_length=args.max_length, truncation=True, padding="max_length")
            
            ### Compute embeddings
            embeddings_a = model(**text1.to(device))
            embeddings_b = model(**text2.to(device))
            
            ### Gather all embeddings (for multi-GPU)
            # NOTE: Remove all_gather for single GPU
            # embeddings_a = torch_xla.core.functions.all_gather(embeddings_a)
            # embeddings_b = torch_xla.core.functions.all_gather(embeddings_b)
            
            ### Compute similarity scores [batch x batch]
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)
            
            ## Symmetric loss as in CLIP - EXACT same as original
            loss = (cross_entropy_loss(scores, labels) + 
                   cross_entropy_loss(scores.transpose(0, 1), labels)) / 2
        
        else:  # (anchor, positive, negative)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", 
                            max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", 
                            max_length=args.max_length, truncation=True, padding="max_length")
            text3 = tokenizer([b[2] for b in batch], return_tensors="pt", 
                            max_length=args.max_length, truncation=True, padding="max_length")
            
            embeddings_a  = model(**text1.to(device))
            embeddings_b1 = model(**text2.to(device))
            embeddings_b2 = model(**text3.to(device))
            
            # NOTE: Remove all_gather for single GPU
            # embeddings_a  = torch_xla.core.functions.all_gather(embeddings_a)
            # embeddings_b1 = torch_xla.core.functions.all_gather(embeddings_b1)
            # embeddings_b2 = torch_xla.core.functions.all_gather(embeddings_b2)
            
            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])
            
            ### Compute similarity scores [batch x 2*batch]
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)
            
            ## One-way loss
            loss = cross_entropy_loss(scores, labels)
        
        # Backward pass - EXACT same as original
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        lr_scheduler.step()
        
        # Save model - EXACT same as original
        if (global_step + 1) % args.save_steps == 0:
            output_path = os.path.join(args.output, str(global_step + 1))
            print(f"Saving model: {output_path}")
            model.save_pretrained(output_path)
    
    output_path = os.path.join(args.output, "final")
    print(f"Saving final model: {output_path}")
    model.save_pretrained(output_path)

# ============================================================================
# DATA PRODUCER - EXACT COPY FROM ORIGINAL
# ============================================================================
def produce_data(args, queue, filepaths, dataset_indices):
    """Data producer - EXACT same as original"""
    global_batch_size = args.batch_size * args.nprocs
    size_per_dataset = int(global_batch_size / args.datasets_per_batch)
    num_same_dataset = int(size_per_dataset / args.batch_size)
    
    print("producer", "global_batch_size", global_batch_size)
    print("producer", "size_per_dataset", size_per_dataset)
    print("producer", "num_same_dataset", num_same_dataset)
    
    datasets = []
    for filepath in filepaths:
        if "reddit_" in filepath:
            data_obj = RedditDataset(filepath)
        else:
            data_obj = Dataset(filepath)
        datasets.append(iter(data_obj))
    
    # Store if dataset is in a 2 col or 3 col format
    num_cols = {idx: len(next(dataset)) for idx, dataset in enumerate(datasets)}
    
    while True:
        texts_in_batch = set()
        batch_format = None
        
        # Add data from several sub datasets
        for _ in range(args.datasets_per_batch):
            valid_dataset = False
            while not valid_dataset:
                data_idx = random.choice(dataset_indices)
                if batch_format is None:
                    batch_format = num_cols[data_idx]
                    valid_dataset = True
                else:
                    valid_dataset = (batch_format == num_cols[data_idx])
            
            # Get data from this dataset
            dataset = datasets[data_idx]
            for _ in range(num_same_dataset):
                for _ in range(args.nprocs):
                    batch_device = []
                    while len(batch_device) < args.batch_size:
                        sample = next(dataset)
                        in_batch = False
                        for text in sample:
                            if text in texts_in_batch:
                                in_batch = True
                                break
                        
                        if not in_batch:
                            for text in sample:
                                texts_in_batch.add(text)
                            batch_device.append(sample)
                    
                    queue.put(batch_device)

class RedditDataset:
    """Reddit dataset handler - EXACT same as original"""
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __iter__(self):
        while True:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if "response" in data and "context" in data:
                        yield [data["response"], data["context"]]

class Dataset:
    """Dataset handler - EXACT same as original"""
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __iter__(self):
        max_dataset_size = 10 * 1000 * 1000
        dataset = []
        data_format = None
        
        while dataset is None or len(dataset) == 0:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        data = data['texts']
                    
                    if data_format is None:
                        data_format = len(data)
                    
                    assert len(data) == data_format
                    
                    if dataset is not None:
                        dataset.append(data)
                        if len(dataset) >= max_dataset_size:
                            dataset = None
                    
                    yield data
        
        # Data loaded. Now stream to the queue
        while True:
            random.shuffle(dataset)
            for data in dataset:
                yield data

# ============================================================================
# MAIN - EXACT COPY FROM ORIGINAL (adapted for GPU)
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Base model for embeddings (warm start)')
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--save_steps', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--nprocs', type=int, default=1,
                       help='Number of GPUs (1 for single GPU)')
    parser.add_argument('--datasets_per_batch', type=int, default=2)
    parser.add_argument('--scale', type=float, default=20,
                       help='Scale for similarity (20 for cossim)')
    parser.add_argument('--data_folder', default="/data")
    parser.add_argument('data_config', nargs='?', default="/workspace/LAM/all-MiniLM-L6-v2/data_config.json",
                       help="A data_config.json file (default: all-MiniLM-L6-v2/data_config.json)")
    parser.add_argument('output', nargs='?', default="/workspace/LAM/deltanet_exact_output",
                       help="Output directory (default: deltanet_exact_output)")
    args = parser.parse_args()
    
    # Validate data_config file exists
    if not os.path.exists(args.data_config):
        print(f"❌ Error: data_config file not found: {args.data_config}")
        print(f"   Please provide a valid data_config.json file or ensure the default exists.")
        sys.exit(1)
    
    # Ensure global batch size is divisible by data_sample_size
    assert (args.batch_size * args.nprocs) % args.datasets_per_batch == 0
    
    logging.info("Output: " + args.output)
    if os.path.exists(args.output):
        print("Output folder already exists.")
        input("Continue?")
    
    # Write train script to output path
    os.makedirs(args.output, exist_ok=True)
    
    data_config_path = os.path.join(args.output, 'data_config.json')
    copyfile(args.data_config, data_config_path)
    
    train_script_path = os.path.join(args.output, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))
    
    # Load data config
    with open(args.data_config) as fIn:
        data_config = json.load(fIn)
    
    queue = mp.Queue(maxsize=100 * args.nprocs)
    
    filepaths = []
    dataset_indices = []
    for idx, data in enumerate(data_config):
        filepaths.append(os.path.join(os.path.expanduser(args.data_folder), data['name']))
        dataset_indices.extend([idx] * data['weight'])
    
    # Start producer
    p = mp.Process(target=produce_data, args=(args, queue, filepaths, dataset_indices))
    p.start()
    
    # Run training (adapted for GPU instead of TPU)
    print("="*80)
    print("DELTANET TRAINING - EXACT ALL-MINILM-L6-V2 METHODOLOGY")
    print("="*80)
    print(f"Architecture: 6 DeltaNet layers (replacing transformer)")
    print(f"Training: Identical to all-MiniLM-L6-v2")
    print(f"  - Contrastive loss with scale={args.scale}")
    print(f"  - Symmetric loss (bidirectional)")
    print(f"  - In-batch negatives")
    print(f"  - {args.steps:,} steps")
    print(f"  - Batch size: {args.batch_size} × {args.nprocs} = {args.batch_size * args.nprocs}")
    print("="*80)
    
    if args.nprocs == 1:
        # Single GPU training
        print("Starting single GPU training...")
        train_function(0, args, queue)
    else:
        # Multi-GPU training
        print(f"Starting {args.nprocs} GPU training...")
        mp.spawn(train_function, args=(args, queue), nprocs=args.nprocs, join=True)
    
    print("Training done")
    p.kill()
    exit()