"""
Train script for a single file - FIXED VERSION

This version handles torch_xla import issues and provides multiple execution modes:
1. TPU mode (if torch_xla works)
2. CPU/GPU mode (if torch_xla fails)
3. Test mode (for verification)

Need to set the TPU address first for TPU mode:
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
"""

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
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch
import os
from shutil import copyfile

# Try to import torch_xla, fallback to CPU/GPU if it fails
try:
    import torch_xla
    import torch_xla.core
    import torch_xla.core.functions
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
    TORCH_XLA_AVAILABLE = True
    print("✅ torch_xla imported successfully - TPU mode available")
except ImportError as e:
    print(f"⚠️  torch_xla import failed: {e}")
    print("   Falling back to CPU/GPU mode")
    TORCH_XLA_AVAILABLE = False
    # Create dummy functions for CPU/GPU mode
    class DummyXM:
        @staticmethod
        def is_master_ordinal():
            return True
        @staticmethod
        def xla_device():
            return torch.device('cpu')
        @staticmethod
        def master_print(msg):
            print(f"[MASTER] {msg}")
        @staticmethod
        def save(state_dict, path):
            torch.save(state_dict, path)
        @staticmethod
        def optimizer_step(optimizer, barrier=True):
            optimizer.step()
    
    xm = DummyXM()
    
    class DummyXMP:
        @staticmethod
        def spawn(func, args, nprocs, start_method='fork'):
            # Run single process for CPU/GPU mode
            func(0, args[0], args[1])
    
    xmp = DummyXMP()
    
    class DummyFunctions:
        @staticmethod
        def all_gather(tensor):
            return tensor
    
    torch_xla = type('torch_xla', (), {'core': type('core', (), {'functions': DummyFunctions()})})()

# Use PyTorch's built-in optimizer instead of transformers to avoid torch_xla conflicts
from torch.optim import AdamW
import math

# Custom learning rate scheduler to replace transformers
class LinearScheduler:
    def __init__(self, optimizer, num_warmup_steps, num_training_steps):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.step_count = 0
        # Store the base learning rate
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self):
        self.step_count += 1
        if self.step_count < self.num_warmup_steps:
            lr_scale = float(self.step_count) / float(max(1, self.num_warmup_steps))
        else:
            lr_scale = max(0.0, float(self.num_training_steps - self.step_count) / 
                          float(max(1, self.num_training_steps - self.num_warmup_steps)))
        
        # Set learning rate to base_lr * scale (not multiply current lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_scale

print("✅ Using PyTorch-only components to avoid torch_xla conflicts")

# Import your LAM components
import sys
sys.path.append('/workspace')
from LAM_model import LAMForSentenceEmbedding
from LAM_config import config as base_lam_config, finalize_config

# Use PyTorch's built-in optimizer and scheduler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

### FIX ###: Added a dedicated evaluation function.
# This is the most critical change. It allows us to measure performance on unseen data.
def evaluate_model(model, tokenizer, validation_data, args, device):
    """Evaluates the model on the validation set."""
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    num_batches = 0
    
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    with torch.no_grad(): # Disable gradient calculations for efficiency
        for i in range(0, len(validation_data), args.batch_size):
            batch = validation_data[i:i + args.batch_size]
            if not batch:
                continue

            # We assume validation data is always pairs for simplicity
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

            embeddings_a = model(**text1.to(device))
            embeddings_b = model(**text2.to(device))

            embeddings_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
            
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)
            
            loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(0, 1), labels)) / 2
            
            total_loss += loss.item()
            num_batches += 1

    model.train() # Set the model back to training mode
    return total_loss / num_batches if num_batches > 0 else 0

def train_function(index, args, queue, validation_data):
    # 1. Load and finalize your custom LAM config
    lam_config = finalize_config(base_lam_config)
    lam_config['max_length'] = args.max_length  # Ensure max length is set
    
    # 2. Instantiate LAM Model (This loads the tokenizer and .bin weights internally!)
    # The LAMForSentenceEmbedding constructor looks for the .bin file in args.model path.
    model = LAMForSentenceEmbedding(lam_config, pretrained_path=args.model)
    
    # 3. Use the tokenizer loaded by your model for data preparation
    tokenizer = model.tokenizer 

    # 4. Attach a compatible save_pretrained method (Crucial for TPU saving)
    # The base LAM model snippet is missing this, so we define a compatible one.
    def lam_save_pretrained(self, output_path):
        if xm.is_master_ordinal():
            self.tokenizer.save_pretrained(output_path)
            # Save the full config for re-loading later
            import json, os
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, "lam_config.json"), 'w') as f:
                json.dump(self.config, f, indent=4)
        
        # Save the full state dict of your LAM model
        xm.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    # Bind the method to the model instance
    model.save_pretrained = lam_save_pretrained.__get__(model, LAMForSentenceEmbedding)

    ### Train Loop
    device = xm.xla_device() if TORCH_XLA_AVAILABLE else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"🚀 Training on device: {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Instantiate optimizer with proper settings for linear models
    optimizer = AdamW(
        params=model.parameters(), 
        lr=args.lr,
        weight_decay=0.1,  # Higher weight decay for linear models
        betas=(0.9, 0.95)  # Better betas for linear models
    )

    ### FIX ###: Replaced custom scheduler with a standard, robust one from `transformers`.
    # This provides a more reliable warmup and decay curve, preventing initial instability.
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.steps,
    )
    
    # Add gradient accumulation and mixed precision
    gradient_accumulation_steps = args.gradient_accumulation
    scaler = torch.amp.GradScaler('cuda', enabled=True)  # Mixed precision
    
    # Now we train the model
    cross_entropy_loss = nn.CrossEntropyLoss()
    max_grad_norm = 1.0

    model.train()
    
    ### FIX ###: Added tracking for the best model based on validation loss.
    best_val_loss = float('inf')
   
    for global_step in tqdm.trange(args.steps, disable=False):
        #### Get the batch data
        while True:
            batch = queue.get()
            
            # Check if all items in batch have the same format
            batch_format = len(batch[0])
            mixed_format = False
            for item in batch:
                if len(item) != batch_format:
                    mixed_format = True
                    break
            
            if not mixed_format:
                break  # We have a consistent batch, proceed with training
        
        if batch_format == 2: #(anchor, positive)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

            ### Compute embeddings with mixed precision
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                embeddings_a = model(**text1.to(device))
                embeddings_b = model(**text2.to(device))
                
                ### Gather all embedings 
                if TORCH_XLA_AVAILABLE:
                    embeddings_a = torch_xla.core.functions.all_gather(embeddings_a)
                    embeddings_b = torch_xla.core.functions.all_gather(embeddings_b)
                else:
                    # For CPU/GPU mode, no gathering needed
                    pass

                ### Normalize embeddings for cosine similarity
                embeddings_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
                embeddings_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
                
                ### Compute similarity scores 512 x 512
                scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
            
                ### Compute cross-entropy loss
                labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
                
                ## Symmetric loss as in CLIP
                loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(0, 1), labels)) / 2
                loss = loss / gradient_accumulation_steps  # Scale loss for gradient accumulation

        elif batch_format == 3:   #(anchor, positive, negative)
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")
            text3 = tokenizer([b[2] for b in batch], return_tensors="pt", max_length=args.max_length, truncation=True, padding="max_length")

            embeddings_a  = model(**text1.to(device))
            embeddings_b1 = model(**text2.to(device))
            embeddings_b2 = model(**text3.to(device))

            if TORCH_XLA_AVAILABLE:
                embeddings_a  = torch_xla.core.functions.all_gather(embeddings_a)
                embeddings_b1 = torch_xla.core.functions.all_gather(embeddings_b1)
                embeddings_b2 = torch_xla.core.functions.all_gather(embeddings_b2)
            else:
                # For CPU/GPU mode, no gathering needed
                pass

            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])

            ### Normalize embeddings for cosine similarity
            embeddings_a = torch.nn.functional.normalize(embeddings_a, p=2, dim=1)
            embeddings_b = torch.nn.functional.normalize(embeddings_b, p=2, dim=1)
            
            ### Compute similarity scores 512 x 1024
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * args.scale
        
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
            
            ## One-way loss
            loss = cross_entropy_loss(scores, labels)

        
        # 1. Backward Pass (Use scaler for CUDA, standard for XLA)
        # NOTE: We DO NOT call optimizer.zero_grad() here. Gradients accumulate.
        
        if TORCH_XLA_AVAILABLE:
            # XLA's internal AMP/optimizer handles unscaling
            loss.backward()
        else:
            # CUDA/Mixed Precision must use the scaler for the backward pass
            scaler.scale(loss).backward() 

        # 2. Check if it's time to step the optimizer
        # (Global step is 0-indexed, so +1 ensures we step on steps 4, 8, 12, etc.)
        if (global_step + 1) % gradient_accumulation_steps == 0:
            
            # 3. Apply Gradient Clipping, Stepping, and Scaling
            if TORCH_XLA_AVAILABLE:
                # XLA step handles unscaling, clipping, and stepping
                # We explicitly pass clip norm to ensure it is applied before the step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) 
                xm.optimizer_step(optimizer, barrier=True)
            else:
                # CUDA/Mixed Precision step logic
                scaler.unscale_(optimizer) # Must unscale before clipping/stepping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update() # Update the scale for the next cycle

            # 4. Update Scheduler and Zero Gradients
            lr_scheduler.step()
            
            # Zero gradients only AFTER the optimizer step is complete
            # NOTE: This replaces the incorrect zero_grad() before loss.backward()
            optimizer.zero_grad() 
        
        # 5. Print progress (Your requested logging fix)
        # Log the full, unscaled loss value, and the current LR for monitoring.
        if global_step % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # The loss was divided by the accumulation steps, so multiply it back for reporting
            xm.master_print(f"Step {global_step}/{args.steps}, Loss: {loss.item()*gradient_accumulation_steps:.4f}, LR: {current_lr:.6e}")

        # Logging and Validation
        if global_step > 0 and (global_step + 1) % args.eval_steps == 0:
            if xm.is_master_ordinal():
                val_loss = evaluate_model(model, tokenizer, validation_data, args, device)
                
                current_lr = optimizer.param_groups[0]['lr']
                train_loss = loss.item() * gradient_accumulation_steps
                
                xm.master_print(f"Step {global_step+1}/{args.steps} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    xm.master_print(f"🎉 New best model found! Val Loss: {best_val_loss:.4f}. Saving to 'best_model'...")
                    output_path = os.path.join(args.output, "best_model")
                    model.save_pretrained(output_path)

        #Save model (only final - no intermediate checkpoints)
        # Removed intermediate saves to avoid creating many files
          
            
    output_path = os.path.join(args.output, "final")
    xm.master_print("save model final: "+ output_path)
    model.save_pretrained(output_path)

def produce_data(args, queue, filepaths, dataset_indices):
    global_batch_size = args.batch_size * args.nprocs
    size_per_dataset = int(global_batch_size / args.datasets_per_batch)
    num_same_dataset = int(size_per_dataset / args.batch_size)
    print("producer", "global_batch_size", global_batch_size)
    print("producer", "size_per_dataset", size_per_dataset)
    print("producer", "num_same_dataset", num_same_dataset)

    # --- FIX: Initialize continuous data streams for full dataset ---
    print("[PRODUCER] Initializing continuous data streams for full dataset...")
    
    # 1. Create a Dataset object for each file
    dataset_objects = []
    for filepath in filepaths:
        print(f"[PRODUCER] Creating Dataset object for {filepath}...")
        if "reddit_" in filepath:
            data_obj = RedditDataset(filepath)
        else:
            data_obj = Dataset(filepath)
        dataset_objects.append(data_obj)

    # 2. Get a live, continuous iterator for each object
    dataset_iterators = [iter(obj) for obj in dataset_objects]
    print(f"[PRODUCER] {len(dataset_iterators)} continuous iterators started. Streaming ALL data.")
    # --- END FIX ---

    while True:
        # We remove the random sampling from all_data, and instead draw from the iterators
        batch_format = None     # 2 vs 3 col format for this batch
        
        # Add data from several sub datasets (controlled by datasets_per_batch)
        for _ in range(args.datasets_per_batch):
            
            # Use random.choice on dataset_indices to select a dataset based on its weight
            data_idx = random.choice(dataset_indices)
            dataset_iterator = dataset_iterators[data_idx]
            
            # Get data from this dataset for the required number of mini-batches
            for _ in range(num_same_dataset):
                for _ in range(args.nprocs):
                    batch_device = []   # A batch for one device
                    
                    # Fill the batch by drawing directly from the continuous stream
                    for _ in range(args.batch_size):
                        try:
                            # Get the next sample from the file iterator
                            data = next(dataset_iterator)
                        except StopIteration:
                            # This should not be hit due to 'while True' in Dataset.__iter__, 
                            # but handles an unexpected end of file by restarting the stream
                            restarted_obj = dataset_objects[data_idx] 
                            dataset_iterator = iter(restarted_obj)
                            dataset_iterators[data_idx] = dataset_iterator
                            print(f"⚠️ Dataset {filepaths[data_idx]} iterator exhausted, restarting stream.")
                            data = next(dataset_iterator)

                        batch_device.append(data)

                    queue.put(batch_device)
                      

class RedditDataset:
    """
    A class that handles the reddit data files
    """
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
    """
    A class that handles one dataset
    """
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        # Stream data directly instead of loading all into memory
        # This prevents hanging on large files like NQ-train_pairs.jsonl.gz
        while True:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        data = data['texts']
                    
                    # Ensure consistent format and at least 2 columns
                    if len(data) >= 2:
                        yield data
                

def create_test_data_config():
    """Create a simple test data config for testing"""
    test_config = [
        {
            "name": "test_data.jsonl.gz",
            "weight": 1
        }
    ]
    
    # Create test data file
    test_data_path = "/workspace/test_data.jsonl.gz"
    with gzip.open(test_data_path, "wt") as f:
        # Create some simple test pairs
        test_pairs = [
            ["Hello world", "Hi there"],
            ["Machine learning is great", "AI is amazing"],
            ["Python is fun", "Coding is enjoyable"],
            ["The weather is nice", "It's a beautiful day"],
            ["I love pizza", "Pizza is delicious"],
        ]
        
        for pair in test_pairs:
            data = {"texts": pair}
            f.write(json.dumps(data) + "\n")
    
    # Create test config file
    config_path = "/workspace/test_data_config.json"
    with open(config_path, "w") as f:
        json.dump(test_config, f, indent=2)
    
    return config_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/workspace/all-MiniLM-L6-v2')
    parser.add_argument('--steps', type=int, default=100)  # Reduced for testing
    # parser.add_argument('--save_steps', type=int, default=50)  # Removed - only saving final model
    parser.add_argument('--batch_size', type=int, default=4)  # Reduced for testing
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--nprocs', type=int, default=1)  # Single process for testing
    parser.add_argument('--datasets_per_batch', type=int, default=1, help="Number of datasets per batch")
    parser.add_argument('--scale', type=float, default=20, help="Use 20 for cossim, and 1 when you work with unnormalized embeddings with dot product")
    parser.add_argument('--data_folder', default="/workspace", help="Folder with your dataset files")
    parser.add_argument('--test_mode', action='store_true', help="Run in test mode with generated data")
    
    ### FIX ###: Added arguments for validation and better scheduler control.
    parser.add_argument('--validation_file', required=True, help="Path to the gzipped validation JSONL file.")
    parser.add_argument('--eval_steps', type=int, default=500, help="Run validation every N steps.")
    parser.add_argument('--lr', type=float, default=4e-4, help="Peak learning rate.")
    parser.add_argument('--warmup_steps', type=int, default=500, help="Number of warmup steps for the scheduler.")
    parser.add_argument('--gradient_accumulation', type=int, default=16, help="Steps to accumulate gradients. Increases effective batch size.")
    parser.add_argument('data_config', nargs='?', help="A data_config.json file")
    parser.add_argument('output', nargs='?', default='/workspace/test_output', help="Output directory")
    args = parser.parse_args()

    effective_batch_size = args.batch_size * args.nprocs * args.gradient_accumulation
    print("="*80)
    print(f"Effective Batch Size: {args.batch_size} (per-device) * {args.nprocs} (devices) * {args.gradient_accumulation} (accum) = {effective_batch_size}")
    if effective_batch_size < 512:
         print("⚠️ WARNING: Effective batch size is small for contrastive loss. Consider increasing batch_size or gradient_accumulation.")
    print("="*80)

    # Test mode - create test data
    if args.test_mode or not args.data_config:
        print("🧪 Running in TEST MODE")
        args.data_config = create_test_data_config()
        args.output = '/workspace/test_output'
        # Don't override steps - use the provided value
        args.batch_size = max(args.batch_size, 2)  # Ensure minimum batch size
        
        # Check test data size and adjust batch size if needed
        import gzip, json
        test_data_size = 0
        with gzip.open('/workspace/test_data.jsonl.gz', 'rt') as f:
            for line in f:
                test_data_size += 1
        
        if args.batch_size > test_data_size:
            print(f"⚠️  Batch size {args.batch_size} > test data size {test_data_size}")
            print(f"   Adjusting batch size to {test_data_size}")
            args.batch_size = test_data_size
            
        args.nprocs = 1
        print(f"   Created test data config: {args.data_config}")
        print(f"   Output directory: {args.output}")
        print(f"   Training steps: {args.steps}")
        print(f"   Batch size: {args.batch_size} (test data has {test_data_size} samples)")

    # Ensure global batch size is divisble by data_sample_size
    assert (args.batch_size*args.nprocs) % args.datasets_per_batch == 0

    logging.info("Output: "+args.output)
    if os.path.exists(args.output):
        print("Output folder already exists. Continuing...")
        # Remove interactive prompt for automated training

    # Write train script to output path
    os.makedirs(args.output, exist_ok=True)

    data_config_path = os.path.join(args.output, 'data_config.json')
    copyfile(args.data_config, data_config_path)

    train_script_path = os.path.join(args.output, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    #Load data config
    with open(args.data_config) as fIn:
        data_config = json.load(fIn)
        
    ### FIX ###: Load validation data before starting processes.
    # We load it all into memory. For very large validation sets, streaming would be needed.
    print(f"Loading validation data from {args.validation_file}...")
    validation_data = []
    with gzip.open(os.path.join(args.data_folder, args.validation_file), "rt") as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data, dict):
                data = data['texts']
            if len(data) >= 2:
                 validation_data.append(data[:2]) # Ensure it's a pair
    print(f"✅ Loaded {len(validation_data)} validation samples.")

    queue = mp.Queue(maxsize=100*args.nprocs)
    
    filepaths = []
    dataset_indices = []
    for idx, data in enumerate(data_config):
        filepaths.append(os.path.join(os.path.expanduser(args.data_folder), data['name']))
        dataset_indices.extend([idx]*data['weight'])

    # Start producer
    p = mp.Process(target=produce_data, args=(args, queue, filepaths, dataset_indices))
    p.start()

    # Run training
    print("Start processes:", args.nprocs)
    if TORCH_XLA_AVAILABLE:
        print("🚀 Using TPU mode")
        xmp.spawn(train_function, args=(args, queue, validation_data), nprocs=args.nprocs, start_method='fork')
    else:
        print("🚀 Using CPU/GPU mode")
        train_function(0, args, queue, validation_data)
    
    print("Training done")
    print("It might be that not all processes exit automatically. In that case you must manually kill this process.")
    print("With 'pkill python' you can kill all remaining python processes")
    p.kill()
    exit()
