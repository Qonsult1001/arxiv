#!/usr/bin/env python3
"""
🚀 QUICK START: Breakthrough Training Launcher

This script makes it easy to start training with automatic:
- Environment validation
- GPU detection and configuration
- Dataset verification
- Best practice recommendations
"""

import sys
import subprocess
import os
from pathlib import Path

def check_environment():
    """Validate training environment"""
    print("="*80)
    print("🔍 ENVIRONMENT CHECK")
    print("="*80)
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("❌ Python 3.8+ required")
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            issues.append("⚠️  No CUDA GPU detected (training will be slow)")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    # Check required packages
    required_packages = {
        'transformers': 'transformers',
        'datasets': 'datasets',
        'scipy': 'scipy',
        'tqdm': 'tqdm',
        'numpy': 'numpy'
    }
    
    for name, package in required_packages.items():
        try:
            __import__(name)
            print(f"✅ {package}")
        except ImportError:
            issues.append(f"❌ {package} not installed")
    
    # Check disk space
    import shutil
    disk = shutil.disk_usage("/workspace" if Path("/workspace").exists() else ".")
    free_gb = disk.free / (1024**3)
    if free_gb < 10:
        issues.append(f"⚠️  Low disk space: {free_gb:.1f} GB free")
    else:
        print(f"✅ Disk space: {free_gb:.1f} GB free")
    
    return issues

def check_data():
    """Check for required datasets"""
    print("\n" + "="*80)
    print("📊 DATASET CHECK")
    print("="*80)
    
    data_dir = Path(__file__).parent / "data"
    
    datasets = {
        "AllNLI.jsonl.gz": "AllNLI (hard negatives)",
        "WikiAnswers_1M.jsonl.gz": "WikiAnswers (paraphrases)",
    }
    
    available = []
    missing = []
    
    for filename, description in datasets.items():
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"✅ {description}: {size_mb:.1f} MB")
            available.append(description)
        else:
            print(f"⚠️  {description}: Not found (will use HuggingFace datasets)")
            missing.append(description)
    
    # Check HuggingFace cache
    cache_dir = Path("/workspace/.cache/huggingface/datasets")
    if cache_dir.exists():
        cache_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        print(f"✅ HuggingFace cache: {cache_size / (1024**3):.1f} GB")
    
    return len(available) > 0 or cache_dir.exists()

def estimate_training_time():
    """Estimate training time"""
    print("\n" + "="*80)
    print("⏱️  TRAINING TIME ESTIMATE")
    print("="*80)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            
            # Rough estimates based on GPU
            if "A100" in gpu_name or "H100" in gpu_name:
                hours = 10-12
                print(f"🚀 GPU: {gpu_name}")
                print(f"   Estimated: {hours} hours for 50K steps")
                print(f"   Speed: ~1.2 it/s")
            elif "RTX" in gpu_name or "V100" in gpu_name:
                hours = 20-25
                print(f"🚀 GPU: {gpu_name}")
                print(f"   Estimated: {hours} hours for 50K steps")
                print(f"   Speed: ~0.6 it/s")
            else:
                hours = 30-40
                print(f"🚀 GPU: {gpu_name}")
                print(f"   Estimated: {hours} hours for 50K steps")
                print(f"   Speed: ~0.4 it/s")
            
            print(f"\n💡 Pro tip: Can reduce to 30K steps for faster training")
            print(f"   (Expected Spearman: 0.81-0.83 instead of 0.82-0.85)")
        else:
            print("⚠️  No GPU detected - training will be very slow (days/weeks)")
    except:
        print("⚠️  Could not estimate training time")

def recommend_config():
    """Recommend configuration based on hardware"""
    print("\n" + "="*80)
    print("⚙️  RECOMMENDED CONFIGURATION")
    print("="*80)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_mem_gb >= 40:
                batch_size = 128
                grad_accum = 4
                print("🔥 Large GPU detected (40+ GB)")
            elif gpu_mem_gb >= 24:
                batch_size = 64
                grad_accum = 8
                print("💪 Medium GPU detected (24-40 GB)")
            elif gpu_mem_gb >= 16:
                batch_size = 32
                grad_accum = 16
                print("✅ Standard GPU detected (16-24 GB)")
            else:
                batch_size = 16
                grad_accum = 32
                print("⚠️  Small GPU detected (<16 GB)")
            
            effective_batch = batch_size * grad_accum
            print(f"\n   Recommended config:")
            print(f"   - batch_size: {batch_size}")
            print(f"   - gradient_accumulation_steps: {grad_accum}")
            print(f"   - Effective batch size: {effective_batch}")
            
            if effective_batch < 512:
                print(f"\n   ⚠️  Warning: Effective batch ({effective_batch}) < 512")
                print(f"   Contrastive learning works best with batch size 512+")
                print(f"   Performance may be slightly reduced")
            else:
                print(f"\n   ✅ Effective batch size is optimal for contrastive learning")
            
            return batch_size, grad_accum
    except:
        pass
    
    return 64, 8

def main():
    """Main launcher"""
    print("\n" + "="*80)
    print("🚀 BREAKTHROUGH TRAINING LAUNCHER")
    print("="*80)
    print("\n📝 This will train your DeltaNet to achieve:")
    print("   🎯 Target: 0.82-0.85 Spearman on STS-B")
    print("   🚀 Using: Latest research from 2024-2025")
    print("   ⏱️  Duration: ~20-40 hours depending on GPU\n")
    
    # Check environment
    issues = check_environment()
    
    if any("❌" in issue for issue in issues):
        print("\n" + "="*80)
        print("❌ CRITICAL ISSUES FOUND")
        print("="*80)
        for issue in issues:
            if "❌" in issue:
                print(issue)
        print("\n💡 Please install missing packages:")
        print("   pip install torch transformers datasets scipy tqdm numpy")
        return 1
    
    # Check data
    has_data = check_data()
    if not has_data:
        print("\n⚠️  Warning: No local datasets found")
        print("   Will download from HuggingFace (may take time)")
    
    # Estimate time
    estimate_training_time()
    
    # Recommend config
    batch_size, grad_accum = recommend_config()
    
    # Warnings
    if issues:
        print("\n" + "="*80)
        print("⚠️  WARNINGS")
        print("="*80)
        for issue in issues:
            if "⚠️" in issue:
                print(issue)
    
    # Ready to start
    print("\n" + "="*80)
    print("✅ ENVIRONMENT READY")
    print("="*80)
    
    print("\n📋 Quick Start Options:")
    print("\n1️⃣  FULL TRAINING (50K steps - Best results)")
    print("   python breakthrough_fine_tune_082plus.py")
    print("   Expected: 0.82-0.85 Spearman")
    print("   Duration: 20-40 hours")
    
    print("\n2️⃣  FAST TRAINING (30K steps - Good results)")
    print("   python breakthrough_fine_tune_082plus.py --total-steps 30000")
    print("   Expected: 0.81-0.83 Spearman")
    print("   Duration: 12-24 hours")
    
    print("\n3️⃣  TEST RUN (5K steps - Verify setup)")
    print("   python breakthrough_fine_tune_082plus.py --total-steps 5000")
    print("   Expected: 0.76-0.78 Spearman")
    print("   Duration: 2-4 hours")
    
    print("\n" + "="*80)
    print("💡 TIPS FOR SUCCESS")
    print("="*80)
    print("""
1. Monitor GPU usage: nvidia-smi -l 1
   Should show >80% utilization

2. Watch training logs for:
   - Contrastive loss decreasing
   - Spearman score increasing
   - No NaN or Inf values

3. Training saves:
   - Best model: pytorch_model.bin (auto-saved)
   - Checkpoints: checkpoint_XXXX.pt (every 2K steps)

4. Resume if interrupted:
   Set config['resume_from_step'] = XXXX
   
5. Expected performance trajectory:
   - Step 10K: ~0.78-0.80 Spearman
   - Step 20K: ~0.80-0.82 Spearman
   - Step 50K: ~0.82-0.85 Spearman

6. If performance plateaus:
   - Check batch size is effective 512+
   - Verify hard negatives are enabled
   - Ensure temperature = 0.05
""")
    
    print("="*80)
    user_input = input("\n🚀 Ready to start? (y/n): ")
    
    if user_input.lower() == 'y':
        print("\n🚀 Starting breakthrough training...")
        print("="*80)
        
        # Run training
        script_path = Path(__file__).parent / "breakthrough_fine_tune_082plus.py"
        if script_path.exists():
            os.system(f"python {script_path}")
        else:
            print("❌ Training script not found!")
            print(f"   Expected at: {script_path}")
            return 1
    else:
        print("\n👋 Exiting. Good luck with your training!")
        return 0

if __name__ == "__main__":
    sys.exit(main())