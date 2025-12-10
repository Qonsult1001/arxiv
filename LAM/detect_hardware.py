#!/usr/bin/env python3
"""
Hardware Detection Script
Detects and reports GPU/CUDA information for training documentation
"""
import torch
import sys
import platform
from pathlib import Path

def detect_hardware():
    """Detect and report hardware configuration"""
    
    print("="*80)
    print("HARDWARE DETECTION REPORT")
    print("="*80)
    print()
    
    # System Information
    print("SYSTEM INFORMATION")
    print("-" * 80)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print()
    
    # CUDA/GPU Information
    print("GPU/CUDA INFORMATION")
    print("-" * 80)
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            
            # Check if TF32 is supported (Ampere+)
            if props.major >= 8:
                print(f"  ✅ TF32 Supported (Ampere+)")
            elif props.major >= 7:
                print(f"  ⚠️  TF32 Not Supported (Volta/Turing)")
            else:
                print(f"  ❌ TF32 Not Supported (Pre-Volta)")
            print()
    else:
        print("⚠️  No CUDA GPU detected - training will use CPU")
        print("   CPU training is significantly slower")
        print()
    
    # PyTorch Configuration
    print("PYTORCH CONFIGURATION")
    print("-" * 80)
    print(f"Float32 Matmul Precision: {torch.get_float32_matmul_precision()}")
    print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print()
    
    # Training Configuration Summary
    print("TRAINING CONFIGURATION SUMMARY")
    print("-" * 80)
    print("Batch Size: 256")
    print("Gradient Accumulation: 4 steps")
    print("Effective Batch Size: 256")
    print("Max Sequence Length: 128 tokens")
    print("Total Steps: 200,000")
    print("Learning Rate: 2e-5 (peak)")
    print()
    
    # Memory Estimation
    if cuda_available:
        print("MEMORY ESTIMATION")
        print("-" * 80)
        # Rough estimation: batch_size * seq_len * d_model * 4 bytes * overhead
        estimated_memory_mb = 256 * 128 * 384 * 4 * 3 / (1024**2)  # Rough estimate with overhead
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Estimated Training Memory: ~{estimated_memory_mb:.0f} MB")
        print(f"Available GPU Memory: {total_memory_gb:.2f} GB")
        if estimated_memory_mb / 1024 < total_memory_gb * 0.8:
            print("✅ Memory should be sufficient")
        else:
            print("⚠️  May need to reduce batch size or use gradient checkpointing")
        print()
    
    print("="*80)
    print("END OF REPORT")
    print("="*80)

if __name__ == "__main__":
    detect_hardware()
    
    # Save to file
    output_file = Path("hardware_report.txt")
    print(f"\n✅ To save report, run: python detect_hardware.py > hardware_report.txt")

