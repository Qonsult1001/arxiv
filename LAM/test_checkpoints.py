#!/usr/bin/env python3
"""
Scan DeltaNet checkpoints in /workspace/LAM/deltanet_all_minilm_replica
and report the highest Pearson on STS-B, comparing against original LAM-base-v1.

Scans for:
- .pt checkpoint files in the directory
- pytorch_model.bin files in subdirectories (e.g., checkpoint-1000/pytorch_model.bin)
"""
import os
from pathlib import Path
import torch
import sys

# Enable TensorFloat32 (TF32) for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

sys.path.append('/workspace/LAM')
from stsb_evaluation import evaluate_checkpoint

# Original LAM-base-v1 performance (reference)
ORIGINAL_VALIDATION = 0.8399
ORIGINAL_TEST = 0.7779
ORIGINAL_VALIDATION_SPEARMAN = 0.8399  # Approximate - update if you have exact value
ORIGINAL_TEST_SPEARMAN = 0.7779  # Approximate - update if you have exact value


def list_checkpoints(directory, min_step=0):
    directory = Path(directory)
    if not directory.exists():
        return []
    
    # Scan for both .pt files and pytorch_model.bin files
    files = []
    
    # First, look for .pt files in the directory
    files.extend([p for p in directory.iterdir() if p.suffix == '.pt' and p.is_file()])
    
    # Also check for pytorch_model.bin in the root directory itself
    root_pytorch_model = directory / "pytorch_model.bin"
    if root_pytorch_model.exists() and root_pytorch_model.is_file():
        files.append(root_pytorch_model)
    
    # Then, recursively look for pytorch_model.bin files in subdirectories
    for subdir in directory.iterdir():
        if subdir.is_dir():
            pytorch_model = subdir / "pytorch_model.bin"
            if pytorch_model.exists():
                files.append(pytorch_model)
    
    def step_key(p):
        # For pytorch_model.bin files, extract step from parent directory name
        if p.name == 'pytorch_model.bin':
            parent_name = p.parent.name
            # If parent is the same as the search directory, it's a root-level file
            # Use a high number to put it at the end (or you could try to extract from filename)
            if parent_name == directory.name:
                return 999997  # Put root-level pytorch_model.bin before other non-numeric files
            # Handle checkpoint-XXXX format
            if parent_name.startswith('checkpoint-'):
                try:
                    step_num = int(parent_name.split('-')[1])
                    return step_num
                except (ValueError, IndexError):
                    return 999999
            # Handle other directory names
            try:
                return int(parent_name)
            except ValueError:
                return 999999
        
        # For .pt files, use existing logic
        name = p.stem
        # Handle step0020_p0.8418.pt format
        if name.startswith('step'):
            try:
                # Extract number after "step" (e.g., "step0020" -> 20)
                step_part = name.split('_')[0]  # "step0020"
                step_num = int(step_part[4:])  # Remove "step" prefix
                return step_num
            except (ValueError, IndexError):
                return 999999
        # Handle checkpoint_best.pt explicitly
        if name == 'checkpoint_best':
            return 999998  # Put checkpoint_best before other non-numeric files
        # Handle checkpoint_XXXX.pt format (e.g., checkpoint_1000.pt)
        if name.startswith('checkpoint_'):
            try:
                # Extract number after "checkpoint_" (e.g., "checkpoint_1000" -> 1000)
                step_num = int(name.split('_')[1])
                return step_num
            except (ValueError, IndexError):
                pass
        # Handle checkpoint_early_XXXX.pt format
        if name.startswith('checkpoint_early_'):
            try:
                step_num = int(name.split('_')[2])
                return step_num
            except (ValueError, IndexError):
                pass
        # Fallback: try to extract from last part
        parts = name.split('_')
        try:
            return int(parts[-1])
        except Exception:
            # For files without numeric steps, use a large number
            return 999999  # Put other non-numeric checkpoints at the end
    
    # Filter checkpoints with step >= min_step (or special files like checkpoint_best, pytorch_model.bin)
    # Include pytorch_model.bin (step_key 999997) and checkpoint_best (step_key 999998) regardless of min_step
    filtered = [f for f in files if step_key(f) >= min_step or step_key(f) >= 999997]
    return sorted(filtered, key=step_key)


def main(directory=None):
    if directory is None:
        ck_dir = "/workspace/LAM/best/"  # Updated to scan 8K extended checkpoints
    else:
        ck_dir = directory
    min_step = 0  # Scan all checkpoints (change to 1000 if you only want step 1000+)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoints = list_checkpoints(ck_dir, min_step=min_step)
    if not checkpoints:
        print(f"No checkpoints (.pt or pytorch_model.bin) found in {ck_dir}")
        return
    
    print("="*80)
    print("CHECKPOINT EVALUATION - Tracking Progress vs Original LAM-base-v1")
    print("="*80)
    print(f"Original LAM-base-v1: Validation={ORIGINAL_VALIDATION:.4f}, Test={ORIGINAL_TEST:.4f}")
    print(f"Testing checkpoints from step {min_step} onward in {ck_dir}")
    print("="*80)
    
    results = {}
    best_val = None
    best_test = None
    best_val_spearman = None
    best_test_spearman = None
    for ck in checkpoints:
        # Load checkpoint metadata to get kernel info (if .pt file)
        kernel_info = None
        if ck.suffix == '.pt':
            try:
                checkpoint_data = torch.load(str(ck), map_location='cpu', weights_only=False)
                use_kernel = checkpoint_data.get('use_kernel_blending', False)
                kernel_alpha = checkpoint_data.get('kernel_blend_alpha', None)
                kernel_trained_on = checkpoint_data.get('kernel_trained_on', None)
                
                if use_kernel:
                    kernel_info = f"Kernel α={kernel_alpha:.2f}"
                    if kernel_trained_on:
                        kernel_info += f" ({kernel_trained_on})"
                else:
                    kernel_info = "No kernel"
            except Exception as e:
                kernel_info = "N/A"
        else:
            # For pytorch_model.bin, we can't easily read metadata
            kernel_info = "?"
        
        # Evaluate on validation split
        res_val = evaluate_checkpoint(str(ck), device=device, split='validation')
        pearson_val = res_val.get('pearson_cosine')
        spearman_val = res_val.get('spearman_cosine')
        
        # Evaluate on test split
        res_test = evaluate_checkpoint(str(ck), device=device, split='test')
        pearson_test = res_test.get('pearson_cosine')
        spearman_test = res_test.get('spearman_cosine')
        
        # Use a readable name for display
        if ck.name == 'pytorch_model.bin':
            # If it's in the root directory, just show the filename
            if ck.parent.name == Path(ck_dir).name:
                display_name = "pytorch_model.bin"
            else:
                display_name = f"{ck.parent.name}/{ck.name}"
        else:
            display_name = ck.name
        
        results[display_name] = {
            'val': pearson_val, 
            'test': pearson_test,
            'val_spearman': spearman_val,
            'test_spearman': spearman_test,
            'kernel_info': kernel_info
        }
        
        if pearson_val is not None:
            gap_val = ORIGINAL_VALIDATION - pearson_val
            if best_val is None or pearson_val > best_val[1]:
                best_val = (display_name, pearson_val, gap_val)
        
        if pearson_test is not None:
            gap_test = ORIGINAL_TEST - pearson_test
            if best_test is None or pearson_test > best_test[1]:
                best_test = (display_name, pearson_test, gap_test)
        
        if spearman_val is not None:
            gap_val_spearman = ORIGINAL_VALIDATION_SPEARMAN - spearman_val
            if best_val_spearman is None or spearman_val > best_val_spearman[1]:
                best_val_spearman = (display_name, spearman_val, gap_val_spearman)
        
        if spearman_test is not None:
            gap_test_spearman = ORIGINAL_TEST_SPEARMAN - spearman_test
            if best_test_spearman is None or spearman_test > best_test_spearman[1]:
                best_test_spearman = (display_name, spearman_test, gap_test_spearman)
    
    print("\n" + "="*80)
    print("SUMMARY: Test Spearman (MTEB Main Score) & Other Metrics per checkpoint (vs Original)")
    print("="*80)
    print(f"{'Checkpoint':<30} {'Kernel':<18} {'Test Spear⭐':<14} {'Test Pearson':<14} {'Val Pearson':<14} {'Val Spear':<14} {'Beat Target':<14} {'Status':<15}")
    print("-" * 145)
    
    def sort_key(x):
        name = x[0]
        # Handle pytorch_model.bin files (checkpoint-XXXX/pytorch_model.bin)
        if '/pytorch_model.bin' in name:
            dir_name = name.split('/')[0]
            if dir_name.startswith('checkpoint-'):
                try:
                    step_num = int(dir_name.split('-')[1])
                    return step_num
                except (ValueError, IndexError):
                    return 999999
            try:
                return int(dir_name)
            except ValueError:
                return 999999
        # Handle step0020_p0.8418.pt format
        if name.startswith('step'):
            try:
                step_part = name.split('_')[0]  # "step0020"
                step_num = int(step_part[4:])  # Remove "step" prefix
                return step_num
            except (ValueError, IndexError):
                return 999999
        # Handle checkpoint_best.pt explicitly
        if name == 'checkpoint_best':
            return 999998  # Put checkpoint_best before other non-numeric files
        # Handle checkpoint_XXXX.pt format (e.g., checkpoint_1000.pt)
        if name.startswith('checkpoint_'):
            try:
                # Extract number after "checkpoint_" (e.g., "checkpoint_1000.pt" -> 1000)
                step_num = int(name.split('_')[1].split('.')[0])
                return step_num
            except (ValueError, IndexError):
                pass
        # Handle checkpoint_early_XXXX.pt format
        if name.startswith('checkpoint_early_'):
            try:
                step_num = int(name.split('_')[2].split('.')[0])
                return step_num
            except (ValueError, IndexError):
                pass
        # Fallback: try to extract from last part
        try:
            return int(name.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return 999999  # Put other non-numeric checkpoints at the end
    
    # Track best scores seen so far (for "beat this" feature)
    best_test_spear_so_far = None
    best_test_spear_checkpoint = None
    best_test_pear_so_far = None
    best_test_pear_checkpoint = None
    
    for name, res in sorted(results.items(), key=sort_key):
        val_pearson = res.get('val')
        val_spearman = res.get('val_spearman')
        test_pearson = res.get('test')
        test_spearman = res.get('test_spearman')
        
        # Track best scores
        if test_spearman is not None:
            if best_test_spear_so_far is None or test_spearman > best_test_spear_so_far:
                best_test_spear_so_far = test_spearman
                best_test_spear_checkpoint = name
        if test_pearson is not None:
            if best_test_pear_so_far is None or test_pearson > best_test_pear_so_far:
                best_test_pear_so_far = test_pearson
                best_test_pear_checkpoint = name
    
    # Now print the table
    for name, res in sorted(results.items(), key=sort_key):
        val_pearson = res.get('val')
        val_spearman = res.get('val_spearman')
        test_pearson = res.get('test')
        test_spearman = res.get('test_spearman')
        
        # Format strings
        val_pearson_str = f"{val_pearson:.4f}" if val_pearson is not None else "N/A"
        val_spearman_str = f"{val_spearman:.4f}" if val_spearman is not None else "N/A"
        test_pearson_str = f"{test_pearson:.4f}" if test_pearson is not None else "N/A"
        test_spearman_str = f"{test_spearman:.4f}" if test_spearman is not None else "N/A"
        
        # Determine what needs to be beaten
        target_to_beat = "N/A"
        if test_spearman is not None and best_test_spear_so_far is not None:
            # Check if this matches or exceeds the best score
            if test_spearman >= best_test_spear_so_far:
                if test_spearman == best_test_spear_so_far:
                    if name == best_test_spear_checkpoint:
                        target_to_beat = "⭐ BEST"
                    else:
                        target_to_beat = "⭐ TIED"
                else:
                    target_to_beat = "✅ NEW BEST"
            else:
                # Show how much more needed to beat best
                gap_to_best = best_test_spear_so_far - test_spearman
                target_to_beat = f"📈 +{gap_to_best:.4f}"
            
            # Status based on Test Spearman (MTEB main score)
            gap_test_spear = ORIGINAL_TEST_SPEARMAN - test_spearman
            if test_spearman > ORIGINAL_TEST_SPEARMAN:
                status = "✅ EXCEEDED"
            elif test_spearman >= best_test_spear_so_far if best_test_spear_so_far else False:
                status = "🎯 BEST"
            else:
                status = f"📉 -{gap_test_spear:.4f}"
        elif test_pearson is not None:
            gap_test = ORIGINAL_TEST - test_pearson
            status = "✅ EXCEEDED" if test_pearson > ORIGINAL_TEST else f"📉 -{gap_test:.4f}"
        else:
            status = "N/A"
        
        # Get kernel info
        kernel_info_str = res.get('kernel_info', 'N/A')
        if kernel_info_str is None:
            kernel_info_str = 'N/A'
        
        # Print in order: Checkpoint, Kernel, Test Spear (MTEB main), Test Pearson, Val Pearson, Val Spear, Target to Beat, Status
        print(f"{name:<30} {kernel_info_str:<18} {test_spearman_str:<14} {test_pearson_str:<14} {val_pearson_str:<14} {val_spearman_str:<14} {target_to_beat:<14} {status:<15}")
    
    print("\n" + "="*80)
    print("TARGETS TO BEAT")
    print("="*80)
    print(f"Original Target (Test Spearman): {ORIGINAL_TEST_SPEARMAN:.4f}")
    if best_test_spear_so_far is not None:
        print(f"Best Score So Far: {best_test_spear_so_far:.4f} (from {best_test_spear_checkpoint})")
        gap_to_original = ORIGINAL_TEST_SPEARMAN - best_test_spear_so_far
        if gap_to_original <= 0:
            print(f"✅ EXCEEDED original target by {abs(gap_to_original):.4f}!")
        else:
            print(f"📊 Need {gap_to_original:.4f} more to beat original target")
            print(f"🎯 Target: Beat {best_test_spear_so_far:.4f} (current best) OR reach {ORIGINAL_TEST_SPEARMAN:.4f} (original)")
    else:
        print("No test scores available yet")
    print("="*80)
    
    print("\n" + "="*80)
    print("BEST CHECKPOINTS (Prioritizing Test Spearman - MTEB Main Score)")
    print("="*80)
    
    # Show Test Spearman first (MTEB main score)
    if best_test_spearman:
        name, spearman, gap = best_test_spearman
        pearson = results[name].get('test')
        print(f"⭐ BEST TEST SPEARMAN (MTEB Main Score): {name}")
        print(f"  Spearman: {spearman:.4f}")
        if pearson is not None:
            print(f"  Pearson: {pearson:.4f}")
        print(f"  Gap to Original: {gap:.4f} ({'✅ EXCEEDED' if spearman > ORIGINAL_TEST_SPEARMAN else '📉 Still behind'})")
        if gap < 0:
            print(f"  🎉 IMPROVEMENT: {abs(gap):.4f} better than original!")
        elif gap < 0.01:
            print(f"  ✅ Very close! Only {gap:.4f} away from original")
        else:
            print(f"  📊 Need {gap:.4f} more to match original")
    
    # Then Test Pearson
    if best_test:
        name, pearson, gap = best_test
        spearman = results[name].get('test_spearman')
        print(f"\nBEST TEST (Pearson): {name}")
        print(f"  Pearson: {pearson:.4f}")
        if spearman is not None:
            print(f"  Spearman: {spearman:.4f}")
        print(f"  Gap to Original: {gap:.4f} ({'✅ EXCEEDED' if pearson > ORIGINAL_TEST else '📉 Still behind'})")
        if gap < 0:
            print(f"  🎉 IMPROVEMENT: {abs(gap):.4f} better than original!")
        elif gap < 0.01:
            print(f"  ✅ Very close! Only {gap:.4f} away from original")
        else:
            print(f"  📊 Need {gap:.4f} more to match original")
    
    # Validation scores (secondary)
    if best_val_spearman:
        name, spearman, gap = best_val_spearman
        pearson = results[name].get('val')
        print(f"\nBEST VALIDATION (Spearman): {name}")
        print(f"  Spearman: {spearman:.4f}")
        if pearson is not None:
            print(f"  Pearson: {pearson:.4f}")
        print(f"  Gap to Original: {gap:.4f} ({'✅ EXCEEDED' if spearman > ORIGINAL_VALIDATION_SPEARMAN else '📉 Still behind'})")
        if gap < 0:
            print(f"  🎉 IMPROVEMENT: {abs(gap):.4f} better than original!")
        elif gap < 0.01:
            print(f"  ✅ Very close! Only {gap:.4f} away from original")
        else:
            print(f"  📊 Need {gap:.4f} more to match original")
    
    if best_val:
        name, pearson, gap = best_val
        spearman = results[name].get('val_spearman')
        print(f"\nBEST VALIDATION (Pearson): {name}")
        print(f"  Pearson: {pearson:.4f}")
        if spearman is not None:
            print(f"  Spearman: {spearman:.4f}")
        print(f"  Gap to Original: {gap:.4f} ({'✅ EXCEEDED' if pearson > ORIGINAL_VALIDATION else '📉 Still behind'})")
        if gap < 0:
            print(f"  🎉 IMPROVEMENT: {abs(gap):.4f} better than original!")
        elif gap < 0.01:
            print(f"  ✅ Very close! Only {gap:.4f} away from original")
        else:
            print(f"  📊 Need {gap:.4f} more to match original")
    
    print("="*80)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Use directory from command line argument
        main(directory=sys.argv[1])
    else:
        # Default to /workspace/LAM/save
        main()