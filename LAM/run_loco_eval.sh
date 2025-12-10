#!/bin/bash
# Run LoCo evaluation on your model
# Usage: ./run_loco_eval.sh [model_path] [max_seq_length]

set -e  # Exit on error

# Default model path
MODEL_PATH="${1:-/workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH}"
MAX_SEQ_LENGTH="${2:-32768}"

echo "=========================================="
echo "Running LoCo Evaluation"
echo "=========================================="
echo "Model Path: $MODEL_PATH"
echo "Max Sequence Length: $MAX_SEQ_LENGTH"
echo "=========================================="
echo ""

# Check if setup has been done
if [ ! -d "m2/bert" ]; then
    echo "❌ M2 repo not found. Running setup first..."
    bash setup_loco_eval.sh
fi

# Navigate to m2/bert directory
cd m2/bert

# Run the evaluation
echo "Starting LoCo evaluation..."
echo "This will take ~30-60 minutes on L40 GPU..."
echo ""

python loco_eval.py \
  --model deltanet \
  --checkpoint "$MODEL_PATH" \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --batch-size 4

echo ""
echo "=========================================="
echo "LoCo Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results should show NDCG@10 scores for each task and an average."
echo "Your linear model with resonance flux should achieve 90%+ average!"
echo ""

