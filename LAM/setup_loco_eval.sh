#!/bin/bash
# Setup script for LoCo (Long Context) evaluation
# This sets up the required dependencies and prepares for running LoCo evaluation

set -e  # Exit on error

echo "=========================================="
echo "Setting up LoCo Evaluation Environment"
echo "=========================================="

# Step 1: Uninstall existing sentence-transformers and clone the special fork
echo ""
echo "Step 1: Installing special Sentence-Transformers fork (required for LoCo dataset loading)..."
pip uninstall sentence-transformers -y || true

if [ -d "sentence-transformers" ]; then
    echo "sentence-transformers directory already exists, removing..."
    rm -rf sentence-transformers
fi

git clone https://github.com/jonsaadfalcon/sentence-transformers.git
cd sentence-transformers
pip install -e .
cd ..

echo "✅ Sentence-Transformers fork installed"

# Step 2: Clone M2 repo for LoCo eval script
echo ""
echo "Step 2: Cloning M2 repo for LoCo evaluation script..."
if [ -d "m2" ]; then
    echo "m2 directory already exists, removing..."
    rm -rf m2
fi

git clone https://github.com/HazyResearch/m2.git
echo "✅ M2 repo cloned"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Navigate to m2/bert directory"
echo "2. Run: python embeddings_eval.py --model_name_or_path /workspace/LAM/deltanet_minilm_6layers_FIXED_FROM_SCRATCH --task loco --max_seq_length 32768"
echo ""
echo "Or use the run_loco_eval.sh script to run it automatically."




