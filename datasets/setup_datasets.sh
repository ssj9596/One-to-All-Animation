#!/bin/bash

set -e

echo "=========================================="
echo "  Setting up One-to-All Training Datasets"
echo "=========================================="

echo ""
echo "[Step 1/3] Downloading datasets from HuggingFace..."
python download_datasets.py

echo ""
echo "[Step 2/3] Preparing training data (unzip & process)..."
python prepare_training_data.py

echo ""
echo "[Step 3/3] Preparing pose pool..."
python prepare_pose_pool.py

echo ""
echo "=========================================="
echo "  ✅ Setup completed successfully!"
echo "=========================================="
echo "Training data: $(pwd)/opensource_dataset/"
echo "Pose pool: $(pwd)/opensource_pose_pool/"
