"""
Configuration Example for MedGMAE Training

This file provides example configurations for training MedGMAE.
Copy this file and modify the paths according to your setup.
"""

import os

# ============================================================================
# Data Configuration
# ============================================================================

# Root directory containing your medical imaging dataset
# Expected structure: DATA_ROOT/patient_id/volume.nii.gz
DATA_ROOT = "/path/to/your/dataset"

# Temporary directory for storing intermediate files during training
# This directory should have sufficient space (recommended: 100GB+)
TMP_DIR = "/path/to/tmp"

# ============================================================================
# Training Configuration
# ============================================================================

# Model architecture
MODEL_TYPE = "vit_large_12p"  # Options: vit_base, vit_large, vit_large_12p

# Input volume size
IMG_SIZE = 96  # 3D volume will be resized to (96, 96, 96)

# Patch size for Vision Transformer
PATCH_SIZE = 12  # Results in 8x8x8 patches for 96^3 volume

# Training hyperparameters
BATCH_SIZE = 4  # Adjust based on GPU memory
NUM_EPOCHS = 2000
LEARNING_RATE = 1.5e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 40

# Masked Autoencoder configuration
MASK_RATIO = 0.75  # Percentage of patches to mask

# Gaussian Splatting configuration
NUM_GAUSSIANS = 512  # Number of Gaussians per patch

# Decoder configuration
DECODER_TYPE = "hierarchical"  # Options: standard, residual, hierarchical, cgs
USE_HIERARCHICAL_GAUSSIANS = True
USE_RESIDUAL_GAUSSIANS = False

# ============================================================================
# Distributed Training Configuration
# ============================================================================

# GPU devices to use (comma-separated)
CUDA_VISIBLE_DEVICES = "0,1,2,3"

# Number of GPUs per node
NUM_GPUS = 4

# Master address and port for distributed training
MASTER_ADDR = "localhost"
MASTER_PORT = 12345

# ============================================================================
# Experiment Tracking Configuration
# ============================================================================

# Weights & Biases configuration
USE_WANDB = True
WANDB_PROJECT = "medgmae"
WANDB_ENTITY = None  # Your WandB username/team (optional)
RUN_NAME = "medgmae_experiment"

# Checkpoint saving
SAVE_CKPT_DIR = "./ckpts"
SAVE_FREQ = 20  # Save checkpoint every N epochs

# ============================================================================
# Environment Variables
# ============================================================================

def setup_environment():
    """
    Set up environment variables for training.
    Call this function before starting training.
    """
    # Set temporary directory
    os.environ['TMPDIR'] = TMP_DIR
    os.environ['WANDB_TEMP'] = TMP_DIR
    os.environ['WANDB_CACHE_DIR'] = TMP_DIR

    # Limit CPU threads to prevent memory issues
    os.environ['OMP_NUM_THREADS'] = '10'
    os.environ['MKL_NUM_THREADS'] = '10'

    # Set CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

    print(f"Environment configured:")
    print(f"  - Data root: {DATA_ROOT}")
    print(f"  - Temp dir: {TMP_DIR}")
    print(f"  - CUDA devices: {CUDA_VISIBLE_DEVICES}")

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Print configuration
    setup_environment()

    print("\nTraining Configuration:")
    print(f"  - Model: {MODEL_TYPE}")
    print(f"  - Image size: {IMG_SIZE}^3")
    print(f"  - Patch size: {PATCH_SIZE}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Epochs: {NUM_EPOCHS}")
    print(f"  - Mask ratio: {MASK_RATIO}")
    print(f"  - Decoder type: {DECODER_TYPE}")
    print(f"  - Num Gaussians: {NUM_GAUSSIANS}")

    print("\nTo start training, run:")
    print(f"CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} torchrun \\")
    print(f"  --nproc_per_node={NUM_GPUS} \\")
    print(f"  --master_addr={MASTER_ADDR} \\")
    print(f"  --master_port={MASTER_PORT} \\")
    print(f"  Gmain.py \\")
    print(f"  --model {MODEL_TYPE} \\")
    print(f"  --batch_size {BATCH_SIZE} \\")
    print(f"  --epochs {NUM_EPOCHS} \\")
    print(f"  --mask_ratio {MASK_RATIO} \\")
    print(f"  --decoder_type {DECODER_TYPE} \\")
    print(f"  --run_name {RUN_NAME}")
