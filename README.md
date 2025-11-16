# MedGMAE: Medical Gaussian Masked Autoencoder

A self-supervised learning framework for 3D medical image analysis using Masked Autoencoders (MAE) with Gaussian Splatting techniques.

## Overview

MedGMAE combines the power of Vision Transformers (ViT) with Gaussian splatting rendering for efficient 3D medical image reconstruction. The framework supports multiple decoder architectures including standard, residual, and hierarchical Gaussian decoders.

## Architecture

```
Input 3D Volume (96×96×96)
    ↓
Vision Transformer Encoder
    ↓ (with masking)
Latent Representations
    ↓
Gaussian Decoder (Multiple Variants)
    ↓
3D Gaussian Rendering
    ↓
Reconstructed Volume
```


## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x or higher
- PyTorch 2.0.1+
- GCC/G++ compiler for CUDA extensions

### Step 1: Create Environment

```bash
conda create -n medgmae python=3.8
conda activate medgmae
```

### Step 2: Install PyTorch

```bash
# For CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install CUDA Extensions

The Gaussian rendering CUDA kernels will be compiled automatically via JIT (Just-In-Time) compilation on first run. Ensure you have:

- CUDA Toolkit installed
- Write permissions to `~/.cache/torch_extensions/`

See INSTALL.md for detailed installation instructions and troubleshooting.

## Usage

### Training

#### Basic Training

```bash
python Gmain.py \
  --model vit_large_12p \
  --batch_size 4 \
  --epochs 2000 \
  --mask_ratio 0.75 \
  --num_gaussians 512
```

#### Distributed Training (Multi-GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=localhost \
  --master_port=12345 \
  Gmain.py \
  --model vit_large_12p \
  --batch_size 4 \
  --epochs 2000 \
  --mask_ratio 0.75 \
  --run_name "medgmae_hierarchical"
```

### Key Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model architecture | `vit_large_12p` |
| `--img_size` | Input volume size | `96` |
| `--patch_size` | Patch size for ViT | `12` |
| `--mask_ratio` | Masking ratio for MAE | `0.75` |
| `--num_gaussians` | Number of Gaussians per patch | `512` |
| `--decoder_type` | Decoder variant | `standard` |
| `--use_hierarchical_gaussians` | Enable hierarchical Gaussians | `False` |
| `--use_residual_gaussians` | Enable residual Gaussians | `False` |
| `--batch_size` | Training batch size | `4` |
| `--epochs` | Number of training epochs | `2000` |

### Decoder Types

- `standard`: Basic Gaussian decoder
- `residual`: Residual Gaussian decoder with skip connections
- `hierarchical`: Hierarchical multi-scale decoder


## Data Preparation

### Dataset Structure

Organize your medical imaging dataset as follows:

```
/path/to/dataset/
├── patient001/
│   └── volume.nii.gz
├── patient002/
│   └── volume.nii.gz
└── ...
```

### Supported Formats

- NIfTI (`.nii`, `.nii.gz`)
- DICOM series
- Other formats supported by MONAI

### Configuration

Update the dataset path in `Gmain.py` or create a config file:

```python
# config_example.py
DATA_ROOT = "/path/to/your/dataset"
TMP_DIR = "/path/to/tmp"  # For temporary files during training
```

## Model Architecture Details

### Vision Transformer Encoder

- Patch size: 12×12×12
- Embedding dimension: 1024 (large), 768 (base)
- Number of layers: 24 (large), 12 (base)
- Attention heads: 16 (large), 12 (base)

### Gaussian Decoder

The decoder predicts Gaussian parameters for each patch:

- **Position** (μ): 3D center of Gaussian
- **Covariance** (Σ): 3D scale and rotation
- **Intensity** (α): Opacity/density value

### CUDA Rendering

Custom CUDA kernels for efficient Gaussian rendering:

- **Dense rendering**: Full volume reconstruction
- **Sparse rendering**: Only computes unmasked regions
- **JIT compilation**: Automatic compilation on first use

## Experiment Tracking

MedGMAE integrates with Weights & Biases (WandB) for experiment tracking:

```bash
# Login to WandB
wandb login

# Training with WandB logging
python Gmain.py --run_name "my_experiment" --wandb_project "medgmae"
```

Tracked metrics include:
- Training/validation loss
- Reconstruction quality (MSE, PSNR, SSIM)
- Learning rate schedules
- GPU memory usage

## Citation

If you use this code in your research, please cite:

```bibtex
@article{medgmae2024,
  title={MedGMAE: Medical Gaussian Masked Autoencoder for 3D Medical Image Analysis},
  author={Anonymous},
  journal={Under Review},
  year={2024}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions and discussions, please open an issue in the GitHub repository.

---

**Note**: This is an anonymous submission for peer review. Full code and pretrained models will be released upon acceptance.
