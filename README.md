# 3D Corset Generation using Pix2Pix and Spine Angle Embeddings

A deep learning pipeline for generating personalized 3D corset models from human photographs using conditional GANs and medical spine data.

## Features

- **Multi-modal Input**: Combines 2D photos with spine X-ray embeddings
- **3D Reconstruction**: Converts 2D projections to 3D STL models  
- **Medical-Aware**: Incorporates real spine angle data for anatomical accuracy
- **Two-Stage Training**: Generator-only -> Full GAN training pipeline

## Architecture

### Generator (U-Net with Custom Components)
- Spine angle conditioning via `rg_angle_activation`
- Dropout-regularized encoder/decoder (encoder: 0.35, decoder: 0.6)
- Style embedding integration from X-ray data

### 3D Processing Pipeline
- 2D projection generation from STL models (`projections.py`)
- Point cloud reconstruction with MLS smoothing (`backprojection.py`)
- Medical image processing for spine features (`prepare_embedding.py`)

## Training Configuration

- **Image Size**: 256x256
- **Batch Size**: 45
- **Learning Rate**: 2e-4 with ReduceLROnPlateau
- **Loss**: L1 (lambda=300) + L2 + GAN (with label smoothing)
- **Augmentations**: Flips, intensity scaling, random crops

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
cd pix2pix
python train.py

# Generate 3D models
python backprojection.py

Project Structure
corset-generation/
├── pix2pix/              # GAN training module
│   ├── models/           # Generator & Discriminator
│   ├── config.py         # Training configuration
│   ├── train.py          # Training script
│   └── evaluate.py       # Model evaluation
├── projections.py        # 3D->2D projection generation
├── backprojection.py     # 2D->3D reconstruction
├── prepare_embedding.py  # X-ray spine feature extraction
├── points_projector.py   # Point cloud operations
├── transformer.py        # 3D coordinate transformations
├── image_model.py        # Image-based 3D model prediction
└── requirements.txt      # Python dependencies

Key Components
Data Processing
PairImageDataset: Handles paired human-corset images with spine angles

Medical image augmentation with Albumentations

X-ray embedding extraction with OpenCV

Neural Networks
Generator: U-Net with 6 down/7 up layers, feature sizes 64-512

Discriminator: PatchGAN with 5 convolutional layers

Custom activation layers for medical data integration

3D Reconstruction
Multi-angle projection (8 slices at 45 degree intervals)

Point cloud processing with vedo/trimesh

Surface reconstruction with radius-based methods

Results
Successfully generates anatomically plausible 3D corset models

Incorporates real spine curvature data from X-rays

Robust to various body types and poses

Technologies Used
Deep Learning: PyTorch, PyTorch Lightning

3D Processing: Trimesh, Vedo

Image Processing: OpenCV, Albumentations, PIL

Medical Data: Spine angle embeddings from X-rays

License
MIT License - see LICENSE file for details
'@ | Out-File -FilePath "README_NEW.md" -Encoding utf8
