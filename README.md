# Diabetic Foot Ulcer Segmentation using U-Net

This repository contains the implementation of a U-Net architecture for diabetic foot ulcer segmentation using PyTorch. The model is trained on the DFUC2022 dataset and achieves accurate segmentation results.

## Features

- **U-Net Architecture**: A specialized medical image segmentation framework based on convolutional neural networks.
- **Symmetric Design**: Employs a symmetric encoder-decoder architecture with skip connections for preserving spatial information.
- **Advanced Components**:
  - BatchNormalization for training stability
  - ReLU activations for non-linearity
  - Double convolution blocks for enhanced feature extraction
  - MaxPooling for spatial dimension reduction
  - Transposed convolutions for upsampling

## Installation

```bash

# Create a conda environment
conda create -n dfuc_seg python=3.8
conda activate dfuc_seg

# Install required packages
pip install torch torchvision
pip install opencv-python
pip install albumentations
pip install matplotlib
pip install numpy
```

## Dataset Preparation

1. Download the DFUC2022 dataset from [Kaggle](https://www.kaggle.com/datasets/pabodhamallawa/dfuc2022-train-release/data)

2. Extract and organize the dataset as follows:
```
DFUC2022/
├── DFUC2022_train_release/
│   ├── DFUC2022_train_images/
│   │   ├── train/
│   │   └── val/
│   └── DFUC2022_train_masks/
│       ├── train/
│       └── val/
```

3. Update the data paths in the code:
```python
train_image_dir = "path/to/DFUC2022_train_images/train"
train_mask_dir  = "path/to/DFUC2022_train_masks/train"
val_image_dir   = "path/to/DFUC2022_train_images/val"
val_mask_dir    = "path/to/DFUC2022_train_masks/val"
```

## Model Architecture

The U-Net architecture consists of:

1. **Encoder Path (Contracting)**:
   - 5 levels of downsampling
   - Double convolution blocks with BatchNorm and ReLU
   - Channel sizes: 3 → 64 → 128 → 256 → 512 → 1024

2. **Decoder Path (Expanding)**:
   - 4 levels of upsampling
   - Skip connections from encoder
   - Channel sizes: 1024 → 512 → 256 → 128 → 64

3. **Output Layer**:
   - 1x1 convolution for binary segmentation
   - Output shape: (2, H, W)

## Training

Training parameters:
- Optimizer: Adam (lr=1e-4)
- Loss: CrossEntropyLoss
- Batch size: 4
- Epochs: 10
- Image size: 256x256

## Results

Our model achieves the following performance metrics:

- **Training Metrics**:
  - Loss: 0.0375
  - Accuracy: 0.9861
  - IoU: 1.9774
  - Dice Coefficient: 2.5700

- **Validation Metrics**:
  - Loss: 0.0358
  - Accuracy: 0.9867
  - IoU: 2.0873
  - Dice Coefficient: 2.6429
  

## Visualization

The training process includes visualization of:
- Training and validation loss curves
- Accuracy metrics
- IoU (Intersection over Union)
- Dice coefficient
- Sample predictions with original images, ground truth masks, and predicted masks

## Citation

Mallawa, P. (2022). DFUC2022 Train Release [Data set]. Kaggle. https://www.kaggle.com/datasets/pabodhamallawa/dfuc2022-train-release/data

## Acknowledgments

- DFUC2022 dataset providers
- U-Net architecture original paper
- PyTorch community



