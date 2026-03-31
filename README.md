# COSMOS-2026 Round 3 – Mind Flayer Control

## Project Title
Semantic Segmentation of Off-Road Desert Environments using U-Net with MobileNetV2

## Overview
This project was developed for **COSMOS-2026 Round 3 – Mind Flayer Control**, sponsored by Denovate and organized by GDGOC GRIET.

The objective of this project is to build a **semantic segmentation model** capable of identifying and classifying terrain and environmental elements in off-road desert scenes using annotated synthetic data.

## Model Used
- **Architecture:** U-Net
- **Encoder:** MobileNetV2
- **Framework:** PyTorch

## Dataset Summary
The dataset contains synthetic off-road images with segmentation masks.

### Dataset Split
- **Training Images:** 2857
- **Validation Images:** 317
- **Test Images:** 1002

## Classes Used
The segmentation masks were mapped into the following 7 classes:

| Class ID | Original Label | Category |
|---|---:|---|
| 0 | 200 | Trees |
| 1 | 300 | Lush Bushes |
| 2 | 500 | Dry Grass |
| 3 | 550 | Dry Bushes |
| 4 | 800 | Ground Clutter |
| 5 | 7100 | Flowers / Logs / Rocks |
| 6 | 10000 | Landscape / Sky |

## Final Performance
- **Best Validation Mean IoU:** **55.89%**

## Execution Instructions

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install opencv-python albumentations segmentation-models-pytorch tqdm
