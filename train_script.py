import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# =========================
# PATHS
# =========================
TRAIN_IMG_DIR = "/content/dataset/train_data/Offroad_Segmentation_Training_Dataset/train/Color_Images"
TRAIN_MASK_DIR = "/content/dataset/train_data/Offroad_Segmentation_Training_Dataset/train/Segmentation"
VAL_IMG_DIR = "/content/dataset/train_data/Offroad_Segmentation_Training_Dataset/val/Color_Images"
VAL_MASK_DIR = "/content/dataset/train_data/Offroad_Segmentation_Training_Dataset/val/Segmentation"

train_images = sorted(glob(os.path.join(TRAIN_IMG_DIR, "*.png")))
train_masks  = sorted(glob(os.path.join(TRAIN_MASK_DIR, "*.png")))
val_images   = sorted(glob(os.path.join(VAL_IMG_DIR, "*.png")))
val_masks    = sorted(glob(os.path.join(VAL_MASK_DIR, "*.png")))

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_VALUES = [200, 300, 500, 550, 800, 7100, 10000]
value_to_class = {v: i for i, v in enumerate(CLASS_VALUES)}
NUM_CLASSES = len(CLASS_VALUES)

# =========================
# MASK CONVERSION
# =========================
def convert_mask(mask):
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for orig_val, class_id in value_to_class.items():
        new_mask[mask == orig_val] = class_id
    return new_mask

# =========================
# TRANSFORMS
# =========================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])

# =========================
# DATASET
# =========================
class OffroadSegDataset(Dataset):
    def __init__(self, image_paths, mask_paths=None, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_paths is not None:
            mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_UNCHANGED)
            mask = convert_mask(mask)
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
            return image, mask
        else:
            augmented = self.transform(image=image)
            image = augmented["image"]
            return image

# =========================
# LOADERS
# =========================
train_dataset = OffroadSegDataset(train_images, train_masks, transform=train_transform)
val_dataset = OffroadSegDataset(val_images, val_masks, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# =========================
# MODEL
# =========================
model = smp.Unet(
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAIN
# =========================
best_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss = {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Best model saved")

print("Training complete")
