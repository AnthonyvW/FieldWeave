from __future__ import annotations

import json
import os
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

import albumentations as A
import segmentation_models_pytorch as smp


DATASET_ROOT = "dataset_patches"


def load_num_classes(dataset_root: str) -> int:
    """
    Derive the number of output classes from the categories.json written by
    extract_patches.py.  The +1 accounts for the background class (label 0).
    """
    categories_path = os.path.join(dataset_root, "categories.json")
    with open(categories_path) as f:
        categories = json.load(f)
    return len(categories) + 1  # background + N foreground classes


class SegmentationDataset(Dataset):
    def __init__(self, image_paths: list[str], mask_paths: list[str], transform: A.Compose | None = None) -> None:
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mask is a uint8 label map: pixel value == category_id (0 = background)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]       # (C, H, W) float32 tensor
            mask = augmented["mask"]         # (H, W) uint8 tensor

        # CrossEntropyLoss expects (H, W) long, not float
        return image, mask.long()


def get_train_transform() -> A.Compose:
    return A.Compose([
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ElasticTransform(p=0.3),

        # Simulate focus/blur variation
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=1.0),
            A.Defocus(radius=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=9, p=1.0),
        ], p=0.4),

        # Simulate sample quality and lighting variation
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.ImageCompression(quality_range=(60, 100), p=0.2),

        # Noise
        A.GaussNoise(p=0.3),

        # Occlusion
        A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    return A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def split_paths(
    image_paths: list[str],
    mask_paths: list[str],
    val_fraction: float = 0.15,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split parallel path lists into train/val without a Subset wrapper."""
    val_size = int(val_fraction * len(image_paths))
    indices = torch.randperm(len(image_paths)).tolist()
    val_idx, train_idx = indices[:val_size], indices[val_size:]

    train_images = [image_paths[i] for i in train_idx]
    train_masks  = [mask_paths[i]  for i in train_idx]
    val_images   = [image_paths[i] for i in val_idx]
    val_masks    = [mask_paths[i]  for i in val_idx]

    return train_images, train_masks, val_images, val_masks


def train() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = load_num_classes(DATASET_ROOT)
    print(f"Number of classes (including background): {num_classes}")

    all_image_paths = sorted(glob(os.path.join(DATASET_ROOT, "images", "*")))
    all_mask_paths  = sorted(glob(os.path.join(DATASET_ROOT, "masks",  "*")))

    train_images, train_masks, val_images, val_masks = split_paths(all_image_paths, all_mask_paths)

    # Each split gets its own Dataset instance so transforms never bleed across
    train_dataset = SegmentationDataset(train_images, train_masks, transform=get_train_transform())
    val_dataset   = SegmentationDataset(val_images,   val_masks,   transform=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    ).to(device)

    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)
    print("Model device:", next(model.parameters()).device)

    # CrossEntropyLoss handles the multi-class case; Dice in multiclass mode
    ce_loss   = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode="multiclass", classes=num_classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_val = float("inf")

    for epoch in range(40):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
            images = images.to(device)
            masks  = masks.to(device)   # (B, H, W) long

            preds = model(images)       # (B, num_classes, H, W) logits
            loss  = ce_loss(preds, masks) + dice_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} val"):
                images = images.to(device)
                masks  = masks.to(device)
                preds  = model(images)
                loss   = ce_loss(preds, masks) + dice_loss(preds, masks)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)

        print(f"Epoch {epoch+1:>3} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  Saved best model")

    print("Training complete")


if __name__ == "__main__":
    train()