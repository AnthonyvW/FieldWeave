from __future__ import annotations

import json
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import load_file

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "model_with_meta.safetensors"
INPUT_DIR = r"D:\Projects\tree-core\output\stitch_test"
OUTPUT_DIR = "predicted_masks"

PATCH_SIZE = 1024
STRIDE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Load categories from model metadata
# -----------------------------
def load_categories_from_model(model_path: str) -> list[dict]:
    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata()
    return json.loads(metadata["categories"])


categories = load_categories_from_model(MODEL_PATH)
num_classes = len(categories) + 1
print(f"Loaded {len(categories)} foreground categories: {[c['name'] for c in categories]}")


# -----------------------------
# Load model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=num_classes,
)

model.load_state_dict(load_file(MODEL_PATH, device=device))
model.to(device)
model.eval()

normalize_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
normalize_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


# -----------------------------
# Inference
# -----------------------------
def get_patch_coords(h: int, w: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    def coords_1d(length: int) -> list[int]:
        positions = list(range(0, length - patch_size + 1, stride))
        if not positions or positions[-1] + patch_size < length:
            positions.append(length - patch_size)
        return positions

    return [(y, x) for y in coords_1d(h) for x in coords_1d(w)]


def predict_label_map(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    pad_h = max(0, PATCH_SIZE - h)
    pad_w = max(0, PATCH_SIZE - w)
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    ph, pw = img.shape[:2]
    prob_map  = torch.zeros(num_classes, ph, pw, device=device)
    count_map = torch.zeros(ph, pw, device=device)

    coords = get_patch_coords(ph, pw, PATCH_SIZE, STRIDE)

    batch = torch.stack([
        torch.from_numpy(img[y : y + PATCH_SIZE, x : x + PATCH_SIZE].transpose(2, 0, 1))
        for y, x in coords
    ]).to(device, dtype=torch.float32, non_blocking=True)
    batch = (batch / 255.0 - normalize_mean) / normalize_std

    with torch.no_grad():
        preds = torch.softmax(model(batch), dim=1)

    for pred, (y, x) in zip(preds, coords):
        prob_map[:, y : y + PATCH_SIZE, x : x + PATCH_SIZE] += pred
        count_map[y : y + PATCH_SIZE, x : x + PATCH_SIZE] += 1

    prob_map /= count_map.clamp(min=1).unsqueeze(0)
    label_map = prob_map[:, :h, :w].argmax(dim=0).cpu().numpy().astype(np.uint8)
    return label_map


# -----------------------------
# Sliding window
# -----------------------------
for img_path in tqdm(glob(os.path.join(INPUT_DIR, "*"))):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)

    label_map = predict_label_map(img)

    mask = (label_map != 0).astype(np.uint8) * 255

    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, base + ".png"), mask)

print("Inference complete.")