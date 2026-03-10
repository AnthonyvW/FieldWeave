import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from glob import glob
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "best_model.pth"
INPUT_DIR = r"D:\Projects\tree-core\misc\stats\samples\2677s12"
OUTPUT_DIR = "predicted_masks"

PATCH_SIZE = 1024
STRIDE = 512
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ImageNet normalization
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def preprocess(img):
    img = img / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)
    return torch.tensor(img, dtype=torch.float32)

# -----------------------------
# Sliding window
# -----------------------------
for img_path in tqdm(glob(os.path.join(INPUT_DIR, "*"))):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = img[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            tensor = preprocess(patch).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

            prob_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    prob_map /= np.maximum(count_map, 1)
    mask = (prob_map > THRESHOLD).astype(np.uint8) * 255

    base = os.path.splitext(os.path.basename(img_path))[0]
    cv2.imwrite(os.path.join(OUTPUT_DIR, base + ".png"), mask)

print("Inference complete.")