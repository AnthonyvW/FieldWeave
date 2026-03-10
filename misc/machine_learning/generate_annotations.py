from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from pycocotools import mask as coco_mask_utils
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "best_model.pth"
INPUT_DIR = r"D:\Projects\tree-core\misc\machine_learning\not_annotated\batch_03"
DATASET_ROOT = "dataset_patches"

# Output zip path — CVAT will import this directly
OUTPUT_ZIP = "cvat_import.zip"

# Name of the COCO subset. CVAT uses this for both the image subfolder
# and the annotation filename (instances_<SUBSET_NAME>.json).
SUBSET_NAME = "default"

PATCH_SIZE = 1024
STRIDE = 512
MIN_COMPONENT_AREA = 100

# Set to the 1-based image index to resume from.
# Set to 1 to start from the beginning.
RESUME_FROM_INDEX = 1


# -----------------------------
# Load category metadata
# -----------------------------
def load_categories(dataset_root: str) -> list[dict]:
    """
    Load the category list written by extract_patches.py.
    Background (label 0) is not included — it has no COCO category entry.
    """
    with open(os.path.join(dataset_root, "categories.json")) as f:
        return json.load(f)


categories = load_categories(DATASET_ROOT)
num_classes = len(categories) + 1  # background + N foreground classes
print(f"Loaded {len(categories)} foreground categories: {[c['name'] for c in categories]}")


# -----------------------------
# Load model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=num_classes,
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    _dummy = torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE, device=device)
    _ = model(_dummy)
print("Model warmed up.")

normalize_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
normalize_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


# -----------------------------
# Inference
# -----------------------------
def get_patch_coords(h: int, w: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    def coords_1d(length: int) -> list[int]:
        positions = list(range(0, length - patch_size + 1, stride))
        # Always include a patch flush with the far edge if not already covered
        if not positions or positions[-1] + patch_size < length:
            positions.append(length - patch_size)
        return positions

    return [(y, x) for y in coords_1d(h) for x in coords_1d(w)]

def predict_label_map(img: np.ndarray) -> np.ndarray:
    """
    Run sliding-window inference and return a uint8 label map where each pixel
    value is the predicted category_id (0 = background).
    """
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


def label_map_to_coco_annotations(
    label_map: np.ndarray,
    image_id: int,
    start_annotation_id: int,
) -> list[dict]:
    """
    Convert a multi-class label map to a list of COCO annotation dicts.
    Each connected component of each foreground class becomes a separate RLE annotation.
    RLE masks require iscrowd=1 per the COCO spec.
    """
    annotations: list[dict] = []
    annotation_id = start_annotation_id

    foreground_ids = np.unique(label_map)
    foreground_ids = foreground_ids[foreground_ids != 0]

    for category_id in foreground_ids:
        class_mask = (label_map == category_id).astype(np.uint8)
        num_labels, labeled = cv2.connectedComponents(class_mask)

        for label in range(1, num_labels):
            component = (labeled == label).astype(np.uint8)
            area = int(component.sum())
            if area < MIN_COMPONENT_AREA:
                continue

            rle = coco_mask_utils.encode(np.asfortranarray(component))
            rle["counts"] = rle["counts"].decode("utf-8")
            bbox_array = coco_mask_utils.toBbox(rle).tolist()

            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(category_id),
                "segmentation": rle,
                "area": area,
                "bbox": bbox_array,
                "iscrowd": 1,  # RLE segmentation requires iscrowd=1 in COCO spec
            })
            annotation_id += 1

    return annotations


# -----------------------------
# Main loop — stage into temp dir, then zip
# -----------------------------
image_paths = sorted(glob(os.path.join(INPUT_DIR, "*")))

with tempfile.TemporaryDirectory() as tmp_dir:
    images_out_dir = os.path.join(tmp_dir, "images", SUBSET_NAME)
    annotations_dir = os.path.join(tmp_dir, "annotations")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    annotation_id_counter = 1

    executor = ThreadPoolExecutor(max_workers=2)

    for image_id, img_path in enumerate(tqdm(image_paths), start=1):
        if image_id < RESUME_FROM_INDEX:
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: could not read {img_path}, skipping.")
            continue

        img_rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        h, w = img_rgb.shape[:2]
        file_name = os.path.basename(img_path)

        # file_name must be the bare filename only.
        # CVAT resolves it relative to images/<SUBSET_NAME>/
        coco_images.append({
            "id": image_id,
            "file_name": file_name,
            "width": w,
            "height": h,
            "date_captured": "",
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
        })

        label_map = predict_label_map(img_rgb)

        # Copy source image into the staging area asynchronously
        executor.submit(shutil.copy2, img_path, os.path.join(images_out_dir, file_name))

        new_annotations = label_map_to_coco_annotations(
            label_map,
            image_id=image_id,
            start_annotation_id=annotation_id_counter,
        )
        coco_annotations.extend(new_annotations)
        annotation_id_counter += len(new_annotations)

    executor.shutdown(wait=True)

    # -----------------------------
    # Write instances_<subset>.json
    # -----------------------------
    coco_dataset: dict = {
        "info": {
            "description": "Generated by gen_coco.py",
            "version": "1.0",
            "year": int(time.strftime("%Y")),
            "date_created": time.strftime("%Y/%m/%d"),
            "contributor": "",
            "url": "",
        },
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }

    annotation_filename = f"instances_{SUBSET_NAME}.json"
    with open(os.path.join(annotations_dir, annotation_filename), "w") as f:
        json.dump(coco_dataset, f, indent=2)

    # -----------------------------
    # Pack everything into a zip
    # -----------------------------
    print(f"Packing into {OUTPUT_ZIP} ...")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(tmp_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                arc_path = os.path.relpath(abs_path, tmp_dir)
                zf.write(abs_path, arc_path)

print(f"\nDone. CVAT-importable archive: {OUTPUT_ZIP}")
print(f"  Structure inside zip:")
print(f"    images/{SUBSET_NAME}/<image files>")
print(f"    annotations/{annotation_filename}")
print(f"  Images:      {len(coco_images)}")
print(f"  Annotations: {len(coco_annotations)}")