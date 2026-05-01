from __future__ import annotations

import json
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_label_mask(coco: COCO, image_id: int, height: int, width: int) -> np.ndarray:
    """
    Build a uint8 label mask for a single image where each pixel value is the
    category_id of the corresponding annotation (0 = background).

    If annotations overlap, the last one drawn wins. Crowd regions are skipped.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=image_id, iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        category_id = ann["category_id"]
        rle_or_poly = ann.get("segmentation")
        if not rle_or_poly:
            continue
        # annToMask returns a binary uint8 mask (0 or 1)
        binary = coco.annToMask(ann).astype(bool)
        mask[binary] = category_id

    return mask


def extract_patches(
    image_dir: str,
    annotations_path: str,
    output_dir: str,
    patch_size: int = 512,
    stride: int = 256,
    min_labeled_fraction: float = 0.0,
) -> None:
    """
    Extract fixed-size patches from a COCO 1.0 dataset.

    Parameters
    ----------
    image_dir:
        Directory containing the raw images referenced in the COCO JSON.
    annotations_path:
        Path to the COCO-format JSON annotations file.
    output_dir:
        Root output directory. Will contain `images/` and `masks/` subdirectories.
    patch_size:
        Side length (pixels) of each square patch.
    stride:
        Step size between patch origins. Use stride < patch_size for overlap.
    min_labeled_fraction:
        Minimum fraction of patch pixels that must be non-background to keep the
        patch. 0.0 (default) keeps any patch with at least one labeled pixel.
    """
    out_images = os.path.join(output_dir, "images")
    out_masks = os.path.join(output_dir, "masks")
    ensure_dir(out_images)
    ensure_dir(out_masks)

    coco = COCO(annotations_path)

    # Write out a category mapping so label IDs are self-documenting
    categories = coco.loadCats(coco.getCatIds())
    categories_path = os.path.join(output_dir, "categories.json")
    with open(categories_path, "w") as f:
        json.dump(categories, f, indent=2)
    print(f"Saved {len(categories)} categories to {categories_path}")

    image_ids = sorted(coco.getImgIds())
    patch_id = 0
    min_pixels = max(1, int(patch_size * patch_size * min_labeled_fraction))

    for image_id in tqdm(image_ids, desc="Images"):
        img_info = coco.loadImgs(image_id)[0]
        img_path = os.path.join(image_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: could not read {img_path}, skipping.")
            continue

        h, w = image.shape[:2]
        label_mask = build_label_mask(coco, image_id, h, w)

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                mask_patch = label_mask[y : y + patch_size, x : x + patch_size]

                # Skip patches with insufficient labeled content
                if np.count_nonzero(mask_patch) < min_pixels:
                    continue

                img_patch = image[y : y + patch_size, x : x + patch_size]
                name = f"patch_{patch_id:06d}"
                cv2.imwrite(os.path.join(out_images, name + ".jpg"), img_patch)
                cv2.imwrite(os.path.join(out_masks, name + ".png"), mask_patch)

                patch_id += 1

    print(f"Total patches saved: {patch_id}")


if __name__ == "__main__":
    extract_patches(
        image_dir="training/merged_dataset/images",
        annotations_path="training/merged_dataset/annotations/instances_default.json",
        output_dir="dataset_patches",
        patch_size=1024,
        stride=512,
        min_labeled_fraction=0.0,
    )