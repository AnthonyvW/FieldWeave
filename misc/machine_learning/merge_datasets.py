"""
Merge multiple COCO 1.0 datasets into a single COCO 1.0 dataset.

Expected layout for each source dataset:
    <dataset_root>/
        images/default/
        annotations/instances_default.json

Usage:
    python merge_coco_datasets.py \\
        --input  /path/to/folder/of/datasets \\
        --output /path/to/merged_output

The script writes:
    <output>/
        images/           (flattened, sequentially renamed image files)
        annotations/instances_default.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ANNOTATIONS_RELPATH = Path("annotations") / "instances_default.json"
IMAGES_SUBDIR = Path("images") / "default"


def load_dataset(root: Path) -> dict:
    ann_path = root / ANNOTATIONS_RELPATH
    if not ann_path.exists():
        raise FileNotFoundError(f"{ANNOTATIONS_RELPATH} not found in {root}")
    return json.loads(ann_path.read_text())


def build_unified_categories(
    per_dataset_coco: list[dict],
) -> tuple[dict[str, int], list[dict]]:
    """
    Merge categories by name across all datasets.
    Returns (name_to_new_id, coco_categories_list).
    """
    names: set[str] = set()
    supercategory_map: dict[str, str] = {}
    for coco in per_dataset_coco:
        for cat in coco.get("categories", []):
            name = cat["name"]
            names.add(name)
            supercategory_map.setdefault(name, cat.get("supercategory", ""))

    name_to_id = {name: idx for idx, name in enumerate(sorted(names), start=1)}
    categories = [
        {"id": cat_id, "name": name, "supercategory": supercategory_map[name]}
        for name, cat_id in sorted(name_to_id.items(), key=lambda x: x[1])
    ]
    return name_to_id, categories


def merge_coco_datasets(input_dir: Path, output_dir: Path) -> None:
    dataset_roots = sorted(
        p for p in input_dir.iterdir()
        if p.is_dir() and (p / ANNOTATIONS_RELPATH).exists()
    )
    if not dataset_roots:
        raise RuntimeError(
            f"No COCO datasets found in {input_dir} "
            f"(looking for {ANNOTATIONS_RELPATH} in each sub-directory)"
        )

    print(f"Found {len(dataset_roots)} dataset(s):")
    for root in dataset_roots:
        print(f"  {root.name}")

    per_dataset_coco = [load_dataset(root) for root in dataset_roots]
    unified_name_to_id, unified_categories = build_unified_categories(per_dataset_coco)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir = output_dir / "images"
    output_images_dir.mkdir(exist_ok=True)
    output_ann_dir = output_dir / "annotations"
    output_ann_dir.mkdir(exist_ok=True)

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    image_id = 1
    ann_id = 1

    for root, coco in zip(dataset_roots, per_dataset_coco):
        src_images_dir = root / IMAGES_SUBDIR

        old_cat_id_to_new: dict[int, int] = {
            cat["id"]: unified_name_to_id[cat["name"]]
            for cat in coco.get("categories", [])
        }
        old_image_id_to_new: dict[int, int] = {}

        images = coco.get("images", [])
        print(f"\nProcessing: {root.name}  ({len(images)} images)")

        for old_img in images:
            src_file = src_images_dir / old_img["file_name"]
            if not src_file.exists():
                print(f"  WARNING: image file missing, skipping — {src_file}")
                continue

            dest_name = f"{image_id:08d}{src_file.suffix}"
            shutil.copy2(src_file, output_images_dir / dest_name)

            new_img = {**old_img, "id": image_id, "file_name": dest_name}
            all_images.append(new_img)
            old_image_id_to_new[old_img["id"]] = image_id
            image_id += 1

        for old_ann in coco.get("annotations", []):
            if old_ann["image_id"] not in old_image_id_to_new:
                continue  # image was skipped
            new_cat_id = old_cat_id_to_new.get(old_ann["category_id"])
            if new_cat_id is None:
                print(
                    f"  WARNING: unknown category_id {old_ann['category_id']}, "
                    "skipping annotation"
                )
                continue

            new_ann = {
                **old_ann,
                "id": ann_id,
                "image_id": old_image_id_to_new[old_ann["image_id"]],
                "category_id": new_cat_id,
            }
            all_annotations.append(new_ann)
            ann_id += 1

    coco_out: dict = {
        "info": {
            "description": "Merged COCO dataset",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "categories": unified_categories,
        "images": all_images,
        "annotations": all_annotations,
    }

    ann_out_path = output_ann_dir / "instances_default.json"
    ann_out_path.write_text(json.dumps(coco_out, indent=2))

    print(
        f"\nDone. "
        f"{len(all_images)} images, "
        f"{len(all_annotations)} annotations, "
        f"{len(unified_categories)} categories."
    )
    print(f"Output written to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a folder of COCO 1.0 datasets into a single COCO 1.0 dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="DIR",
        help="Folder containing one sub-directory per COCO dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Output directory for the merged dataset.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    merge_coco_datasets(input_dir, Path(args.output))


if __name__ == "__main__":
    main()