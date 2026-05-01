from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image


# Maximum hamming distance to consider two images duplicates.
# 10 is a reasonable threshold for "very similar / minor JPEG artifacts".
HASH_THRESHOLD = 10


def hash_directory(directory: Path) -> list[tuple[imagehash.ImageHash, Path]]:
    """Return a list of (hash, path) pairs for all image files in a directory."""
    results: list[tuple[imagehash.ImageHash, Path]] = []
    if not directory.exists():
        return results
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            try:
                h = imagehash.phash(Image.open(path))
                results.append((h, path))
            except Exception as e:
                print(f"Warning: could not hash {path}: {e}")
    return results


def find_mask(image_path: Path, mask_dir: Path) -> Path | None:
    """Return the mask file for an image if it exists, matching by stem."""
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / (image_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def delete_file(path: Path, reason: str) -> None:
    print(f"  Deleted: {path}  ({reason})")
    path.unlink()


def is_duplicate(
    h: imagehash.ImageHash,
    reference: list[tuple[imagehash.ImageHash, Path]],
) -> Path | None:
    """Return the path of the first reference image within the hash threshold, or None."""
    for ref_hash, ref_path in reference:
        if h - ref_hash <= HASH_THRESHOLD:
            return ref_path
    return None


def deduplicate(training_dir: str | Path = "training") -> None:
    training = Path(training_dir)

    curated_images = training / "curated" / "JPEGImages"
    not_annotated  = training / "curated" / "not_annotated"
    good_images    = training / "good" / "JPEGImages"
    good_masks     = training / "good" / "SegmentationClass"
    bad_images     = training / "bad" / "JPEGImages"
    bad_masks      = training / "bad" / "SegmentationClass"

    print("Hashing curated/JPEGImages...")
    curated_hashes = hash_directory(curated_images)
    print(f"  {len(curated_hashes)} images hashed")

    print("Hashing curated/not_annotated...")
    not_annotated_hashes = hash_directory(not_annotated)
    print(f"  {len(not_annotated_hashes)} images hashed")

    total_deleted = 0

    for images_dir, masks_dir, label in [
        (good_images, good_masks, "good"),
        (bad_images,  bad_masks,  "bad"),
    ]:
        print(f"\nChecking {label}/JPEGImages...")
        if not images_dir.exists():
            print(f"  Warning: {images_dir} not found, skipping.")
            continue

        for h, path in hash_directory(images_dir):
            # First check against annotated curated images — delete from good/bad
            match = is_duplicate(h, curated_hashes)
            if match:
                print(f"  Duplicate of curated image: {match}")
                delete_file(path, f"duplicate of curated image — found in {label}")
                total_deleted += 1
                mask = find_mask(path, masks_dir)
                if mask:
                    delete_file(mask, f"mask of duplicate removed from {label}")
                    total_deleted += 1
                continue

            # Then check against not_annotated — delete from not_annotated, keep good/bad
            match = is_duplicate(h, not_annotated_hashes)
            if match:
                print(f"  Duplicate of not_annotated image: {match}")
                delete_file(match, f"unannotated duplicate superseded by {label} image")
                total_deleted += 1
                # Remove from the reference list so it isn't matched again
                not_annotated_hashes = [(rh, rp) for rh, rp in not_annotated_hashes if rp != match]

    print(f"\nDone. Deleted {total_deleted} file(s) in total.")


if __name__ == "__main__":
    deduplicate()