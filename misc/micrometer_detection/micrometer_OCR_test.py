from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
import torch

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,.]*\d|\d")


def gpu_available() -> bool:
    return torch.cuda.is_available()


def load_reader(languages: list[str] | None = None, use_gpu: bool | None = None) -> easyocr.Reader:
    if languages is None:
        languages = ["en"]
    if use_gpu is None:
        use_gpu = gpu_available()
    print(f"EasyOCR device: {'GPU' if use_gpu else 'CPU'}")
    return easyocr.Reader(languages, gpu=use_gpu)


ROTATION_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    return cv2.rotate(image, ROTATION_CODES[degrees])


def binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def longest_runs(matrix: np.ndarray) -> np.ndarray:
    h, _ = matrix.shape
    mask = matrix > 0
    padded = np.concatenate(
        [np.zeros((h, 1), dtype=bool), mask, np.zeros((h, 1), dtype=bool)], axis=1,
    )
    diff = np.diff(padded.astype(np.int8), axis=1)
    runs = np.zeros(h, dtype=np.int32)
    for r in range(h):
        starts = np.where(diff[r] == 1)[0]
        ends = np.where(diff[r] == -1)[0]
        if starts.size:
            runs[r] = int((ends - starts).max())
    return runs


def find_bar_axis(binary: np.ndarray) -> tuple[str, int]:
    row_runs = longest_runs(binary)
    col_runs = longest_runs(binary.T)
    if col_runs.max() >= row_runs.max():
        return "vertical", int(col_runs.argmax())
    return "horizontal", int(row_runs.argmax())


def detect_rotation(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)
    h, w = image.shape[:2]
    axis, baseline_index = find_bar_axis(binary)
    band_half = (h if axis == "horizontal" else w) // 10
    numbers_only = binary.copy()
    if axis == "horizontal":
        y0 = max(0, baseline_index - band_half)
        y1 = min(h, baseline_index + band_half + 1)
        numbers_only[y0:y1, :] = 0
        ys, _ = np.where(numbers_only > 0)
        offset = float(ys.mean()) - h / 2.0 if ys.size > 0 else 0.0
        return 0 if offset < 0 else 180
    x0 = max(0, baseline_index - band_half)
    x1 = min(w, baseline_index + band_half + 1)
    numbers_only[:, x0:x1] = 0
    _, xs = np.where(numbers_only > 0)
    offset = float(xs.mean()) - w / 2.0 if xs.size > 0 else 0.0
    return 90 if offset < 0 else 270


def mask_long_lines(binary: np.ndarray, max_frac: float = 0.2, check_height: bool = True) -> np.ndarray:
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    masked = binary.copy()
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        too_wide = cw / w > max_frac
        too_tall = check_height and ch / h > max_frac
        if too_wide or too_tall:
            cv2.drawContours(masked, [c], -1, 0, cv2.FILLED)
    return masked


def crop_to_number_band(image: np.ndarray, padding: int = 150) -> tuple[np.ndarray, int]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)
    axis, baseline_index = find_bar_axis(binary)
    if axis != "horizontal":
        return image, 0

    h, w = binary.shape
    band_half = h // 10
    cleaned = mask_long_lines(binary)
    y0 = max(0, baseline_index - band_half)
    y1 = min(h, baseline_index + band_half + 1)
    cleaned[y0:y1, :] = 0
    ys, _ = np.where(cleaned > 0)
    if ys.size == 0:
        return image, 0

    top = max(0, int(ys.min()) - padding)
    bottom = min(h, int(ys.max()) + padding)
    return image[top:bottom, :], top


def deskew(gray: np.ndarray) -> np.ndarray:
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def isolate_digit_blobs(binary: np.ndarray, min_area_ratio: float = 0.15, min_aspect: float = 0.25) -> np.ndarray:
    cleaned = mask_long_lines(binary, check_height=False)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cleaned

    max_area = max(cv2.contourArea(c) for c in contours)
    mask = np.zeros_like(cleaned)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h else 0.0
        if cv2.contourArea(c) >= max_area * min_area_ratio and aspect >= min_aspect:
            mask[y:y + h, x:x + w] = cleaned[y:y + h, x:x + w]
    return mask


def preprocess(
    image: np.ndarray, upscale_factor: float = 2.0, deskew_enabled: bool = False,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    straightened = deskew(gray) if deskew_enabled else gray
    denoised = cv2.fastNlMeansDenoising(straightened, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(denoised)
    new_size = (
        int(contrasted.shape[1] * upscale_factor),
        int(contrasted.shape[0] * upscale_factor),
    )
    upscaled = cv2.resize(contrasted, new_size, interpolation=cv2.INTER_CUBIC)
    binary = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return binary


def recognize_numbers(
    reader: easyocr.Reader, image: np.ndarray,
    text_threshold: float = 0.4, low_text: float = 0.3, link_threshold: float = 0.3,
) -> list[dict]:
    results = reader.readtext(
        image,
        allowlist="0123456789.,-+",
        detail=1,
        paragraph=False,
        canvas_size=max(image.shape[:2]),
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
    )
    detections = []
    for bbox, text, confidence in results:
        cleaned = text.strip()
        if not cleaned or not NUMBER_PATTERN.search(cleaned):
            continue
        detections.append({
            "text": cleaned,
            "confidence": round(float(confidence), 4),
            "bbox": [[int(x), int(y)] for x, y in bbox],
        })
    return detections


def scale_bbox(bbox: list[list[int]], factor: float, y_offset: int = 0) -> list[list[int]]:
    return [[int(round(x / factor)), int(round(y / factor)) + y_offset] for x, y in bbox]


def draw_annotations(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        points = np.array(detection["bbox"], dtype=np.int32)
        cv2.polylines(annotated, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        origin = tuple(points[0])
        cv2.putText(
            annotated, detection["text"], origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )
    return annotated


def process_folder(
    input_dir: Path, output_dir: Path, save_annotated: bool = True,
    rotate: int | str = "auto", deskew_enabled: bool = False, use_gpu: bool | None = None,
    upscale_factor: float = 2.0, crop_to_numbers: bool = True, isolate_digits: bool = True,
    text_threshold: float = 0.4, low_text: float = 0.3, link_threshold: float = 0.3,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = load_reader(use_gpu=use_gpu)
    all_results = []

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable file: {path.name}")
            continue

        degrees = detect_rotation(image) if rotate == "auto" else rotate
        image = rotate_image(image, degrees)

        if crop_to_numbers:
            region, y_offset = crop_to_number_band(image)
        else:
            region, y_offset = image, 0

        processed = preprocess(region, upscale_factor=upscale_factor, deskew_enabled=deskew_enabled)
        if isolate_digits:
            processed = cv2.bitwise_not(isolate_digit_blobs(cv2.bitwise_not(processed)))
        detections = recognize_numbers(
            reader, processed,
            text_threshold=text_threshold, low_text=low_text, link_threshold=link_threshold,
        )
        for detection in detections:
            detection["bbox"] = scale_bbox(detection["bbox"], upscale_factor, y_offset)

        if detections:
            readings = ", ".join(d["text"] for d in detections)
            print(f"{path.name}: {readings}")
        else:
            print(f"{path.name}: no numbers detected")

        for detection in detections:
            all_results.append({"file": path.name, **detection})

        if save_annotated:
            annotated = draw_annotations(image, detections)
            out_path = output_dir / f"annotated_{path.name}"
            cv2.imwrite(str(out_path), annotated)

    return all_results


def save_results_csv(results: list[dict], output_path: Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "text", "confidence", "bbox"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def parse_rotate_arg(value: str) -> int | str:
    if value == "auto":
        return "auto"
    if value in {"0", "90", "180", "270"}:
        return int(value)
    raise argparse.ArgumentTypeError("rotate must be 'auto', 0, 90, 180, or 270")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and read numbers in images.")
    parser.add_argument("input_dir", type=Path, help="Folder of images to process")
    parser.add_argument("output_dir", type=Path, help="Folder to write results to")
    parser.add_argument("--no-annotate", action="store_true", help="Skip saving annotated images")
    parser.add_argument(
        "--rotate", type=parse_rotate_arg, default="auto",
        help=(
            "Rotation to apply before processing. 'auto' (default) detects it per "
            "image from the scale bar's position; or force a fixed rotation with "
            "0, 90, 180, or 270."
        ),
    )
    parser.add_argument(
        "--no-crop", action="store_true",
        help="Skip cropping to the detected number band and process the full image",
    )
    parser.add_argument(
        "--no-isolate", action="store_true",
        help=(
            "Skip isolating individual digit blobs before recognition. Off by "
            "default this filters out tick marks, dust, and line remnants by "
            "shape and size, which otherwise cause false-positive readings once "
            "detection thresholds are lowered."
        ),
    )
    parser.add_argument(
        "--deskew", action="store_true",
        help=(
            "Auto-correct small skew angles before recognition. Off by default: "
            "on sparse images dominated by long straight lines (rulers, scales, "
            "table borders) this can misjudge the angle and rotate a straight "
            "image into a crooked one."
        ),
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU even if a GPU is available (auto-detected by default)",
    )
    parser.add_argument(
        "--text-threshold", type=float, default=0.4,
        help="EasyOCR text detection confidence threshold (default 0.4, lower catches more but risks false positives)",
    )
    parser.add_argument(
        "--low-text", type=float, default=0.3,
        help="EasyOCR low-bound text score for region growing (default 0.3)",
    )
    parser.add_argument(
        "--link-threshold", type=float, default=0.3,
        help="EasyOCR character-linking threshold (default 0.3)",
    )
    args = parser.parse_args()

    results = process_folder(
        args.input_dir, args.output_dir,
        save_annotated=not args.no_annotate, rotate=args.rotate,
        deskew_enabled=args.deskew, use_gpu=(False if args.cpu else None),
        crop_to_numbers=not args.no_crop, isolate_digits=not args.no_isolate,
        text_threshold=args.text_threshold, low_text=args.low_text, link_threshold=args.link_threshold,
    )
    save_results_csv(results, args.output_dir / "results.csv")
    print(f"Processed {len(results)} number detections. Results saved to {args.output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()