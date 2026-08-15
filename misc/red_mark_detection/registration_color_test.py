# red_registration_analysis.py
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import os

img_dir = Path('./test_img')

# Supported extensions
exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

images = []
if img_dir.exists():
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])

if not images:
    # optional: create a demo image so you can see how it looks
    img_dir.mkdir(parents=True, exist_ok=True)
    demo_path = img_dir / 'demo_registration.png'
    W, H = 800, 400
    bg_color = (240, 240, 240)
    img = Image.new('RGB', (W, H), bg_color)
    draw = ImageDraw.Draw(img)
    slot_w = 120
    slot_h = 200
    x_margin = 40
    y_margin = 50
    for i in range(5):
        x0 = x_margin + i * (slot_w + 10)
        y0 = y_margin
        draw.rectangle([x0, y0, x0 + slot_w, y0 + slot_h], outline=(100,100,100))
        reg_color = (150, 10, 10)  # dark red to simulate printed mark
        cx = x0 + slot_w//2
        cy = y0 + slot_h//2
        r = 12
        draw.rectangle([cx-r, cy-r, cx+r, cy+r], fill=reg_color)
    img.save(demo_path)
    images = [demo_path]
    print(f"No images found in ./test_img/. Created demo image at {demo_path}.")

for img_path in images:
    img = Image.open(img_path).convert('RGB')
    arr = np.array(img).astype(np.int32)
    R = arr[..., 0]
    G = arr[..., 1]
    B = arr[..., 2]

    # R - (G + B)
    diff = R - (G + B)
    diff_clipped = np.clip(diff, 0, None)

    # normalize clipped diff to 0-255 for visibility
    lo = float(diff_clipped.min())
    hi = float(diff_clipped.max())
    if hi == lo:
        norm = np.zeros_like(diff_clipped, dtype=np.uint8)
    else:
        norm = ((diff_clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)

    # sums for distributions
    sum_per_x = diff_clipped.sum(axis=0)  # columns -> X
    sum_per_y = diff_clipped.sum(axis=1)  # rows -> Y

    # display images
    fig, axs = plt.subplots(1, 4, figsize=(16, 6))
    fig.suptitle(f"File: {img_path.name}", fontsize=14)

    axs[0].imshow(np.clip(arr, 0, 255).astype(np.uint8))
    axs[0].set_title('Original (RGB)')
    axs[0].axis('off')

    axs[1].imshow(np.clip(R, 0, 255).astype(np.uint8))
    axs[1].set_title('Red channel (0-255)')
    axs[1].axis('off')

    axs[2].imshow(np.clip(diff_clipped, 0, 255).astype(np.uint8))
    axs[2].set_title('Red - (Green+Blue) (clipped)')
    axs[2].axis('off')

    axs[3].imshow(norm)
    axs[3].set_title('Normalized (0-255)')
    axs[3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # X distribution (per-column)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(np.arange(len(sum_per_x)), sum_per_x)
    ax.set_title('Sum of (R - (G+B)) per column — X axis distribution')
    ax.set_xlabel('Column (X)')
    ax.set_ylabel('Sum of positive red-dominance values')
    plt.tight_layout()
    plt.show()

    # Y distribution (per-row)
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.barh(np.arange(len(sum_per_y)), sum_per_y)
    ax.set_title('Sum of (R - (G+B)) per row — Y axis distribution')
    ax.set_xlabel('Sum of positive red-dominance values')
    ax.set_ylabel('Row (Y)')
    plt.tight_layout()
    plt.show()

    total_positive = int((diff_clipped > 0).sum())
    pct_positive = total_positive / diff_clipped.size * 100.0
    print(f"Image: {img_path.name} — min diff: {int(diff.min())}, max diff: {int(diff.max())}, "
          f"positive-pixel-count: {total_positive} ({pct_positive:.2f}%)")
