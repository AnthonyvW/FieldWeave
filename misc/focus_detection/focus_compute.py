import sys
import cv2
import numpy as np


def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img.astype(np.float32)


# ------------------------------------------------------------
# 1. Laplacian Variance
# ------------------------------------------------------------
def laplacian_variance(img: np.ndarray) -> float:
    lap = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F, ksize=3)
    return float(lap.var())


# ------------------------------------------------------------
# 2. Tenengrad (Sobel gradient magnitude energy)
# ------------------------------------------------------------
def tenengrad(img: np.ndarray) -> float:
    img64 = img.astype(np.float64)
    gx = cv2.Sobel(img64, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img64, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = gx**2 + gy**2
    return float(np.mean(grad_mag_sq))


# ------------------------------------------------------------
# 3. FFT High-Frequency Energy
# ------------------------------------------------------------
def fft_high_frequency_energy(img: np.ndarray, cutoff_ratio: float = 0.1) -> float:
    """
    cutoff_ratio: fraction of image size used as low-frequency radius.
    Example: 0.1 removes central 10% radius as low frequency.
    """

    h, w = img.shape
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)

    magnitude_sq = np.abs(fft_shifted) ** 2

    # Create radial mask
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((x - cx)**2 + (y - cy)**2)

    max_radius = np.sqrt(cx**2 + cy**2)
    cutoff = cutoff_ratio * max_radius

    high_freq_mask = radius > cutoff

    high_freq_energy = magnitude_sq[high_freq_mask].sum()
    total_energy = magnitude_sq.sum()

    if total_energy == 0:
        return 0.0

    return float(high_freq_energy / total_energy)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def compute_metrics(image_path: str):
    img = load_grayscale(image_path)

    lap_var = laplacian_variance(img)
    teng = tenengrad(img)
    fft_energy = fft_high_frequency_energy(img)

    print(f"Image: {image_path}")
    print(f"Laplacian Variance:      {lap_var:.6f}")
    print(f"Tenengrad (mean grad^2): {teng:.6f}")
    print(f"FFT High-Freq Energy:    {fft_energy:.6f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python focus_metrics.py <image_path>")
        sys.exit(1)

    compute_metrics(sys.argv[1])