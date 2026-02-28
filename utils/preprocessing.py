"""
utils/preprocessing.py
Image preprocessing pipeline + quality scoring
"""

import cv2
import numpy as np


def preprocess_image(image_path: str) -> tuple:
    """
    Load and preprocess an image for forensic analysis.

    Returns:
        (preprocessed_img: np.ndarray, quality_score: float 0-1)

    quality_score drives adaptive fusion weights:
        High (≥0.4)  → trust visual forensics more  (w_visual=0.68)
        Low  (<0.4)  → trust OCR more               (w_visual=0.45)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # 1. Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Bilateral denoise — preserves edges (important for forensics)
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3. Adaptive threshold — binarise for OCR-friendly output
    thresholded = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11, C=2
    )

    # 4. Quality score: Laplacian variance (sharpness) + noise estimate
    quality_score = _compute_quality_score(gray)

    return thresholded, quality_score


def _compute_quality_score(gray: np.ndarray) -> float:
    """
    Compute a 0-1 image quality score from sharpness and noise.

    Sharpness: Laplacian variance — high var = sharp image
    Noise:     std of Gaussian residuals — low std = clean image
    """
    # Sharpness via Laplacian variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalise: clip at 1000 (typical sharp document), map to 0-1
    sharpness = min(lap_var / 1000.0, 1.0)

    # Noise: difference between original and Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_std = np.std(gray.astype(np.float32) - blurred.astype(np.float32))
    # Low noise_std = clean image; clip at 20
    cleanliness = max(0.0, 1.0 - noise_std / 20.0)

    quality_score = round(0.6 * sharpness + 0.4 * cleanliness, 4)
    return float(quality_score)


def load_image_rgb(image_path: str) -> np.ndarray:
    """Load image as RGB numpy array (for ELA and CNN input)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_gray(image_path: str) -> np.ndarray:
    """Load as grayscale."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return img


def resize_for_cnn(img: np.ndarray, size: int = 128) -> np.ndarray:
    """Resize and normalise image for CNN input (bottom 30% ROI)."""
    h, w = img.shape[:2]
    roi = img[int(h * 0.7):h, 0:w]
    roi_resized = cv2.resize(roi, (size, size))
    if len(roi_resized.shape) == 2:
        roi_resized = roi_resized.reshape(1, size, size, 1)
    else:
        roi_resized = roi_resized.reshape(1, size, size, 3)
    return roi_resized / 255.0
