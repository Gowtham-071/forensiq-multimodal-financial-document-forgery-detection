"""
vision/vision_model.py
6-Signal Visual Forensics Engine
Detectors: ELA · Noise · JPEG Ghost · Copy-Paste · Edge · CNN
"""

import cv2
import numpy as np
import os
import io
import tempfile
from pathlib import Path
from PIL import Image

BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "fraud_document_cnn.h5"
IMG_SIZE   = 128

# Load CNN once at module level
_cnn_model = None

def _get_cnn():
    global _cnn_model
    if _cnn_model is None and MODEL_PATH.exists():
        import tensorflow as tf  # lazy import — only loaded on first bill analysis
        _cnn_model = tf.keras.models.load_model(str(MODEL_PATH))
    return _cnn_model


# ── 1. ELA (Error Level Analysis) ──────────────────────────────────────────

def _ela_score(image_path: str, quality: int = 90) -> tuple:
    """
    Compute ELA score and return (score 0-1, heatmap_path).
    High score = compression inconsistency = likely tampered.
    """
    original = Image.open(image_path).convert("RGB")

    # Re-save at target quality
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    # Compute absolute difference
    orig_arr = np.array(original, dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    ela_arr = np.abs(orig_arr - recomp_arr)

    # Normalise scale
    max_diff = ela_arr.max() if ela_arr.max() > 0 else 1.0
    ela_norm = (ela_arr / max_diff * 255).astype(np.uint8)

    # Score = mean intensity of top 5% pixels (tampered regions are bright)
    flat = ela_norm.flatten()
    threshold = np.percentile(flat, 95)
    score = float(np.mean(flat[flat >= threshold]) / 255.0)

    # Save heatmap
    heatmap_dir  = BASE_DIR / "uploads" / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = str(heatmap_dir / (Path(image_path).stem + "_ela.png"))
    ela_gray = cv2.cvtColor(ela_norm, cv2.COLOR_RGB2GRAY)
    ela_colored = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_path, ela_colored)

    return round(min(score * 2.0, 1.0), 4), heatmap_path  # scale up, cap at 1


# ── 2. Noise Inconsistency ──────────────────────────────────────────────────

def _noise_score(image_path: str, block_size: int = 64) -> float:
    """
    Block-wise noise analysis. Tampered regions have different noise texture.
    Returns score 0-1 (high = inconsistent noise across image blocks).
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    stds = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = gray[y:y+block_size, x:x+block_size]
            blurred = cv2.GaussianBlur(block, (5, 5), 0)
            noise   = block.astype(np.float32) - blurred.astype(np.float32)
            stds.append(float(np.std(noise)))

    if len(stds) < 2:
        return 0.0

    # High variance in per-block noise = inconsistency = tampering
    global_std = float(np.std(stds))
    score = min(global_std / 8.0, 1.0)
    return round(score, 4)


# ── 3. JPEG Ghost ───────────────────────────────────────────────────────────

def _jpeg_ghost_score(image_path: str) -> float:
    """
    Multi-quality JPEG recompression ghost detection.
    Regions saved at different quality levels leave 'ghosts'.
    Returns score 0-1.
    """
    original = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
    min_diff = float("inf")

    for q in [50, 65, 75, 85, 95]:
        buf = io.BytesIO()
        Image.fromarray(original.astype(np.uint8)).save(buf, "JPEG", quality=q)
        buf.seek(0)
        recomp = np.array(Image.open(buf).convert("L"), dtype=np.float32)
        diff = np.mean(np.abs(original - recomp))
        if diff < min_diff:
            min_diff = diff

    # Low min_diff means image was probably saved at one of these qualities (ghost match)
    score = min(min_diff / 20.0, 1.0)
    return round(score, 4)


# ── 4. Copy-Paste Detection ─────────────────────────────────────────────────

def _copypaste_score(image_path: str) -> float:
    """
    SIFT keypoint + RANSAC homography copy-paste detector.
    Returns score 0-1 (high = suspicious repeated regions).
    """
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=500)
    kps, descs = sift.detectAndCompute(gray, None)

    if descs is None or len(kps) < 10:
        return 0.0

    # Self-match: detect duplicate keypoint clusters
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descs, descs, k=2)

    # Filter out self-matches and very close spatial neighbours
    suspicious = 0
    for m, n in matches:
        if m.trainIdx == m.queryIdx:
            continue
        if m.distance < 0.7 * n.distance:
            pt1 = np.array(kps[m.queryIdx].pt)
            pt2 = np.array(kps[m.trainIdx].pt)
            dist = np.linalg.norm(pt1 - pt2)
            if dist > 30:  # not adjacent; likely copied from elsewhere
                suspicious += 1

    score = min(suspicious / 50.0, 1.0)
    return round(score, 4)


# ── 5. Edge / Lighting Inconsistency ───────────────────────────────────────

def _edge_score(image_path: str, block_size: int = 64) -> float:
    """
    Sobel gradient block-wise inconsistency.
    Tampered regions often have sharper or different edge density.
    Returns score 0-1.
    """
    gray  = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    sobel = np.abs(sobel)
    h, w  = gray.shape

    densities = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = sobel[y:y+block_size, x:x+block_size]
            densities.append(float(np.mean(block)))

    if len(densities) < 2:
        return 0.0

    cv_score = np.std(densities) / (np.mean(densities) + 1e-6)  # coefficient of variation
    score = min(cv_score / 2.0, 1.0)
    return round(score, 4)


# ── 6. CNN Visual Score ─────────────────────────────────────────────────────

def _cnn_score(image_path: str) -> float:
    """
    Existing CNN model inference on bottom-30% ROI.
    Returns probability of fraud 0-1.
    """
    model = _get_cnn()
    if model is None:
        return 0.0

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0
    h, w   = gray.shape
    roi    = gray[int(h * 0.7):h, 0:w]
    roi    = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) / 255.0
    roi    = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    score  = float(model.predict(roi, verbose=0)[0][0])
    return round(score, 4)


# ── Main entry point ────────────────────────────────────────────────────────

def run_visual_forensics(image_path: str) -> dict:
    """
    Run all 6 visual forensics signals on an image.

    Returns:
        dict with individual scores, composite_visual_score,
        image_quality_score, and ela_heatmap_path.
    """
    from utils.preprocessing import _compute_quality_score, load_image_gray

    # Individual detector scores
    ela, heatmap_path = _ela_score(image_path)
    noise    = _noise_score(image_path)
    jpeg     = _jpeg_ghost_score(image_path)
    copypaste= _copypaste_score(image_path)
    edge     = _edge_score(image_path)
    cnn      = _cnn_score(image_path)

    # Quality score (for adaptive fusion)
    gray = load_image_gray(image_path)
    quality = _compute_quality_score(gray)

    # Weighted composite visual score
    # Weights: ELA=0.25, CNN=0.25, Noise=0.15, JPEG=0.15, CopyPaste=0.10, Edge=0.10
    composite = (
        0.25 * ela +
        0.25 * cnn +
        0.15 * noise +
        0.15 * jpeg +
        0.10 * copypaste +
        0.10 * edge
    )

    return {
        "ela":                   ela,
        "noise":                 noise,
        "jpeg":                  jpeg,
        "copypaste":             copypaste,
        "edge":                  edge,
        "cnn":                   cnn,
        "composite_visual_score":round(composite, 4),
        "image_quality_score":   round(quality, 4),
        "ela_heatmap_path":      heatmap_path,
    }
