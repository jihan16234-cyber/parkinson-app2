"""Shared helpers: image preprocessing + feature extraction (used by train.py and app.py)."""
import zipfile
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog

SIZES = {"spiral": (128, 128), "wave": (256, 128)}  # (width, height)
CLASSES = ["healthy", "parkinson"]


def preprocess(path, size):
    """Gray -> resize -> Otsu binarize (stroke = white on black)."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise ValueError("ছবি পড়া যায়নি")
    g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
    return cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def features(b):
    """HOG (shape/tremor pattern) + simple stroke statistics."""
    h = hog(b, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    ink = b > 0
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    length = sum(cv2.arcLength(c, False) for c in cnts)
    ys, xs = np.nonzero(ink)
    spread = [xs.std() / b.shape[1], ys.std() / b.shape[0]] if len(xs) else [0, 0]
    return np.concatenate([h, [ink.mean(), length / b.size, ink.sum() / (length + 1e-6), len(cnts), *spread]])


def ensure_data(base):
    """If data/ folder is missing, unzip data.zip (used on the cloud)."""
    base = Path(base)
    if not (base / "data" / "spiral").exists() and (base / "data.zip").exists():
        zipfile.ZipFile(base / "data.zip").extractall(base / "data")
