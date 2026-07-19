"""Dual-engine OCR: ChandraOCR CLI when installed, Tesseract fallback.

Digital PDFs skip OCR entirely (PyMuPDF text layer). Scanned pages go through
CLAHE/denoise/threshold preprocessing before Tesseract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import cv2
import fitz
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

CHANDRA_AVAILABLE = shutil.which("chandra") is not None

MIN_DIGITAL_CHARS = 50
MIN_OCR_CHARS = 50


def preprocess_for_ocr(image: Image.Image | np.ndarray) -> np.ndarray:
    """Grayscale -> Gaussian denoise -> CLAHE -> adaptive threshold."""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array

    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gaussian)
    thresh = cv2.adaptiveThreshold(
        clahe_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh


def run_tesseract_ocr(image: Image.Image | np.ndarray) -> str:
    processed = preprocess_for_ocr(image)
    return pytesseract.image_to_string(processed, config=r"--oem 3 --psm 6")


def run_chandra_ocr(image_path: str) -> str:
    """ChandraOCR via its CLI; raises on failure so the caller can fall back."""
    result = subprocess.run(
        ["chandra", "ocr", image_path],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return result.stdout


def ocr_page_intelligent(image_path: str) -> tuple[str, str]:
    """ChandraOCR first when available; Tesseract otherwise. Returns (text, engine)."""
    if CHANDRA_AVAILABLE:
        try:
            chandra_text = run_chandra_ocr(image_path)
            if chandra_text and len(chandra_text.strip()) > MIN_OCR_CHARS:
                return chandra_text, "ChandraOCR"
        except Exception:
            pass

    text = run_tesseract_ocr(Image.open(image_path))
    return text, "Tesseract"


def is_digital_pdf(pdf_path: str, min_chars: int = MIN_DIGITAL_CHARS) -> bool:
    with fitz.open(pdf_path) as doc:
        return len(doc) > 0 and len(doc[0].get_text().strip()) > min_chars


def extract_text_pipeline(pdf_path: str, dpi: int = 300, progress=None) -> tuple[str, str]:
    """Extract full text from a PDF. Returns (text, method).

    Digital PDFs use the PyMuPDF text layer; scanned PDFs are rendered and OCR'd
    page by page.
    """
    if is_digital_pdf(pdf_path):
        full_text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text() + "\n"
        return full_text, "PyMuPDF"

    images = convert_from_path(pdf_path, dpi=dpi)
    full_text = ""
    with tempfile.TemporaryDirectory(prefix="mrag_ocr_") as temp_dir:
        for i, img in enumerate(images):
            img_path = os.path.join(temp_dir, f"page_{i}.png")
            img.save(img_path)
            page_text, engine = ocr_page_intelligent(img_path)
            full_text += page_text + "\n"
            if progress:
                progress((i + 1) / len(images), desc=f"OCR page {i + 1} ({engine})")

    return full_text, "Hybrid-OCR"


def extract_page_texts(pdf_path: str, dpi: int = 300) -> list[str]:
    """Per-page text extraction with per-page OCR fallback for mixed PDFs."""
    texts = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if not text.strip():
                try:
                    images = convert_from_path(
                        pdf_path, first_page=i + 1, last_page=i + 1, dpi=dpi
                    )
                    if images:
                        text = run_tesseract_ocr(images[0])
                except Exception:
                    text = ""
            texts.append(text)
    return texts
