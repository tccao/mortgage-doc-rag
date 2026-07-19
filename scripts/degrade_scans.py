"""Produce scan-degraded variants of every clean PDF: render pages, apply
rotation/noise/blur/low-DPI, reassemble as image-only PDFs.

Deterministic: per-file RNG seeded from the filename, so re-runs are
byte-stable in content. Pages are capped so multi-hundred-page guides don't
bloat the repo; the OCR eval compares page-for-page against ground truth.

Run from the repo (uses the project venv for fitz/cv2/PIL):
    uv run python scripts/degrade_scans.py
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
DEGRADED_DIR = ROOT / "data" / "degraded"

DPI = 150
MAX_PAGES = 4
JPEG_QUALITY = 45


def degrade_page(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img.convert("L"))

    # Slight skew, as if the page was fed crooked into a scanner
    angle = float(rng.uniform(-2.0, 2.0))
    h, w = arr.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    arr = cv2.warpAffine(arr, m, (w, h), borderValue=255)

    # Sensor noise
    noise = rng.normal(0, 9, arr.shape)
    arr = np.clip(arr.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    # Mild optical blur on some pages
    if rng.random() < 0.5:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)

    # Contrast wash-out (weak toner)
    if rng.random() < 0.4:
        arr = np.clip(arr.astype(np.float64) * 0.85 + 30, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def degrade_pdf(src: Path, dest: Path) -> int:
    seed = int.from_bytes(hashlib.sha256(src.name.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)

    pages = []
    with fitz.open(src) as doc:
        for i in range(min(len(doc), MAX_PAGES)):
            pix = doc[i].get_pixmap(dpi=DPI)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(degrade_page(img, rng))

    if not pages:
        return 0

    # JPEG round-trip inside the PDF, like a real low-quality scanner
    jpeg_pages = []
    for p in pages:
        buf = io.BytesIO()
        p.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
        jpeg_pages.append(Image.open(io.BytesIO(buf.getvalue())))

    dest.parent.mkdir(parents=True, exist_ok=True)
    jpeg_pages[0].save(dest, format="PDF", save_all=True, append_images=jpeg_pages[1:])
    return len(pages)


def main() -> None:
    total_files = 0
    total_pages = 0
    for src in sorted(CLEAN_DIR.rglob("*.pdf")):
        rel = src.relative_to(CLEAN_DIR)
        dest = DEGRADED_DIR / rel.parent / f"{src.stem}_scan.pdf"
        n = degrade_pdf(src, dest)
        total_files += 1
        total_pages += n
        print(f"  ok  {rel.parent}/{dest.name} ({n} pages)")
    print(f"\nDegraded {total_files} files, {total_pages} pages")


if __name__ == "__main__":
    main()
