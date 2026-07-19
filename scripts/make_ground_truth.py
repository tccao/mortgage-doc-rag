"""Freeze deterministic plain-text ground truth for every clean PDF.

Text comes from the digital PDF's own text layer via PyMuPDF — no OCR, no
model in the loop — so it is a stable reference for scoring OCR output,
chunker behavior, and final answers. Pages are separated by form-feed (\\f)
so page-level comparisons (against page-capped degraded scans) line up.

Run: uv run python scripts/make_ground_truth.py
"""

from __future__ import annotations

from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
GT_DIR = ROOT / "data" / "ground_truth"


def main() -> None:
    count = 0
    empty = []
    for src in sorted(CLEAN_DIR.rglob("*.pdf")):
        rel = src.relative_to(CLEAN_DIR)
        dest = GT_DIR / rel.parent / f"{src.stem}.txt"
        dest.parent.mkdir(parents=True, exist_ok=True)

        with fitz.open(src) as doc:
            pages = [page.get_text() for page in doc]
        text = "\f".join(pages)
        dest.write_text(text)
        count += 1
        if not text.strip():
            empty.append(str(rel))

    print(f"Wrote {count} ground-truth files to {GT_DIR}")
    if empty:
        print(f"\nNo text layer ({len(empty)} files — image-only PDFs, excluded from OCR CER eval):")
        for e in empty:
            print(f"  {e}")


if __name__ == "__main__":
    main()
