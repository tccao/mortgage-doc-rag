# /// script
# requires-python = ">=3.11"
# dependencies = ["requests"]
# ///
"""Download the public mortgage-document corpus defined in data/manifest.csv.

Every file is a US-government work (CFPB, HUD, IRS, VA, FTC — public domain).
Files land in data/clean/<doc_type>/<filename>. Existing files with matching
recorded checksums are skipped; checksums are recorded in data/checksums.csv
on first download so the corpus is byte-reproducible.
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
CHECKSUMS = ROOT / "data" / "checksums.csv"
CLEAN_DIR = ROOT / "data" / "clean"

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) mortgage-doc-rag corpus builder"}
MIN_BYTES = 10_000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_checksums() -> dict[str, str]:
    if not CHECKSUMS.exists():
        return {}
    with open(CHECKSUMS) as f:
        return {row["filename"]: row["sha256"] for row in csv.DictReader(f)}


def save_checksums(checksums: dict[str, str]) -> None:
    with open(CHECKSUMS, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "sha256"])
        for name in sorted(checksums):
            writer.writerow([name, checksums[name]])


def main() -> int:
    checksums = load_checksums()
    ok, skipped, failed = [], [], []

    with open(MANIFEST) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        dest = CLEAN_DIR / row["doc_type"] / row["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            data = dest.read_bytes()
            recorded = checksums.get(row["filename"])
            if recorded and sha256(data) == recorded:
                skipped.append(row["filename"])
                continue

        try:
            resp = requests.get(row["url"], headers=HEADERS, timeout=60)
            resp.raise_for_status()
            data = resp.content
            if not data.startswith(b"%PDF"):
                raise ValueError(f"not a PDF (starts with {data[:8]!r})")
            if len(data) < MIN_BYTES:
                raise ValueError(f"suspiciously small ({len(data)} bytes)")
            dest.write_bytes(data)
            checksums[row["filename"]] = sha256(data)
            ok.append(row["filename"])
            print(f"  ok  {row['doc_type']}/{row['filename']} ({len(data) // 1024} KB)")
        except Exception as e:
            failed.append((row["filename"], row["url"], str(e)))
            print(f"FAIL  {row['filename']}: {e}")

    save_checksums(checksums)

    print(f"\nDownloaded {len(ok)}, skipped (cached) {len(skipped)}, failed {len(failed)}")
    if failed:
        print("\nFailures (substitute or fix URLs in manifest.csv):")
        for name, url, err in failed:
            print(f"  {name}: {err}\n    {url}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
