"""
prepare_dataset_for_kaggle.py
==============================================
Run this ONCE on your local machine to zip the
LEVIR-CD dataset into a format ready for Kaggle upload.

Usage:
    python prepare_dataset_for_kaggle.py

Output:
    levir_cd_kaggle.zip  (upload this to Kaggle Datasets)
==============================================
"""

import os
import zipfile
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────
DATASET_ROOT = r"e:\SIH'26\Dataset\LEVIR CD"
OUTPUT_ZIP   = r"e:\SIH'26\Dataset\levir_cd_kaggle.zip"
# ────────────────────────────────────────────────────────

def zip_levir_cd(src_root: str, out_zip: str):
    src = Path(src_root)
    assert src.exists(), f"Dataset not found at: {src_root}"

    splits = ["train", "val", "test"]
    folders = ["A", "B", "label"]

    print(f"[ZIP] Zipping LEVIR-CD dataset from: {src_root}")
    print(f"[OUT] Output: {out_zip}\n")

    total_files = 0

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for split in splits:
            for folder in folders:
                dir_path = src / split / folder
                if not dir_path.exists():
                    print(f"  [SKIP] Missing: {dir_path}")
                    continue

                files = sorted(dir_path.glob("*.png"))
                print(f"  [OK] {split}/{folder}: {len(files)} files")

                for file in files:
                    # Archive path inside zip: LEVIR_CD/train/A/train_1.png
                    arcname = f"LEVIR_CD/{split}/{folder}/{file.name}"
                    zf.write(file, arcname=arcname)
                    total_files += 1

    size_mb = os.path.getsize(out_zip) / (1024 * 1024)
    print(f"\n[DONE] {total_files} files zipped.")
    print(f"[SIZE] ZIP size: {size_mb:.1f} MB")
    print(f"\n[NEXT] Upload '{out_zip}' to https://www.kaggle.com/datasets/new")
    print("   Dataset name suggestion: 'levir-cd-building-change-detection'")


if __name__ == "__main__":
    zip_levir_cd(DATASET_ROOT, OUTPUT_ZIP)
