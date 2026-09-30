"""
upload_to_kaggle.py
==============================================
Upload the LEVIR-CD dataset directly to Kaggle
using kagglehub (no zipping required).

SETUP (run once):
    pip install kagglehub

    Then set your Kaggle API credentials:
    Either:
      - Place kaggle.json in C:/Users/<you>/.kaggle/kaggle.json
      - OR set env vars: KAGGLE_USERNAME and KAGGLE_KEY

HOW TO GET kaggle.json:
    kaggle.com -> Your Profile -> Settings -> API -> Create New Token

RUN:
    python upload_to_kaggle.py
==============================================
"""

import kagglehub

# ── CONFIG: Fill these in ───────────────────────────────────
KAGGLE_USERNAME   = "your_kaggle_username"        # <-- change this
DATASET_SLUG      = "levir-cd-statquery"           # <-- dataset name on Kaggle (new)
LOCAL_DATASET_DIR = r"e:\SIH'26\Dataset\LEVIR CD"  # local folder to upload
# ────────────────────────────────────────────────────────────

HANDLE = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"

print(f"[INFO] Kaggle handle : {HANDLE}")
print(f"[INFO] Local source  : {LOCAL_DATASET_DIR}")
print()

# -- Step 1: Create the dataset for the first time
print("[STEP 1] Creating new Kaggle dataset (first upload)...")
kagglehub.dataset_upload(
    handle          = HANDLE,
    local_dataset_dir = LOCAL_DATASET_DIR,
)
print(f"[DONE] Dataset created at: https://www.kaggle.com/datasets/{HANDLE}")
print()

# -- Step 2 (optional): Push a new version with notes
# Uncomment this block when you want to update the dataset later:
#
# print("[STEP 2] Pushing updated version...")
# kagglehub.dataset_upload(
#     handle            = HANDLE,
#     local_dataset_dir = LOCAL_DATASET_DIR,
#     version_notes     = "Added preprocessed patches",
# )
# print("[DONE] New version pushed.")

# -- Step 3 (optional): Upload while ignoring specific files/dirs
# Uncomment if you want to exclude certain files:
#
# kagglehub.dataset_upload(
#     handle            = HANDLE,
#     local_dataset_dir = LOCAL_DATASET_DIR,
#     ignore_patterns   = ["*.zip", "*.tmp", "__pycache__/"],
# )
