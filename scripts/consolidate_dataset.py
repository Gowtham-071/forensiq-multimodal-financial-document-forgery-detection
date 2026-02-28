"""
Standalone dataset consolidation script — same logic as notebooks/15_dataset_consolidate.ipynb
Run: python scripts/consolidate_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

BASE_DIR = Path(r"c:\Users\saigo\Desktop\fraud_document_ai")
DATASET_DIR = BASE_DIR / "dataset"

# Create directories
print("Creating directory structure...")
for split in ["raw/fraud", "raw/genuine", "genuine_extended",
              "train/fraud", "train/genuine",
              "val/fraud",   "val/genuine",
              "test/fraud",  "test/genuine"]:
    (DATASET_DIR / split).mkdir(parents=True, exist_ok=True)
print("✅ Directories created")

# ── Collect fraud images ──────────────────────────────────────────────────

FRAUD_SOURCES = [
    BASE_DIR / "Main_Dataset" / "train"           / "fraud",
    BASE_DIR / "Main_Dataset" / "val"             / "fraud",
    BASE_DIR / "Main_Dataset" / "test"            / "fraud",
    BASE_DIR / "Main_Dataset" / "augmented"       / "fraud",
    BASE_DIR / "Main_Dataset" / "augmented_debug" / "fraud",
]

all_fraud = []
for src in FRAUD_SOURCES:
    if src.exists():
        imgs = [p for p in src.iterdir() if p.suffix.lower() in [".jpg",".jpeg",".png"]]
        print(f"  {src.relative_to(BASE_DIR)}: {len(imgs)} fraud images")
        all_fraud.extend(imgs)

print(f"Total fraud found: {len(all_fraud)}")

# ── Collect genuine images ────────────────────────────────────────────────

GENUINE_SOURCES = [
    BASE_DIR / "Main_Dataset" / "augmented"       / "genuine",
    BASE_DIR / "Main_Dataset" / "augmented_debug" / "genuine",
    BASE_DIR / "Main_Dataset" / "train"           / "genuine",
    BASE_DIR / "Main_Dataset" / "val"             / "genuine",
    BASE_DIR / "Main_Dataset" / "test"            / "genuine",
]

# Add Geniune Document DS batches
genuine_ds = BASE_DIR / "Geniune Document DS"
if genuine_ds.exists():
    for batch_dir in sorted(genuine_ds.iterdir()):
        if batch_dir.is_dir():
            GENUINE_SOURCES.append(batch_dir)
            for sub in batch_dir.iterdir():
                if sub.is_dir():
                    GENUINE_SOURCES.append(sub)

all_genuine = []
seen_names = set()
for src in GENUINE_SOURCES:
    if src.exists():
        imgs = [p for p in src.iterdir()
                if p.suffix.lower() in [".jpg",".jpeg",".png"] and p.name not in seen_names]
        for p in imgs:
            seen_names.add(p.name)
        all_genuine.extend(imgs)

print(f"Total unique genuine found: {len(all_genuine)}")

# ── Deduplicate + cap ─────────────────────────────────────────────────────

seen_fraud = set()
fraud_unique = []
for p in all_fraud:
    if p.name not in seen_fraud:
        seen_fraud.add(p.name)
        fraud_unique.append(p)

random.seed(42)
random.shuffle(fraud_unique)
random.shuffle(all_genuine)

n_fraud   = len(fraud_unique)
n_genuine_cap = min(n_fraud * 2, len(all_genuine))

genuine_selected = all_genuine[:n_genuine_cap]
genuine_extended = all_genuine[n_genuine_cap:]

print(f"\n✅ Fraud (deduped):    {n_fraud}")
print(f"✅ Genuine (selected): {n_genuine_cap} (2x cap)")
print(f"📦 Genuine (extended): {len(genuine_extended)} → genuine_extended/")
print(f"📊 TOTAL in dataset:   {n_fraud + n_genuine_cap}")

# ── Copy to raw/ ──────────────────────────────────────────────────────────

def safe_copy(src, dest_dir):
    dest = dest_dir / src.name
    if dest.exists():
        dest = dest_dir / (src.parent.name + "_" + src.name)
    shutil.copy2(str(src), str(dest))

print("\nCopying fraud → dataset/raw/fraud/ ...")
for p in fraud_unique:
    safe_copy(p, DATASET_DIR / "raw" / "fraud")

print("Copying genuine → dataset/raw/genuine/ ...")
for p in genuine_selected:
    safe_copy(p, DATASET_DIR / "raw" / "genuine")

print("Copying extended genuine → dataset/genuine_extended/ ...")
for p in genuine_extended[:5000]:  # cap at 5K for disk space
    safe_copy(p, DATASET_DIR / "genuine_extended")

print("✅ Copy complete")

# ── 60/20/20 Stratified Split ─────────────────────────────────────────────

from sklearn.model_selection import train_test_split

fraud_files   = sorted((DATASET_DIR / "raw" / "fraud").iterdir())
genuine_files = sorted((DATASET_DIR / "raw" / "genuine").iterdir())
all_files     = fraud_files + genuine_files
all_labels    = [0]*len(fraud_files) + [1]*len(genuine_files)

X_train, X_temp, y_train, y_temp = train_test_split(
    all_files, all_labels, test_size=0.40, stratify=all_labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)

print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

CLASS_MAP = {0: "fraud", 1: "genuine"}

for split_name, flist, lbls in [("train",X_train,y_train),("val",X_val,y_val),("test",X_test,y_test)]:
    for fpath, lbl in zip(flist, lbls):
        dest = DATASET_DIR / split_name / CLASS_MAP[lbl] / fpath.name
        shutil.copy2(str(fpath), str(dest))

print("\n" + "="*55)
print("FINAL VERIFICATION")
print("="*55)
total = 0
for split in ["train","val","test"]:
    for cls in ["fraud","genuine"]:
        n = len(list((DATASET_DIR / split / cls).iterdir()))
        total += n
        print(f"  dataset/{split}/{cls}: {n}")
ext = len(list((DATASET_DIR / "genuine_extended").iterdir()))
print(f"\n  dataset/genuine_extended/: {ext}")
print(f"  TOTAL (train+val+test): {total}")
print("="*55)
print("✅ Dataset consolidation complete!")
print("Old folders (Main_Dataset/, Geniune Document DS/) kept as archive.")
