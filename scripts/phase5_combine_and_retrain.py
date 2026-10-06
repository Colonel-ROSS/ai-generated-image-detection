"""
Phase 5 — Combine all feature files and retrain FusionHead.

Combines:
  features_fixed/  (80,060 original with corrected NSS)
  features_new/*/  (any additional downloaded datasets)

Then retrains streams_23_combined model.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

FEATURES_FIXED = ROOT / "features_fixed"
FEATURES_NEW   = ROOT / "features_new"
OUT_DIR        = ROOT / "features_combined"

print("=== Phase 5.1: Combine all features ===")

all_s2, all_s3, all_labels = [], [], []

# Core fixed features
print(f"Loading features_fixed/ ...")
s2 = np.load(str(FEATURES_FIXED / "stream2" / "features.npy"))
s3 = np.load(str(FEATURES_FIXED / "stream3" / "features.npy"))
y  = np.load(str(FEATURES_FIXED / "labels.npy"))
print(f"  features_fixed: S2={s2.shape} S3={s3.shape} labels={y.shape} real={( y==0).sum()} fake={(y==1).sum()}")
all_s2.append(s2); all_s3.append(s3); all_labels.append(y)

# Any additional feature files in features_new/
if FEATURES_NEW.exists():
    for subdir in sorted(FEATURES_NEW.iterdir()):
        s2_path = subdir / "stream2" / "features.npy"
        s3_path = subdir / "stream3" / "features.npy"
        y_path  = subdir / "labels.npy"
        if s2_path.exists() and s3_path.exists() and y_path.exists():
            _s2 = np.load(str(s2_path))
            _s3 = np.load(str(s3_path))
            _y  = np.load(str(y_path))
            print(f"  {subdir.name}: S2={_s2.shape} S3={_s3.shape} real={(_y==0).sum()} fake={(_y==1).sum()}")
            all_s2.append(_s2); all_s3.append(_s3); all_labels.append(_y)
        else:
            print(f"  {subdir.name}: missing files, skipping")
else:
    print("  No features_new/ directory found.")

# Concatenate
X2 = np.concatenate(all_s2, axis=0)
X3 = np.concatenate(all_s3, axis=0)
y_all = np.concatenate(all_labels, axis=0)

print(f"\nCombined: S2={X2.shape} S3={X3.shape} labels={y_all.shape}")
print(f"  Real: {(y_all==0).sum()} ({100*(y_all==0).mean():.1f}%)")
print(f"  Fake: {(y_all==1).sum()} ({100*(y_all==1).mean():.1f}%)")

# Save
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "stream2").mkdir(exist_ok=True)
(OUT_DIR / "stream3").mkdir(exist_ok=True)
np.save(str(OUT_DIR / "stream2" / "features.npy"), X2)
np.save(str(OUT_DIR / "stream3" / "features.npy"), X3)
np.save(str(OUT_DIR / "labels.npy"), y_all)

# Copy encoder weights for robustness eval
import shutil
shutil.copy2(str(FEATURES_FIXED / "stream2" / "stream2_encoder.pth"),
             str(OUT_DIR / "stream2" / "stream2_encoder.pth"))
shutil.copy2(str(FEATURES_FIXED / "stream3" / "nss_encoder.pth"),
             str(OUT_DIR / "stream3" / "nss_encoder.pth"))
print(f"Saved combined features to {OUT_DIR}")

print("\n=== Phase 5.3: Retrain FusionHead on combined features ===")
cmd = [
    "python3", "src/fusion/train.py",
    "--streams", "23",
    "--features_dir", "features_combined",
    "--exp_name", "streams_23_combined",
    "--epochs", "50",
    "--batch_size", "256",
    "--patience", "10",
    "--skip_extract",
]
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, cwd=str(ROOT))
if result.returncode != 0:
    print("Training failed!")
    sys.exit(1)

print("\n=== Phase 5.4: Full evaluation ===")
# Find the best checkpoint
ckpt_dir = ROOT / "models" / "checkpoints"
candidates = sorted(ckpt_dir.glob("streams_23_combined_best*.pth"))
if not candidates:
    candidates = sorted(ckpt_dir.glob("streams_23_combined*.pth"))
ckpt = candidates[-1] if candidates else None

if ckpt:
    eval_cmd = [
        "python3", "src/fusion/evaluate.py",
        "--full_eval",
        "--streams", "23",
        "--features_dir", "features_combined",
        "--checkpoint", str(ckpt.relative_to(ROOT)),
    ]
    print(f"Running: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, cwd=str(ROOT))
else:
    print("No checkpoint found for evaluation!")

print("\n=== Phase 5 complete ===")
