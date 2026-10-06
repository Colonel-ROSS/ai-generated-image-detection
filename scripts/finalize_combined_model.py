"""
Post-phase5: compute Youden's J threshold for streams_23_combined_best.pth
and update gradio_app_v2.py to point at the combined model.
Run after scripts/phase5_combine_and_retrain.py completes.
"""
import sys, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

FEATURES_DIR = ROOT / "features_combined"
CKPT_DIR     = ROOT / "models" / "checkpoints"
APP_FILE     = ROOT / "app" / "gradio_app_v2.py"

# Find checkpoint
candidates = sorted(CKPT_DIR.glob("streams_23_combined_best*.pth"))
if not candidates:
    candidates = sorted(CKPT_DIR.glob("streams_23_combined*.pth"))
if not candidates:
    print("ERROR: No streams_23_combined checkpoint found!")
    sys.exit(1)
CKPT_PATH = candidates[-1]
print(f"Using checkpoint: {CKPT_PATH.name}")

# Load features
labels = np.load(str(FEATURES_DIR / "labels.npy")).astype(np.float32)
s2 = np.load(str(FEATURES_DIR / "stream2" / "features.npy")).astype(np.float32)
s3 = np.load(str(FEATURES_DIR / "stream3" / "features.npy")).astype(np.float32)
X = np.concatenate([s2, s3], axis=1)
print(f"Features: {X.shape}  real={(labels==0).sum()}  fake={(labels==1).sum()}")

# Apply normalisation
stats = np.load(str(FEATURES_DIR / "feature_stats.npz"))
X = ((X - stats["mean"]) / np.maximum(stats["std"], 1e-2)).astype(np.float32)

# Load FusionHead
class FusionHead(nn.Module):
    def __init__(self, d=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

model = FusionHead(X.shape[1])
state = torch.load(str(CKPT_PATH), map_location="cpu")
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]
model.load_state_dict(state)
model.eval()

# Predict on ALL samples
loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=512, shuffle=False)
probs = []
with torch.no_grad():
    for (x,) in loader:
        probs.append(model(x).squeeze(1).numpy())
probs = np.concatenate(probs)

auc = roc_auc_score(labels, probs)
print(f"Full-set AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(labels, probs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
opt_threshold = float(thresholds[best_idx])
tpr_at_opt = float(tpr[best_idx])
tnr_at_opt = 1.0 - float(fpr[best_idx])
balanced_acc = (tpr_at_opt + tnr_at_opt) / 2

print(f"Optimal threshold (Youden's J): {opt_threshold:.4f}")
print(f"  TPR={tpr_at_opt:.4f}  TNR={tnr_at_opt:.4f}  balanced_acc={balanced_acc:.4f}")

pred_05  = (probs >= 0.5).astype(int)
pred_opt = (probs >= opt_threshold).astype(int)
acc_05   = accuracy_score(labels, pred_05)
acc_opt  = accuracy_score(labels, pred_opt)
print(f"\nAt threshold=0.5:     acc={acc_05:.4f}")
print(f"At threshold={opt_threshold:.4f}: acc={acc_opt:.4f}")
print(f"\nPrevious (fixed) threshold: 0.196")

# Save threshold JSON
result = {
    "checkpoint": str(CKPT_PATH),
    "features_dir": str(FEATURES_DIR),
    "n_samples": int(len(labels)),
    "n_real": int((labels==0).sum()),
    "n_fake": int((labels==1).sum()),
    "full_set_auc": round(auc, 4),
    "optimal_threshold": round(opt_threshold, 4),
    "at_optimal": {
        "TPR": round(tpr_at_opt, 4),
        "TNR": round(tnr_at_opt, 4),
        "balanced_acc": round(balanced_acc, 4),
    },
    "at_0_5": {"acc": round(acc_05, 4)},
}
out_path = ROOT / "experiments" / "threshold_combined.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {out_path}")

# Update gradio_app_v2.py
print("\nUpdating gradio_app_v2.py ...")
content = APP_FILE.read_text()

# checkpoint path (from fixed → combined)
content = content.replace('"streams_23_fixed_best.pth"', '"streams_23_combined_best.pth"')
# features dir (from features_fixed → features_combined)
content = content.replace('"features_fixed"', '"features_combined"')
# threshold value
content = re.sub(r'DECISION_THRESHOLD = [0-9.]+',
                 f'DECISION_THRESHOLD = {opt_threshold}', content)
# model docstring
content = re.sub(
    r'Model: streams_23_fixed_best\.pth.*?full-set\.',
    f'Model: streams_23_combined_best.pth  —  F2 (FFT) + F3 (NSS, fixed), AUC={round(auc,4)} full-set.',
    content
)
APP_FILE.write_text(content)

# Verify
updated = APP_FILE.read_text()
assert "streams_23_combined_best.pth" in updated, "checkpoint update failed"
assert "features_combined" in updated, "features dir update failed"
assert f"DECISION_THRESHOLD = {opt_threshold}" in updated, "threshold update failed"
print(f"Updated {APP_FILE}")
print(f"  checkpoint → streams_23_combined_best.pth")
print(f"  features   → features_combined")
print(f"  threshold  → {opt_threshold}")
print("Verification passed.")
print("\n=== finalize_combined_model complete ===")
