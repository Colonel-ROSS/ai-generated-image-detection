"""
Compute optimal decision threshold for the fixed model (streams_23_fixed_best.pth)
using Youden's J on the full feature set, then print the result.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_curve, roc_auc_score

FEATURES_DIR = ROOT / "features_fixed"
CKPT_PATH    = ROOT / "models" / "checkpoints" / "streams_23_fixed_best.pth"

# Load features
labels = np.load(str(FEATURES_DIR / "labels.npy")).astype(np.float32)
s2 = np.load(str(FEATURES_DIR / "stream2" / "features.npy")).astype(np.float32)
s3 = np.load(str(FEATURES_DIR / "stream3" / "features.npy")).astype(np.float32)
X = np.concatenate([s2, s3], axis=1)

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

# Compute AUC
auc = roc_auc_score(labels, probs)
print(f"Full-set AUC: {auc:.4f}")

# Youden's J: argmax(TPR - FPR) = argmax(TPR + TNR - 1)
fpr, tpr, thresholds = roc_curve(labels, probs)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
opt_threshold = float(thresholds[best_idx])
tpr_at_opt = float(tpr[best_idx])
tnr_at_opt = 1.0 - float(fpr[best_idx])
balanced_acc = (tpr_at_opt + tnr_at_opt) / 2

print(f"Optimal threshold (Youden's J): {opt_threshold:.4f}")
print(f"  TPR={tpr_at_opt:.4f}  TNR={tnr_at_opt:.4f}  balanced_acc={balanced_acc:.4f}")

# Compare to 0.5
pred_05 = (probs >= 0.5).astype(int)
pred_opt = (probs >= opt_threshold).astype(int)
from sklearn.metrics import accuracy_score
acc_05  = accuracy_score(labels, pred_05)
acc_opt = accuracy_score(labels, pred_opt)
print(f"\nAt threshold=0.5:     acc={acc_05:.4f}")
print(f"At threshold={opt_threshold:.4f}: acc={acc_opt:.4f}")

# Also compare to old model threshold
print(f"\nOld model threshold was 0.2342")
print(f"New optimal threshold:   {opt_threshold:.4f}")

import json
result = {
    "checkpoint": str(CKPT_PATH),
    "full_set_auc": round(auc, 4),
    "optimal_threshold": round(opt_threshold, 4),
    "at_optimal": {
        "TPR": round(tpr_at_opt, 4),
        "TNR": round(tnr_at_opt, 4),
        "balanced_acc": round(balanced_acc, 4),
    },
    "at_0_5": {
        "acc": round(acc_05, 4),
    },
}
out_path = ROOT / "experiments" / "threshold_fixed.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved → {out_path}")
