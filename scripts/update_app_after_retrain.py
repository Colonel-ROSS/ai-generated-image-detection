"""
Update gradio_app_v2.py to point to the newly trained streams_23_fixed model.
Run after compute_threshold.py has saved experiments/threshold_fixed.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

THRESHOLD_FILE = ROOT / "experiments" / "threshold_fixed.json"
APP_FILE = ROOT / "app" / "gradio_app_v2.py"

if not THRESHOLD_FILE.exists():
    print(f"ERROR: {THRESHOLD_FILE} not found — run compute_threshold.py first")
    sys.exit(1)

with open(THRESHOLD_FILE) as f:
    threshold_data = json.load(f)

new_threshold = threshold_data["optimal_threshold"]
auc = threshold_data["full_set_auc"]
tpr = threshold_data["at_optimal"]["TPR"]
tnr = threshold_data["at_optimal"]["TNR"]
bal = threshold_data["at_optimal"]["balanced_acc"]
print(f"New threshold: {new_threshold} (AUC={auc}, balanced_acc={bal})")

# Read app source
content = APP_FILE.read_text()

# 1. Update checkpoint path
content = content.replace(
    '"streams_23_full_best.pth"',
    '"streams_23_fixed_best.pth"'
)
content = content.replace(
    '"streams_23_fixed_best.pth"',
    '"streams_23_fixed_best.pth"'
)

# 2. Update features dir
content = content.replace(
    '"features_full"',
    '"features_fixed"'
)

# 3. Update DECISION_THRESHOLD value
import re
content = re.sub(
    r'DECISION_THRESHOLD = [0-9.]+',
    f'DECISION_THRESHOLD = {new_threshold}',
    content
)

# 4. Update the docstring/comment above threshold (update metric values)
content = content.replace(
    '80k feature set. The model was trained on 79% real / 21% fake data so the raw\n# sigmoid output is biased low; threshold 0.5 catches only 47% of fakes.\n# At 0.2342: TPR=0.772, TNR=0.821, balanced_acc=0.797 (vs 0.715 at 0.5).',
    f'80k feature set (fixed NSS). The model was trained on 79% real / 21% fake data so the raw\n# sigmoid output is biased low; threshold 0.5 catches only 47% of fakes.\n# At {new_threshold}: TPR={tpr}, TNR={tnr}, balanced_acc={bal} (fixed NSS model).'
)

# 5. Update the docstring at the top of the file
content = content.replace(
    'Model: streams_23_full_best.pth  —  F2 (FFT) + F3 (NSS), AUC=0.8624 in-dist / 0.9172 fresh.',
    f'Model: streams_23_fixed_best.pth  —  F2 (FFT) + F3 (NSS, fixed), AUC={auc} full-set.'
)

# 6. Update decision threshold display in the performance section
old_line = '**Decision threshold:** 0.2342 (Youden\'s J on 80k set; training imbalance 79%/21% real/fake shifts optimal threshold well below 0.5).'
new_line = f'**Decision threshold:** {new_threshold} (Youden\'s J on 80k set; NSS bug fixed model. Training imbalance 79%/21% shifts threshold below 0.5).'
content = content.replace(old_line, new_line)

APP_FILE.write_text(content)
print(f"Updated {APP_FILE}")
print(f"  checkpoint  → streams_23_fixed_best.pth")
print(f"  features    → features_fixed")
print(f"  threshold   → {new_threshold}")

# Verify changes applied
updated = APP_FILE.read_text()
assert "streams_23_fixed_best.pth" in updated, "checkpoint update failed"
assert "features_fixed" in updated, "features dir update failed"
assert f"DECISION_THRESHOLD = {new_threshold}" in updated, "threshold update failed"
print("Verification passed.")
