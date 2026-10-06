"""
Update experiments/final_results.json with combined model results.
Run after finalize_combined_model.py, evaluate_per_generator.py,
and evaluate_cross_benchmark.py all complete.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FINAL_JSON = ROOT / "experiments" / "final_results.json"
ABLATION_DIR = ROOT / "experiments" / "ablation"
BENCHMARKS_DIR = ROOT / "experiments" / "benchmarks"
THRESHOLD_JSON = ROOT / "experiments" / "threshold_combined.json"

def latest_file(pattern, search_dir):
    files = sorted(search_dir.glob(pattern))
    return files[-1] if files else None

# Load existing final_results.json
with open(FINAL_JSON) as f:
    results = json.load(f)

print("=== Updating final_results.json with combined model results ===")

# ── Combined model full eval ─────────────────────────────────────────────────
full_eval_path = latest_file("streams_23_combined_full_eval_*.json", ABLATION_DIR)
if not full_eval_path:
    full_eval_path = latest_file("streams_23_combined*.json", ABLATION_DIR)

if full_eval_path:
    with open(full_eval_path) as f:
        combined_eval = json.load(f)
    print(f"Loaded full eval: {full_eval_path.name}")
else:
    combined_eval = {}
    print("WARNING: No combined model eval JSON found in experiments/ablation/")

# ── Threshold calibration ────────────────────────────────────────────────────
threshold_data = {}
if THRESHOLD_JSON.exists():
    with open(THRESHOLD_JSON) as f:
        threshold_data = json.load(f)
    print(f"Loaded threshold: {THRESHOLD_JSON.name}")
else:
    print("WARNING: threshold_combined.json not found")

# ── Per-generator results ────────────────────────────────────────────────────
per_gen_path = latest_file("streams_23_combined_per_generator_*.json", ABLATION_DIR)
if not per_gen_path:
    per_gen_path = latest_file("streams_23_per_generator_*.json", ABLATION_DIR)
per_gen_data = {}
if per_gen_path:
    with open(per_gen_path) as f:
        per_gen_data = json.load(f)
    print(f"Loaded per-gen: {per_gen_path.name}")
else:
    print("WARNING: No per-generator results found")

# ── Cross-benchmark results ──────────────────────────────────────────────────
cross_bench_path = latest_file("cross_bench_*.json", BENCHMARKS_DIR)
cross_bench_data = {}
if cross_bench_path:
    with open(cross_bench_path) as f:
        cross_bench_data = json.load(f)
    print(f"Loaded cross-bench: {cross_bench_path.name}")
else:
    print("WARNING: No cross-benchmark results found")

# ── Build combined_model entry ───────────────────────────────────────────────
test_eval = combined_eval.get("test_evaluation", {})
rob_eval = combined_eval.get("robustness", {})

combined_entry = {
    "status": "COMPLETE",
    "session": 11,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "description": (
        "F2+F3 fusion model trained on combined dataset: "
        "COCO(63560) + COCO-extra(54257) + Flickr30k(~31783) real + "
        "SD1.5(16000) + DiffusionDB-SD2(~10000) fake = ~175600 images total."
    ),
    "training": {
        "checkpoint": "models/checkpoints/streams_23_combined_best.pth",
        "features_dir": "features_combined/",
        "streams": "23",
        "epochs": 50,
        "batch_size": 256,
        "dataset_size": {
            "real_approx": 149600,
            "fake_approx": 26500,
            "total_approx": 176100,
            "note": "~85% real / ~15% fake — class imbalance compensated by Youden threshold"
        }
    },
    "test_evaluation": test_eval,
    "threshold_calibration": threshold_data,
    "robustness": rob_eval,
    "vs_fixed_model": {
        "fixed_test_auc": 0.8639,
        "fixed_full_set_auc": 0.8837,
        "note": "streams_23_fixed was trained on 80k images; combined adds ~95k more"
    }
}

# Per-generator breakdown
if per_gen_data:
    per_gen_breakdown = per_gen_data.get("per_generator", per_gen_data.get("generators", {}))
    combined_entry["per_generator"] = per_gen_breakdown

# Cross-benchmark
if cross_bench_data:
    overall = cross_bench_data.get("overall", {})
    per_source = cross_bench_data.get("per_source", {})
    combined_entry["cross_benchmark"] = {
        "dataset": cross_bench_data.get("dataset", "marco-willi/cnnspot-small"),
        "overall_auc": overall.get("auc"),
        "overall_acc": overall.get("acc"),
        "overall_ap": overall.get("ap"),
        "n_sources": len(per_source),
        "n_samples": cross_bench_data.get("n_samples"),
        "per_source": per_source,
    }

results["combined_model"] = combined_entry

# Update metadata
results["metadata"]["last_updated"] = datetime.utcnow().strftime("%Y-%m-%d") + " (Session 11: combined model)"
results["metadata"]["dataset"] = {
    "real": {
        "coco_original": 63560,
        "coco_extra": 54257,
        "flickr30k": "~31783",
        "total_real_approx": "~149600"
    },
    "fake": {
        "stable_diffusion_1_5": 16000,
        "stylegan3": 500,
        "diffusiondb_sd2": "~10000",
        "total_fake_approx": "~26500"
    },
    "combined_total_approx": "~176100",
    "class_balance": "~85% real / ~15% fake"
}

# Save
with open(FINAL_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved updated final_results.json ({FINAL_JSON})")

# Print summary
print("\n=== Combined model summary ===")
if test_eval:
    print(f"Test AUC: {test_eval.get('auc', 'N/A')}")
    print(f"Test ACC: {test_eval.get('acc', 'N/A')}")
    print(f"Test AP:  {test_eval.get('ap', 'N/A')}")
if threshold_data:
    print(f"Full-set AUC: {threshold_data.get('full_set_auc', 'N/A')}")
    print(f"Optimal threshold: {threshold_data.get('optimal_threshold', 'N/A')}")
    at_opt = threshold_data.get('at_optimal', {})
    print(f"At threshold: TPR={at_opt.get('TPR','N/A')} TNR={at_opt.get('TNR','N/A')} balanced_acc={at_opt.get('balanced_acc','N/A')}")

print("\n=== update_final_results complete ===")
