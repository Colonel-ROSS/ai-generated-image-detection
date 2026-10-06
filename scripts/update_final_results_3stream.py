#!/usr/bin/env python3
"""
scripts/update_final_results_3stream.py

Update experiments/final_results.json with completed 3-stream evaluation results.
Run after all evaluation scripts have finished.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FINAL_RESULTS = ROOT / "experiments" / "final_results.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def latest_file(pattern_dir: Path, glob: str):
    files = sorted(pattern_dir.glob(glob))
    return files[-1] if files else None


def main():
    final = load_json(FINAL_RESULTS)
    section = final.setdefault("three_stream_trained_prnu_model", {})

    # ── Per-generator (feature-store, fast eval) ──────────────────────────────
    per_gen_dir = ROOT / "experiments" / "ablation"
    per_gen_f = latest_file(per_gen_dir, "streams_123_per_generator_202606*.json")
    if per_gen_f:
        pg = load_json(per_gen_f)
        section["per_generator_stored"] = {
            "source": per_gen_f.name,
            "n_samples": pg.get("n_samples"),
            "overall": pg.get("overall"),
            "per_generator": pg.get("per_generator"),
        }
        print(f"[OK] per_generator_stored from {per_gen_f.name}")
    else:
        print("[SKIP] No per-generator results found")

    # ── ForenSynths ───────────────────────────────────────────────────────────
    forensynths_f = ROOT / "experiments" / "cross_benchmark" / "forensynths_3stream_results.json"
    if forensynths_f.exists():
        fs = load_json(forensynths_f)
        section["cross_benchmark"]["forensynths"] = {
            "source": "forensynths_3stream_results.json",
            "overall_auc": fs.get("overall_auc"),
            "overall_acc": fs.get("overall_acc"),
            "threshold": fs.get("threshold"),
            "n_generators": fs.get("n_generators"),
            "n_total": fs.get("n_total"),
            "best_generator": max(
                ((k, v.get("auc", 0)) for k, v in fs.get("per_generator", {}).items()),
                key=lambda x: x[1] if isinstance(x[1], float) else 0,
                default=("?", 0)
            ),
            "worst_generator": min(
                ((k, v.get("auc", 1)) for k, v in fs.get("per_generator", {}).items()),
                key=lambda x: x[1] if isinstance(x[1], float) else 1,
                default=("?", 0)
            ),
            "per_generator": {
                k: {"auc": v.get("auc"), "acc": v.get("acc")}
                for k, v in fs.get("per_generator", {}).items()
            },
        }
        print(f"[OK] ForenSynths results: AUC={fs.get('overall_auc'):.4f}")
    else:
        print("[SKIP] ForenSynths results not found")

    # ── CNNSpot cross-benchmark ───────────────────────────────────────────────
    bench_dir = ROOT / "experiments" / "benchmarks"
    crossbench_f = latest_file(bench_dir, "cross_bench_marco-willi_cnnspot-small_2026061*.json")
    if crossbench_f:
        cb = load_json(crossbench_f)
        section["cross_benchmark"]["cnnspot_small"] = {
            "source": crossbench_f.name,
            "overall_auc": cb.get("overall_auc"),
            "overall_acc": cb.get("overall_acc"),
            "n_generators": cb.get("n_generators") or cb.get("n_sources"),
            "n_total": cb.get("n_total"),
            "best_generator": cb.get("best_generator") or cb.get("best_source"),
            "worst_generator": cb.get("worst_generator") or cb.get("worst_source"),
        }
        print(f"[OK] CNNSpot results: AUC={cb.get('overall_auc'):.4f}")
    else:
        print("[SKIP] CNNSpot cross-benchmark results not found (looking for Jun 10 file)")

    # ── Robustness sweep ──────────────────────────────────────────────────────
    rob_dir = ROOT / "experiments" / "robustness"
    rob_f = latest_file(rob_dir, "robustness_streams_123_*.json")
    if rob_f:
        rob = load_json(rob_f)
        section["robustness"] = {
            "source": rob_f.name,
            "baseline_clean_auc": rob.get("baseline_auc") or rob.get("baseline"),
            "results": rob.get("robustness") or rob.get("results") or rob,
        }
        print(f"[OK] Robustness from {rob_f.name}")
    else:
        print("[SKIP] No 3-stream robustness results found")

    # ── Update best_model pointer ─────────────────────────────────────────────
    fs_auc = (section.get("cross_benchmark", {}).get("forensynths", {}) or {}).get("overall_auc")
    cb_auc = (section.get("cross_benchmark", {}).get("cnnspot_small", {}) or {}).get("overall_auc")
    final["best_model"].update({
        "forensynths_auc": fs_auc,
        "cnnspot_small_auc": cb_auc,
    })

    # ── Save ──────────────────────────────────────────────────────────────────
    final["metadata"]["last_updated"] = datetime.now().isoformat()
    with open(FINAL_RESULTS, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n[SAVED] {FINAL_RESULTS}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"{'Model':<35} {'Test AUC':>9} {'ForenSynths':>12} {'CNNSpot':>8} {'Thr':>6}")
    print("="*70)
    print(f"{'streams_23_balanced':<35} {'0.885':>9} {'0.539':>12} {'0.566':>8} {'0.480':>6}")
    thr = 0.4587
    print(f"{'streams_123_trained_prnu':<35} {'0.9685':>9} {str(fs_auc or 'TBD'):>12} {str(cb_auc or 'TBD'):>8} {thr:>6.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
