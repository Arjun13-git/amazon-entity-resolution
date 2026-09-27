"""
End-to-end, restartable production pipeline:

    candidates -> validate -> features -> scores -> decisions -> submission

Every stage writes Parquet parts atomically with per-part sidecars and a
final manifest; re-running the same command resumes where it stopped
(finished parts are skipped; a candidate partition whose parts are all done
is not refitted). ``run_manifest.json`` records code, environment, model
and input hashes, and configuration.

Full test run (see outputs/diagnostics/production_readiness/production_pipeline_design.md):

    python -m src.inference.run_pipeline --dataset test --out outputs/production/test_v1

Smoke run on a small S1 sample:

    python -m src.inference.run_pipeline --dataset test --s1-sample 600 \\
        --out outputs/production/test_smoke --stages candidates validate features
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from src.blocking import candidate_pipeline
from src.blocking.candidate_pipeline import PipelineConfig, atomic_write_json, code_version
from src.data.loader import dataset_dir
from src.features import pairwise
from src.inference import decide, predict_base, write_outputs
from src.inference.predict_base import sha256_file


ROOT = Path(__file__).resolve().parents[2]
STAGES = ["candidates", "validate", "features", "scores", "decisions", "submission"]
DATASET_DIR = dataset_dir()

PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "xgboost", "rapidfuzz", "unidecode"]


def environment() -> dict:

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: md.version(p) for p in PACKAGES},
    }


def input_hashes(dataset: str) -> dict:

    return {
        f.name: sha256_file(f)
        for f in sorted((DATASET_DIR / dataset).glob(f"{dataset}_source*.tsv"))
    }


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="test", choices=["train", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    parser.add_argument("--base-experiment", type=Path, default=ROOT / "outputs/experiments/xgb_skeleton")
    parser.add_argument("--second-stage", type=Path, default=ROOT / "outputs/experiments/s1_context")
    parser.add_argument("--lexicon-dir", type=Path, default=ROOT / "outputs/candidates/sample20k_a",
                        help="Training candidate run whose city lexicons the model was trained with.")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--s1-sample", type=int, default=None, help="Smoke runs only.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-input-hash", action="store_true")
    args = parser.parse_args()

    run_dir = args.out
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        s1_sample=args.s1_sample, seed=args.seed, chunk_size=args.chunk_size,
        workers=args.workers, dataset=args.dataset, lexicon_dir=str(args.lexicon_dir),
    )

    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text()) if run_manifest_path.exists() else {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate_config": asdict(cfg),
        "base_experiment": str(args.base_experiment),
        "second_stage": str(args.second_stage),
        "artifacts_sha256": {
            "base_model.json": sha256_file(args.base_experiment / "model.json"),
            "base_features.json": sha256_file(args.base_experiment / "features.json"),
            "second_stage_model.json": sha256_file(args.second_stage / "second_stage_model.json"),
            "second_stage_metadata.json": sha256_file(args.second_stage / "metadata.json"),
            **{f"lexicon/{p.name}": sha256_file(p) for p in sorted(args.lexicon_dir.glob("lexicon_*.txt"))},
        },
        "inputs_sha256": {} if args.skip_input_hash else input_hashes(args.dataset),
        "stages": {},
    }
    if run_manifest["candidate_config"] != json.loads(json.dumps(asdict(cfg))):
        raise SystemExit("run directory was created with a different configuration")
    run_manifest.setdefault("invocations", []).append(
        {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stages": args.stages,
         "code": code_version(), "environment": environment()}
    )
    atomic_write_json(run_manifest_path, run_manifest)

    for stage in STAGES:
        if stage not in args.stages:
            continue
        t0 = time.perf_counter()
        print(f"\n=== {stage} ===", flush=True)

        if stage == "candidates":
            out = run_dir / "candidates"
            candidate_pipeline.run(cfg, out, resume=out.exists())
        elif stage == "validate":
            result = subprocess.run([sys.executable, "-m", "src.blocking.validate_candidates", str(run_dir / "candidates")])
            if result.returncode != 0:
                raise SystemExit("candidate validation failed")
        elif stage == "features":
            out = run_dir / "features"
            pairwise.run(run_dir / "candidates", out, 0.2, args.seed, resume=out.exists())
        elif stage == "scores":
            predict_base.run(run_dir / "features", args.base_experiment, run_dir / "scores")
        elif stage == "decisions":
            decide.run(run_dir, args.second_stage)
        elif stage == "submission":
            write_outputs.run(run_dir)

        run_manifest["stages"][stage] = {"finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                         "seconds": round(time.perf_counter() - t0, 1)}
        atomic_write_json(run_manifest_path, run_manifest)

    print(f"\nDone: {run_dir}")


if __name__ == "__main__":
    main()
