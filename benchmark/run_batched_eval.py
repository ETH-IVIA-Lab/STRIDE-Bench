"""Watch benchmark_small_200.json and trigger all-model eval at each 50-scene milestone."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BENCHMARK = EVAL_DIR / "benchmark" / "benchmark_small_200.json"
SNAPSHOT_DIR = EVAL_DIR / "benchmark"
RESULTS_DIR = EVAL_DIR / "results"
THRESHOLDS = [50, 100, 150, 200]
POLL_SECONDS = 30


def n_scenes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with open(path) as f:
            d = json.load(f)
        return len(d.get("scenes", []))
    except Exception:
        return 0


def snapshot(path: Path, n: int) -> Path:
    with open(path) as f:
        d = json.load(f)
    d_sub = {**d, "n_scenes": n, "scenes": d["scenes"][:n]}
    out = SNAPSHOT_DIR / f"benchmark_small_{n}.json"
    with open(out, "w") as f:
        json.dump(d_sub, f, indent=2)
    return out


def run_eval(benchmark_path: Path, results_root: Path) -> int:
    cmd = [
        sys.executable,
        str(EVAL_DIR / "eval_all_models_on_benchmark.py"),
        "--benchmark-path", str(benchmark_path),
        "--results-root", str(results_root),
        "--overwrite",
    ]
    print(f"[batched-eval] running: {' '.join(cmd)}", flush=True)
    log_path = results_root / "run.log"
    results_root.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
    return proc.returncode


def main() -> None:
    done: set[int] = set()
    while len(done) < len(THRESHOLDS):
        cur = n_scenes(BENCHMARK)
        for i, t in enumerate(THRESHOLDS, start=1):
            if t in done:
                continue
            if cur >= t:
                print(f"[batched-eval] threshold {t} reached (n_scenes={cur}); snapshotting + evaluating", flush=True)
                snap = snapshot(BENCHMARK, t)
                out_root = RESULTS_DIR / f"all_small_batch_{i}"
                rc = run_eval(snap, out_root)
                print(f"[batched-eval] batch_{i} done rc={rc} -> {out_root}", flush=True)
                done.add(t)
        if len(done) == len(THRESHOLDS):
            break
        time.sleep(POLL_SECONDS)
    print("[batched-eval] all batches complete.", flush=True)


if __name__ == "__main__":
    main()
