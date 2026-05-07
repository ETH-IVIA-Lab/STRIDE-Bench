"""Evaluate multiple trajectory models against eval/benchmark/benchmark.json.

Per model this script:
  1. reads each scene trajectory file
  2. loads the shared benchmark.json
  3. computes local expected-result benchmark scores
  4. runs the  stride analysis and saves plots/CSVs in a dedicated result folder

Outputs live under:
  eval/results/all_in_one/
    logs/
    <model_name>_eval_result/
      scene_eval/<scene_id>/benchmark_eval_local.json
       stride_scores.csv
       stride_by_category.csv
       stride_summary.json
      *.png
    overall_results.csv
    overall_results.json
    overall_results.txt
    model_mean_comparison.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_RUNTIME_CACHE_DIR = Path(__file__).resolve().parent / "results" / ".runtime_cache"
_RUNTIME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
(_RUNTIME_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(_RUNTIME_CACHE_DIR / "xdg").mkdir(parents=True, exist_ok=True)
(_RUNTIME_CACHE_DIR / "xdg" / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_RUNTIME_CACHE_DIR / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
RESULTS_MODULE_DIR = EVAL_DIR / "results"
DEFAULT_BENCHMARK_PATH = EVAL_DIR / "benchmark" / "benchmark.json"
DEFAULT_RESULTS_ROOT = EVAL_DIR / "results" / "all_in_one"

sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(RESULTS_MODULE_DIR))

import benchmark_eval  # noqa: E402
from analyze_ stride import run_analysis  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset_root: Path


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"[{_timestamp()}] {message}\n")


def _sanitize_name(raw: str) -> str:
    sanitized = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in raw.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "model"


def _default_model_name(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT)
    parts = rel.parts
    if parts == ("dataset",):
        return "sfm_dataset"
    if parts == ("baselines", "random", "dataset"):
        return "random"
    if parts == ("baselines", "random", "dataset_random_walk_subset"):
        return "random_subset"
    if parts and parts[0] == "baselines":
        return _sanitize_name("_".join(parts[1:]))
    return _sanitize_name("_".join(parts))


def _dedupe_model_names(specs: list[ModelSpec]) -> list[ModelSpec]:
    seen: dict[str, int] = {}
    output: list[ModelSpec] = []
    for spec in specs:
        count = seen.get(spec.name, 0)
        seen[spec.name] = count + 1
        name = spec.name if count == 0 else f"{spec.name}_{count + 1}"
        output.append(ModelSpec(name=name, dataset_root=spec.dataset_root))
    return output


def discover_model_specs(sim_filename: str = "sim.npz") -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    seen_paths: set[Path] = set()

    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    if list(glob.glob(str(dataset_root / "????" / sim_filename))):
        specs.append(ModelSpec(name="sfm_dataset", dataset_root=dataset_root))
        seen_paths.add(dataset_root)

    baselines_root = PROJECT_ROOT / "baselines"
    if baselines_root.exists():
        for candidate in sorted(baselines_root.rglob("*")):
            if not candidate.is_dir():
                continue
            candidate_resolved = candidate.resolve()
            if candidate_resolved in seen_paths:
                continue
            if not list(glob.glob(str(candidate / "????" / sim_filename))):
                continue
            specs.append(
                ModelSpec(
                    name=_default_model_name(candidate_resolved),
                    dataset_root=candidate_resolved,
                )
            )
            seen_paths.add(candidate_resolved)

    return _dedupe_model_names(specs)


def parse_model_specs(model_roots: list[str] | None, sim_filename: str = "sim.npz") -> list[ModelSpec]:
    if not model_roots:
        return discover_model_specs(sim_filename=sim_filename)

    specs: list[ModelSpec] = []
    for spec in model_roots:
        if "=" not in spec:
            raise ValueError(f"Invalid --model-root '{spec}'. Use name=path.")
        name, raw_path = spec.split("=", 1)
        dataset_root = Path(raw_path).expanduser().resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Model dataset root not found: {dataset_root}")
        specs.append(ModelSpec(name=_sanitize_name(name), dataset_root=dataset_root))
    return _dedupe_model_names(specs)


def filter_model_specs(specs: list[ModelSpec], selected: str | None) -> list[ModelSpec]:
    if not selected:
        return specs
    wanted = {_sanitize_name(part) for part in selected.split(",") if part.strip()}
    filtered = [spec for spec in specs if spec.name in wanted]
    missing = sorted(wanted - {spec.name for spec in filtered})
    if missing:
        raise ValueError(f"Unknown model name(s): {', '.join(missing)}")
    return filtered


def load_benchmark_scene_ids(benchmark: dict) -> set[str]:
    scene_ids: set[str] = set()
    for entry in benchmark.get("scenes", []):
        if entry.get("source_scene_id") is not None:
            scene_ids.add(str(entry["source_scene_id"]).zfill(4))
        elif entry.get("scene_index") is not None:
            scene_ids.add(f"{int(entry['scene_index']):04d}")
    return scene_ids


def aggregate_scene_eval_dir(scene_eval_root: Path) -> dict:
    files = sorted(scene_eval_root.glob("????/benchmark_eval_local.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark_eval_local.json found under {scene_eval_root}")

    per_scene = []
    total = 0.0
    total_questions = 0

    for path in files:
        with open(path) as f:
            report = json.load(f)
        scored = [
            q["question_score_by_expected_result"]
            for q in report.get("questions", [])
            if q.get("question_score_by_expected_result") is not None
        ]
        total += sum(scored)
        total_questions += len(scored)
        per_scene.append(
            {
                "scene_id": path.parent.name,
                "category": report.get("category", "unknown"),
                "location": report.get("location", "unknown"),
                " stride_score_by_expected_result": report.get(" stride_score_by_expected_result"),
                "n_questions": len(scored),
            }
        )

    scene_scores = [
        row[" stride_score_by_expected_result"]
        for row in per_scene
        if row[" stride_score_by_expected_result"] is not None
    ]
    return {
        "metric": "benchmark local expected-result score",
        "scene_eval_root": str(scene_eval_root),
        "overall_accuracy_by_expected_result": round(total / total_questions, 4) if total_questions else 0.0,
        "n_scenes": len(per_scene),
        "total_questions": total_questions,
        "mean_scene_score": round(float(np.mean(scene_scores)), 4) if scene_scores else None,
        "std_scene_score": round(float(np.std(scene_scores, ddof=1)), 4) if len(scene_scores) > 1 else 0.0,
        "median_scene_score": round(float(np.median(scene_scores)), 4) if scene_scores else None,
        "min_scene_score": round(float(np.min(scene_scores)), 4) if scene_scores else None,
        "max_scene_score": round(float(np.max(scene_scores)), 4) if scene_scores else None,
        "per_scene": per_scene,
    }


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def evaluate_model(
    spec: ModelSpec,
    benchmark: dict,
    benchmark_scene_ids: set[str],
    results_root: Path,
    overwrite: bool,
    no_plots: bool,
    limit_scenes: int | None,
    sim_filename: str = "sim.npz",
) -> dict:
    model_result_root = results_root / f"{spec.name}_eval_result"
    scene_eval_root = model_result_root / "scene_eval"
    log_path = results_root / "logs" / f"{spec.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")
    _write_log(log_path, f"START model={spec.name} dataset_root={spec.dataset_root}")

    available_scene_ids = benchmark_eval.list_scenes(dataset_root=str(spec.dataset_root), sim_filename=sim_filename)
    if limit_scenes is not None:
        available_scene_ids = available_scene_ids[:limit_scenes]

    counters = {
        "available_scenes": len(available_scene_ids),
        "evaluated_now": 0,
        "skipped_existing": 0,
        "missing_benchmark": 0,
        "errors": 0,
    }

    progress = None
    iterator = available_scene_ids
    if tqdm is not None:
        progress = tqdm(available_scene_ids, total=len(available_scene_ids), desc=spec.name, unit="scene")
        iterator = progress

    for index, scene_id in enumerate(iterator, start=1):
        if progress is None and (index == 1 or index % 25 == 0 or index == len(available_scene_ids)):
            print(f"[{spec.name}] {index}/{len(available_scene_ids)}")

        out_path = scene_eval_root / scene_id / "benchmark_eval_local.json"
        if out_path.exists() and not overwrite:
            counters["skipped_existing"] += 1
            if progress is not None:
                progress.set_postfix(skip=counters["skipped_existing"], err=counters["errors"], refresh=False)
            continue

        try:
            if scene_id not in benchmark_scene_ids:
                raise KeyError(f"No benchmark entry found for scene {scene_id}")
            report = benchmark_eval.evaluate_scene(
                scene_id,
                benchmark,
                dataset_root=str(spec.dataset_root),
                sim_filename=sim_filename,
            )
            benchmark_eval.save_scene_result(
                scene_id,
                report,
                output_root=str(scene_eval_root),
                output_suffix="",
            )
            counters["evaluated_now"] += 1
        except KeyError as exc:
            counters["missing_benchmark"] += 1
            _write_log(log_path, f"{scene_id} SKIP missing benchmark: {exc}")
        except Exception as exc:  # pragma: no cover
            counters["errors"] += 1
            _write_log(log_path, f"{scene_id} ERROR {type(exc).__name__}: {exc}")
            _write_log(log_path, traceback.format_exc().rstrip())

        if progress is not None:
            progress.set_postfix(
                skip=counters["skipped_existing"],
                miss=counters["missing_benchmark"],
                err=counters["errors"],
                refresh=False,
            )

    if progress is not None:
        progress.close()

    scene_eval_files = sorted(scene_eval_root.glob("????/benchmark_eval_local.json"))
    if not scene_eval_files:
        _write_log(log_path, "END no scene eval files produced")
        return {
            "model": spec.name,
            "dataset_root": str(spec.dataset_root),
            "result_root": str(model_result_root),
            "scene_eval_root": str(scene_eval_root),
            "log_path": str(log_path),
            **counters,
            "scene_count": 0,
            "overall_question_accuracy": None,
            "mean_scene_score": None,
            "std_scene_score": None,
            "median_scene_score": None,
            "q25_scene_score": None,
            "q75_scene_score": None,
            "min_scene_score": None,
            "max_scene_score": None,
            "total_questions": 0,
            "analysis_summary_json": None,
        }

    aggregate_summary = aggregate_scene_eval_dir(scene_eval_root)
    save_json(model_result_root / "benchmark_eval_local_summary.json", aggregate_summary)

    analysis = run_analysis(
        dataset_dir=scene_eval_root,
        eval_file="benchmark_eval_local.json",
        output_dir=model_result_root,
        prefix="",
        no_plots=no_plots,
        verbose=False,
    )
    score_stats = analysis["overall"]["expected_result"]
    std_score = score_stats["std"]

    result = {
        "model": spec.name,
        "dataset_root": str(spec.dataset_root),
        "result_root": str(model_result_root),
        "scene_eval_root": str(scene_eval_root),
        "log_path": str(log_path),
        **counters,
        "scene_count": int(score_stats["count"]),
        "overall_question_accuracy": aggregate_summary["overall_accuracy_by_expected_result"],
        "mean_scene_score": round(float(score_stats["mean"]), 4),
        "std_scene_score": round(float(std_score) if pd.notna(std_score) else 0.0, 4),
        "median_scene_score": round(float(score_stats["median"]), 4),
        "q25_scene_score": round(float(score_stats["q25"]), 4),
        "q75_scene_score": round(float(score_stats["q75"]), 4),
        "min_scene_score": round(float(score_stats["min"]), 4),
        "max_scene_score": round(float(score_stats["max"]), 4),
        "total_questions": int(aggregate_summary["total_questions"]),
        "analysis_summary_json": analysis["summary_json_path"],
    }
    _write_log(
        log_path,
        "END "
        f"scene_count={result['scene_count']} "
        f"mean_scene_score={result['mean_scene_score']} "
        f"overall_question_accuracy={result['overall_question_accuracy']}",
    )
    return result


def save_overall_outputs(results_root: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    csv_path = results_root / "overall_results.csv"
    json_path = results_root / "overall_results.json"
    txt_path = results_root / "overall_results.txt"

    df.to_csv(csv_path, index=False)
    save_json(json_path, rows)
    txt_path.write_text(df.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n")


def plot_model_comparison(results_root: Path, rows: list[dict]) -> Path | None:
    df = pd.DataFrame(rows)
    if df.empty or df["mean_scene_score"].dropna().empty:
        return None

    plot_df = df.dropna(subset=["mean_scene_score"]).sort_values("mean_scene_score", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, len(plot_df) * 1.2), 4.5))
    x = np.arange(len(plot_df))
    ax.bar(
        x,
        plot_df["mean_scene_score"],
        yerr=plot_df["std_scene_score"].fillna(0.0),
        capsize=5,
        color="#4C72B0",
        alpha=0.85,
        error_kw=dict(elinewidth=1.2, ecolor="gray"),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=20, ha="right")
    ax.set_ylabel("Mean scene  stride score ± std")
    ax.set_ylim(0, 1.05)
    ax.set_title("Benchmark Result Comparison Across Models")
    for idx, row in plot_df.reset_index(drop=True).iterrows():
        ax.text(idx, row["mean_scene_score"] + 0.015, f"n={int(row['scene_count'])}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    plot_path = results_root / "model_mean_comparison.png"
    fig.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return plot_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate all models against eval/benchmark/benchmark.json")
    parser.add_argument("--benchmark-path", default=str(DEFAULT_BENCHMARK_PATH), help="Path to benchmark.json")
    parser.add_argument(
        "--results-root",
        default=str(DEFAULT_RESULTS_ROOT),
        help="Directory to store per-model outputs and combined summaries",
    )
    parser.add_argument(
        "--model-root",
        action="append",
        default=None,
        help="Explicit model root in the form name=/abs/or/relative/path. Repeatable.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated subset of discovered model names to run, e.g. sfm_dataset,random",
    )
    parser.add_argument("--overwrite", action="store_true", help="Recompute scene eval files even if they already exist")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-model plot generation")
    parser.add_argument("--limit-scenes", type=int, default=None, help="Debug helper: evaluate only the first N scenes per model")
    parser.add_argument("--list-models", action="store_true", help="List discovered model names and dataset roots, then exit")
    parser.add_argument(
        "--sim-filename",
        default="sim.npz",
        help="Trajectory filename to load per scene (default: sim.npz). Use subsample_sim.npz for subsampled data.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_path).expanduser().resolve()
    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_path}")

    model_specs = filter_model_specs(parse_model_specs(args.model_root, sim_filename=args.sim_filename), args.models)
    if args.list_models:
        for spec in model_specs:
            print(f"{spec.name}\t{spec.dataset_root}")
        return

    results_root = Path(args.results_root).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "logs").mkdir(parents=True, exist_ok=True)

    with open(benchmark_path) as f:
        benchmark = json.load(f)
    benchmark_scene_ids = load_benchmark_scene_ids(benchmark)

    rows = []
    for spec in model_specs:
        print(f"\n[{spec.name}] dataset_root={spec.dataset_root}")
        row = evaluate_model(
            spec=spec,
            benchmark=benchmark,
            benchmark_scene_ids=benchmark_scene_ids,
            results_root=results_root,
            overwrite=args.overwrite,
            no_plots=args.no_plots,
            limit_scenes=args.limit_scenes,
            sim_filename=args.sim_filename,
        )
        rows.append(row)
        print(
            f"  scenes={row['scene_count']} "
            f"mean={row['mean_scene_score']} "
            f"std={row['std_scene_score']} "
            f"overall_q={row['overall_question_accuracy']} "
            f"errors={row['errors']} "
            f"missing_benchmark={row['missing_benchmark']}"
        )

    save_overall_outputs(results_root, rows)
    comparison_plot = plot_model_comparison(results_root, rows)

    summary_df = pd.DataFrame(rows)
    print("\nOverall result")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if comparison_plot is not None:
        print(f"\nComparison plot: {comparison_plot}")
    print(f"Combined outputs: {results_root}")


if __name__ == "__main__":
    main()
