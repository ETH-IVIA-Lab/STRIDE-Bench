import argparse
import glob
import json
import os
import re
from typing import Any

import numpy as np

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
BENCHMARK_PATH = os.path.join(EVAL_DIR, "benchmark", "benchmark.json")
DEFAULT_FOLDER_PATTERN = "????"

import sys

sys.path.insert(0, EVAL_DIR)
# from traj_eval_func import FUNCTION_REGISTRY
from traj_eval_func_norm import FUNCTION_REGISTRY as FUNCTION_REGISTRY_NORM


def _serialize(val):
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_serialize(x) for x in val]
    if isinstance(val, (np.floating, float)):
        return round(float(val), 4)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def _normalize_function_name(name: str) -> str:
    if not isinstance(name, str):
        return name
    return name.split(".")[-1]


def _normalize_identifier(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve_function_name(func_name: str, registry: dict | None = None) -> str | None:
    registry = registry or FUNCTION_REGISTRY
    raw = _normalize_function_name(func_name)
    if raw in registry:
        return raw
    normalized = _normalize_identifier(raw)
    for candidate in registry:
        if _normalize_identifier(candidate) == normalized:
            return candidate
    return None


def _sanitize_params(func, params: dict) -> tuple[dict, dict]:
    sig = __import__("inspect").signature(func)
    param_names = [p for p in sig.parameters if p != "trajectories"]
    normalized_map = {_normalize_identifier(name): name for name in param_names}
    sanitized = {}
    dropped = {}
    for key, value in (params or {}).items():
        normalized_key = _normalize_identifier(key)
        actual_key = normalized_map.get(normalized_key)
        if actual_key is None:
            dropped[key] = value
            continue
        sanitized[actual_key] = value
    return sanitized, dropped


def _is_number(val) -> bool:
    return isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool)


def _resolve_expected_key(expected_key: str, actual: dict) -> str | None:
    aliases = {"count": "n_clusters"}
    if expected_key in actual:
        return expected_key
    aliased = aliases.get(expected_key)
    if aliased in actual:
        return aliased
    return None


def _augment_actual_dict(actual: dict) -> dict:
    augmented = dict(actual)

    if "per_goal" in augmented and isinstance(augmented["per_goal"], list):
        per_goal = augmented["per_goal"]
        augmented.setdefault("per_goal_nonzero_count", sum(float(v) > 0 for v in per_goal))
        if per_goal:
            augmented.setdefault("per_goal_max_fraction", float(max(per_goal)))
            augmented.setdefault("per_goal_index_max", int(np.argmax(per_goal)))
            top_two = sorted((float(v) for v in per_goal), reverse=True)[:2]
            dominance_gap = top_two[0] - (top_two[1] if len(top_two) > 1 else 0.0)
            augmented.setdefault("single_goal_dominance_gap", dominance_gap)

    if "per_point" in augmented and isinstance(augmented["per_point"], list):
        per_point = augmented["per_point"]
        if per_point:
            augmented.setdefault("per_point_max_fraction", float(max(per_point)))
            augmented.setdefault("per_point_index_max", int(np.argmax(per_point)))

    if "n_clusters" in augmented and "sizes" in augmented and isinstance(augmented["sizes"], list):
        sizes = [int(s) for s in augmented["sizes"]]
        n_points = int(augmented.get("n_points", sum(sizes)))
        clustered = int(sum(sizes))
        largest = int(max(sizes)) if sizes else 0
        augmented.setdefault("largest_cluster_size", largest)
        augmented.setdefault("largest_cluster_fraction", float(largest / n_points) if n_points else 0.0)
        augmented.setdefault("clustered_fraction", float(clustered / n_points) if n_points else 0.0)
        augmented.setdefault("noise_fraction", float(max(n_points - clustered, 0) / n_points) if n_points else 0.0)
        augmented.setdefault("sizes_min", int(min(sizes)) if sizes else 0)
        augmented.setdefault("sizes_max", int(max(sizes)) if sizes else 0)

    return augmented


def _merge_range_constraint(target: dict, key: str, bound: str, value) -> None:
    current = target.get(key)
    if current is None:
        target[key] = {bound: value}
        return
    if isinstance(current, dict):
        current[bound] = value
        return
    target[key] = {"approx": current, bound: value}


def _normalize_expected_dict(actual: dict, expected: dict) -> dict:
    range_keys = {"min", "max", "value", "approx", "abs_tol", "rel_tol", "one_of", "delta_value"}

    if set(expected).issubset(range_keys) and set(expected):
        if "overall_fraction" in actual and _is_number(actual["overall_fraction"]):
            return {"overall_fraction": expected}
        return expected

    if "n_clusters" not in actual or "sizes" not in actual:
        return expected

    normalized = {}
    for key, value in expected.items():
        if key in {"min_clusters", "min_n_clusters"}:
            _merge_range_constraint(normalized, "n_clusters", "min", value)
        elif key in {"max_clusters", "max_n_clusters"}:
            _merge_range_constraint(normalized, "n_clusters", "max", value)
        elif key in {"largest_cluster_fraction_min", "min_largest_cluster_fraction"}:
            _merge_range_constraint(normalized, "largest_cluster_fraction", "min", value)
        elif key == "max_largest_cluster_fraction":
            _merge_range_constraint(normalized, "largest_cluster_fraction", "max", value)
        elif key == "min_largest_cluster_size":
            _merge_range_constraint(normalized, "largest_cluster_size", "min", value)
        elif key == "max_largest_cluster_size":
            _merge_range_constraint(normalized, "largest_cluster_size", "max", value)
        elif key == "min_clustered_fraction":
            _merge_range_constraint(normalized, "clustered_fraction", "min", value)
        elif key == "max_clustered_fraction":
            _merge_range_constraint(normalized, "clustered_fraction", "max", value)
        elif key in {"max_noise_fraction", "noise_fraction_max", "max_noise_ratio"}:
            _merge_range_constraint(normalized, "noise_fraction", "max", value)
        elif key in {"min_noise_fraction"}:
            _merge_range_constraint(normalized, "noise_fraction", "min", value)
        else:
            normalized[key] = value
    return normalized


def _matches_expected(actual, expected) -> bool | None:
    if expected is None:
        return None

    if isinstance(expected, dict):
        if isinstance(actual, dict):
            actual = _augment_actual_dict(actual)
            expected = _normalize_expected_dict(actual, expected)

        range_keys = {"min", "max", "value", "approx", "abs_tol", "rel_tol", "one_of", "delta_value"}
        if set(expected).issubset(range_keys) and set(expected):
            # Effective tolerance is at least ABS_SLACK_FLOOR so near-misses
            # within numerical noise still pass.
            ABS_SLACK_FLOOR = 1e-2
            delta = max(float(expected.get("delta_value", 0)), ABS_SLACK_FLOOR)
            if "one_of" in expected:
                return any(_matches_expected(actual, item) for item in expected["one_of"])
            if "value" in expected:
                expected = {
                    "approx": expected["value"],
                    "abs_tol": expected.get("abs_tol", 1e-3),
                    "rel_tol": expected.get("rel_tol", 0.0),
                }
            if "approx" in expected:
                if not _is_number(actual):
                    return False
                abs_tol = expected.get("abs_tol", 1e-3) + delta
                rel_tol = expected.get("rel_tol", 0.0)
                target = expected["approx"]
                return abs(float(actual) - float(target)) <= max(abs_tol, rel_tol * abs(float(target)))
            if not _is_number(actual):
                return False
            if "min" in expected and float(actual) < float(expected["min"]) - delta:
                return False
            if "max" in expected and float(actual) > float(expected["max"]) + delta:
                return False
            return True

        if not isinstance(actual, dict):
            return False
        for key, exp_val in expected.items():
            actual_key = _resolve_expected_key(key, actual)
            if actual_key is None:
                return False
            matched = _matches_expected(actual[actual_key], exp_val)
            if matched is not True:
                return matched
        return True

    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            return False
        for actual_item, expected_item in zip(actual, expected):
            matched = _matches_expected(actual_item, expected_item)
            if matched is not True:
                return matched
        return True

    if _is_number(expected):
        if not _is_number(actual):
            return False
        tol = 1e-6 if isinstance(expected, int) and isinstance(actual, int) else 1e-3
        return abs(float(actual) - float(expected)) <= tol

    if isinstance(expected, bool):
        return actual is expected

    return actual == expected


def resolve_trajectory_path(scene_id: str, traj_path: str | None = None, dataset_root: str = DATASET_DIR, sim_filename: str = "sim.npz") -> str:
    path = traj_path or os.path.join(dataset_root, scene_id, sim_filename)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    return path


def load_trajectories(scene_id: str, traj_path: str | None = None, dataset_root: str = DATASET_DIR, sim_filename: str = "sim.npz") -> tuple[np.ndarray, str]:
    path = resolve_trajectory_path(scene_id, traj_path, dataset_root=dataset_root, sim_filename=sim_filename)
    data = np.load(path, allow_pickle=True)
    return data["states"], path


def _extract_benchmark_entry(scene_file: str) -> dict:
    with open(scene_file) as f:
        report = json.load(f)
    source_scene_id = os.path.basename(os.path.dirname(scene_file))
    scene_json_path = os.path.join(os.path.dirname(scene_file), "scene.json")
    benchmark_n_agents = report.get("n_agents")
    if os.path.exists(scene_json_path):
        with open(scene_json_path) as f:
            scene_data = json.load(f)
        if scene_data.get("ungrouped_agents") is not None:
            benchmark_n_agents = int(scene_data["ungrouped_agents"])
        elif benchmark_n_agents is None and scene_data.get("crowd_size") is not None:
            benchmark_n_agents = int(scene_data["crowd_size"])

    questions = []
    for question in report.get("questions", []):
        functions = []
        for func in question.get("functions", []):
            functions.append({
                "name": _normalize_function_name(func.get("name")),
                "params": func.get("params", {}),
                "expected_result": func.get("expected_result"),
                "expected_result_reasoning": func.get("expected_result_reasoning", ""),
            })
        questions.append({
            "id": question.get("id"),
            "question": question.get("question"),
            "functions": functions,
        })

    return {
        "source_scene_id": source_scene_id,
        "scenario_id": report.get("scenario_id"),
        "scene_index": report.get("scene_index"),
        "description": report.get("description"),
        "category": report.get("category"),
        "location": report.get("location"),
        "n_timesteps": report.get("n_timesteps"),
        "n_agents": benchmark_n_agents,
        "decomposition_reasoning": report.get("decomposition_reasoning", ""),
        "questions": questions,
    }


def build_benchmark_dataset(
    output_path: str = BENCHMARK_PATH,
    dataset_root: str = DATASET_DIR,
    folder_pattern: str = DEFAULT_FOLDER_PATTERN,
) -> dict:
    files = sorted(glob.glob(os.path.join(dataset_root, folder_pattern, "benchmark_eval.json")))
    scenes = [_extract_benchmark_entry(path) for path in files]
    benchmark = {"n_scenes": len(scenes), "scenes": scenes}
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    return benchmark


def load_benchmark(benchmark_path: str = BENCHMARK_PATH) -> dict:
    with open(benchmark_path) as f:
        return json.load(f)


def _is_norm_benchmark(benchmark: dict) -> bool:
    """Detect if the benchmark uses the normalized format (measurements[] with function key)."""
    for entry in benchmark.get("scenes", []):
        for q in entry.get("questions", []):
            if "measurements" in q:
                return True
            if "functions" in q:
                return False
    return False


def _normalize_benchmark_entry(entry: dict, is_norm: bool) -> dict:
    """Convert a norm-format benchmark entry to the standard format expected by evaluate_scene."""
    if not is_norm:
        return entry
    questions = []
    for q in entry.get("questions", []):
        functions = []
        for m in q.get("measurements", []):
            expected = m.get("expected_result")
            if isinstance(expected, dict) and "delta_value" in m and "delta_value" not in expected:
                expected = {**expected, "delta_value": m["delta_value"]}
            functions.append({
                "name": m.get("function", ""),
                "params": m.get("params", {}),
                "expected_result": expected,
                "expected_result_reasoning": m.get("expected_result_reasoning", ""),
            })
        questions.append({
            "id": q.get("id"),
            "question": q.get("question"),
            "functions": functions,
        })
    normalized = dict(entry)
    normalized["questions"] = questions
    return normalized


def _scene_id_to_index(scene_id: str) -> int:
    try:
        return int(scene_id)
    except (TypeError, ValueError):
        return -1


def _find_benchmark_entry(benchmark: dict, scene_id: str) -> dict:
    for entry in benchmark.get("scenes", []):
        if entry.get("source_scene_id") == scene_id:
            return entry

    # Backward-compatible fallback for older benchmark.json files that did not
    # store the source folder id. This is ambiguous when scene_index is reused.
    scene_index = _scene_id_to_index(scene_id)
    for entry in benchmark.get("scenes", []):
        if entry.get("scene_index") == scene_index:
            return entry
    raise KeyError(f"No benchmark entry found for scene {scene_id}")


def _run_function(func_spec: dict, trajectories: np.ndarray, registry: dict | None = None) -> tuple[Any, str | None, dict]:
    registry = registry or FUNCTION_REGISTRY
    resolved_name = _resolve_function_name(func_spec["name"], registry=registry)
    if resolved_name is None:
        return None, f"Unknown function: {func_spec['name']}", {}
    func = registry[resolved_name]
    params, dropped = _sanitize_params(func, func_spec.get("params", {}))
    params = {k: v for k, v in params.items() if v is not None}
    try:
        result = func(trajectories, **params)
        return _serialize(result), None, dropped
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}", dropped


def evaluate_scene(
    scene_id: str,
    benchmark: dict,
    traj_path: str | None = None,
    dataset_root: str = DATASET_DIR,
    sim_filename: str = "sim.npz",
) -> dict:
    is_norm = _is_norm_benchmark(benchmark)
    registry = FUNCTION_REGISTRY_NORM if is_norm else FUNCTION_REGISTRY
    entry = _find_benchmark_entry(benchmark, scene_id)
    entry = _normalize_benchmark_entry(entry, is_norm)
    trajectories, trajectory_path = load_trajectories(scene_id, traj_path, dataset_root=dataset_root, sim_filename=sim_filename)

    questions = []
    for question in entry.get("questions", []):
        functions = []
        local_scores = []
        for func_spec in question.get("functions", []):
            result, error, dropped = _run_function(func_spec, trajectories, registry=registry)
            local_match = None if error else _matches_expected(result, func_spec.get("expected_result"))
            functions.append({
                "name": _normalize_function_name(func_spec.get("name")),
                "params": func_spec.get("params", {}),
                "expected_result": func_spec.get("expected_result"),
                "expected_result_reasoning": func_spec.get("expected_result_reasoning", ""),
                "result": result,
                "error": error,
                "dropped_params": dropped if dropped else None,
                "consistent_by_expected_result": (
                    int(local_match) if local_match is not None else None
                ),
            })
            if local_match is not None:
                local_scores.append(int(local_match))

        questions.append({
            "id": question.get("id"),
            "question": question.get("question"),
            "functions": functions,
            "question_score_by_expected_result": (
                round(sum(local_scores) / len(local_scores), 4) if local_scores else None
            ),
        })

    scored = [
        q["question_score_by_expected_result"]
        for q in questions
        if q["question_score_by_expected_result"] is not None
    ]
    return {
        "scenario_id": entry.get("scenario_id"),
        "scene_index": entry.get("scene_index"),
        "trajectory_path": trajectory_path,
        "description": entry.get("description"),
        "category": entry.get("category"),
        "location": entry.get("location"),
        "n_timesteps": int(trajectories.shape[0]),
        "n_agents": (
            int(entry["n_agents"])
            if entry.get("n_agents") is not None
            else int(trajectories.shape[1])
        ),
        "decomposition_reasoning": entry.get("decomposition_reasoning", ""),
        "questions": questions,
        " stride_score_by_expected_result": (
            round(sum(scored) / len(scored), 4) if scored else None
        ),
    }


def save_scene_result(
    scene_id: str,
    report: dict,
    output_root: str = DATASET_DIR,
    output_suffix: str = "",
) -> str:
    out_dir = os.path.join(output_root, scene_id)
    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_{output_suffix}" if output_suffix else ""
    path = os.path.join(out_dir, f"benchmark_eval_local{suffix}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def aggregate_local_results(
    output_root: str = DATASET_DIR,
    output_suffix: str = "",
    folder_pattern: str = DEFAULT_FOLDER_PATTERN,
) -> dict:
    suffix = f"_{output_suffix}" if output_suffix else ""
    files = sorted(glob.glob(os.path.join(output_root, folder_pattern, f"benchmark_eval_local{suffix}.json")))
    if not files:
        return {"error": f"No benchmark_eval_local{suffix}.json found in {output_root}/*/"}

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
        per_scene.append({
            "scene_id": os.path.basename(os.path.dirname(path)),
            "category": report.get("category", "unknown"),
            " stride_score_by_expected_result": report.get(" stride_score_by_expected_result"),
            "n_questions": len(scored),
        })

    summary = {
        "metric": "benchmark local expected-result score",
        "output_suffix": output_suffix,
        "output_root": output_root,
        "overall_accuracy_by_expected_result": round(total / total_questions, 4) if total_questions else 0.0,
        "n_scenes": len(per_scene),
        "total_questions": total_questions,
        "per_scene": per_scene,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_suffix = f"_{output_suffix}" if output_suffix else ""
    summary_path = os.path.join(RESULTS_DIR, f"benchmark_eval_local_summary{summary_suffix}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def list_scenes(
    dataset_root: str = DATASET_DIR,
    sim_filename: str = "sim.npz",
    folder_pattern: str = DEFAULT_FOLDER_PATTERN,
) -> list[str]:
    sim_files = sorted(glob.glob(os.path.join(dataset_root, folder_pattern, sim_filename)))
    return [os.path.basename(os.path.dirname(p)) for p in sim_files]


def main():
    parser = argparse.ArgumentParser(description="Build benchmark.json and run local benchmark evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="Build eval/benchmark/benchmark.json from dataset/*/benchmark_eval.json")
    group.add_argument("--scene", type=str, help="Evaluate a single scene locally from benchmark.json")
    group.add_argument("--all", action="store_true", help="Evaluate all scenes locally from benchmark.json")
    group.add_argument("--aggregate", action="store_true", help="Aggregate existing benchmark_eval_local.json files")
    group.add_argument("--list", action="store_true", help="List available scene ids")

    parser.add_argument("--benchmark-path", default=BENCHMARK_PATH, help="Path to benchmark.json")
    parser.add_argument("--traj-path", default=None, help="Trajectory .npz path for a single-scene run")
    parser.add_argument("--dataset-root", default=DATASET_DIR, help="Root directory containing per-scene sim.npz files")
    parser.add_argument("--output-root", default=None, help="Root directory to save local evaluation files (defaults to --dataset-root)")
    parser.add_argument("--output-suffix", default="", help="Suffix for output filenames, e.g. random -> benchmark_eval_local_random.json")
    parser.add_argument("--folder-pattern", default=DEFAULT_FOLDER_PATTERN, help="Glob pattern for scene folder names (e.g. 'lyon-????')")
    args = parser.parse_args()

    if args.traj_path and not args.scene:
        parser.error("--traj-path can only be used with --scene")

    if args.list:
        scenes = list_scenes(dataset_root=args.dataset_root, folder_pattern=args.folder_pattern)
        print(f"Found {len(scenes)} scenes")
        for scene_id in scenes:
            print(f"  {scene_id}")
        return

    if args.build:
        benchmark = build_benchmark_dataset(
            output_path=args.benchmark_path,
            dataset_root=args.dataset_root,
            folder_pattern=args.folder_pattern,
        )
        print(f"Saved benchmark dataset with {benchmark['n_scenes']} scenes to {args.benchmark_path}")
        return

    if args.aggregate:
        output_root = args.output_root or args.dataset_root
        print(json.dumps(
            aggregate_local_results(
                output_root=output_root,
                output_suffix=args.output_suffix,
                folder_pattern=args.folder_pattern,
            ),
            indent=2,
        ))
        return

    benchmark = load_benchmark(args.benchmark_path)
    scenes = [args.scene] if args.scene else list_scenes(
        dataset_root=args.dataset_root,
        folder_pattern=args.folder_pattern,
    )
    output_root = args.output_root or args.dataset_root
    for scene_id in scenes:
        try:
            report = evaluate_scene(
                scene_id,
                benchmark,
                traj_path=args.traj_path if args.scene else None,
                dataset_root=args.dataset_root,
            )
        except KeyError as exc:
            print(f"{scene_id} -> skip ({exc})")
            continue

        path = save_scene_result(
            scene_id,
            report,
            output_root=output_root,
            output_suffix=args.output_suffix,
        )
        score = report.get(" stride_score_by_expected_result")
        score_str = f"  stride_score_by_expected_result={score}" if score is not None else ""
        print(f"{scene_id} -> {path}{score_str}")


if __name__ == "__main__":
    main()
