---
license: mit
task_categories:
  - text-generation
  - question-answering
language:
  - en
tags:
  - crowd-simulation
  - trajectory-evaluation
  - pedestrian-dynamics
  - benchmark
  - behavioral-evaluation
pretty_name: STRIDE-Bench
size_categories:
  - 1K<n<10K
configs:
  - config_name: benchmark
    data_files: benchmark.jsonl
  - config_name: scenes
    data_files: scenes.parquet
  - config_name: maps
    data_files: maps.parquet
---

# STRIDE Benchmark

## Dataset Description

STRIDE Benchmark is an evaluation benchmark for assessing the behavioral realism of crowd trajectory generation and simulation models. Rather than comparing trajectories point-by-point, it evaluates whether generated trajectories exhibit behaviors consistent with a given scenario description — measuring **trajectory-context consistency** through decomposed behavioral questions.

### Dataset Summary

The dataset is distributed as three files, all aligned by scene:

| File | Format | Rows | Purpose |
|------|--------|------|---------|
| `benchmark.jsonl` | JSON Lines | 936 | Behavioral evaluation specs (questions + measurements per scene) |
| `scenes.parquet`  | Parquet    | 936 | Scenario metadata, initial agent states, group structure, goals |
| `maps.parquet`    | Parquet    | 936 | Per-scene obstacle rectangles + rendered obstacle PNG |

| Property | Value |
|----------|-------|
| Scenes | 936 |
| Total Questions | 6,633 |
| Total Measurements | 11,696 |
| Evaluation Functions | 22 |
| Behavioral Categories | 11 |
| Unique Scenarios | 31 |
| Locations | 30 real-world locations |

### Languages

English

## Dataset Structure

The dataset has three configs — `benchmark`, `scenes`, `maps` — each backed by one file. All three are aligned: row `i` in any file corresponds to the same scene, joinable on `scene_id` / `source_scene_id`.

### `benchmark.jsonl` — behavioral evaluation specs

Each line is one scene. Fields:

- **`source_scene_id`** (`string`): Unique scene identifier.
- **`scenario_id`** (`string`): Reference to the physical scenario / environment.
- **`description`** (`string`): Natural-language description of the crowd behavior scenario.
- **`category`** (`string`): Behavioral category label (one of 11).
- **`location`** (`string`): Real-world location name.
- **`decomposition_reasoning`** (`string`): LLM reasoning for how the description maps to measurable behavioral properties.
- **`questions`** (`list`): Behavioral evaluation questions, each containing:
  - `id` (`string`): Question identifier (e.g., "Q1").
  - `question` (`string`): Natural-language behavioral question.
  - `measurements` (`list`): Quantitative measurements to answer the question:
    - `function` (`string`): Name of the evaluation function.
    - `params` (`object`): Function parameters.
    - `expected_result` (`object`): Expected value range with `min` and/or `max` bounds.
    - `expected_result_reasoning` (`string`): Justification for the expected range.
    - `delta_value` (`number`): Tolerance/margin for soft evaluation.
- **`_model`** (`string`): The LLM used for benchmark generation (carried per-row from the original top-level `model` field).
- **`_n_scenes`** (`int`): Total number of scenes in the dataset (carried per-row).

### `scenes.parquet` — scene state and metadata

One row per scene. Selected columns (full schema in the Parquet file):

- **`scene_id`** (`string`): Scene identifier (joins with `benchmark.source_scene_id` and `maps.scene_id`).
- **`scene_index`** (`int`): Numeric index of the scene.
- **`scenario`** (`string`): Free-form scenario description.
- **`category`** (`string`): Behavioral category label.
- **`crowd_size`** (`int`), **`crowd_size_label`** (`string`): Number of agents and bucketed label.
- **`ungrouped_agents`** (`int`): Count of agents not in a group.
- **`event_center_px`** / **`event_center_m`** (`list<float>`): Event center in pixel and meter coordinates.
- **`initial_state`** (`list<list<float>>`): Per-agent starting state.
- **`groups`** (`list<list<int>>`): Agent grouping (each inner list is one group's member indices).
- **`towards_event`**, **`towards_goal`** (`bool`): Whether agents move toward the event / goal.
- **`desired_speed_range`**, **`collision_rate_early`**, **`collision_rate_late`**, **`relavance_to_event_range`**, **`lingering_fraction_range`** (`list<float>`): Expected behavioral ranges used during simulation.
- **`goal_location_raw`**, **`goals_px`**, **`goals_m`**, **`goal_location_px`**, **`goal_location_m`** (`string`): Goal specifications. Stored as JSON-encoded strings because their runtime types vary across scenes (string / list / dict / null).
- **`assets`** (`struct`): Paths/refs to anchored obstacles, homography, and obstacle PNG.

### `maps.parquet` — per-scene obstacle data

One row per scene. Columns:

- **`scene_id`** (`string`): Scene identifier (joins with the other tables).
- **`obstacles_key`** (`string`): Original npz key (always `"obstacles"`).
- **`obstacles`** (`list<list<float>>`): Obstacle rectangles, shape `(N, 4)` — the contents of `map.npz["obstacles"]` from the source layout.
- **`obstacles_shape`** (`list<int>`): Original array shape, for reconstruction.
- **`obstacle_map_png`** (`bytes`): Raw PNG bytes of the rendered obstacle map.

### Behavioral Categories

| Category | Count | Description |
|----------|-------|-------------|
| Escaping | 123 | Emergency/evacuation behavior |
| Violent | 122 | Conflict scenarios |
| Demonstrator | 106 | Organized group/protest movement |
| Dense | 103 | High-density crowd movement |
| Aggressive | 89 | Aggressive crowd behavior |
| Rushing | 85 | Fast-paced movement |
| Expressive | 76 | Emotional/expressive movement |
| Participatory | 75 | Event-based/interactive |
| Cohesive | 67 | Group-walking behavior |
| Ambulatory | 58 | Normal walking/commuting |
| Disability | 32 | Mobility-constrained movement |

### Evaluation Functions (20 total)


**V-Velocy:**
- `mean_speed` — Average walking speed (m/s)
- `speed_variation_coeff` — Speed heterogeneity (coefficient of variation)

**R-Realism:**
- `collision_fraction` — Fraction of near-collision events
- `lingering_fraction` — Fraction of slow/stationary agents


**D-Direction:**
- `flow_alignment` — Movement directionality alignment (0–1)
- `path_linearity` — Straightness of paths (0–1)
- `directional_entropy_normalized` — Direction diversity (0–1)

**S-Spatial:**
- `spatial_concentration` — Non-uniform spatial distribution (Gini-based, 0–1)
- `mean_local_density` — Average neighbor count within adaptive radius
- `peak_local_density` — Maximum local density (hotspot intensity)
- `dispersal_score` — Radial divergence from initial centroid
- `convergence_score` — Inward convergence toward final centroid


**T-Temporal:**
- `clustering_trend` — Group formation dynamics
- `lingering_trend` — Agents stopping/mobilizing over time
- `entropy_trend` — Movement organization vs. chaos over time
- `speed_trend` — Temporal acceleration/deceleration pattern
- `density_trend` — Crowd compacting or dispersing over time
- `collision_trend` — Collision escalation/resolution over time
- `flow_alignment_trend` — Movement becoming organized or chaotic

### Example

One row of `benchmark.jsonl`:

```json
{
  "source_scene_id": "0000",
  "scenario_id": "00_Zurich_HB_simplified_obstacle",
  "description": "Morning commuter flow at Zurich HB with people casually walking between ticket machines and platforms, navigating the concourse and open corridors.",
  "category": "Ambulatory",
  "location": "Zurich",
  "questions": [
    {
      "id": "Q1",
      "question": "Are pedestrians moving at normal-but-constrained commuter walking speeds (not rushing/running)?",
      "measurements": [
        {
          "function": "mean_speed",
          "params": {},
          "expected_result": {"min": 0.8, "max": 1.3},
          "expected_result_reasoning": "Morning commuters are walking casually; in a station concourse with obstacles, average walking speed is typically 0.8–1.2 m/s.",
          "delta_value": 0.07
        }
      ]
    }
  ],
  "_model": "...",
  "_n_scenes": 936
}
```

## Usage

Each file is exposed as a separate config. Load whichever you need:

```python
from datasets import load_dataset

REPO = "eth-ivia-lab/STRIDE-Bench"

benchmark = load_dataset(REPO, "benchmark", split="train")
scenes    = load_dataset(REPO, "scenes",    split="train")
maps      = load_dataset(REPO, "maps",      split="train")

print(benchmark[0]["description"])
print(scenes[0]["scene_id"], scenes[0]["crowd_size"])
print(maps[0]["scene_id"], maps[0]["obstacles_shape"])
```

Rows are aligned across the three configs and joinable on `scene_id` (or `source_scene_id` in `benchmark`).

### Reconstructing maps from Parquet

```python
import io, numpy as np
from PIL import Image

row = maps[0]
obstacles = np.array(row["obstacles"], dtype=np.float64).reshape(row["obstacles_shape"])
img = Image.open(io.BytesIO(row["obstacle_map_png"]))
```

### Intended Use

This benchmark is designed to evaluate crowd trajectory generation models by measuring whether their outputs exhibit behaviorally consistent movement patterns given a textual scenario description. It enables:

- **Model comparison**: Rank trajectory generators by behavioral realism across diverse scenarios.
- **Category-level analysis**: Identify model strengths/weaknesses per behavior type (e.g., evacuation vs. normal walking).
- **Trajectory Question Answering (STRIDE)**: Frame trajectory evaluation as a QA task — does the trajectory "answer" behavioral questions correctly?

### Evaluation Protocol

For each model and scene:
1. Generate/simulate a trajectory given the scene description.
2. Execute evaluation functions on the generated trajectory.
3. Compare actual function outputs to expected ranges (with `delta_value` tolerance).
4. Compute pass/fail per question and aggregate STRIDE accuracy.

### Trajectory Format

Models are expected to produce trajectories as arrays of shape `(timesteps, agents, 7)` with columns: `[x, y, vx, vy, goal_x, goal_y, radius]`.

## Limitations

- Expected value ranges are LLM-estimated and may not perfectly match all real-world instances of a described scenario.
- The benchmark focuses on aggregate behavioral statistics, not individual trajectory plausibility.
- Location geometry (obstacles, boundaries) is simplified from real environments.
