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
pretty_name: TrajQA-Bench
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files: benchmark.jsonl
---

# TrajQA Benchmark

## Dataset Description

TrajQA Benchmark is an evaluation benchmark for assessing the behavioral realism of crowd trajectory generation and simulation models. Rather than comparing trajectories point-by-point, it evaluates whether generated trajectories exhibit behaviors consistent with a given scenario description — measuring **trajectory-context consistency** through decomposed behavioral questions.

### Dataset Summary

| Property | Value |
|----------|-------|
| File | `benchmark.jsonl` (one scene per line) |
| Scenes | 1,142 |
| Total Questions | 9,706 |
| Total Measurements | 15,056 |
| Evaluation Functions | 22 |
| Behavioral Categories | 11 |
| Unique Scenarios | 31 |
| Locations | 30 real-world locations |

### Languages

English

## Dataset Structure

The dataset is distributed as JSON Lines: **each line is one scene**. The companion file `benchmark_norm.json` contains the same content as a single nested object (top-level `model`, `n_scenes`, `scenes[]`) and is provided for convenience; loaders should prefer `benchmark.jsonl`.

### Data Fields

Each row (one scene) has the following fields:

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

### Behavioral Categories

| Category | Count | Description |
|----------|-------|-------------|
| Escaping | 158 | Emergency/evacuation behavior |
| Violent | 153 | Conflict scenarios |
| Demonstrator | 131 | Organized group/protest movement |
| Dense | 128 | High-density crowd movement |
| Aggressive | 118 | Aggressive crowd behavior |
| Rushing | 107 | Fast-paced movement |
| Expressive | 89 | Emotional/expressive movement |
| Participatory | 89 | Event-based/interactive |
| Cohesive | 73 | Group-walking behavior |
| Ambulatory | 60 | Normal walking/commuting |
| Disability | 36 | Mobility-constrained movement |

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
  "_n_scenes": 1142
}
```

## Usage

```python
from datasets import load_dataset
ds = load_dataset("anonymous1ads34/trajQA-Bench", split="train")
print(ds[0]["description"])
```

### Intended Use

This benchmark is designed to evaluate crowd trajectory generation models by measuring whether their outputs exhibit behaviorally consistent movement patterns given a textual scenario description. It enables:

- **Model comparison**: Rank trajectory generators by behavioral realism across diverse scenarios.
- **Category-level analysis**: Identify model strengths/weaknesses per behavior type (e.g., evacuation vs. normal walking).
- **Trajectory Question Answering (TrajQA)**: Frame trajectory evaluation as a QA task — does the trajectory "answer" behavioral questions correctly?

### Evaluation Protocol

For each model and scene:
1. Generate/simulate a trajectory given the scene description.
2. Execute evaluation functions on the generated trajectory.
3. Compare actual function outputs to expected ranges (with `delta_value` tolerance).
4. Compute pass/fail per question and aggregate TrajQA accuracy.

### Trajectory Format

Models are expected to produce trajectories as arrays of shape `(timesteps, agents, 7)` with columns: `[x, y, vx, vy, goal_x, goal_y, radius]`.

## Limitations

- Expected value ranges are LLM-estimated and may not perfectly match all real-world instances of a described scenario.
- The benchmark focuses on aggregate behavioral statistics, not individual trajectory plausibility.
- Location geometry (obstacles, boundaries) is simplified from real environments.
