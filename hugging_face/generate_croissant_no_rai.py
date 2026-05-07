"""
Generate three Croissant 1.0 JSON-LD metadata files for the STRIDE-Bench
HuggingFace dataset:

    hugging_face/croissant/benchmark.croissant.json   <- benchmark.jsonl
    hugging_face/croissant/scenes.croissant.json      <- scenes.parquet
    hugging_face/croissant/maps.croissant.json        <- maps.parquet

Each file is a complete, standalone Croissant document with sha256 + size
computed from the local data file under hugging_face/data/.

Usage:
    python hugging_face/generate_croissant.py
    python hugging_face/generate_croissant.py --repo myuser/STRIDE-Bench

Validate:
    mlcroissant validate --jsonld hugging_face/croissant/benchmark.croissant.json
    mlcroissant validate --jsonld hugging_face/croissant/scenes.croissant.json
    mlcroissant validate --jsonld hugging_face/croissant/maps.croissant.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

DEFAULT_REPO = " 0/ stride-benchmark"
DATA_DIR = Path("hugging_face/data")
OUT_DIR = Path("hugging_face/croissant")


# ---------- helpers ----------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_object(file_id: str, name: str, repo: str, sha: str, size: int,
                encoding: str, description: str) -> dict:
    return {
        "@type": "cr:FileObject",
        "@id": file_id,
        "name": name,
        "description": description,
        "contentUrl": f"https://huggingface.co/datasets/{repo}/resolve/main/{name}",
        "encodingFormat": encoding,
        "sha256": sha,
        "contentSize": f"{size} B",
    }


def context_block() -> dict:
    return {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "citeAs": "cr:citeAs",
        "column": "cr:column",
        "conformsTo": "dct:conformsTo",
        "cr": "http://mlcommons.org/croissant/",
        "rai": "http://mlcommons.org/croissant/RAI/",
        "data": {"@id": "cr:data", "@type": "@json"},
        "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
        "dct": "http://purl.org/dc/terms/",
        "examples": {"@id": "cr:examples", "@type": "@json"},
        "extract": "cr:extract",
        "field": "cr:field",
        "fileProperty": "cr:fileProperty",
        "fileObject": "cr:fileObject",
        "fileSet": "cr:fileSet",
        "format": "cr:format",
        "includes": "cr:includes",
        "jsonPath": "cr:jsonPath",
        "key": "cr:key",
        "parentField": "cr:parentField",
        "path": "cr:path",
        "recordSet": "cr:recordSet",
        "references": "cr:references",
        "repeated": "cr:repeated",
        "sc": "https://schema.org/",
        "source": "cr:source",
        "subField": "cr:subField",
        "transform": "cr:transform",
    }


def base_dataset(name: str, description: str, repo: str) -> dict:
    return {
        "@context": context_block(),
        "@type": "sc:Dataset",
        "name": name,
        "description": description,
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": "https://choosealicense.com/licenses/mit/",
        "url": f"https://huggingface.co/datasets/{repo}",
        "version": "1.0.0",
        "datePublished": "2026-04-20",
        "keywords": [
            "crowd-simulation",
            "trajectory-evaluation",
            "pedestrian-dynamics",
            "benchmark",
            "behavioral-evaluation",
        ],
        "creator": {
            "@type": "Person",
            "name": "Anonymous (NeurIPS 2026 D&B submission, double-blind)",
        },
        "rai:dataCollectionType": ["Synthetic", "Simulation"],
        "rai:personalSensitiveInformation": (
            "None. The dataset contains no personal data, no images of identifiable "
            "individuals, and no real pedestrian tracks. All scenes are synthetic."
        ),
        "rai:hasSyntheticData": True,
    }


# ---------- type mapping for parquet ----------

def pa_to_croissant_type(pa_type) -> str:
    """Map a pyarrow type to a Croissant sc: type. Nested types -> sc:Text
    (consumers parse the nested structure via the Parquet schema)."""
    import pyarrow as pa
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        return "sc:Text"
    if pa.types.is_integer(pa_type):
        return "sc:Integer"
    if pa.types.is_floating(pa_type):
        return "sc:Float"
    if pa.types.is_boolean(pa_type):
        return "sc:Boolean"
    if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
        return "sc:ImageObject"
    return "sc:Text"


def parquet_fields(parquet_path: Path, file_id: str, record_id: str) -> list[dict]:
    schema = pq.read_schema(parquet_path)
    fields = []
    for f in schema:
        fields.append({
            "@type": "cr:Field",
            "@id": f"{record_id}/{f.name}",
            "name": f.name,
            "description": f"{f.name} (parquet type: {f.type})",
            "dataType": pa_to_croissant_type(f.type),
            "source": {
                "fileObject": {"@id": file_id},
                "extract": {"column": f.name},
            },
        })
    return fields


# ---------- build per-file Croissant docs ----------

def build_benchmark(repo: str, jsonl_path: Path) -> dict:
    sha = sha256_of(jsonl_path)
    size = jsonl_path.stat().st_size
    fid = "benchmark-jsonl"

    fo = file_object(
        file_id=fid,
        name=jsonl_path.name,
        repo=repo,
        sha=sha,
        size=size,
        encoding="application/jsonlines",
        description=("JSON Lines: one scene per line. Each line contains the scene's "
                     "scenario description, behavioral category, decomposition reasoning, "
                     "and a list of behavioral evaluation questions with measurement specs."),
    )

    # Top-level fields per JSONL row.
    field_specs = [
        ("source_scene_id", "sc:Text", "Unique scene identifier."),
        ("scenario_id", "sc:Text", "Reference to the physical scenario / environment."),
        ("description", "sc:Text", "Natural-language description of the crowd behavior scenario."),
        ("category", "sc:Text", "Behavioral category label (one of 11)."),
        ("location", "sc:Text", "Real-world location name."),
        ("decomposition_reasoning", "sc:Text", "LLM reasoning for description -> behavioral properties."),
        ("questions", "sc:Text", ("Nested list of {id, question, measurements:[{function, params, "
                                  "expected_result, expected_result_reasoning, delta_value}]}.")),
        ("_model", "sc:Text", "LLM used to generate the benchmark (carried per-row)."),
        ("_n_scenes", "sc:Integer", "Total scenes in the dataset (carried per-row)."),
    ]
    fields = []
    for name, dtype, desc in field_specs:
        fields.append({
            "@type": "cr:Field",
            "@id": f"benchmark/{name}",
            "name": name,
            "description": desc,
            "dataType": dtype,
            "source": {
                "fileObject": {"@id": fid},
                "extract": {"jsonPath": f"$.{name}"},
            },
        })

    record_set = {
        "@type": "cr:RecordSet",
        "@id": "benchmark",
        "name": "benchmark",
        "description": "One record per scene (one JSONL line).",
        "field": fields,
    }

    doc = base_dataset(
        name="STRIDE-Bench-benchmark",
        description=(
            "Behavioral evaluation specs for the STRIDE-Bench crowd trajectory benchmark. "
            "Each row is one scene with a natural-language scenario description and a "
            "decomposition into behavioral evaluation questions (Trajectory Question "
            "Answering /  stride), each grounded in one or more measurement functions with "
            "expected value ranges and tolerances."
        ),
        repo=repo,
    )
    doc["distribution"] = [fo]
    doc["recordSet"] = [record_set]
    return doc


def build_scenes(repo: str, parquet_path: Path) -> dict:
    sha = sha256_of(parquet_path)
    size = parquet_path.stat().st_size
    fid = "scenes-parquet"

    fo = file_object(
        file_id=fid,
        name=parquet_path.name,
        repo=repo,
        sha=sha,
        size=size,
        encoding="application/x-parquet",
        description=("Parquet table of crowd-simulation scenes: scenario metadata, "
                     "initial agent states, group structure, goal locations, and "
                     "expected behavioral ranges. One row per scene."),
    )

    record_set = {
        "@type": "cr:RecordSet",
        "@id": "scenes",
        "name": "scenes",
        "description": "One record per scene; columns derived from the Parquet schema.",
        "field": parquet_fields(parquet_path, fid, "scenes"),
    }

    doc = base_dataset(
        name="STRIDE-Bench-scenes",
        description=(
            "Crowd-simulation scenes for STRIDE-Bench: scenario metadata, initial agent "
            "states (position/velocity/goal/radius), group structure, and per-scene "
            "expected behavioral ranges. Distributed as a single Parquet table with one "
            "row per scene; nested fields whose runtime types vary (e.g. goal_location_*) "
            "are stored as JSON-encoded strings."
        ),
        repo=repo,
    )
    doc["distribution"] = [fo]
    doc["recordSet"] = [record_set]
    return doc


def build_maps(repo: str, parquet_path: Path) -> dict:
    sha = sha256_of(parquet_path)
    size = parquet_path.stat().st_size
    fid = "maps-parquet"

    fo = file_object(
        file_id=fid,
        name=parquet_path.name,
        repo=repo,
        sha=sha,
        size=size,
        encoding="application/x-parquet",
        description=("Parquet table of per-scene map data: obstacle rectangles "
                     "(N x 4 float64) and a rendered obstacle PNG. One row per scene."),
    )

    record_set = {
        "@type": "cr:RecordSet",
        "@id": "maps",
        "name": "maps",
        "description": "One record per scene; obstacles array + rendered obstacle PNG.",
        "field": parquet_fields(parquet_path, fid, "maps"),
    }

    doc = base_dataset(
        name="STRIDE-Bench-maps",
        description=(
            "Per-scene map data for STRIDE-Bench: obstacle rectangles (originally stored "
            "as map.npz with key 'obstacles', shape (N, 4)) and a rendered obstacle PNG, "
            "packed into a single Parquet table with one row per scene."
        ),
        repo=repo,
    )
    doc["distribution"] = [fo]
    doc["recordSet"] = [record_set]
    return doc


# ---------- driver ----------

def write_doc(doc: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}  ({out_path.stat().st_size} B)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help="HuggingFace dataset repo (user/name) used in contentUrl + url.")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    targets = [
        ("benchmark.jsonl", "benchmark.croissant.json", build_benchmark),
        ("scenes.parquet", "scenes.croissant.json", build_scenes),
        ("maps.parquet", "maps.croissant.json", build_maps),
    ]
    for in_name, out_name, builder in targets:
        in_path = args.data_dir / in_name
        if not in_path.exists():
            raise SystemExit(f"Missing data file: {in_path}")
        doc = builder(args.repo, in_path)
        write_doc(doc, args.out_dir / out_name)


if __name__ == "__main__":
    main()
