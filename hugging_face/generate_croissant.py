"""
Generate a Croissant 1.0 JSON-LD metadata file for sweetspot000/tqa-benchmark.

Usage:
    # If you have benchmark_norm.json locally (recommended — gives a real sha256):
    python generate_croissant.py --data benchmark_norm.json --out croissant.json

    # Without the file (uses placeholder sha256 you fill in later):
    python generate_croissant.py --out croissant.json

Then validate:
    mlcroissant validate --jsonld croissant.json

Then commit croissant.json to the root of your HF dataset repo.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

REPO = "sweetspot000/tqa-benchmark"
FILE_NAME = "benchmark_norm.json"
CONTENT_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{FILE_NAME}"
HF_URL = f"https://huggingface.co/datasets/{REPO}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(sha256: str, num_bytes: int | None) -> dict:
    file_object = {
        "@type": "cr:FileObject",
        "@id": "benchmark-file",
        "name": FILE_NAME,
        "description": "Single JSON file containing all benchmark scenes, questions, and measurement specs.",
        "contentUrl": CONTENT_URL,
        "encodingFormat": "application/json",
        "sha256": sha256,
    }
    if num_bytes is not None:
        file_object["contentSize"] = f"{num_bytes} B"

    # Top-level RecordSet: one record per scene.
    # Nested questions/measurements stay inside the JSON value of each scene
    # (Croissant's JSONPath extract works well for the scene level; deeper
    # nesting can be parsed by the consumer).
    scenes_record_set = {
        "@type": "cr:RecordSet",
        "@id": "scenes",
        "name": "scenes",
        "description": "One record per crowd-behavior scene.",
        "field": [
            {
                "@type": "cr:Field",
                "@id": "scenes/source_scene_id",
                "name": "source_scene_id",
                "description": "Unique scene identifier.",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": "benchmark-file"},
                    "extract": {"jsonPath": "$.scenes[*].source_scene_id"},
                },
            },
            {
                "@type": "cr:Field",
                "@id": "scenes/scenario_id",
                "name": "scenario_id",
                "description": "Reference to the physical scenario / environment.",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": "benchmark-file"},
                    "extract": {"jsonPath": "$.scenes[*].scenario_id"},
                },
            },
            {
                "@type": "cr:Field",
                "@id": "scenes/description",
                "name": "description",
                "description": "Natural-language description of the crowd behavior scenario.",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": "benchmark-file"},
                    "extract": {"jsonPath": "$.scenes[*].description"},
                },
            },
            {
                "@type": "cr:Field",
                "@id": "scenes/category",
                "name": "category",
                "description": "Behavioral category label (one of 11: Escaping, Violent, Demonstrator, Dense, Aggressive, Rushing, Expressive, Participatory, Cohesive, Ambulatory, Disability).",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": "benchmark-file"},
                    "extract": {"jsonPath": "$.scenes[*].category"},
                },
            },
            {
                "@type": "cr:Field",
                "@id": "scenes/location",
                "name": "location",
                "description": "Real-world location name.",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": "benchmark-file"},
                    "extract": {"jsonPath": "$.scenes[*].location"},
                },
            },
            {
                "@type": "cr:Field",
                "@id": "scenes/decomposition_reasoning",
                "name": "decomposition_reasoning",
                "description": "LLM reasoning for how the description maps to measurable behavioral properties.",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": "benchmark-file"},
                    "extract": {"jsonPath": "$.scenes[*].decomposition_reasoning"},
                },
            },
            {
                "@type": "cr:Field",
                "@id": "scenes/questions",
                "name": "questions",
                "description": "List of behavioral evaluation questions for this scene. Each question has an id, natural-language text, and one or more measurement specs (function, params, expected_result range, delta_value tolerance). Consumers parse this nested structure directly.",
                "dataType": "sc:Text",
                "source": {
                    "fileObject": {"@id": "benchmark-file"},
                    "extract": {"jsonPath": "$.scenes[*].questions"},
                },
            },
        ],
    }

    metadata = {
        "@context": {
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
            "equivalentProperty": "wd:P1628",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "samplingRate": "cr:samplingRate",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            "wd": "https://www.wikidata.org/wiki/",
        },
        "@type": "sc:Dataset",
        "name": "tqa-benchmark",
        "description": (
            "CRDTraj-Bench (TQA Benchmark) is an evaluation benchmark for context-consistent "
            "crowd trajectory generation. Rather than comparing trajectories point-by-point, it "
            "measures whether generated trajectories exhibit behaviors consistent with a given "
            "natural-language scenario description, by decomposing each description into "
            "measurable behavioral questions (Trajectory Question Answering / TQA). The benchmark "
            "spans 1,142 scenes across 11 behavioral categories (Escaping, Violent, Demonstrator, "
            "Dense, Aggressive, Rushing, Expressive, Participatory, Cohesive, Ambulatory, "
            "Disability) and 30 real-world locations, with 9,706 questions and 15,056 measurement "
            "specs computed via 22 evaluation functions covering speed, density, spatial "
            "structure, flow, path, and behavioral dynamics."
        ),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": "https://choosealicense.com/licenses/mit/",
        "url": HF_URL,
        "version": "1.0.0",
        "datePublished": "2026-04-20",
        "keywords": [
            "crowd-simulation",
            "trajectory-evaluation",
            "pedestrian-dynamics",
            "benchmark",
            "behavioral-evaluation",
            "trajectory-question-answering",
            "context-consistency",
        ],
        "creator": {
            "@type": "Person",
            "name": "Anonymous (NeurIPS 2026 D&B submission, double-blind)",
        },
        "citeAs": (
            "@misc{crdtrajbench2026,\n"
            "  title  = {CRDTraj-Bench: Trajectory Question Answering for Context-Consistent "
            "Crowd Trajectory Generation},\n"
            "  author = {Anonymous},\n"
            "  year   = {2026},\n"
            "  note   = {NeurIPS 2026 Datasets and Benchmarks submission. "
            "Dataset: https://huggingface.co/datasets/sweetspot000/tqa-benchmark}\n"
            "}"
        ),
        # ----- RAI (Responsible AI) fields -----
        "rai:dataCollection": (
            "Scenes were synthetically constructed: an LLM authored natural-language crowd "
            "behavior descriptions, then translated them into Social Force Model (SFM) parameters "
            "to simulate physically-grounded multi-agent trajectories. Real-world location "
            "geometry (30 locations) was used as scene context but simplified into obstacle maps. "
            "No real pedestrian trajectories or personal data were collected."
        ),
        "rai:dataCollectionType": ["Synthetic", "Simulation"],
        "rai:dataCollectionRawData": (
            "Inputs to the pipeline: (a) LLM-generated behavior descriptions, "
            "(b) SFM parameters derived from those descriptions, (c) simplified real-world "
            "location geometries used as scene scaffolding."
        ),
        "rai:annotationsPerItem": (
            "Each scene is annotated by an LLM with: a category label (1 of 11), a decomposition "
            "of the description into ~7-9 behavioral questions, and per-question measurement "
            "specs (target function, parameters, expected value range, tolerance delta)."
        ),
        "rai:annotationDemographics": (
            "Annotations are LLM-generated, not produced by human annotators."
        ),
        "rai:personalSensitiveInformation": (
            "None. The dataset contains no personal data, no images of identifiable individuals, "
            "and no real pedestrian tracks. All scenes are synthetic."
        ),
        "rai:dataBiases": (
            "Behavior descriptions and decomposition specs were authored by a single LLM, which "
            "introduces model-specific biases in (a) which behavioral aspects are deemed salient "
            "for a given description and (b) the expected value ranges. Location coverage is "
            "biased toward urban/transit settings. The 11 behavioral categories are imbalanced "
            "(Escaping: 158, Disability: 36)."
        ),
        "rai:dataLimitations": (
            "Expected value ranges are LLM-estimated and may not perfectly match all real-world "
            "instances of a described scenario. Aggregate behavioral statistics are evaluated, "
            "not individual trajectory plausibility. Simplified location geometry omits "
            "fine-grained obstacles."
        ),
        "rai:dataUseCases": (
            "Intended uses: (1) ranking crowd trajectory generation models by behavioral "
            "context-consistency, (2) per-category analysis of model strengths/weaknesses, "
            "(3) framing trajectory evaluation as a question-answering task. "
            "Out-of-scope: training crowd generators (the benchmark is for evaluation only); "
            "deployment of any generator-evaluator pair to real-world safety-critical systems "
            "without further validation."
        ),
        # ----- Distribution + RecordSet -----
        "distribution": [file_object],
        "recordSet": [scenes_record_set],
    }
    return metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None,
                    help="Path to benchmark_norm.json (used to compute sha256 + size).")
    ap.add_argument("--out", type=Path, default=Path("croissant.json"))
    args = ap.parse_args()

    if args.data and args.data.exists():
        sha = sha256_of(args.data)
        size = args.data.stat().st_size
        print(f"Computed sha256: {sha}")
        print(f"Size: {size} B")
    else:
        sha = "TODO_REPLACE_WITH_SHA256_OF_benchmark_norm.json"
        size = None
        print("WARNING: --data not provided. Run on the actual file to fill in sha256.")
        print("         Quick command:  sha256sum benchmark_norm.json")

    metadata = build(sha, size)
    args.out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
