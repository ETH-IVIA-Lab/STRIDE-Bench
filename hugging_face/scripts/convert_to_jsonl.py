"""
Convert benchmark_norm.json (single nested JSON object) to benchmark.jsonl
(one scene per line) so HuggingFace's dataset viewer renders 1,142 rows
instead of 1.

Top-level metadata (model name, scene count) is carried into each row
under _model and _n_scenes, so the information is preserved.

Usage:
    python convert_to_jsonl.py
    python convert_to_jsonl.py --in benchmark_norm.json --out benchmark.jsonl
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=Path, default=Path("benchmark_norm.json"))
    ap.add_argument("--out", dest="dst", type=Path, default=Path("benchmark.jsonl"))
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Input file not found: {args.src}")

    data = json.loads(args.src.read_text())
    model = data.get("model")
    scenes = data.get("scenes", [])
    n = len(scenes)

    with open(args.dst, "w") as f:
        for scene in scenes:
            # Carry top-level metadata into each row so it stays queryable.
            scene["_model"] = model
            scene["_n_scenes"] = n
            f.write(json.dumps(scene, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(args.dst.read_bytes()).hexdigest()
    size = args.dst.stat().st_size

    print(f"Wrote {args.dst}")
    print(f"  rows:   {n}")
    print(f"  bytes:  {size}")
    print(f"  sha256: {sha}")
    print()
    print("Next steps:")
    print(f"  1. Update croissant.json:")
    print(f"       distribution[0].name        -> '{args.dst.name}'")
    print(f"       distribution[0].contentUrl  -> '.../resolve/main/{args.dst.name}'")
    print(f"       distribution[0].sha256      -> '{sha}'")
    print(f"       distribution[0].contentSize -> '{size} B'")
    print(f"       distribution[0].encodingFormat -> 'application/jsonlines'")
    print(f"  2. Upload both files to the Hub:")
    print(f"       hf upload <repo> {args.dst.name} {args.dst.name} --repo-type dataset")
    print(f"       hf upload <repo> README.md README.md --repo-type dataset")
    print(f"       hf upload <repo> croissant.json croissant.json --repo-type dataset")


if __name__ == "__main__":
    main()
