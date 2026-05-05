"""
Convert scenes/*.json and maps/<id>/{map.npz, obstacle_map.png} to Parquet
for HuggingFace dataset auto-discovery (Datasets Viewer + auto-Croissant).

Outputs (relative to --out-dir, default = hugging_face/data):
    scenes.parquet  -- one row per scenes/<id>.json
    maps.parquet    -- one row per maps/<id>/, with obstacles array + PNG bytes

Usage:
    python convert_to_parquet.py
    python convert_to_parquet.py --scenes-dir scenes --maps-dir maps --out-dir hugging_face/data
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _stringify_mixed_columns(df: pd.DataFrame) -> list[str]:
    """JSON-encode columns whose non-null values mix multiple types
    (e.g. str + dict + list). Parquet requires a single type per column.
    Returns the list of columns that were stringified."""
    stringified = []
    for col in df.columns:
        types = {type(v).__name__ for v in df[col] if v is not None}
        if len(types) > 1:
            df[col] = df[col].map(lambda v: None if v is None else json.dumps(v, ensure_ascii=False))
            stringified.append(col)
    return stringified


def convert_scenes(scenes_dir: Path, out_path: Path) -> int:
    files = sorted(scenes_dir.glob("*.json"))
    rows = [json.loads(p.read_text()) for p in files]
    df = pd.DataFrame(rows)
    mixed = _stringify_mixed_columns(df)
    if mixed:
        print(f"  json-encoded mixed-type columns: {mixed}")
    df.to_parquet(out_path, index=False)
    return len(rows)


def convert_maps(maps_dir: Path, out_path: Path) -> int:
    rows = []
    for d in sorted(p for p in maps_dir.iterdir() if p.is_dir()):
        npz = np.load(d / "map.npz")
        key = npz.files[0]
        obstacles = npz[key].astype(np.float64)
        png_bytes = (d / "obstacle_map.png").read_bytes()
        rows.append({
            "scene_id": d.name,
            "obstacles_key": key,
            "obstacles": obstacles.tolist(),
            "obstacles_shape": list(obstacles.shape),
            "obstacle_map_png": png_bytes,
        })
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    return len(rows)


def report(path: Path, n: int) -> None:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    size = path.stat().st_size
    print(f"Wrote {path}")
    print(f"  rows:   {n}")
    print(f"  bytes:  {size}")
    print(f"  sha256: {sha}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-dir", type=Path, default=Path("scenes"))
    ap.add_argument("--maps-dir", type=Path, default=Path("maps"))
    ap.add_argument("--out-dir", type=Path, default=Path("hugging_face/data"))
    ap.add_argument("--skip-scenes", action="store_true")
    ap.add_argument("--skip-maps", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_scenes:
        if not args.scenes_dir.exists():
            raise SystemExit(f"Scenes dir not found: {args.scenes_dir}")
        out = args.out_dir / "scenes.parquet"
        n = convert_scenes(args.scenes_dir, out)
        report(out, n)

    if not args.skip_maps:
        if not args.maps_dir.exists():
            raise SystemExit(f"Maps dir not found: {args.maps_dir}")
        out = args.out_dir / "maps.parquet"
        n = convert_maps(args.maps_dir, out)
        report(out, n)


if __name__ == "__main__":
    main()
