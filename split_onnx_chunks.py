#!/usr/bin/env python3
"""Split a large .onnx into <100MB binary chunks + JSON manifest for GitHub Pages.

Browser loads the manifest, fetches each part, concatenates ArrayBuffers, then
passes the result to onnxruntime-web InferenceSession.create() — no model change.

Usage:
  python split_onnx_chunks.py amodal_dino_q4.onnx
  python split_onnx_chunks.py amodal_dino_q4.onnx --max-mb 95
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def split_file(src: Path, max_bytes: int, out_dir: Path | None = None) -> Path:
    src = src.resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    out_dir = (out_dir or src.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = src.read_bytes()
    total = len(data)
    stem = src.stem  # amodal_dino_q4
    n_parts = max(1, (total + max_bytes - 1) // max_bytes)
    chunks: list[dict] = []
    sha = hashlib.sha256()

    for i in range(n_parts):
        start = i * max_bytes
        end = min(total, start + max_bytes)
        part = data[start:end]
        sha.update(part)
        name = f"{stem}.part{i:03d}.bin"
        path = out_dir / name
        path.write_bytes(part)
        chunks.append({"file": name, "bytes": len(part), "index": i})
        print(f"  wrote {name}  {len(part) / (1024 * 1024):.1f} MB")

    manifest = {
        "format": "raw-byte-chunks-v1",
        "model": src.name,
        "stem": stem,
        "total_bytes": total,
        "sha256": sha.hexdigest(),
        "max_chunk_bytes": max_bytes,
        "n_chunks": len(chunks),
        "chunks": chunks,
        "note": "Fetch chunks in order, concatenate, pass ArrayBuffer to ORT.",
    }
    man_path = out_dir / f"{stem}.manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest → {man_path.name}  ({n_parts} parts, {total / (1024 * 1024):.1f} MB total)")
    return man_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("onnx", type=Path, help="source .onnx path")
    ap.add_argument(
        "--max-mb",
        type=float,
        default=95.0,
        help="max chunk size in MiB (default 95, under GitHub 100MB hard limit)",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    max_bytes = int(args.max_mb * 1024 * 1024)
    if max_bytes <= 0 or max_bytes > 100 * 1024 * 1024:
        raise SystemExit("--max-mb must be in (0, 100]")
    split_file(args.onnx, max_bytes, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
