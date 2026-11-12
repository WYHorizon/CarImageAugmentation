#!/usr/bin/env python3
"""Prepare a reusable RGBA occluder without a generative model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from occlusion.material import prepare_material


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--background", choices=("alpha", "light-checker"), default="alpha")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if output.suffix.lower() != '.png':
        parser.error('output must use a .png extension')
    metadata_path = output.with_suffix(".json")
    if source == output:
        parser.error("output must differ from input")
    if output.exists() or metadata_path.exists():
        parser.error("output and its metadata sidecar must not already exist")
    with Image.open(source) as image:
        rgba, metadata = prepare_material(image, args.background)
    output.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(output, format="PNG")
    metadata.update({"source_filename": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "output_filename": output.name, "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "metadata": str(metadata_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
