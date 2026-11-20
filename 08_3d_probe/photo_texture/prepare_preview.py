"""Create a compact, UV-preserving chat preview from an existing textured GLB.

The source geometry and its fixed texture atlas remain unchanged. Seam vertices
are welded geometrically before texture-aware simplification, while per-corner
UVs retain the chart discontinuities. Vertices are split by (position, UV) again
for the indexed preview mesh. This script runs entirely on the CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from PIL import Image, ImageDraw

from bake_texture import camera_for_angle, render_textured


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--texture", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-faces", type=int, default=15000)
    parser.add_argument("--texture-weight", type=float, default=2.0)
    parser.add_argument("--elevation", type=float, default=2.0)
    parser.add_argument("--validation-width", type=int, default=640)
    parser.add_argument("--validation-angles", default="0,45,90,135,180,225,270,315")
    args = parser.parse_args()
    if args.target_faces < 4:
        raise SystemExit("--target-faces must be at least 4")
    source_path = args.source.resolve()
    texture_path = (args.texture or source_path.parent / "texture.png").resolve()
    output = args.output_dir or source_path.parent / "preview"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty directory: {output}")

    start = time.perf_counter()
    source_sha = sha256_file(source_path)
    texture_sha = sha256_file(texture_path)
    scene = trimesh.load(source_path, force="scene")
    if len(scene.geometry) != 1:
        raise ValueError("Preview preparation currently expects one textured mesh")
    source = next(iter(scene.geometry.values()))
    if source.visual.kind != "texture":
        raise ValueError("Source mesh does not have texture coordinates")
    source_uv = np.asarray(source.visual.uv)
    texture = np.asarray(Image.open(texture_path).convert("RGB"))
    embedded = source.visual.material.baseColorTexture
    if embedded is None or not np.array_equal(texture, np.asarray(embedded.convert("RGB"))):
        raise ValueError("External texture does not exactly match the GLB's embedded atlas")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(source_path))
    if not ms.current_mesh().has_wedge_tex_coord():
        raise ValueError("GLB importer did not retain per-corner texture coordinates")
    # UV seams are represented in wedge UVs. Welding only geometry allows the
    # texture-aware filter to reduce topology without arbitrarily joining charts.
    ms.meshing_remove_duplicate_vertices()
    welded_vertices = ms.current_mesh().vertex_number()
    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=args.target_faces,
        extratcoordw=args.texture_weight,
        preserveboundary=True,
        preservenormal=True,
        optimalplacement=True,
    )
    simplified = ms.current_mesh()
    vertices = simplified.vertex_matrix()
    faces = simplified.face_matrix()
    wedge_uv = simplified.wedge_tex_coord_matrix()
    corners = np.column_stack((vertices[faces.reshape(-1)], wedge_uv))
    unique, inverse = np.unique(corners, axis=0, return_inverse=True)
    vertices = unique[:, :3].astype(np.float32)
    uv = unique[:, 3:].astype(np.float32)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    if not np.isfinite(vertices).all() or not np.isfinite(uv).all():
        raise ValueError("Non-finite preview positions or UVs")
    if uv.min() < -1e-5 or uv.max() > 1 + 1e-5:
        raise ValueError("Simplification produced UVs outside the fixed atlas")
    preview = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    output.mkdir(parents=True, exist_ok=True)
    npz_path = output / "preview_mesh.npz"
    np.savez_compressed(npz_path, vertices=vertices, faces=faces, uv=uv)
    record = {
        "method": "UV-aware quadric decimation with geometric weld and position/UV seam split",
        "source": str(source_path), "source_sha256": source_sha,
        "texture": str(texture_path), "texture_sha256": texture_sha,
        "texture_dimensions": [texture.shape[1], texture.shape[0]],
        "source_vertices": len(source.vertices), "source_faces": len(source.faces),
        "welded_vertices_before_simplification": welded_vertices,
        "requested_faces": args.target_faces, "vertices": len(vertices), "faces": len(faces),
        "mesh_npz": "preview_mesh.npz", "mesh_npz_bytes": npz_path.stat().st_size,
        "mesh_npz_sha256": sha256_file(npz_path),
        "uv_convention": "Bottom-origin v: sample image row (1-v)*(height-1); no atlas rebake or image flip.",
        "params": {"texture_weight": args.texture_weight, "preserveboundary": True,
                   "preservenormal": True, "default_elevation_degrees": args.elevation},
        "pymeshlab": importlib.metadata.version("pymeshlab"),
        "validation": [],
    }

    angles = [float(value) for value in args.validation_angles.split(",")]
    width = args.validation_width
    height = width * 3 // 4
    sheet = Image.new("RGB", (width * 2, height * len(angles)), (235, 237, 240))
    draw = ImageDraw.Draw(sheet)
    for row, azimuth in enumerate(angles):
        camera = camera_for_angle(source.vertices, azimuth, elevation=args.elevation,
                                  width=width, height=height)
        full_image, full_mask, _ = render_textured(source, source_uv, texture, camera, width, height)
        small_image, small_mask, _ = render_textured(preview, uv, texture, camera, width, height)
        common = full_mask & small_mask
        error = np.abs(np.asarray(full_image).astype(np.float32)
                       - np.asarray(small_image).astype(np.float32)).mean(axis=-1)
        record["validation"].append({
            "azimuth": azimuth,
            "silhouette_iou": float(common.sum() / (full_mask | small_mask).sum()),
            "common_foreground_rgb_mae_0_255": float(error[common].mean()),
            "common_foreground_fraction_rgb_mae_gt30": float((error[common] > 30).mean()),
        })
        sheet.paste(full_image, (0, row * height))
        sheet.paste(small_image, (width, row * height))
        draw.text((10, row * height + 10), f"Original: {len(source.faces):,} faces; az {azimuth:g}", fill=(10, 10, 10))
        draw.text((width + 10, row * height + 10), f"Preview: {len(faces):,} faces; same atlas", fill=(10, 10, 10))
    sheet.save(output / "comparison.png")
    record["source_unchanged"] = source_sha == sha256_file(source_path)
    record["texture_unchanged"] = texture_sha == sha256_file(texture_path)
    if not (record["source_unchanged"] and record["texture_unchanged"]):
        raise ValueError("Source files changed while preparing the preview")
    record["seconds"] = round(time.perf_counter() - start, 3)
    (output / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: record[key] for key in ["vertices", "faces", "mesh_npz_bytes", "seconds", "validation"]}, indent=2))


if __name__ == "__main__":
    main()
