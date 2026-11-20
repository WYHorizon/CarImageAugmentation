"""Build a conservative material approximation from reviewed semantic regions.

This is NOT inverse rendering. Paint, glass and rubber use explicit material
priors; protected photo details retain their original baked appearance. Geometry
and UV coordinates are unchanged. Two complementary alpha masks make the hybrid
representation portable as a standard GLB without a custom shader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import distance_transform_edt

from material_normals import welded_vertex_normals, smooth_vertex_normals


MATERIALS = {
    "1": {"name": "paint", "rgb": [30, 36, 48], "roughness": .32,
          "metalness": 0., "clearcoat": 1.},
    "2": {"name": "glass_opaque_approximation", "rgb": [12, 20, 25],
          "roughness": .18, "metalness": 0., "clearcoat": .8},
    "3": {"name": "rubber", "rgb": [18, 20, 22], "roughness": .85,
          "metalness": 0., "clearcoat": 0.},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--material-ids", type=Path, required=True)
    parser.add_argument("--valid-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normal-radius-fraction", type=float, default=0.,
                        help="Optional shading-normal smoothing radius as a fraction of longest mesh extent")
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    sources = [args.source_dir / name for name in
               ("texture.png", "textured_mesh.npz", "textured_mesh.glb", "manifest.json")]
    sources += [args.material_ids, args.valid_mask, Path(__file__).with_name("material_normals.py")]
    source_hashes = {str(p.resolve()): sha256(p) for p in sources}
    photo = np.asarray(Image.open(args.source_dir / "texture.png").convert("RGB"))
    ids = np.asarray(Image.open(args.material_ids).convert("L")).copy()
    valid = np.asarray(Image.open(args.valid_mask).convert("L")) > 127
    if ids.shape != photo.shape[:2] or valid.shape != ids.shape:
        raise ValueError("Semantic atlas, valid mask and source texture dimensions must match")
    if not set(np.unique(ids)).issubset({0, 1, 2, 3, 4, 5, 255}):
        raise ValueError("Unexpected semantic material label")
    # Padding is strictly outside charts. Unknown interior texels remain photo
    # appearance unless the region pipeline explicitly provided an inferred ID.
    distance, nearest = distance_transform_edt(~valid, return_indices=True)
    padding = (~valid) & (distance <= 12)
    ids[padding] = ids[nearest[0][padding], nearest[1][padding]]
    ids[(~valid) & ~padding] = 0
    clean = np.isin(ids, [1, 2, 3])
    base = np.zeros((*ids.shape, 4), np.uint8)
    details = np.zeros_like(base)
    controls = np.zeros_like(photo)
    for key, material in MATERIALS.items():
        selected = ids == int(key)
        base[selected, :3] = material["rgb"]
        base[selected, 3] = 255
        controls[selected] = np.rint(np.array([
            material["clearcoat"], material["roughness"], material["metalness"]]) * 255).astype(np.uint8)
    details[:, :, :3] = photo
    details[:, :, 3] = np.where(clean, 0, 255)
    # Complementary masks prevent overlapping photo/specular lighting. Both
    # materials use alpha MASK, not transparency or screen-order blending.
    assert np.all(base[:, :, 3].astype(np.int16) + details[:, :, 3] == 255)
    base_image = Image.fromarray(base)
    detail_image = Image.fromarray(details)
    control_image = Image.fromarray(controls)
    for name, image in (("basecolor.png", base_image), ("photo_details.png", detail_image),
                        ("material_controls.png", control_image), ("material_ids.png", Image.fromarray(ids)),
                        ("atlas_valid_mask.png", Image.fromarray(valid.astype(np.uint8) * 255))):
        image.save(out / name)
    mesh_data = np.load(args.source_dir / "textured_mesh.npz")
    vertices, faces, uv = (mesh_data[k] for k in ("vertices", "faces", "uv"))
    if args.normal_radius_fraction > 0:
        normals, normal_info = smooth_vertex_normals(vertices, faces,
            radius_fraction=args.normal_radius_fraction, iterations=2, return_info=True)
    else:
        normals, normal_info = welded_vertex_normals(vertices, faces, return_info=True)
    np.savez_compressed(out / "mesh.npz", vertices=vertices, faces=faces, uv=uv, normals=normals)
    pbr = trimesh.visual.material.PBRMaterial(
        name="approximate_paint_glass_rubber", baseColorTexture=base_image,
        metallicRoughnessTexture=control_image, metallicFactor=1., roughnessFactor=1.,
        alphaMode="MASK", alphaCutoff=.5, doubleSided=True)
    detail_material = trimesh.visual.material.PBRMaterial(
        name="protected_photo_details", baseColorTexture=detail_image,
        alphaMode="MASK", alphaCutoff=.5, metallicFactor=0., roughnessFactor=1., doubleSided=True)
    scene = trimesh.Scene()
    for name, material in (("material_surfaces", pbr), ("photo_details", detail_material)):
        geometry = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals,
                                  visual=trimesh.visual.TextureVisuals(uv=uv, material=material), process=False)
        scene.add_geometry(geometry, node_name=name, geom_name=name)

    def extend_materials(tree):
        for material in tree["materials"]:
            if material["name"] == "protected_photo_details":
                material.setdefault("extensions", {})["KHR_materials_unlit"] = {}
            else:
                # Clearcoat reads R from the same linear texture whose G/B are
                # roughness/metallic. No new image, texture or custom shader.
                index = material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"]
                material.setdefault("extensions", {})["KHR_materials_clearcoat"] = {
                    "clearcoatFactor": 1., "clearcoatTexture": {"index": index},
                    "clearcoatRoughnessFactor": .22}
        tree["extensionsUsed"] = list(dict.fromkeys(tree.get("extensionsUsed", []) +
                                                    ["KHR_materials_clearcoat", "KHR_materials_unlit"]))

    (out / "material_mesh.glb").write_bytes(trimesh.exchange.gltf.export_glb(
        scene, include_normals=True, tree_postprocessor=extend_materials))
    manifest = {
        "method": "semantic_material_prior_with_protected_unlit_photo_details",
        "source_hashes": source_hashes,
        "material_parameters": MATERIALS, "clearcoatRoughness": .22,
        "labels": {"0": "unknown_photo_fallback", "1": "paint", "2": "opaque_glass_approximation",
                   "3": "rubber", "4": "wheel_chrome_photo", "5": "protected_photo_detail"},
        "counts_on_valid_atlas": {str(i): int(((ids == i) & valid).sum()) for i in range(6)},
        "clean_material_fraction_of_valid_texels": float(clean[valid].mean()),
        "geometry_unchanged": True, "normal_repair": normal_info,
        "protected_rgb_exact_before_texture_filtering": bool(np.array_equal(details[~clean, :3], photo[~clean])),
        "source_unchanged": all(sha256(p) == h for p, h in source_hashes.items()),
        "limitations": [
            "Material parameters are explicit approximations, not recovered physical albedo or BRDF.",
            "Glass is opaque dark reflective material; no recovered cabin or true transmission.",
            "Protected lamps, wheels, chrome, grille and badges retain baked photo illumination/reflections.",
            "Manual semantic polygons and inferred unobserved labels are specific to these BMW images.",
            "Original geometry/camera errors, photo-detail distortion and missing surfaces remain.",
        ],
    }
    (out / "build_script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (out / "normal_script_snapshot.py").write_bytes(Path(__file__).with_name("material_normals.py").read_bytes())
    manifest["outputs"] = {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts_on_valid_atlas", "clean_material_fraction_of_valid_texels", "source_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
