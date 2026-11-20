"""Independent integrity, material-export and browser-response checks.

These checks establish that the material experiment implements its declared
priors, preserves protected photo texels and responds to lighting. They do not
establish recovered physical reflectance, geometry accuracy or vehicle identity.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from material_normals import normal_seam_statistics
from verify_outputs import accessor, check_hash, read_glb, read_png, require, sha256


def embedded_image(tree, binary, texture_index, mode):
    texture = tree["textures"][texture_index]
    image = tree["images"][texture["source"]]
    require("uri" not in image and image.get("mimeType") == "image/png", "Expected embedded PNG material texture")
    view = tree["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    return np.asarray(Image.open(io.BytesIO(binary[start:start + view["byteLength"]])).convert(mode))


def verify_browser(directory):
    report_path = directory / "browser_verification.json"
    browser = json.loads(report_path.read_text())
    require(not browser.get("errors"), "Browser reported JavaScript or rendering errors")
    records = browser.get("lightChecks", [])
    require(len(records) == 4, "Expected photo/PBR captures under two environment rotations")
    images, states = {}, {}
    for record in records:
        state = record["state"]
        key = (state["mode"], int(state["environmentYaw"]))
        require(key not in images, "Duplicate light-check state")
        path = directory / record["file"]
        check_hash(path, record["sha256"])
        images[key] = np.asarray(Image.open(path).convert("RGB"))
        states[key] = state
    for mode in ("photo", "pbr"):
        require((mode, 0) in images and (mode, 90) in images, "Missing requested environment rotation")
        a = {k: v for k, v in states[(mode, 0)].items() if k != "environmentYaw"}
        b = {k: v for k, v in states[(mode, 90)].items() if k != "environmentYaw"}
        require(a == b, "Light check changed camera, exposure or another declared state")
    photo_same = np.array_equal(images[("photo", 0)], images[("photo", 90)])
    require(photo_same, "Original unlit photo render changes when only environment rotation changes")
    a = images[("pbr", 0)].astype(np.float32)
    b = images[("pbr", 90)].astype(np.float32)
    require(a.shape == b.shape, "PBR light-check dimensions differ")
    delta = np.abs(a - b).mean(axis=-1)
    require(float(delta.max()) > 5 and int((delta > 2).sum()) > 100,
            "PBR lighting change is absent or too small to verify")
    result = {
        "directory": str(directory.resolve()),
        "fixed_camera_and_declared_state": True,
        "photo_mode_pixel_identical": True,
        "pbr_changed_pixels_rgb_mae_gt2": int((delta > 2).sum()),
        "pbr_full_frame_mean_rgb_absolute_difference": float(delta.mean()),
        "pbr_max_pixel_rgb_absolute_difference": float(delta.max()),
        "scope": "Two actual browser captures at fixed camera; visual review must distinguish reflections from background or ground changes.",
    }
    mask_path = directory / "lightcheck_object_mask.png"
    if mask_path.exists():
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        require(mask.shape == delta.shape and mask.any(), "Invalid light-check object mask")
        require(int((delta[mask] > 2).sum()) > 100, "Only non-object pixels changed under lighting")
        result["object_mask_sha256"] = sha256(mask_path)
        result["pbr_object_mean_rgb_absolute_difference"] = float(delta[mask].mean())
        result["pbr_object_changed_pixels_rgb_mae_gt2"] = int((delta[mask] > 2).sum())
    return result


def verify(source, output, baseline_path, protected_mask_path=None, browser_directory=None):
    baseline = json.loads(baseline_path.read_text())
    require(Path(baseline["source_dir"]).resolve() == source.resolve(), "Baseline refers to another source asset")
    for name, file_record in baseline["files"].items():
        check_hash(source / name, file_record["sha256"])
    manifest = json.loads((output / "manifest.json").read_text())
    for path, digest in manifest["source_hashes"].items():
        check_hash(Path(path), digest)
    for name, digest in manifest["outputs"].items():
        check_hash(output / name, digest)
    source_tree, _ = read_glb(source / "textured_mesh.glb")
    require(all("KHR_materials_unlit" in m.get("extensions", {}) for m in source_tree["materials"]),
            "Original source asset is no longer unlit")

    with np.load(source / "textured_mesh.npz", allow_pickle=False) as original, np.load(output / "mesh.npz", allow_pickle=False) as current:
        vertices, faces, uv, normals = [current[k].copy() for k in ("vertices", "faces", "uv", "normals")]
        for key in ("vertices", "faces", "uv"):
            require(np.array_equal(original[key], current[key]), f"Material conversion changed {key}")
    require(normals.shape == vertices.shape and np.isfinite(normals).all(), "Nonfinite or incorrectly shaped normals")
    require(np.max(np.abs(np.linalg.norm(normals.astype(np.float64), axis=1) - 1)) < 1e-6, "Normals are not unit length")
    seams = normal_seam_statistics(vertices, normals)
    require(seams["same_position_normals_bitwise_equal"], "UV seams still have inconsistent normals")

    photo = read_png(source / "texture.png", mode="RGB")
    size = [photo.shape[1], photo.shape[0]]
    base = read_png(output / "basecolor.png", size, "RGBA")
    details = read_png(output / "photo_details.png", size, "RGBA")
    controls = read_png(output / "material_controls.png", size, "RGB")
    ids = read_png(output / "material_ids.png", size, "L")
    valid = read_png(output / "atlas_valid_mask.png", size, "L") > 127
    require(valid.any(), "No valid atlas texels")
    require(set(np.unique(ids)).issubset({0, 1, 2, 3, 4, 5, 255}), "Unknown material label")
    clean = np.isin(ids, [1, 2, 3])
    protected = (~clean) & valid
    require(np.array_equal(base[..., 3].astype(np.int16) + details[..., 3], np.full(ids.shape, 255)),
            "Material/photo alpha masks are not complementary")
    require(np.all(base[clean, 3] == 255) and np.all(base[~clean, 3] == 0), "PBR alpha does not follow material labels")
    require(np.array_equal(details[protected, :3], photo[protected]), "Protected/fallback photo texels changed")
    require(np.all(details[protected, 3] == 255), "Protected photo texels are not fully visible")
    if protected_mask_path:
        explicit_protected = read_png(protected_mask_path, size, "L") > 127
        require(not np.any(explicit_protected & valid & clean), "A protected source region was overwritten with a material prior")
    counts = {str(i): int(((ids == i) & valid).sum()) for i in range(6)}
    require(counts == manifest["counts_on_valid_atlas"], "Material-count manifest does not match masks")
    for key, prior in manifest["material_parameters"].items():
        selected = (ids == int(key)) & valid
        require(selected.any(), f"Declared material {key} has no valid texels")
        expected_control = np.rint(np.array([prior["clearcoat"], prior["roughness"], prior["metalness"]]) * 255).astype(np.uint8)
        require(np.all(base[selected, :3] == np.asarray(prior["rgb"])), f"Material {key} base color differs from declared prior")
        require(np.all(controls[selected] == expected_control), f"Material {key} control channels differ from declared prior")

    tree, binary = read_glb(output / "material_mesh.glb")
    primitives = [p for mesh in tree["meshes"] for p in mesh["primitives"]]
    require(len(primitives) == 2, "Expected two complementary material layers")
    expected_uv = uv.copy()
    expected_uv[:, 1] = 1 - expected_uv[:, 1]
    material_roles = []
    for primitive in primitives:
        require(primitive.get("mode", 4) == 4, "GLB primitive is not triangles")
        attributes = primitive["attributes"]
        require(np.array_equal(accessor(tree, binary, attributes["POSITION"]), vertices), "GLB position accessor changed geometry")
        require(np.array_equal(accessor(tree, binary, primitive["indices"]).reshape(-1, 3), faces), "GLB face accessor differs from NPZ")
        require(np.allclose(accessor(tree, binary, attributes["TEXCOORD_0"]), expected_uv, rtol=0, atol=1e-7), "GLB UV origin or values differ")
        require(np.allclose(accessor(tree, binary, attributes["NORMAL"]), normals, rtol=0, atol=1e-7), "GLB normals differ from validated normals")
        material = tree["materials"][primitive["material"]]
        require(material.get("alphaMode") == "MASK" and material.get("alphaCutoff", .5) == .5,
                "Layer is not a complementary alpha-cutoff material")
        require(material.get("doubleSided") is True, "Expected double-sided layers")
        pbr = material["pbrMetallicRoughness"]
        require(np.allclose(pbr.get("baseColorFactor", [1, 1, 1, 1]), [1, 1, 1, 1]), "Material color/alpha factor alters texture")
        embedded_base = embedded_image(tree, binary, pbr["baseColorTexture"]["index"], "RGBA")
        extensions = material.get("extensions", {})
        if "KHR_materials_unlit" in extensions:
            material_roles.append("photo")
            require("KHR_materials_clearcoat" not in extensions, "Unlit photo layer incorrectly also declares clearcoat")
            require(np.array_equal(embedded_base, details), "Embedded photo details differ from PNG")
        else:
            material_roles.append("pbr")
            require(np.array_equal(embedded_base, base), "Embedded PBR base color differs from PNG")
            require(pbr.get("roughnessFactor", 1) == 1 and pbr.get("metallicFactor", 1) == 1,
                    "Factors unexpectedly scale roughness/metallic texture channels")
            control_index = pbr["metallicRoughnessTexture"]["index"]
            require(np.array_equal(embedded_image(tree, binary, control_index, "RGB"), controls), "Embedded G-roughness/B-metallic texture differs")
            coat = extensions["KHR_materials_clearcoat"]
            require(coat["clearcoatTexture"]["index"] == control_index and coat.get("clearcoatFactor") == 1,
                    "Clearcoat does not read the shared R control channel")
            require(abs(coat["clearcoatRoughnessFactor"] - manifest["clearcoatRoughness"]) < 1e-8,
                    "Clearcoat roughness differs from declared prior")
    require(sorted(material_roles) == ["pbr", "photo"], "Missing PBR or protected-photo layer")
    require({"KHR_materials_clearcoat", "KHR_materials_unlit"}.issubset(tree.get("extensionsUsed", [])),
            "GLB extensionsUsed does not declare its material extensions")
    for node in tree.get("nodes", []):
        require(np.allclose(np.asarray(node.get("matrix", np.eye(4))).reshape(4, 4), np.eye(4)), "GLB node transforms geometry")
        for key, default in (("translation", [0, 0, 0]), ("rotation", [0, 0, 0, 1]), ("scale", [1, 1, 1])):
            require(np.allclose(node.get(key, default), default), f"Unexpected node {key}")

    result = {
        "status": "asset_checks_passed_render_pending",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "source_baseline_files_unchanged": len(baseline["files"]),
        "source_and_output_hashes_match": True,
        "vertices_faces_uv_unchanged": True, "normal_seams": seams,
        "protected_photo_pixels_exact": int(protected.sum()),
        "photo_fallback_and_protected_pixels_exact": int(protected.sum()),
        "protected_ids_4_5_pixels_exact": int((np.isin(ids, [4, 5]) & valid).sum()),
        "photo_preservation_count_note": "The combined photo count includes unknown/fallback ID 0 as well as protected IDs 4 and 5.",
        "explicit_protected_mask_checked": str(protected_mask_path) if protected_mask_path else None,
        "complementary_alpha_layers": True,
        "material_counts_on_valid_atlas": counts,
        "glb_geometry_uv_normals_and_embedded_textures_match": True,
        "pbr_G_roughness_B_metallic_R_clearcoat_semantics": True,
        "photo_layer_unlit_and_original_source_unlit": True,
        "scope": "Artifact consistency, protected texel preservation and declared material semantics; not physical inverse-rendering or identity validation.",
    }
    if browser_directory:
        result["browser_lighting"] = verify_browser(browser_directory)
        result["status"] = "asset_and_browser_checks_passed_pending_visual_review"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).parent / "outputs/materials_review_v1/source_baseline.json")
    parser.add_argument("--protected-mask", type=Path)
    parser.add_argument("--browser-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report_path = args.report or args.output_dir / "material_verification.json"
    result = verify(args.source_dir, args.output_dir, args.baseline, args.protected_mask, args.browser_dir)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
