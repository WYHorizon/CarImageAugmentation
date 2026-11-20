"""Independently verify replacement of the approved v5 GLB image only.

This verifier does not import the texture baker, re-fit cameras, load a model or
modify any original asset. A new PNG may be appended to the GLB; geometry,
normals, UV, material settings, scene structure and source selection stay fixed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import struct

import cv2
import numpy as np
from PIL import Image


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_glb(path):
    data = Path(path).read_bytes()
    require(len(data) >= 12, "Truncated GLB")
    magic, version, length = struct.unpack_from("<III", data)
    require(magic == 0x46546C67 and version == 2 and length == len(data), "Invalid GLB header")
    offset, tree, binary = 12, None, None
    while offset < len(data):
        size, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset:offset + size]
        require(len(chunk) == size, "Truncated GLB chunk")
        if kind == 0x4E4F534A:
            tree = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += size
    require(offset == len(data) and tree is not None and binary is not None, "Missing GLB chunks")
    require(len(tree["buffers"]) == 1 and "uri" not in tree["buffers"][0], "Expected one embedded buffer")
    require(tree["buffers"][0]["byteLength"] <= len(binary), "GLB buffer length exceeds BIN chunk")
    return tree, binary


def view_bytes(tree, binary, index):
    view = tree["bufferViews"][index]
    require(view.get("buffer", 0) == 0, "External GLB buffer")
    offset, length = view.get("byteOffset", 0), view["byteLength"]
    require(0 <= offset <= offset + length <= tree["buffers"][0]["byteLength"], "Invalid GLB buffer view")
    return binary[offset:offset + length]


def accessor(tree, binary, index):
    a = tree["accessors"][index]
    require("sparse" not in a and not a.get("normalized", False), "Unsupported sparse/normalized accessor")
    dims = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
    columns, dtype = dims[a["type"]], np.dtype(types[a["componentType"]])
    view = tree["bufferViews"][a["bufferView"]]
    data = view_bytes(tree, binary, a["bufferView"])
    stride = view.get("byteStride", columns * dtype.itemsize)
    offset = a.get("byteOffset", 0)
    need = offset + max(0, a["count"] - 1) * stride + columns * dtype.itemsize
    require(need <= len(data), "Accessor exceeds buffer view")
    return np.ndarray((a["count"], columns), dtype=dtype, buffer=data,
                      offset=offset, strides=(stride, dtype.itemsize)).copy()


def image_rgb(path):
    with Image.open(path) as image:
        require(image.mode == "RGB", f"Expected RGB texture: {path}")
        return np.asarray(image).copy()


def image_from_glb(tree, binary):
    require(len(tree["images"]) == 1, "Expected the original single-image GLB")
    record = tree["images"][0]
    require(record.get("mimeType") == "image/png" and "uri" not in record, "Expected embedded PNG")
    payload = view_bytes(tree, binary, record["bufferView"])
    with Image.open(io.BytesIO(payload)) as image:
        require(image.mode == "RGB", "Embedded image mode changed")
        return np.asarray(image).copy(), hashlib.sha256(payload).hexdigest()


PROFILES = {
    "mild": {"mix": .45, "true_cfg_scale": 1., "negative_required": False, "side_variant": "mild"},
    "strong": {"mix": .85, "true_cfg_scale": 2.5, "negative_required": True, "side_variant": "strong"},
}


def verify_profile_settings(compositing, generation, profile):
    """Compare reported settings with caller-selected requirements, not itself."""
    expected = PROFILES[profile]
    mix = float(compositing["mix"])
    require(mix == expected["mix"], f"{profile}: compositing mix must be {expected['mix']}")
    require(float(generation["intended_compositing_mix"]) == expected["mix"], "Generation/compositing plan differs")
    require(float(generation["true_cfg_scale"]) == expected["true_cfg_scale"], f"{profile}: wrong generation CFG")
    negative = generation.get("negative_prompt")
    if expected["negative_required"]:
        require(isinstance(negative, str) and bool(negative.strip()), "Strong generation requires an explicit negative prompt")
    require([generation["width"], generation["height"]] == [1536, 1024], "Generation dimensions changed")
    return mix


def verify_continuations(edited_root, generation):
    """Verify imported predictions against the separate saved run and files."""
    checked = {}
    for item in generation["views"]:
        provenance = item.get("continuation")
        if not provenance:
            continue
        name = item["id"]
        resume_dir = Path(provenance["resume_directory"])
        resume_manifest = resume_dir / "manifest.json"
        require(sha256(resume_manifest) == provenance["resume_manifest_sha256"],
                f"Continuation manifest changed: {name}")
        resume = json.loads(resume_manifest.read_text())
        for key in ("model_revision", "prepared_manifest_sha256", "prompt", "seed", "steps",
                    "true_cfg_scale", "negative_prompt", "width", "height", "use_kv_cache",
                    "vae_tiling", "pipeline_source_sha256", "scheduler_config", "sample_sigmas"):
            require(resume[key] == generation[key], f"Continuation setting differs: {name}/{key}")
        matching = [view for view in resume["views"] if view["id"] == name]
        require(len(matching) == 1, f"Missing or duplicate continuation record: {name}")
        source = matching[0]
        for key in ("source_path", "source_sha256", "raw_sha256", "mode"):
            require(source[key] == item[key], f"Continuation view record differs: {name}/{key}")
        raw_source = Path(provenance["raw_source_path"])
        require(raw_source.resolve() == (resume_dir / name / "raw.png").resolve(),
                f"Continuation raw source path differs: {name}")
        require(sha256(raw_source) == provenance["raw_sha256"] == item["raw_sha256"] ==
                sha256(edited_root / name / "raw.png"), f"Imported continuation raw differs: {name}")
        require(sha256(resume_dir / name / "source.png") == item["source_sha256"],
                f"Continuation input file differs: {name}")
        checked[name] = {"resume_manifest_sha256": provenance["resume_manifest_sha256"],
                         "settings_and_source_identical": True, "imported_raw_bytes_exact": True}
    top = generation.get("continuation", {})
    require(set(top.get("resumed_views", [])) == set(checked), "Continuation summary and view provenance differ")
    return checked


def verify_protected_photos(edited_root, prepared_root, profile="mild", extra_baseline=None):
    record = json.loads((edited_root / "compositing.json").read_text())
    generation = json.loads((edited_root / "manifest.json").read_text())
    continuations = verify_continuations(edited_root, generation)
    masks = Path(record["mask_directory"])
    mix = verify_profile_settings(record, generation, profile)
    generated = {item["id"]: item for item in generation["views"]}
    require(len(generated) == len(generation["views"]) == len(record["views"]) == 8,
            "Expected eight distinct generation/compositing records")
    require(set(generated) == {item["id"] for item in record["views"]}, "Generation/compositing view IDs differ")
    accepted_side = edited_root.parent / "bmw5_side_qwen_v2" / PROFILES[profile]["side_variant"]
    if profile == "strong":
        require(extra_baseline is not None, "Strong profile requires a pre-run additional baseline")
        requirement = extra_baseline["expected_profile"]
        require(requirement["name"] == "strong" and requirement["true_cfg_scale"] == 2.5 and
                requirement["compositing_mix"] == .85 and requirement["negative_prompt_required"] is True,
                "Pre-run baseline does not contain the authorized strong requirements")
        require(edited_root.resolve() == Path(requirement["expected_photo_output"]).resolve(), "Unexpected strong photo directory")
        accepted_side = Path(requirement["accepted_side_directory"])
        require(accepted_side.name == "strong" and accepted_side.parent.name == "bmw5_side_qwen_v2", "Wrong accepted strong source")
        previous_manifest_path = edited_root.parent / "bmw5_multiview_qwen_mild_v1" / "manifest.json"
        previous_record = extra_baseline["files"].get(str(previous_manifest_path.resolve()))
        require(previous_record is not None and sha256(previous_manifest_path) == previous_record["sha256"],
                "Frozen previous generation manifest is unavailable or changed")
        previous_generation = json.loads(previous_manifest_path.read_text())
        for key in ("model_revision", "prepared_manifest_sha256", "prompt", "seed", "steps", "width", "height"):
            require(generation[key] == previous_generation[key], f"Unexpected strong generation setting change: {key}")
    result = {}
    for item in record["views"]:
        name = item["id"]
        source_path, raw_path, clean_path = [edited_root / name / filename for filename in ("source.png", "raw.png", "clean.png")]
        alpha_path = masks / name / "soft_alpha.png"
        for path, key in ((source_path, "source_sha256"), (raw_path, "raw_sha256"),
                          (clean_path, "clean_sha256"), (alpha_path, "mask_sha256")):
            require(sha256(path) == item[key], f"Edited photo/compositing hash mismatch: {path}")
        require(sha256(raw_path) == generated[name]["raw_sha256"], "Raw output differs from generation record")
        require(sha256(source_path) == generated[name]["source_sha256"], "Source differs from generation record")
        if name != "side_004":
            require(generated[name]["mode"] == "Qwen_generated", "A new view was silently replaced with a reused candidate")
            require(float(generated[name].get("true_cfg_scale", generation["true_cfg_scale"])) == PROFILES[profile]["true_cfg_scale"],
                    f"Effective view CFG differs: {name}")
            if profile == "strong":
                negative = generated[name].get("negative_prompt", generation.get("negative_prompt"))
                require(isinstance(negative, str) and bool(negative.strip()), f"Missing strong negative prompt: {name}")
                previous_raw = edited_root.parent / "bmw5_multiview_qwen_mild_v1" / name / "raw.png"
                prior = extra_baseline["files"].get(str(previous_raw.resolve()))
                require(prior is not None and sha256(raw_path) != prior["sha256"],
                        f"Strong raw is identical to previous mild prediction: {name}")
        source, clean = image_rgb(source_path), image_rgb(clean_path)
        require(np.array_equal(source, image_rgb(prepared_root / name / "source_rgb.png")), "Working reference changed")
        with Image.open(raw_path) as im:
            raw = np.asarray(im.convert("RGB"))
        require(source.shape == raw.shape == clean.shape, "Clean/raw/reference dimensions differ")
        alpha = np.asarray(Image.open(alpha_path).convert("L"))
        protected = np.asarray(Image.open(masks / name / "protected_mask.png").convert("L")) > 127
        require(np.array_equal(protected, alpha == 0), "Photo protection mask does not match alpha")
        require(np.array_equal(clean[protected], source[protected]), f"Protected photo pixels changed: {name}")
        a = alpha.astype(np.float64)[..., None] / 255 * mix
        expected = np.rint(source * (1-a) + raw * a).astype(np.uint8)
        error = np.abs(clean.astype(np.int16) - expected.astype(np.int16))
        require(error.max() <= 1, "Clean photograph is not the stated protected composite")
        geometry = np.asarray(Image.open(prepared_root / name / "mask_geometry.png").convert("L")) > 127
        edge = geometry & ~cv2.erode(geometry.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        require(np.array_equal(clean[~geometry | edge], source[~geometry | edge]), "Background/silhouette band changed")
        # The mask's reviewed source semantic IDs provide an independent check
        # that glass/interior, tires and alloys are restored despite raw edits.
        mask_manifest = json.loads((masks / name / "manifest.json").read_text())
        labels_path = Path(mask_manifest["inputs"]["material_ids"]["path"])
        labels = np.asarray(Image.open(labels_path).convert("L"))
        glass_rubber_alloy = np.isin(labels, [2, 3, 4])
        require(np.array_equal(clean[glass_rubber_alloy], source[glass_rubber_alloy]),
                f"Glass, rubber or wheel photo pixels changed: {name}")
        side_exact = None
        if name == "side_004":
            require(generated[name]["mode"] in {"reused_accepted_side_candidate", "reused_reviewed_side_candidate"},
                    "Expected reviewed side_004 reuse")
            side_exact = (sha256(clean_path) == sha256(edited_root / name / "accepted_clean.png") ==
                          sha256(accepted_side / "clean.png"))
            require(side_exact and sha256(raw_path) == sha256(accepted_side / "raw.png"),
                    f"side_004 did not reuse accepted {profile} raw/clean files")
            if profile == "strong":
                require(sha256(clean_path) == requirement["accepted_side_clean_sha256"] and
                        sha256(raw_path) == requirement["accepted_side_raw_sha256"], "Accepted strong baseline hashes differ")
                side_run = json.loads((accepted_side.parent / "manifest.json").read_text())["variants"]["strong"]
                require(side_run["true_cfg_scale"] == 2.5 and bool(side_run.get("negative_prompt", "").strip()),
                        "Accepted side source was not the strong CFG/negative candidate")
                require(generation["negative_prompt"] == side_run["negative_prompt"],
                        "Multiview negative prompt differs from the reviewed strong source")
        result[name] = {"locked_pixels_exact": int(protected.sum()),
            "glass_rubber_alloy_pixels_exact": int(glass_rubber_alloy.sum()),
            "background_and_silhouette_band_exact": True,
            "maximum_compositing_error_lsb": int(error.max()),
            "accepted_side_004_file_exact": side_exact}
    return {"requested_profile": profile, "compositing_mix": mix,
        "generation_cfg_matches_requested_profile": True,
        "negative_prompt_required_and_present": PROFILES[profile]["negative_required"],
        "continuation_import_checks": continuations,
        "accepted_side_source": str(accepted_side.resolve()), "views": result,
        "scope": "Exact source-region restoration, declared generation provenance and measured 2D compositing. This does not replay/attest model inference; blend weight is not native Qwen strength."}


def verify_replay(candidate, baseline, original_texture, candidate_texture, source_ids, report_path):
    """Check frozen choices and independently resample the stored pixel map."""
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    report_path = report_path or Path(candidate_manifest["replay_dir"]) / "manifest.json"
    replay_dir = report_path.parent
    record = json.loads(report_path.read_text())
    require(sha256(report_path) == candidate_manifest["replay_manifest_sha256"], "Replay manifest hash changed")
    require(record["status"] == "ready" and record["old_photo_replay"]["exact_rgb"], "Original replay did not pass")
    require(record["old_photo_replay"]["changed_pixels"] == 0 and record["old_photo_replay"]["max_channel_error"] == 0,
            "Replay report contains nonzero original-texture errors")
    require(record["source_order"] == candidate_manifest["source_order"] == baseline["source_order"],
            "Source-view order changed")
    map_path = replay_dir / "replay_map.npz"
    require(sha256(map_path) == record["replay_map_sha256"] == candidate_manifest["replay_map_sha256"],
            "Frozen replay map hash changed")
    require(np.array_equal(image_rgb(replay_dir / "replayed_original_texture.png"), original_texture),
            "Saved original-image replay differs from v5 texture")
    with np.load(map_path) as archive:
        ids, coords, sizes, donor, flat = [archive[k].copy() for k in
            ("observed_source_ids", "source_pixels_uv", "source_sizes_wh", "donor_observed_index", "observed_atlas_flat_indices")]
    require(np.array_equal(flat, np.flatnonzero(source_ids < len(baseline["source_order"]))), "Observed texel order changed")
    require(np.array_equal(ids, source_ids.ravel()[flat]), "Saved source winners changed")
    require(donor.shape == source_ids.shape and donor.min() >= -1 and donor.max() < len(ids), "Invalid donor map")
    require(np.array_equal(donor.ravel()[flat], np.arange(len(ids))), "Observed texels no longer retain their own source samples")
    require(coords.shape == (len(ids), 2) and np.isfinite(coords).all(), "Invalid source coordinates")
    require(len(candidate_manifest["sources"]) == len(baseline["source_order"]), "Source count changed")
    original_paths, clean_paths = [], []
    for i, item in enumerate(candidate_manifest["sources"]):
        require(item["id"] == baseline["source_order"][i], "Source record order differs")
        original = Path(record["source_paths"][i])
        working, clean = Path(item["source"]), Path(item["clean"])
        require(sha256(working) == item["source_sha256"] and sha256(clean) == item["clean_sha256"], "Edited source file changed")
        require(np.array_equal(image_rgb(working), image_rgb(original)), "Working photo coordinates or pixels changed")
        original_paths.append(original)
        clean_paths.append(clean)

    def resample(paths):
        colors = np.empty((len(ids), 3), np.uint8)
        for i, path in enumerate(paths):
            picture = image_rgb(path)
            require(list(picture.shape[:2][::-1]) == sizes[i].tolist(), "A source image size changed")
            rows = np.flatnonzero(ids == i)
            for start in range(0, len(rows), 16000):
                selected = rows[start:start + 16000]
                sampled = cv2.remap(picture, coords[selected, 0, None].astype(np.float32),
                    coords[selected, 1, None].astype(np.float32), cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                colors[selected] = sampled[:, 0]
        atlas = np.zeros(donor.shape + (3,), np.uint8)
        valid = donor >= 0
        atlas[valid] = colors[donor[valid]]
        return atlas

    require(np.array_equal(resample(original_paths), original_texture), "Independent old-photo replay is not pixel-exact")
    require(np.array_equal(resample(clean_paths), candidate_texture), "Candidate texture differs from edited photos sampled through frozen map")
    return {"manifest": str(report_path.resolve()), "manifest_sha256": sha256(report_path),
        "map_sha256": sha256(map_path), "source_winners_equal_baseline": True,
        "source_images_aligned_and_hashes_match": True, "old_photo_replay_independently_exact": True,
        "new_texture_independently_matches_frozen_map": True, "sources": len(clean_paths)}


def verify_render_shapes(candidate, source):
    manifest = json.loads((candidate / "manifest.json").read_text())
    views = manifest["renders"]
    require(sorted(x["azimuth"] for x in views) == list(range(0, 360, 30)), "Expected twelve fixed-angle renders")
    results = []
    for item in views:
        filename = Path(item["file"]).name
        new_path, old_path = candidate / "renders" / filename, source / "renders" / filename
        require(sha256(new_path) == item["sha256"], "New render hash differs from its manifest")
        require(sha256(candidate / "old_renders" / filename) == sha256(old_path), "Old render copy changed")
        old, new = image_rgb(old_path), image_rgb(new_path)
        require(old.shape == new.shape, "Paired render dimensions differ")
        masks = []
        for image in (old, new):
            foreground = np.any(image != np.array([235, 237, 240], np.uint8), axis=-1).astype(np.uint8)
            contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(foreground)
            cv2.drawContours(filled, contours, -1, 1, thickness=-1)
            masks.append(filled > 0)
        require(np.array_equal(masks[0], masks[1]), f"Rendered outer silhouette changed: {filename}")
        require(np.array_equal(old[~masks[0]], new[~masks[0]]), "Rendered background changed")
        delta = np.abs(old.astype(np.int16) - new.astype(np.int16)).mean(axis=-1)
        results.append({"azimuth": item["azimuth"], "render_sha256": item["sha256"],
            "external_foreground_mask_exact": True, "external_foreground_iou": 1.,
            "background_pixels_exact": True,
            "foreground_mean_rgb_absolute_difference": float(delta[masks[0]].mean()),
            "foreground_changed_fraction": float((delta[masks[0]] > 0).mean())})
    return {"status": "passed", "render_count": len(results),
        "all_external_foreground_masks_exact": True, "all_background_pixels_exact": True,
        "method": "Known-background external foreground masks from actual paired CPU renders at identical copied cameras; old/new render hashes checked.",
        "scope": "Displayed silhouette preservation, not physical reconstruction accuracy or appearance-quality acceptance.",
        "views": results}


def verify(baseline_dir, candidate, replay_report=None, profile="mild", additional_baseline=None):
    baseline = json.loads((baseline_dir / "baseline.json").read_text())
    extra = None
    if additional_baseline:
        extra = json.loads(additional_baseline.read_text())
        require(sha256(baseline_dir / "baseline.json") == extra["original_117_baseline_sha256"],
                "Original 117-file baseline changed after the strong experiment was planned")
        for path, expected in extra["files"].items():
            require(sha256(path) == expected["sha256"], f"A protected previous experiment file changed: {path}")
    if profile == "strong":
        require(extra is not None, "Strong verification requires --additional-baseline")
        require(candidate.resolve() == Path(extra["expected_profile"]["expected_asset_output"]).resolve(),
                "Strong verification must target the new strong asset directory")
    source = Path(baseline["source_dir"])
    for path, expected in baseline["files"].items():
        require(sha256(path) == expected["sha256"], f"An original v5/source file changed: {path}")
    frozen_glb = baseline_dir / "v5_textured_mesh.glb"
    require(sha256(frozen_glb) == baseline["files"][str(source / "textured_mesh.glb")]["sha256"],
            "Frozen baseline GLB itself changed")
    old_tree, old_bin = read_glb(frozen_glb)
    new_tree, new_bin = read_glb(candidate / "textured_mesh.glb")
    require(new_bin[:len(old_bin)] == old_bin, "Original GLB binary prefix changed")
    mutable = {"buffers", "bufferViews", "images"}
    require({k: v for k, v in old_tree.items() if k not in mutable} ==
            {k: v for k, v in new_tree.items() if k not in mutable},
            "GLB scene, accessors, mesh, material, sampler or texture settings changed")
    require([{k: v for k, v in b.items() if k != "byteLength"} for b in old_tree["buffers"]] ==
            [{k: v for k, v in b.items() if k != "byteLength"} for b in new_tree["buffers"]],
            "GLB buffer metadata changed beyond byte length")
    old_images, new_images = old_tree["images"], new_tree["images"]
    require(len(old_images) == len(new_images) == 1, "GLB image count changed")
    require([{k: v for k, v in x.items() if k != "bufferView"} for x in old_images] ==
            [{k: v for k, v in x.items() if k != "bufferView"} for x in new_images],
            "GLB image metadata changed beyond its buffer view")
    old_image_views = {x["bufferView"] for x in old_images}
    new_image_views = {x["bufferView"] for x in new_images}
    non_image_count = 0
    for index, old_view in enumerate(old_tree["bufferViews"]):
        if index in old_image_views:
            continue
        require(index < len(new_tree["bufferViews"]), "Original non-image buffer view disappeared")
        new_view = new_tree["bufferViews"][index]
        require({k: v for k, v in old_view.items() if k != "byteOffset"} ==
                {k: v for k, v in new_view.items() if k != "byteOffset"}, "Non-image buffer descriptor changed")
        require(view_bytes(old_tree, old_bin, index) == view_bytes(new_tree, new_bin, index),
                f"Non-image buffer view {index} bytes changed")
        non_image_count += 1
    require(all(i in new_image_views for i in range(len(old_tree["bufferViews"]), len(new_tree["bufferViews"]))),
            "A new non-image buffer view was added")
    for index in range(len(old_tree["accessors"])):
        require(np.array_equal(accessor(old_tree, old_bin, index), accessor(new_tree, new_bin, index)),
                f"Accessor {index} values changed")
    require(all("KHR_materials_unlit" in m.get("extensions", {}) for m in new_tree["materials"]),
            "Approved unlit material was replaced")
    with np.load(baseline_dir / "v5_textured_mesh.npz") as old_npz, np.load(candidate / "textured_mesh.npz") as new_npz:
        require(set(old_npz.files) == set(new_npz.files), "NPZ fields changed")
        for key in old_npz.files:
            require(old_npz[key].dtype == new_npz[key].dtype and np.array_equal(old_npz[key], new_npz[key]),
                    f"NPZ {key} changed")
    frozen = ["texture_source_ids.png", "observed_texel_mask.png", "render_cameras.json"]
    for name in frozen:
        require(sha256(candidate / name) == sha256(baseline_dir / name), f"Frozen mapping/camera changed: {name}")
    require(sha256(candidate / "input_cameras.json") == sha256(baseline_dir / "cameras.json"), "Input-fitted cameras changed")
    preview = candidate / "preview" / "preview_mesh.npz"
    preview_equal = None
    if preview.exists():
        preview_equal = sha256(preview) == sha256(baseline_dir / "v5_preview_mesh.npz")
        require(preview_equal, "Preview mesh was regenerated or changed")
    texture = image_rgb(candidate / "texture.png")
    original_texture = image_rgb(source / "texture.png")
    embedded, embedded_sha = image_from_glb(new_tree, new_bin)
    require(texture.shape == original_texture.shape == (2048, 2048, 3), "Atlas size changed")
    require(np.array_equal(texture, embedded), "Embedded GLB PNG differs from texture.png")
    changed = np.any(texture != original_texture, axis=-1)
    require(changed.any(), "No texture pixels changed")
    ids = np.asarray(Image.open(baseline_dir / "texture_source_ids.png"))
    replay = verify_replay(candidate, baseline, original_texture, texture, ids, replay_report)
    candidate_manifest = json.loads((candidate / "manifest.json").read_text())
    edited_root = Path(candidate_manifest["sources"][0]["source"]).parent.parent
    protected_photos = verify_protected_photos(edited_root, Path(baseline["prepared_root"]), profile, extra)
    render_shapes = verify_render_shapes(candidate, source)
    return {
        "status": "texture_only_integrity_passed_pending_visual_review",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str((baseline_dir / "baseline.json").resolve()),
        "candidate": str(candidate.resolve()), "unchanged_original_files": len(baseline["files"]),
        "requested_profile": profile,
        "additional_baseline": str(additional_baseline.resolve()) if additional_baseline else None,
        "unchanged_previous_experiment_files": len(extra["files"]) if extra else 0,
        "non_image_buffer_views_byte_identical": non_image_count,
        "accessors_identical": len(old_tree["accessors"]),
        "mesh_positions_indices_uv_normals_unchanged": True,
        "glb_scene_material_sampler_texture_settings_unchanged": True,
        "npz_arrays_exact": True, "source_ids_observed_mask_turntable_cameras_exact": True,
        "unlit_material_retained": True, "preview_mesh_byte_identical_if_copied": preview_equal,
        "embedded_texture_matches_png": True, "embedded_png_sha256": embedded_sha,
        "old_binary_prefix_preserved": new_bin[:len(old_bin)] == old_bin,
        "changed_atlas_pixels": int(changed.sum()),
        "changed_observed_pixels": int((changed & (ids < len(baseline["source_order"]))).sum()),
        "changed_unobserved_fill_pixels": int((changed & (ids == 254)).sum()),
        "changed_outside_chart_pixels": int((changed & (ids == 255)).sum()),
        "changed_observed_pixels_by_frozen_source": {name: int((changed & (ids == i)).sum())
            for i, name in enumerate(baseline["source_order"])},
        "bake_settings_frozen": baseline["bake_settings"],
        "independent_frozen_map_replay": replay,
        "independent_source_photo_protection": protected_photos,
        "actual_paired_render_shape_verification": render_shapes,
        "scope": "Exact file/GLB structure preservation and image substitution; no guarantee of photorealism, physical albedo or improved vehicle identity.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="mild")
    parser.add_argument("--additional-baseline", type=Path,
                        help="Pre-run baseline protecting previous mild/accepted-side assets; required for strong")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.baseline_dir, args.candidate_dir, args.replay_report,
                    args.profile, args.additional_baseline)
    report = args.report or args.candidate_dir / "texture_only_verification.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.candidate_dir / "independent_render_shape_check.json").write_text(
        json.dumps(result["actual_paired_render_shape_verification"], ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
