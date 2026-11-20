"""Replay v5's saved one-hot texture sources, then replace only its photo colors.

CPU only. Never reruns xatlas, fitting, visibility ranking, geometry processing,
or model inference. First `prepare` must reproduce the old texture byte for byte.
`replace` uses that frozen replay map and patches only the GLB's embedded image.
The original GLB binary is retained intact as a prefix, including every normal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

METHOD = Path(__file__).resolve().parent
PROJECT = METHOD.parent
PHOTO = PROJECT / "08_3d_probe/photo_texture"
DEFAULT_BASE = PHOTO / "outputs/bmw5_mv_textured_v5"
DEFAULT_FIT = PHOTO / "outputs/bmw5_mv_proportion_fit_v2"
DEFAULT_PREPARED = PHOTO / "outputs/bmw5_inputs_v2"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def new_output(path):
    path = path.resolve()
    require(path.is_relative_to(METHOD / "outputs"), "Write only into 09_reflection_cleanup/outputs")
    require(not path.exists(), f"Select a fresh output directory: {path}")
    path.mkdir(parents=True)
    return path


def load_local_raster(output, source=PHOTO / "raster.py"):
    # Import a read-only source copy from 09: Python and Numba caches must never
    # create or update __pycache__ files in the frozen 08 directory.
    snapshot = output / "raster_snapshot.py"
    shutil.copyfile(source, snapshot)
    name = "frozen_replay_raster_" + hashlib.sha256(str(output).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, snapshot)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.render


def sample(image, u, v):
    result = []
    for i in range(0, len(u), 20000):
        result.append(cv2.remap(image, u[i:i + 20000, None].astype(np.float32),
                               v[i:i + 20000, None].astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)[:, 0])
    return np.concatenate(result) if result else np.empty((0, 3), np.uint8)


def project(points, camera):
    pc = points @ np.asarray(camera["R"]).T + np.asarray(camera["t"])
    focal = np.asarray(camera["focal_px"])
    fx, fy = (float(focal), float(focal)) if focal.ndim == 0 else focal
    require(camera.get("projection", "perspective") == "perspective", "Expected frozen perspective cameras")
    return pc[:, 0] / pc[:, 2] * fx + camera["cx"], pc[:, 1] / pc[:, 2] * fy + camera["cy"]


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def compose_texture(replay, colors):
    donor = replay["donor_observed_index"]
    texture = np.zeros(donor.shape + (3,), np.uint8)
    valid = donor >= 0
    texture[valid] = colors[donor[valid]]
    return texture


def sample_sources(replay, source_paths):
    ids, coordinates = replay["observed_source_ids"], replay["source_pixels_uv"]
    colors = np.empty((len(ids), 3), np.uint8)
    for source_index, path in enumerate(source_paths):
        selected = ids == source_index
        image = read_rgb(path)
        require(list(image.shape[:2][::-1]) == replay["source_sizes_wh"][source_index].tolist(),
                f"Photo size changed: {path}")
        colors[selected] = sample(image, coordinates[selected, 0], coordinates[selected, 1])
    return colors


def check_inputs(records):
    for path, digest in records.items():
        require(sha256(path) == digest, f"Frozen input changed: {path}")


def prepare(args):
    started = time.perf_counter()
    base, fit, prepared = args.base_dir.resolve(), args.fit_dir.resolve(), args.prepared_root.resolve()
    manifest = read_json(base / "manifest.json")
    require(manifest["blend_mode"] == "best", "Only saved best-source / one-hot v5 textures are supported")
    require(sha256(fit / "cameras.json") == manifest["cameras_sha256"], "Wrong fitted cameras")
    require(sha256(fit / "canonical_mesh.glb") == manifest["mesh_sha256"], "Wrong canonical mesh")
    require(sha256(prepared / "manifest.json") == manifest["prepared_manifest_sha256"], "Wrong prepared photos")
    refs = {view["id"]: view for view in read_json(prepared / "manifest.json")["views"]}
    cameras = {view["id"]: view for view in read_json(fit / "cameras.json")["views"]}
    source_ids = [source["id"] for source in manifest["sources"]]
    sources, sizes = [], []
    frozen = {str(path): sha256(path) for path in base.rglob("*") if path.is_file()}
    for path in [fit / "cameras.json", fit / "canonical_mesh.glb", prepared / "manifest.json", PHOTO / "raster.py"]:
        frozen[str(path)] = sha256(path)
    for source in manifest["sources"]:
        ref = refs[source["id"]]
        rgb = prepared / ref["files"]["source_rgb.png"]["path"]
        mask = prepared / ref["files"]["mask_geometry.png"]["path"]
        require(sha256(rgb) == source["rgb_sha256"] and sha256(mask) == source["geometry_mask_sha256"], "Prepared source differs from the v5 bake")
        require(sha256(ref["source_path"]) == source["source_sha256"], "Original photo changed")
        for path in [rgb, mask, Path(ref["source_path"])]:
            frozen[str(path)] = sha256(path)
        sources.append(rgb)
        sizes.append(ref["working_size"])
    output = new_output(args.output_dir)
    render = load_local_raster(output)
    record = {"method": "frozen_v5_best_source_replay", "status": "preparing", "script_sha256": sha256(__file__),
              "base_dir": str(base), "fit_dir": str(fit), "prepared_root": str(prepared),
              "frozen_inputs": frozen, "source_order": source_ids, "source_paths": list(map(str, sources)),
              "weight_policy": "Saved source IDs are exact one-hot choices. No angles, visibility, source priors or weights are recomputed."}
    save_json(output / "manifest.json", record)
    arrays = np.load(base / "textured_mesh.npz")
    vertices, faces, uv = (arrays[key] for key in ("vertices", "faces", "uv"))
    size = manifest["texture_size"]
    source_map = np.asarray(Image.open(base / "texture_source_ids.png").convert("L"))
    observed_map = np.asarray(Image.open(base / "observed_texel_mask.png").convert("L")) > 127
    require(source_map.shape == (size, size), "Source-ID atlas has the wrong dimensions")
    uv_screen = np.column_stack([uv[:, 0] * (size - 1), (1 - uv[:, 1]) * (size - 1), np.ones(len(uv))])
    _, uv_faces, uv_bary = render(uv_screen, faces,
        dict(projection="orthographic", focal_px=1, cx=0, cy=0), (size, size))
    valid = uv_faces >= 0
    require(np.array_equal(valid, source_map != 255), "Saved UV chart membership did not replay exactly")
    require(np.array_equal(observed_map, source_map < len(source_ids)), "Observed mask disagrees with saved source IDs")
    point_faces = faces[uv_faces[valid]]
    points = np.einsum("ni,nij->nj", uv_bary[valid], vertices[point_faces])
    observed = observed_map[valid]
    obs_ids = source_map[valid][observed]
    coordinates = np.empty((observed.sum(), 2), np.float32)
    observed_points = points[observed]
    for i, view_id in enumerate(source_ids):
        selected = obs_ids == i
        coordinates[selected, 0], coordinates[selected, 1] = project(observed_points[selected], cameras[view_id]["camera"])
    # Reconstruct and freeze the original nearest-observed donor rule. These
    # donor indices were not logged by v5; exact old-texture replay gates reuse.
    donor_valid = np.empty(len(points), np.int32)
    donor_valid[observed] = np.arange(observed.sum(), dtype=np.int32)
    if (~observed).any():
        tree = cKDTree(observed_points)
        _, nearest = tree.query(points[~observed], k=1, workers=4)
        donor_valid[~observed] = nearest
    donor = np.full((size, size), -1, np.int32)
    donor[valid] = donor_valid
    padding_distance, nearest = distance_transform_edt(~valid, return_indices=True)
    padding = (~valid) & (padding_distance <= 12)
    donor[padding] = donor[nearest[0][padding], nearest[1][padding]]
    replay = {"observed_source_ids": obs_ids, "source_pixels_uv": coordinates,
              "source_sizes_wh": np.asarray(sizes, np.int32), "donor_observed_index": donor,
              "observed_atlas_flat_indices": np.flatnonzero(observed_map).astype(np.int32)}
    reconstructed = compose_texture(replay, sample_sources(replay, sources))
    original = read_rgb(base / "texture.png")
    difference = np.abs(reconstructed.astype(np.int16) - original.astype(np.int16))
    record["old_photo_replay"] = {"exact_rgb": bool(np.array_equal(reconstructed, original)),
        "changed_pixels": int(np.any(difference, axis=2).sum()), "max_channel_error": int(difference.max()),
        "mean_channel_error": float(difference.mean()), "observed_texels": int(observed.sum()),
        "unobserved_texels": int((~observed).sum()), "outside_chart_padding_texels": int(padding.sum())}
    Image.fromarray(reconstructed).save(output / "replayed_original_texture.png")
    save_json(output / "manifest.json", record)
    require(record["old_photo_replay"]["exact_rgb"], "Old photo replay is not pixel-exact; replacement is blocked")
    np.savez_compressed(output / "replay_map.npz", **replay)
    check_inputs(frozen)
    record.update(status="ready", seconds=round(time.perf_counter() - started, 3),
                  replay_map_sha256=sha256(output / "replay_map.npz"),
                  map_arrays={key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_hash(value)} for key, value in replay.items()},
                  frozen_inputs_unchanged=True, raster_snapshot_sha256=sha256(output / "raster_snapshot.py"))
    save_json(output / "manifest.json", record)
    print(json.dumps({"output": str(output), **record["old_photo_replay"], "seconds": record["seconds"]}, indent=2), flush=True)


def parse_glb(data):
    magic, version, length = struct.unpack_from("<III", data)
    require(magic == 0x46546C67 and version == 2 and length == len(data), "Invalid GLB header")
    chunks, offset = {}, 12
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunks[kind] = data[offset + 8:offset + 8 + length]
        offset += 8 + length
    require(set(chunks) == {0x4E4F534A, 0x004E4942}, "Expected one JSON and one BIN GLB chunk")
    return json.loads(chunks[0x4E4F534A]), chunks[0x004E4942]


def accessor_view_hashes(tree, binary):
    result = []
    for accessor in tree["accessors"]:
        view = tree["bufferViews"][accessor["bufferView"]]
        start = view.get("byteOffset", 0)
        result.append(hashlib.sha256(binary[start:start + view["byteLength"]]).hexdigest())
    return result


def patch_glb(original, texture_bytes):
    old_tree, old_binary = parse_glb(original)
    tree = copy.deepcopy(old_tree)
    require(len(tree["buffers"]) == 1 and len(tree["images"]) == 1, "Expected v5's single embedded texture")
    image = tree["images"][0]
    require(image.get("mimeType") == "image/png" and "bufferView" in image, "Expected an embedded PNG")
    image_view_index = image["bufferView"]
    require(all(accessor["bufferView"] != image_view_index for accessor in tree["accessors"]), "Image view unexpectedly stores geometry")
    offset = len(old_binary)
    require(offset % 4 == 0, "Original GLB binary is not aligned")
    new_binary = old_binary + texture_bytes
    new_binary += b"\0" * (-len(new_binary) % 4)
    tree["bufferViews"][image_view_index]["byteOffset"] = offset
    tree["bufferViews"][image_view_index]["byteLength"] = len(texture_bytes)
    tree["buffers"][0]["byteLength"] = len(new_binary)
    encoded = json.dumps(tree, separators=(",", ":"), ensure_ascii=True).encode()
    encoded += b" " * (-len(encoded) % 4)
    result = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(encoded) + 8 + len(new_binary))
    result += struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
    result += struct.pack("<II", len(new_binary), 0x004E4942) + new_binary
    loaded_tree, loaded_binary = parse_glb(result)
    require(loaded_binary[:len(old_binary)] == old_binary, "Original binary prefix changed")
    require(accessor_view_hashes(loaded_tree, loaded_binary) == accessor_view_hashes(old_tree, old_binary), "A geometry/UV/normal accessor changed")
    restored = copy.deepcopy(loaded_tree)
    restored["bufferViews"][image_view_index] = old_tree["bufferViews"][image_view_index]
    restored["buffers"] = old_tree["buffers"]
    require(restored == old_tree, "GLB metadata changed beyond the texture view and buffer length")
    return result, {"original_binary_prefix_exact": True, "accessor_buffer_views_exact": True,
                    "geometry_normals_uv_scene_material_metadata_exact": True,
                    "accessor_view_sha256": accessor_view_hashes(old_tree, old_binary),
                    "embedded_png_exact": loaded_binary[offset:offset + len(texture_bytes)] == texture_bytes,
                    "note": "The original embedded PNG remains as unreachable bytes in the preserved binary prefix."}


def render_texture(render, vertices, faces, uv, texture, camera, width, height):
    _, face_id, bary = render(vertices, faces, camera, (height, width))
    hit = face_id >= 0
    coords = (uv[faces[face_id[hit]]] * bary[hit, :, None]).sum(axis=1)
    image = np.full((height, width, 3), (235, 237, 240), np.uint8)
    image[hit] = sample(texture, coords[:, 0] * (texture.shape[1] - 1), (1 - coords[:, 1]) * (texture.shape[0] - 1))
    return Image.fromarray(image)


def three_way_side_comparison(render, vertices, faces, base, mild, output):
    """Use one geometry-derived crop without resizing or recoloring any view."""
    mild = mild.resolve()
    for filename in ["textured_mesh.npz", "render_cameras.json"]:
        require(sha256(mild / filename) == sha256(base / filename), f"Mild comparison changed frozen {filename}")
    view = next(view for view in read_json(base / "render_cameras.json") if view["azimuth"] == 90)
    _, face_id, _ = render(vertices, faces, view["camera"], (view["height"], view["width"]), return_barycentric=False)
    ys, xs = np.nonzero(face_id >= 0)
    require(len(xs) > 0, "No foreground in the frozen 90-degree camera")
    box = [max(0, int(xs.min()) - 25), max(0, int(ys.min()) - 25),
           min(view["width"], int(xs.max()) + 26), min(view["height"], int(ys.max()) + 26)]
    labels = ["原 v5", "轻度去反光", "重度去反光"]
    paths = [directory / "renders/az_090.png" for directory in [base, mild, output]]
    sources = [{"label": label, "path": str(path.resolve()), "sha256": sha256(path)} for label, path in zip(labels, paths)]
    crops = []
    for path in paths:
        with Image.open(path) as image:
            require(image.size == (view["width"], view["height"]), "Three-way comparison canvases differ")
            crops.append(image.convert("RGB").crop(box))
    width, height = crops[0].size
    label_height = 48
    panel = Image.new("RGB", (width, 3 * (height + label_height)), (235, 237, 240))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 28)
    for index, (label, image) in enumerate(zip(labels, crops)):
        top = index * (height + label_height)
        draw.text((12, top + 8), label, font=font, fill=(25, 25, 25))
        panel.paste(image, (0, top + label_height))
    filename = "comparison_three_az_090.png"
    panel.save(output / filename)
    require(all(sha256(source["path"]) == source["sha256"] for source in sources), "A comparison source changed")
    report = {"file": filename, "sha256": sha256(output / filename), "azimuth": 90,
              "crop_xyxy": box, "crop_origin": "Original v5 geometry silhouette under its frozen 90-degree camera, expanded by 25 pixels",
              "margin_pixels": 25, "resize": "none", "color_adjustment": "none", "layout": "vertical, original/mild/strong",
              "size_wh": list(panel.size), "source_size_wh": [view["width"], view["height"]], "camera": view["camera"], "sources": sources}
    save_json(output / "comparison_three_az_090.json", report)
    return report


def replace(args):
    started = time.perf_counter()
    replay_dir, edited = args.replay_dir.resolve(), args.edited_root.resolve()
    replay_record = read_json(replay_dir / "manifest.json")
    require(replay_record["status"] == "ready" and replay_record["old_photo_replay"]["exact_rgb"], "Replay did not pass")
    require(sha256(replay_dir / "replay_map.npz") == replay_record["replay_map_sha256"], "Frozen replay map changed")
    check_inputs(replay_record["frozen_inputs"])
    base = Path(replay_record["base_dir"])
    replay = np.load(replay_dir / "replay_map.npz")
    paths, source_records = [], []
    for view_id, original_path in zip(replay_record["source_order"], replay_record["source_paths"]):
        source, clean = edited / view_id / "source.png", edited / view_id / "clean.png"
        require(source.is_file() and clean.is_file(), f"Missing edited view: {view_id}")
        require(np.array_equal(read_rgb(source), read_rgb(original_path)), f"Edited source coordinates/pixels changed for {view_id}")
        with Image.open(clean) as image:
            require(image.mode == "RGB", f"Expected reviewed RGB clean image: {clean}")
        paths.append(clean)
        source_records.append({"id": view_id, "source": str(source), "source_sha256": sha256(source),
                               "clean": str(clean), "clean_sha256": sha256(clean)})
    output = new_output(args.output_dir)
    record = {"method": "frozen_v5_texture_only_replacement", "status": "running", "script_sha256": sha256(__file__),
              "variant_label": args.variant_label,
              "replay_dir": str(replay_dir), "replay_manifest_sha256": sha256(replay_dir / "manifest.json"),
              "replay_map_sha256": replay_record["replay_map_sha256"], "old_photo_replay": replay_record["old_photo_replay"],
              "base_dir": str(base), "sources": source_records, "source_order": replay_record["source_order"],
              "limitations": "Photo cleanup is generative and may alter paint appearance. Original source seams, fit errors and unobserved-neighbor fill remain. This is not physical albedo or identity validation."}
    save_json(output / "manifest.json", record)
    colors = sample_sources(replay, paths)
    texture = compose_texture(replay, colors)
    Image.fromarray(texture).save(output / "texture.png")
    glb, glb_check = patch_glb((base / "textured_mesh.glb").read_bytes(), (output / "texture.png").read_bytes())
    (output / "textured_mesh.glb").write_bytes(glb)
    unchanged = ["textured_mesh.npz", "texture_source_ids.png", "observed_texel_mask.png", "render_cameras.json"]
    for filename in unchanged:
        shutil.copyfile(base / filename, output / filename)
    shutil.copyfile(Path(replay_record["fit_dir"]) / "cameras.json", output / "input_cameras.json")
    if (base / "preview/preview_mesh.npz").is_file():
        (output / "preview").mkdir()
        shutil.copyfile(base / "preview/preview_mesh.npz", output / "preview/preview_mesh.npz")
        unchanged.append("preview/preview_mesh.npz")
    render = load_local_raster(output, replay_dir / "raster_snapshot.py")
    data = np.load(output / "textured_mesh.npz")
    vertices, faces, uv = (data[key] for key in ("vertices", "faces", "uv"))
    for folder in ["renders", "old_renders", "comparisons"]:
        (output / folder).mkdir()
    renders = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    for view in read_json(base / "render_cameras.json"):
        filename = f"az_{int(view['azimuth']):03d}.png"
        image = render_texture(render, vertices, faces, uv, texture, view["camera"], view["width"], view["height"])
        image.save(output / "renders" / filename)
        old_path = base / "renders" / filename
        shutil.copyfile(old_path, output / "old_renders" / filename)
        old = Image.open(old_path).convert("RGB")
        require(old.size == image.size, "Original/new render canvas mismatch")
        panel = Image.new("RGB", (image.width * 2, image.height + 40), (235, 237, 240))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 8), "Original v5 texture", fill=(25, 25, 25), font=font)
        draw.text((image.width + 12, 8), f"Qwen {args.variant_label} photos; frozen v5 geometry + UV + camera", fill=(25, 25, 25), font=font)
        panel.paste(old, (0, 40)); panel.paste(image, (image.width, 40))
        panel.save(output / "comparisons" / filename)
        renders.append({"azimuth": view["azimuth"], "file": "renders/" + filename,
                        "sha256": sha256(output / "renders" / filename)})
        print(f"Rendered {filename}", flush=True)
    require(len(renders) == 12, "Expected the original twelve v5 cameras")
    if args.variant_label == "strong":
        record["three_way_comparison"] = three_way_side_comparison(render, vertices, faces, base, args.mild_reference, output)
    check_inputs(replay_record["frozen_inputs"])
    for source in source_records:
        require(sha256(source["source"]) == source["source_sha256"] and sha256(source["clean"]) == source["clean_sha256"], "Edited input changed during baking")
    for filename in unchanged:
        require(sha256(output / filename) == sha256(base / filename), f"Frozen copied asset changed: {filename}")
    difference = np.abs(texture.astype(np.int16) - read_rgb(base / "texture.png").astype(np.int16))
    record.update(status="complete_pending_visual_review", seconds=round(time.perf_counter() - started, 3),
                  glb_checks=glb_check, original_inputs_unchanged=True,
                  unchanged_copies={filename: sha256(output / filename) for filename in unchanged},
                  texture_sha256=sha256(output / "texture.png"), glb_sha256=sha256(output / "textured_mesh.glb"),
                  texture_changed_pixels=int(np.any(difference, axis=2).sum()), texture_mean_channel_change=float(difference.mean()),
                  renders=renders)
    save_json(output / "manifest.json", record)
    save_json(output / "verification.json", {"status": "passed", "source_choice_and_donor_map_frozen": True,
        "old_texture_replay_exact": True, "source_images_aligned": True, "original_inputs_unchanged": True,
        "copied_geometry_uv_masks_cameras_exact": True, "render_count": len(renders), **glb_check,
        "scope": "Exact asset/provenance invariants. This does not establish improved appearance or identity preservation."})
    print(json.dumps({"output": str(output), "seconds": record["seconds"], "texture_changed_pixels": record["texture_changed_pixels"], **glb_check}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare_parser = modes.add_parser("prepare", help="Build the immutable sample/donor map and pass exact old-photo replay")
    prepare_parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    prepare_parser.add_argument("--fit-dir", type=Path, default=DEFAULT_FIT)
    prepare_parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    replace_parser = modes.add_parser("replace", help="Sample reviewed edited photos through the saved replay map")
    replace_parser.add_argument("--replay-dir", type=Path, required=True)
    replace_parser.add_argument("--edited-root", type=Path, required=True)
    replace_parser.add_argument("--output-dir", type=Path, required=True)
    replace_parser.add_argument("--variant-label", choices=["mild", "strong"], default="mild",
                                help="Comparison caption/manifest label only; does not alter photo colors or blend weights")
    replace_parser.add_argument("--mild-reference", type=Path, default=METHOD / "outputs/bmw5_v5_qwen_mild_texture_v1",
                                help="For strong only: frozen mild asset used in a uniformly cropped three-way side comparison")
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else replace)(args)


if __name__ == "__main__":
    main()
