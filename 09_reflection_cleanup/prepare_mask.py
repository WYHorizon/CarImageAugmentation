"""Prepare a conservative 2D edit mask for the reviewed BMW side_004 photo.

No RGB pixels, geometry, camera, UVs, or 08_3d_probe files are changed.
The output alpha is intended for compositing a raw edited image with the exact
original RGB. Keep original pixels wherever alpha is zero, even if a model
changes them. Manual protection marks are tied to the 1536 x 1024 source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manual_protection(size):
    """Protect recognizable panel/trim edges, never a generic high-pass map."""
    entries = [
        {"name": "visible_hood_panel_seam", "kind": "line", "width": 3,
         "points": [(190, 480), (285, 464), (437, 449), (489, 432)]},
        {"name": "front_bumper_fender_join", "kind": "line", "width": 3,
         "points": [(185, 513), (241, 529)]},
        {"name": "rear_bumper_quarter_join", "kind": "line", "width": 3,
         "points": [(1364, 477), (1316, 516), (1266, 529)]},
        {"name": "front_lower_intake_trim", "kind": "polygon",
         "points": [(117, 568), (145, 567), (163, 585), (172, 626), (147, 638), (120, 632)]},
    ]
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for entry in entries:
        if entry["kind"] == "line":
            draw.line(entry["points"], fill=255, width=entry["width"], joint="curve")
        else:
            draw.polygon(entry["points"], fill=255)
    return np.asarray(image) > 0, entries


def draw_regions(entries, size=(1536, 1024)):
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for entry in entries:
        points = [tuple(p) for p in entry.get("points", [])]
        if entry["kind"] == "line":
            draw.line(points, fill=255, width=entry["width"], joint="curve")
        elif entry["kind"] == "polygon":
            draw.polygon(points, fill=255)
        elif entry["kind"] == "ellipse":
            draw.ellipse(entry["box"], fill=255)
        else:
            raise ValueError(f"Unknown annotation shape {entry['kind']}")
    return np.asarray(image) > 0


def split_protection(labels, geometry, annotation_path):
    """Replace coarse internal boxes/lines; retain the reviewed outer details."""
    annotations = json.loads(annotation_path.read_text())["annotations"]["side_004"]
    replace_names = {"front_handle", "rear_handle", "front_door_seam",
                     "middle_door_seam", "rear_door_seam", "rocker_trim"}
    old_internal = draw_regions([a for a in annotations if a["name"] in replace_names])
    # Reconstruct outer label-5 polygons as well: overlapping chrome/window masks
    # must stay protected even where an old internal polygon overlaps them.
    outer_details = draw_regions([a for a in annotations if a["label"] == 5 and a["name"] not in replace_names])
    outer = ~geometry | np.isin(labels, [0, 2, 3, 4]) | outer_details | ((labels == 5) & ~old_internal)
    _, previous_marks = manual_protection((1536, 1024))
    outer |= draw_regions([a for a in previous_marks if a["name"] == "front_lower_intake_trim"])
    features = [
        {"name": "front_handle_refined", "kind": "polygon",
         "points": [(717, 467), (722, 461), (729, 460), (773, 457), (780, 461),
                    (784, 468), (777, 471), (773, 477), (735, 478), (728, 473)]},
        {"name": "rear_handle_refined", "kind": "polygon",
         "points": [(1018, 457), (1022, 451), (1070, 450), (1079, 453), (1083, 459),
                    (1077, 465), (1069, 470), (1032, 471), (1021, 466)]},
        {"name": "front_door_seam_refined", "kind": "line", "width": 2,
         "points": [(518, 442), (508, 451), (501, 464), (498, 480), (497, 515),
                    (498, 558), (499, 596), (503, 628)]},
        {"name": "middle_door_seam_refined", "kind": "line", "width": 2,
         "points": [(825, 431), (818, 451), (813, 470), (809, 491), (807, 520),
                    (807, 552), (808, 588), (809, 614), (812, 626)]},
        {"name": "rear_door_seam_refined", "kind": "line", "width": 2,
         "points": [(1149, 417), (1137, 441), (1122, 467), (1104, 492),
                    (1082, 522), (1058, 552), (1034, 585), (1009, 623)]},
        {"name": "rocker_upper_seam_refined", "kind": "line", "width": 2,
         "points": [(427, 630), (625, 628), (812, 627), (1008, 625), (1047, 645)]},
        {"name": "rocker_lower_edge_refined", "kind": "line", "width": 2,
         "points": [(433, 658), (642, 660), (842, 660), (1045, 658)]},
    ]
    for mark in previous_marks:
        if mark["name"] != "front_lower_intake_trim":
            features.append({**mark, "width": 2})
    feature = draw_regions(features) & geometry
    return outer, feature, features, replace_names, old_internal


def smoothstep_distance(distance, inset, feather):
    ramp = np.clip((distance - inset) / feather, 0, 1)
    return ramp * ramp * (3 - 2 * ramp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rgb", type=Path, required=True)
    parser.add_argument("--material-ids", type=Path, required=True)
    parser.add_argument("--geometry-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inset-px", type=float, default=8.)
    parser.add_argument("--feather-px", type=float, default=8.)
    parser.add_argument("--profile", choices=["uniform-v1", "split-v2"], default="split-v2")
    parser.add_argument("--feature-inset-px", type=float, default=0.)
    parser.add_argument("--feature-feather-px", type=float, default=3.)
    parser.add_argument("--reference-annotations", type=Path, default=Path(__file__).resolve().parents[1] /
                        "08_3d_probe/photo_texture/outputs/material_regions_v1/annotations.json")
    args = parser.parse_args()
    if args.inset_px < 0 or args.feather_px <= 0 or args.feature_inset_px < 0 or args.feature_feather_px <= 0:
        parser.error("Use inset-px >= 0 and feather-px > 0")
    inputs = {"source_rgb": args.source_rgb, "material_ids": args.material_ids,
              "geometry_mask": args.geometry_mask}
    if args.profile == "split-v2":
        inputs["reference_annotations"] = args.reference_annotations
    input_records = {k: {"path": str(p.resolve()), "sha256": sha256(p)} for k, p in inputs.items()}
    photo = np.asarray(Image.open(args.source_rgb).convert("RGB"))
    labels = np.asarray(Image.open(args.material_ids).convert("L"))
    geometry = np.asarray(Image.open(args.geometry_mask).convert("L")) > 127
    if photo.shape != (1024, 1536, 3) or labels.shape != photo.shape[:2] or geometry.shape != labels.shape:
        raise ValueError("These reviewed annotations require aligned 1536 x 1024 inputs")
    if not set(np.unique(labels)).issubset(set(range(6))):
        raise ValueError("Expected semantic IDs 0 unknown, 1 paint, 2 glass, 3 rubber, 4 alloy, 5 detail")
    manual, marks = manual_protection((1536, 1024))
    replacement_names = set()
    old_internal = np.zeros_like(geometry)
    if args.profile == "split-v2":
        outer, feature, marks, replacement_names, old_internal = split_protection(labels, geometry, args.reference_annotations)
        manual = feature
        eligible = ~outer & ~feature
        distance = distance_transform_edt(~outer)
        feature_distance = distance_transform_edt(~feature)
        outer_alpha = smoothstep_distance(distance, args.inset_px, args.feather_px)
        feature_alpha = smoothstep_distance(feature_distance, args.feature_inset_px, args.feature_feather_px)
        alpha_float = np.minimum(outer_alpha, feature_alpha)
    else:
        eligible = geometry & (labels == 1) & ~manual
        outer, feature = ~eligible, np.zeros_like(geometry)
        distance = distance_transform_edt(eligible)
        outer_alpha = smoothstep_distance(distance, args.inset_px, args.feather_px)
        feature_alpha = np.ones_like(outer_alpha)
        alpha_float = outer_alpha
    alpha = np.rint(alpha_float * 255).astype(np.uint8)
    support = alpha > 0
    locked = ~support
    detail = geometry & (outer | feature)
    if not support.any():
        raise ValueError("No editable paint survives the conservative inset")
    assert not np.any(support & ~eligible)
    assert np.all(alpha[locked] == 0)
    assert not np.any(support & detail)
    assert np.all(distance[support] > args.inset_px)
    assert all(sha256(p) == input_records[k]["sha256"] for k, p in inputs.items())

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"hard_support.png": support.astype(np.uint8) * 255,
              "soft_alpha.png": alpha,
              "protected_mask.png": locked.astype(np.uint8) * 255,
              "detail_protection_mask.png": detail.astype(np.uint8) * 255,
              "manual_protection_mask.png": manual.astype(np.uint8) * 255,
              "outer_protection_mask.png": outer.astype(np.uint8) * 255,
              "feature_protection_mask.png": feature.astype(np.uint8) * 255,
              "outer_soft_alpha.png": np.rint(outer_alpha * 255).astype(np.uint8),
              "feature_soft_alpha.png": np.rint(feature_alpha * 255).astype(np.uint8),
              "replaced_coarse_feature_mask.png": old_internal.astype(np.uint8) * 255,
              "refined_paint_domain.png": eligible.astype(np.uint8) * 255}
    for name, array in arrays.items():
        Image.fromarray(array).save(out / name)
    overlay = photo.astype(np.float32).copy()
    blend = alpha.astype(np.float32)[..., None] / 255 * .44
    overlay = overlay * (1 - blend) + np.array([30, 220, 145]) * blend
    overlay[manual & geometry] = .30 * overlay[manual & geometry] + .70 * np.array([235, 80, 175])
    Image.fromarray(np.rint(overlay).astype(np.uint8)).save(out / "overlay.png")
    for name, selected, tint in [("outer_protection_overlay.png", outer & geometry, [240, 155, 50]),
                                 ("feature_protection_overlay.png", feature, [235, 80, 175])]:
        diagnostic = photo.astype(np.float32).copy()
        diagnostic[selected] = .25 * diagnostic[selected] + .75 * np.asarray(tint)
        Image.fromarray(np.rint(diagnostic).astype(np.uint8)).save(out / name)
    # An adversarial synthetic edit verifies the compositing contract without
    # invoking or judging an image model. Unknown/background/details stay exact.
    simulated = 255 - photo
    a = alpha.astype(np.float32)[..., None] / 255
    composite = np.rint(photo * (1 - a) + simulated * a).astype(np.uint8)
    assert np.array_equal(composite[locked], photo[locked])
    record = {
        "status": "complete_pending_visual_review",
        "method": "split_outer_and_refined_internal_feature_distance_masks" if args.profile == "split-v2" else "uniform_reviewed_paint_inset",
        "script_sha256": sha256(__file__), "inputs": input_records,
        "size_wh": [1536, 1024],
        "parameters": {"profile": args.profile, "paint_label": 1, "inset_px": args.inset_px, "feather_px": args.feather_px,
                       "feature_inset_px": args.feature_inset_px, "feature_feather_px": args.feature_feather_px,
                       "feather": "smoothstep(distance); combined alpha=min(outer alpha, feature alpha)"},
        "manual_protection": marks,
        "replaced_coarse_annotations": sorted(replacement_names),
        "counts": {"geometry_pixels": int(geometry.sum()), "original_paint_pixels": int((geometry & (labels == 1)).sum()),
                   "eligible_after_manual_protection": int(eligible.sum()), "editable_support_pixels": int(support.sum()),
                   "fully_editable_alpha255_pixels": int((alpha == 255).sum()), "locked_original_pixels": int(locked.sum()),
                   "soft_alpha_effective_area_pixels": float(alpha.sum() / 255)},
        "contracts": {
            "soft_alpha": "8-bit L; divide by 255. Composite edited*alpha + original*(1-alpha); explicitly copy original where alpha==0.",
            "hard_support": "255 iff quantized soft alpha > 0; every pixel outside this support must remain bitwise identical to original RGB.",
            "protected_mask": "Complement of hard_support over the entire image, including background; 255 means locked original pixel.",
            "detail_protection_mask": "Union of retained outer details and refined internal feature shapes inside vehicle, before their separate inset rules.",
            "outer_protection_mask": "Background, silhouette, glass, wheels and retained external lamp/chrome/trim shapes; receives wide outer inset/feather.",
            "feature_protection_mask": "Refined handle silhouettes and narrow actual panel/door/rocker seams; receives short feature inset/feather.",
            "refined_paint_domain": "Original paint plus reclaimed portions of old coarse internal feature boxes/lines, minus refined feature shapes; this deliberately updates conservative class-5 annotation boundaries.",
            "source_coordinates": "No cropping, warping, rescaling, or inferred/unobserved material labels; this is a 2D source-photo mask.",
        },
        "checks": {"support_only_in_refined_paint_domain": True, "support_excludes_known_details": True,
                   "minimum_distance_from_outer_forbidden_pixels": float(distance[support].min()),
                   "synthetic_extreme_edit_leaves_locked_pixels_exact": True, "source_hashes_unchanged": True},
        "limitations": [
            "Material polygons remain approximate; the inset provides a conservative boundary margin.",
            "Only explicit panel/trim edges are protected; no high-frequency image threshold is used because reflections also carry high frequencies.",
            "Shading and broad styling creases inside paint must still be preserved by the image-edit prompt and judged in the resulting photograph.",
            "This mask does not make an edited photograph a physical albedo estimate and changes no 3D asset.",
        ],
        "outputs": {name: {"sha256": sha256(out / name)} for name in [*arrays, "overlay.png", "outer_protection_overlay.png", "feature_protection_overlay.png"]},
    }
    (out / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out.resolve()), "counts": record["counts"], "checks": record["checks"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
