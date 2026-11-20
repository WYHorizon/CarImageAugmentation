"""Make a photo-specific mask for visible background through front_002 glass.

This is a manual background-region annotation, not a whole-glass semantic mask.
All RGB outside the soft-alpha support must be copied from the original photo.
The image coordinates and source hash are fixed to the reviewed 1536x1024 image.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt


SOURCE_SHA256 = "7ed8197993348f8b53375e49886f27f1eb4ace2ab7932912b881f0ca834ecfcc"
SIZE = (1536, 1024)
CROP = (470, 240, 1060, 385)


def polygon(name, points):
    return {"name": name, "kind": "polygon", "points": points}


def annotations():
    # Coordinates were traced against the enlarged original, not mirrored from
    # another view or inherited from the coarse material class-2 polygons.
    edit = [polygon("confirmed_shutter_background_between_headrests", [
        (616, 279), (644, 278), (665, 277), (690, 280), (720, 282),
        (741, 284), (761, 284), (793, 284), (814, 284), (835, 282),
        (857, 282), (875, 280), (871, 285), (864, 290), (865, 295),
        (864, 300), (863, 305), (862, 310), (861, 315), (861, 320),
        (858, 325), (854, 330), (850, 335), (847, 340), (846, 346),
        (820, 346), (805, 344), (787, 342),
        (770, 342), (749, 343), (731, 341), (716, 340), (704, 341),
        (698, 345), (683, 346), (682, 340), (680, 335), (676, 330),
        (672, 325), (670, 320), (669, 315), (668, 310), (666, 305),
        (663, 300), (658, 295), (658, 290), (657, 287), (642, 285), (615, 286)
    ])]
    protected = [
        polygon("yellow_windshield_sticker", [
            (557, 272), (617, 271), (605, 313), (544, 314)]),
        {"name": "oval_certification_sticker", "kind": "ellipse",
         "box": [559, 261, 584, 273]},
        polygon("interior_rear_view_mirror_and_sensor", [
            (713, 251), (826, 251), (826, 279), (818, 285),
            (817, 291), (820, 298), (819, 305), (812, 309),
            (777, 311), (761, 310), (735, 308), (728, 304),
            (729, 294), (731, 288), (741, 285), (714, 280)]),
        polygon("left_seat_and_headrest", [
            (592, 287), (650, 287), (655, 290), (655, 295), (660, 300),
            (663, 305), (665, 310), (666, 315), (667, 320), (669, 325),
            (673, 330), (677, 335), (679, 340), (680, 345), (695, 375), (578, 375)]),
        polygon("right_seat_headrest_and_steering_wheel", [
            (875, 276), (890, 267), (937, 263), (968, 274),
            (987, 309), (1013, 376), (835, 376), (849, 345), (850, 340),
            (853, 335), (857, 330), (861, 325), (864, 320), (864, 315),
            (865, 310), (866, 305), (867, 300), (868, 295), (867, 290),
            (874, 285), (878, 280)]),
        polygon("white_interior_cover_and_dashboard", [
            (700, 339), (718, 340), (737, 344), (750, 343),
            (774, 341), (790, 343), (807, 346), (815, 353),
            (811, 375), (691, 375), (694, 348)]),
        polygon("lower_glass_black_border_dashboard_and_any_visible_wiper", [
            (487, 369), (1044, 369), (1052, 392), (480, 392)]),
        # This narrow bright feature is not confidently distinguishable from
        # a seat-belt/interior edge, so the first sample deliberately keeps it.
        polygon("uncertain_left_bright_slit_preserved", [
            (567, 310), (586, 310), (583, 349), (564, 349)]),
    ]
    return edit, protected


def draw_regions(regions):
    image = Image.new("L", SIZE, 0)
    draw = ImageDraw.Draw(image)
    for region in regions:
        if region["kind"] == "polygon":
            draw.polygon(region["points"], fill=255)
        elif region["kind"] == "ellipse":
            draw.ellipse(region["box"], fill=255)
        else:
            raise ValueError(region["kind"])
    return np.asarray(image) > 0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-rgb", type=Path, default=root /
                        "08_3d_probe/photo_texture/outputs/bmw5_inputs_v2/front_002/source_rgb.png")
    parser.add_argument("--output-dir", type=Path, default=root /
                        "09_reflection_cleanup/outputs/bmw5_front_glass_mask_v1")
    parser.add_argument("--inset-px", type=float, default=0.5)
    parser.add_argument("--feather-px", type=float, default=2.0)
    args = parser.parse_args()
    if args.inset_px < 0 or args.feather_px <= 0:
        parser.error("Require inset >= 0 and feather > 0")
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite a completed mask; choose a new output directory")
    source_sha = sha256(args.source_rgb)
    if source_sha != SOURCE_SHA256:
        raise ValueError("Source differs from the image reviewed for these annotations")
    source = Image.open(args.source_rgb).convert("RGB")
    if source.size != SIZE:
        raise ValueError("Annotations require the exact 1536x1024 source")
    rgb = np.asarray(source)
    edit_marks, protection_marks = annotations()
    proposed = draw_regions(edit_marks)
    features = draw_regions(protection_marks)
    eligible = proposed & ~features
    distance = distance_transform_edt(eligible)
    ramp = np.clip((distance - args.inset_px) / args.feather_px, 0, 1)
    alpha = np.rint(255 * ramp * ramp * (3 - 2 * ramp)).astype(np.uint8)
    support = alpha > 0
    protected = ~support
    assert support.any() and not np.any(support & features)
    assert np.all(alpha[~proposed] == 0)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"soft_alpha.png": alpha,
              "hard_support.png": support.astype(np.uint8) * 255,
              "protected_mask.png": protected.astype(np.uint8) * 255,
              "feature_protection_mask.png": features.astype(np.uint8) * 255,
              "proposed_background_region.png": proposed.astype(np.uint8) * 255}
    for name, array in arrays.items():
        Image.fromarray(array).save(out / name)
    blend = alpha[..., None].astype(np.float32) / 255 * .56
    overlay = rgb * (1 - blend) + np.array([30, 230, 145]) * blend
    Image.fromarray(np.rint(overlay).astype(np.uint8)).save(out / "overlay.png")
    detail_overlay = rgb.astype(np.float32).copy()
    detail_overlay[features] = .55 * detail_overlay[features] + .45 * np.array([230, 70, 170])
    detail_overlay[support] = .5 * detail_overlay[support] + .5 * np.array([30, 230, 145])
    Image.fromarray(np.rint(detail_overlay).astype(np.uint8)).crop(CROP).resize(
        (1770, 435), Image.Resampling.NEAREST).save(out / "detail_protection_overlay.png")
    Image.open(out / "overlay.png").crop(CROP).resize(
        (1770, 435), Image.Resampling.NEAREST).save(out / "overlay_windshield_crop.png")
    source.crop(CROP).resize((1770, 435), Image.Resampling.NEAREST).save(out / "source_windshield_crop.png")
    guide = source.crop(CROP).resize((1770, 435), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(guide)
    for x in range(480, 1060, 20):
        xx = (x - CROP[0]) * 3
        draw.line([(xx, 0), (xx, 435)], fill=(120, 120, 120), width=1)
        draw.text((xx + 2, 3), str(x), fill="yellow")
    for y in range(250, 385, 10):
        yy = (y - CROP[1]) * 3
        draw.line([(0, yy), (1770, yy)], fill=(120, 120, 120), width=1)
        draw.text((1, yy + 1), str(y), fill="yellow")
    guide.save(out / "coordinate_guide.png")

    # Contract check: even an adversarial edited image cannot change the source
    # outside the support when the documented composition is followed.
    a = alpha[..., None].astype(np.float32) / 255
    trial = np.rint(rgb.astype(np.float32) * (1-a) + (255-rgb).astype(np.float32) * a).astype(np.uint8)
    assert np.array_equal(trial[protected], rgb[protected])
    sample_points = {"yellow_sticker": (590, 290), "oval_sticker": (570, 267),
                     "mirror": (770, 293), "left_headrest": (630, 320),
                     "right_headrest": (920, 320), "steering_wheel": (950, 363),
                     "white_interior_cover": (755, 357), "lower_black_border": (770, 374),
                     "shutter_hole_left": (692, 289), "shutter_hole_center": (781, 316),
                     "shutter_hole_right": (832, 316)}
    samples = {name: {"xy": list(xy), "alpha": int(alpha[xy[1], xy[0]])}
               for name, xy in sample_points.items()}
    assert all(v["alpha"] == 0 for k, v in samples.items() if not k.startswith("shutter_hole"))
    assert all(v["alpha"] == 255 for k, v in samples.items() if k.startswith("shutter_hole"))
    ys, xs = np.where(support)
    record = {
        "status": "complete_pending_visual_review", "view_id": "front_002", "size": list(SIZE),
        "method": "manual_confirmed_background_only_inside_front_windshield",
        "inputs": {"source_rgb": {"path": str(args.source_rgb.resolve()), "sha256": source_sha}},
        "annotation_coordinate_size_wh": list(SIZE),
        "edit_regions": edit_marks, "protected_features": protection_marks,
        "inset_px": args.inset_px, "feather_px": args.feather_px,
        "support_pixels": int(support.sum()), "full_alpha_pixels": int((alpha == 255).sum()),
        "support_bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
        "protected_pixels": int(protected.sum()), "sample_points": samples,
        "contract": {"soft_alpha": "uint8: zero preserves source exactly; 255 permits full replacement; intermediate values feather inward",
                     "hard_support": "255 exactly where soft_alpha > 0",
                     "protected_mask": "255 exactly where soft_alpha == 0",
                     "composite": "round(source*(1-soft_alpha/255*mix) + rawRGB*(soft_alpha/255*mix)); then copy source exactly wherever soft_alpha == 0",
                     "geometric_transform": "none"},
        "checks": {"source_hash_matches_reviewed_photo": True, "support_avoids_named_features": True,
                   "adversarial_composite_protected_exact": True, "sample_checks_passed": True},
        "limitations": ["Only unambiguously visible central shutter/background is editable; this is not a full-glass cleanup mask.",
                        "The left narrow bright slit and uncertain edge pixels are deliberately preserved.",
                        "Preserving seat, mirror and interior boundaries may retain narrow original background fringes; inspect the actual composite.",
                        "No geometry, camera, UV, 3D texture, source photo, or previous experiment is modified."],
        "outputs": {name: {"path": str((out/name).resolve()), "sha256": sha256(out/name)}
                    for name in [*arrays, "overlay.png", "detail_protection_overlay.png", "overlay_windshield_crop.png",
                                 "source_windshield_crop.png", "coordinate_guide.png"]},
    }
    assert sha256(args.source_rgb) == source_sha
    (out / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ["status", "support_pixels", "full_alpha_pixels", "support_bbox_xyxy"]}, indent=2))


if __name__ == "__main__":
    main()
