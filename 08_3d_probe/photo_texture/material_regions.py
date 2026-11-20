"""Manual material regions for the eight BMW5 references, projected to v5 UVs.

The annotations are geometric polygons/ellipses, not learned segmentation.
Observed labels follow the exact source photo chosen by the existing v5 bake.
Unobserved material guesses are exported separately with lower confidence;
they are not measurements and must not be reported as recovered appearance.
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from bake_texture import project, sample, sha256
from raster import render


CLASSES = {0: "unknown", 1: "paint", 2: "glass", 3: "tire_rubber",
           4: "wheel_alloy", 5: "protected_lights_grille_badges_trim_seams"}
PALETTE = np.array([[45, 45, 45], [70, 190, 85], [45, 155, 230],
                    [165, 100, 65], [240, 185, 45], [230, 60, 160]], dtype=np.uint8)


def poly(label, points, name):
    return {"kind": "polygon", "label": label, "points": points, "name": name}


def ellipse(label, box, name):
    return {"kind": "ellipse", "label": label, "box": box, "name": name}


def line(points, name, width=6):
    return {"kind": "line", "label": 5, "points": points, "width": width, "name": name}


def box(label, rect, name):
    x0, y0, x1, y1 = rect
    return poly(label, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], name)


def glass(points, name):
    # Preserve actual chrome/rubber window outlines while allowing glass cleanup.
    return [poly(2, points, name), line(points + [points[0]], name + "_frame", 9)]


def wheel(outer, inner, name):
    return [ellipse(3, outer, name + "_tire"), ellipse(4, inner, name + "_rim")]


def mirrored(shapes):
    out = copy.deepcopy(shapes)
    for shape in out:
        if "points" in shape:
            shape["points"] = [(1535 - x, y) for x, y in shape["points"]]
        else:
            x0, y0, x1, y1 = shape["box"]
            shape["box"] = [1535 - x1, y0, 1535 - x0, y1]
    return out


def annotations():
    """Coordinates are in the original prepared 1536x1024 pixel frame."""
    front = []
    front += glass([(486, 375), (552, 251), (988, 251), (1044, 376)], "windshield")
    front += [box(5, [540, 270, 617, 318], "windshield_sticker"),
              poly(5, [(409, 468), (546, 492), (533, 534), (404, 520)], "left_lamp"),
              poly(5, [(998, 494), (1134, 472), (1137, 523), (1015, 535)], "right_lamp"),
              poly(5, [(551, 488), (600, 475), (734, 481), (771, 507), (796, 483),
                       (931, 480), (984, 504), (966, 559), (928, 602), (615, 604), (563, 569)], "grille"),
              ellipse(5, [748, 458, 797, 492], "hood_badge"),
              poly(5, [(487, 583), (574, 609), (1016, 608), (1052, 589), (1005, 719), (534, 718)], "lower_grille_plate"),
              poly(5, [(388, 587), (427, 592), (464, 687), (396, 697)], "left_intake"),
              poly(5, [(1119, 590), (1154, 600), (1151, 697), (1094, 687)], "right_intake"),
              box(5, [363, 339, 445, 410], "left_mirror"), box(5, [1093, 339, 1177, 411], "right_mirror"),
              box(3, [380, 719, 496, 811], "left_tire"), box(3, [1032, 721, 1159, 822], "right_tire"),
              line([(420, 463), (444, 409), (485, 382)], "left_hood_seam", 4),
              line([(1114, 466), (1090, 410), (1045, 383)], "right_hood_seam", 4)]
    rear = []
    rear += glass([(492, 342), (550, 245), (980, 246), (1040, 342)], "rear_window")
    rear += [poly(5, [(421, 408), (593, 417), (651, 445), (636, 465), (410, 456)], "left_tail_lamp"),
             poly(5, [(908, 419), (1104, 415), (1134, 461), (906, 466), (890, 446)], "right_tail_lamp"),
             box(5, [669, 456, 871, 531], "rear_plate"), ellipse(5, [748, 380, 790, 426], "rear_badge"),
             box(5, [510, 386, 570, 410], "left_model_text"), box(5, [950, 385, 1031, 412], "right_model_text"),
             box(5, [421, 531, 455, 615], "left_reflector"), box(5, [1080, 538, 1114, 623], "right_reflector"),
             poly(5, [(425, 638), (585, 640), (660, 675), (894, 675), (939, 637), (1117, 647),
                      (1130, 735), (416, 738)], "rear_diffuser_exhaust"),
             box(3, [376, 699, 489, 825], "left_tire"), box(3, [1036, 699, 1161, 826], "right_tire"),
             box(5, [385, 360, 454, 418], "left_mirror"), box(5, [1087, 363, 1154, 425], "right_mirror"),
             line([(488, 349), (495, 447), (511, 500), (554, 543), (985, 547), (1038, 491), (1050, 350)], "trunk_seam", 5)]
    side = []
    side += glass([(530, 443), (650, 366), (746, 336), (842, 327), (1009, 330),
                   (1110, 354), (1154, 377), (1157, 400), (1129, 414), (799, 433)], "side_windows")
    side += [poly(5, [(835, 327), (866, 328), (832, 429), (802, 432)], "B_pillar"),
             poly(5, [(1047, 339), (1068, 345), (1055, 418), (1036, 421)], "quarter_pillar"),
             poly(5, [(1145, 360), (1186, 377), (1217, 397), (1209, 408), (1147, 416), (1160, 394)], "C_pillar_badge"),
             box(5, [715, 454, 785, 484], "front_handle"), box(5, [1013, 446, 1085, 475], "rear_handle"),
             poly(5, [(553, 397), (604, 397), (628, 411), (625, 437), (572, 448), (550, 427)], "mirror"),
             poly(5, [(102, 496), (228, 477), (219, 498), (125, 531), (105, 521)], "front_lamp"),
             poly(5, [(1343, 445), (1395, 442), (1429, 454), (1432, 481), (1346, 476)], "rear_lamp"),
             poly(5, [(77, 508), (112, 506), (122, 548), (107, 629), (78, 643)], "grille_edge"),
             line([(501, 445), (499, 500), (500, 642)], "front_door_seam", 5),
             line([(825, 433), (811, 485), (800, 628)], "middle_door_seam", 5),
             line([(1152, 416), (1124, 475), (1066, 554), (1001, 628)], "rear_door_seam", 5),
             line([(423, 634), (1005, 634), (1052, 650), (437, 656)], "rocker_trim", 7)]
    side += wheel([216, 521, 422, 716], [242, 548, 394, 696], "front_wheel")
    side += wheel([1051, 523, 1267, 720], [1078, 548, 1238, 698], "rear_wheel")
    oblique = []
    oblique += glass([(504, 384), (609, 288), (694, 274), (942, 280), (967, 293), (924, 389)], "windshield")
    oblique += glass([(976, 391), (1008, 286), (1042, 281), (1144, 294), (1223, 340), (1236, 379), (1092, 400)], "side_windows")
    oblique += [poly(5, [(1047, 280), (1065, 283), (1102, 398), (1082, 400)], "B_pillar"),
                line([(1184, 317), (1201, 385)], "quarter_pillar", 7),
                poly(5, [(1220, 335), (1241, 356), (1258, 387), (1238, 395), (1228, 374)], "C_pillar_badge"),
                box(5, [1073, 430, 1115, 462], "front_handle"), box(5, [1211, 427, 1251, 456], "rear_handle"),
                poly(5, [(984, 358), (1037, 354), (1076, 373), (1076, 401), (1019, 414), (982, 401)], "mirror"),
                poly(5, [(583, 487), (776, 453), (783, 481), (749, 511), (625, 534)], "near_lamp"),
                poly(5, [(210, 464), (254, 478), (249, 514), (211, 509)], "far_lamp"),
                poly(5, [(252, 480), (300, 471), (383, 475), (400, 490), (429, 476),
                         (527, 476), (568, 494), (565, 550), (526, 580), (294, 574), (259, 549)], "grille"),
                ellipse(5, [402, 452, 440, 477], "hood_badge"),
                poly(5, [(214, 556), (311, 579), (592, 579), (617, 567), (554, 680), (251, 674)], "lower_grille_plate"),
                poly(5, [(678, 575), (725, 577), (751, 651), (673, 675), (652, 649)], "near_intake"),
                poly(5, [(185, 559), (215, 565), (236, 654), (183, 668)], "far_intake"),
                box(5, [574, 292, 655, 339], "windshield_sticker"),
                line([(1098, 409), (1120, 476), (1141, 646)], "middle_door_seam", 5),
                line([(1250, 397), (1270, 453), (1251, 510)], "rear_door_seam", 5),
                line([(926, 675), (1227, 650)], "rocker_trim", 7)]
    oblique += wheel([714, 513, 933, 768], [742, 542, 910, 739], "near_front_wheel")
    oblique += wheel([1228, 514, 1333, 723], [1247, 541, 1313, 700], "near_rear_wheel")
    oblique += [poly(3, [(223, 678), (371, 685), (362, 732), (278, 740), (238, 715)], "far_front_tire")]
    back_oblique = []
    back_oblique += glass([(774, 282), (961, 275), (1111, 313), (1160, 357), (880, 366)], "rear_window")
    back_oblique += glass([(391, 417), (455, 348), (528, 302), (577, 286), (673, 282), (725, 307), (736, 342), (716, 376), (519, 407)], "side_windows")
    back_oblique += [poly(5, [(563, 287), (584, 286), (532, 403), (510, 407)], "B_pillar"),
                     line([(689, 296), (659, 387)], "quarter_pillar", 7),
                     poly(5, [(725, 307), (746, 327), (760, 355), (751, 373), (711, 383), (736, 342)], "C_pillar_badge"),
                     box(5, [420, 439, 467, 473], "front_handle"), box(5, [579, 425, 632, 459], "rear_handle"),
                     poly(5, [(332, 381), (378, 376), (390, 399), (382, 426), (349, 429), (327, 409)], "mirror"),
                     poly(5, [(795, 445), (843, 413), (878, 413), (931, 425), (1026, 432), (1046, 453), (1021, 471), (795, 463)], "near_tail_lamp"),
                     poly(5, [(1228, 440), (1269, 427), (1328, 431), (1340, 469), (1230, 469)], "far_tail_lamp"),
                     ellipse(5, [1119, 397, 1154, 438], "rear_badge"),
                     box(5, [938, 400, 986, 419], "left_model_text"), box(5, [1251, 400, 1309, 425], "right_model_text"),
                     box(5, [1057, 460, 1204, 524], "rear_plate"),
                     poly(5, [(871, 526), (903, 535), (892, 592), (872, 598)], "near_reflector"),
                     box(5, [1332, 532, 1351, 586], "far_reflector"),
                     poly(5, [(878, 603), (1013, 602), (1046, 631), (1248, 628), (1271, 600),
                              (1345, 606), (1353, 666), (937, 687), (887, 660)], "rear_diffuser"),
                     line([(932, 428), (947, 482), (988, 536), (1281, 545), (1308, 496), (1315, 431)], "trunk_seam", 5),
                     line([(522, 415), (481, 480), (473, 650)], "middle_door_seam", 5),
                     line([(714, 384), (669, 442), (605, 526)], "rear_door_seam", 5),
                     line([(373, 655), (599, 679)], "rocker_trim", 7)]
    back_oblique += wheel([182, 523, 378, 726], [203, 550, 331, 704], "far_front_wheel")
    back_oblique += wheel([602, 527, 793, 770], [623, 555, 717, 747], "near_rear_wheel")
    back_oblique += [poly(3, [(1076, 676), (1260, 674), (1251, 713), (1219, 750), (1162, 757), (1120, 739)], "far_rear_tire")]
    out = {"front_002": front, "rear_006": rear, "side_004": side,
           "front_001": oblique, "rear_005": back_oblique,
           "side_008": mirrored(side), "front_003": mirrored(oblique), "rear_007": mirrored(back_oblique)}
    out["side_008"] += [line([(297, 427), (343, 423), (359, 433), (371, 457), (365, 467), (304, 468), (297, 427)], "fuel_cap", 5)]
    out["front_003"] += [line([(248, 397), (274, 393), (279, 406), (264, 437), (235, 440), (248, 397)], "fuel_cap", 5)]
    out["rear_007"] += [line([(788, 396), (819, 397), (841, 413), (861, 448), (846, 454), (805, 445), (788, 396)], "fuel_cap", 5)]
    return out


def make_labels(shapes, geometry_mask):
    image = Image.fromarray(geometry_mask.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for shape in shapes:
        if shape["kind"] == "polygon":
            draw.polygon([tuple(p) for p in shape["points"]], fill=shape["label"])
        elif shape["kind"] == "ellipse":
            draw.ellipse(shape["box"], fill=shape["label"])
        else:
            draw.line([tuple(p) for p in shape["points"]], fill=shape["label"], width=shape["width"], joint="curve")
    labels = np.asarray(image).copy()
    labels[~geometry_mask] = 0
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--textured-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared = json.loads((args.prepared_root / "manifest.json").read_text())
    cameras = {v["id"]: v for v in json.loads((args.fit_dir / "cameras.json").read_text())["views"]}
    baked = json.loads((args.textured_dir / "manifest.json").read_text())
    shapes_by_id = annotations()
    (output / "annotations.json").write_text(json.dumps({"coordinate_size_wh": [1536, 1024],
        "classes": CLASSES, "annotations": shapes_by_id}, indent=2))
    labels_by_id, safe_by_id, confidence_by_id, tiles = {}, {}, {}, []
    for view in prepared["views"]:
        vid = view["id"]
        rgb = np.asarray(Image.open(args.prepared_root / view["files"]["source_rgb.png"]["path"]).convert("RGB"))
        mask = np.asarray(Image.open(args.prepared_root / view["files"]["mask_geometry.png"]["path"])) > 127
        if mask.shape != (1024, 1536):
            raise ValueError("These manual annotations require prepared images at 1536x1024")
        labels = make_labels(shapes_by_id[vid], mask)
        safe = distance_transform_edt(labels == 1) >= 4
        confidence = np.zeros(mask.shape, np.float32)
        for label in range(1, 6):
            distance = distance_transform_edt(labels == label)
            confidence[labels == label] = np.clip(distance[labels == label] / 4, .55, 1.)
        labels_by_id[vid], safe_by_id[vid], confidence_by_id[vid] = labels, safe, confidence
        folder = output / vid
        folder.mkdir(exist_ok=True)
        Image.fromarray(labels).save(folder / "material_ids.png")
        Image.fromarray((safe * 255).astype(np.uint8)).save(folder / "paint_safe_mask.png")
        overlay = rgb.copy()
        overlay[mask] = (.67 * rgb[mask] + .33 * PALETTE[labels[mask]]).astype(np.uint8)
        annotated = Image.fromarray(overlay)
        ImageDraw.Draw(annotated).text((20, 20), vid + " | green paint / blue glass / brown tire / yellow alloy / pink protected", fill="red")
        annotated.save(folder / "overlay.png")
        annotated.thumbnail((768, 512))
        tiles.append(annotated)
    contact = Image.new("RGB", (1536, 512 * ((len(tiles) + 1) // 2)), "white")
    for index, tile in enumerate(tiles):
        contact.paste(tile, ((index % 2) * 768, (index // 2) * 512))
    contact.save(output / "annotation_contact_sheet.jpg", quality=95)
    if args.annotations_only:
        print("Annotations saved; UV projection skipped")
        return

    print("Projecting annotations through the exact v5 selected-source map", flush=True)
    with np.load(args.textured_dir / "textured_mesh.npz") as data:
        vertices, faces, uv = (data[k].copy() for k in ("vertices", "faces", "uv"))
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    source_ids = np.asarray(Image.open(args.textured_dir / "texture_source_ids.png"))
    observed_input = np.asarray(Image.open(args.textured_dir / "observed_texel_mask.png")) > 127
    height, width = source_ids.shape
    uv_screen = np.column_stack((uv[:, 0] * (width - 1), (1 - uv[:, 1]) * (height - 1), np.ones(len(uv))))
    _, uv_face, bary = render(uv_screen, faces, {"projection": "orthographic", "focal_px": 1, "cx": 0, "cy": 0}, (height, width))
    valid = uv_face >= 0
    if not np.array_equal(valid, source_ids != 255):
        raise RuntimeError("UV raster validity differs from v5 source-id atlas")
    ids, weights = uv_face[valid], bary[valid]
    points = (vertices[faces[ids]] * weights[:, :, None]).sum(axis=1)
    normals = (np.asarray(mesh.vertex_normals)[faces[ids]] * weights[:, :, None]).sum(axis=1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    selected_source = source_ids[valid]
    observed_ids = np.zeros(len(points), np.uint8)
    observed_conf = np.zeros(len(points), np.float32)
    safe_paint = np.zeros(len(points), bool)
    for index, source in enumerate(baked["sources"]):
        selected = (selected_source == index) & observed_input[valid]
        if not selected.any():
            continue
        vid = source["id"]
        u, v, _, _ = project(points[selected], cameras[vid]["camera"])
        observed_ids[selected] = sample(labels_by_id[vid], u, v, cv2.INTER_NEAREST)
        observed_conf[selected] = sample(confidence_by_id[vid], u, v)
        safe_paint[selected] = sample(safe_by_id[vid].astype(np.uint8), u, v, cv2.INTER_NEAREST) > 0
    safe_paint &= observed_ids == 1
    # Bilinear confidence may cross the nearest-label foreground boundary.
    # An unlabelled texel must never claim an observed semantic confidence.
    observed_conf[observed_ids == 0] = 0
    known = observed_ids > 0
    missing = ~known
    inferred = observed_ids.copy()
    inferred_conf = np.zeros(len(points), np.float32)
    print(f"Observed semantic labels: {known.mean():.1%} of valid atlas; inferring remaining material only", flush=True)
    tree = cKDTree(points[known])
    distance, nearest = tree.query(points[missing], k=1, workers=4)
    nearest_ids = observed_ids[known][nearest]
    nearest_normals = normals[known][nearest]
    normal_agreement = np.einsum("ij,ij->i", normals[missing], nearest_normals)
    p = points[missing]
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    span = hi - lo
    cabin = ((p[:, 1] > lo[1] + .43 * span[1]) &
             (p[:, 2] > lo[2] + .15 * span[2]) & (p[:, 2] < lo[2] + .81 * span[2]))
    body = (p[:, 1] > lo[1] + .12 * span[1]) & (normals[missing, 1] > -.45)
    guess = np.zeros(len(p), np.uint8)
    guess[body] = 1
    locally_consistent = (distance < .06 * span[2]) & (normal_agreement > .1)
    guess[locally_consistent & (nearest_ids == 1)] = 1
    guess[locally_consistent & (nearest_ids == 2) & cabin] = 2
    # Protected markings only propagate a very short surface distance.
    close_detail = (distance < .012 * span[2]) & (normal_agreement > .3)
    guess[close_detail & (nearest_ids == 5)] = 5
    # Wheel colors must not spread to the doors, roof or engine cover.
    wheel_region = np.zeros(len(p), bool)
    wheel_points = points[np.isin(observed_ids, [3, 4])]
    for x_sign in (-1, 1):
        for z_sign in (-1, 1):
            local = wheel_points[(wheel_points[:, 0] * x_sign > 0) & (wheel_points[:, 2] * z_sign > 0)]
            if not len(local):
                continue
            center = np.median(local, axis=0)
            yz_distance = np.linalg.norm(p[:, [1, 2]] - center[[1, 2]], axis=1)
            wheel_region |= ((p[:, 0] * x_sign > .20 * span[0]) &
                             (yz_distance < .105 * span[2]) & (p[:, 1] < lo[1] + .49 * span[1]))
    guess[wheel_region & np.isin(nearest_ids, [3, 4])] = nearest_ids[wheel_region & np.isin(nearest_ids, [3, 4])]
    inferred[missing] = guess
    confidence = .15 + .45 * np.exp(-distance / (.035 * span[2])) * np.clip(normal_agreement, 0, 1)
    confidence[guess == 0] = 0
    inferred_conf[missing] = confidence

    def atlas(values, dtype=np.uint8):
        array = np.zeros((height, width), dtype=dtype)
        array[valid] = values
        return array

    outputs = {
        "atlas_valid_mask.png": valid.astype(np.uint8) * 255,
        "observed_material_ids.png": atlas(observed_ids),
        "inferred_material_ids.png": atlas(inferred),
        "observed_confidence.png": atlas(np.rint(observed_conf * 255).astype(np.uint8)),
        "inferred_confidence.png": atlas(np.rint(inferred_conf * 255).astype(np.uint8)),
        "inferred_only_mask.png": atlas(missing.astype(np.uint8) * 255),
        "paint_safe_mask.png": atlas(safe_paint.astype(np.uint8) * 255),
        "protected_mask.png": atlas(np.isin(observed_ids, [4, 5]).astype(np.uint8) * 255),
        "inferred_protected_mask.png": atlas(np.isin(inferred, [4, 5]).astype(np.uint8) * 255),
    }
    for name, array in outputs.items():
        Image.fromarray(array).save(output / name)
    Image.fromarray(PALETTE[outputs["observed_material_ids.png"]]).save(output / "observed_material_preview.png")
    Image.fromarray(PALETTE[outputs["inferred_material_ids.png"]]).save(output / "inferred_material_preview.png")
    record = {
        "method": "manual_reference_material_annotations_following_v5_selected_source_uv_projection",
        "status": "complete_pending_visual_review", "classes": CLASSES,
        "script_sha256": sha256(__file__), "annotation_sha256": sha256(output / "annotations.json"),
        "input_paths": {"prepared_root": str(args.prepared_root.resolve()), "fit_dir": str(args.fit_dir.resolve()),
                        "textured_dir": str(args.textured_dir.resolve())},
        "input_sha256": {"prepared_manifest": sha256(args.prepared_root / "manifest.json"),
                         "cameras": sha256(args.fit_dir / "cameras.json"),
                         "textured_mesh_npz": sha256(args.textured_dir / "textured_mesh.npz"),
                         "texture_source_ids": sha256(args.textured_dir / "texture_source_ids.png")},
        "atlas_size_wh": [width, height], "atlas_valid_texels": int(valid.sum()),
        "observed_semantic_fraction_of_valid": float(known.mean()),
        "paint_safe_fraction_of_valid": float(safe_paint.mean()),
        "observed_class_counts": {CLASSES[i]: int((observed_ids == i).sum()) for i in CLASSES},
        "inferred_combined_class_counts": {CLASSES[i]: int((inferred == i).sum()) for i in CLASSES},
        "contracts": {
            "observed_material_ids": "0 remains unknown/unobserved; labels follow the actual photo selected by v5, whose visibility was checked at baking",
            "inferred_material_ids": "Contains observed labels plus low-confidence guessed labels for missing regions; NOT recovered material or appearance",
            "paint_safe_mask": "Observed paint only; eroded >=4 source pixels away from all protected/glass boundaries",
            "protected_mask": "Observed wheel alloy (4) and detail/trim/handle/seam (5) pixels; tire rubber (3) is separate for PBR cleaning, and glass interiors (2) are not photo-protected",
            "confidence": "Heuristic region trust, not a calibrated probability; observed and inferred confidence are separate, inferred <=0.60",
            "inference": "3D nearest known semantics with cabin constraints for glass, wheel-local constraints for rubber/alloy, short-distance propagation for details; exposed upper body fallback paint, underside ambiguous stays 0",
        },
        "limitations": ["Manual masks are approximate and tied to the prepared 1536x1024 BMW5 images.",
                        "Pose/shape errors can move semantic boundaries in UV space; original labels and overlays are retained for review.",
                        "Glass labels include photographed reflection/transmission; material inference does not recover original transmission or unseen roof appearance."],
        "outputs": {name: {"sha256": sha256(output / name)} for name in outputs},
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(json.dumps({"output": str(output), "observed": float(known.mean()), "safe_paint": float(safe_paint.mean()),
                      "inferred_counts": record["inferred_combined_class_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
