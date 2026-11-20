"""Render fixed-camera comparisons without changing any reconstruction outputs.

All columns share the SAME world-to-camera transform, focal length, image size,
background and relative azimuth/elevation. Texture images contain photographed
lighting/reflections; the gray mesh instead uses an explicitly documented
Lambert diagnostic shader. This is a pipeline comparison, not a single-factor
ablation or an independent identity evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from bake_texture import camera_for_angle, render_textured, sha256
from raster import render


def load_textured(root):
    with np.load(root / "textured_mesh.npz") as arrays:
        vertices, faces, uv = (arrays[k].copy() for k in ("vertices", "faces", "uv"))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh, uv, np.asarray(Image.open(root / "texture.png").convert("RGB"))


def render_gray(mesh, camera, width, height, background):
    _, ids, bary = render(mesh.vertices, mesh.faces, camera, (height, width))
    hit = ids >= 0
    face_vertices = mesh.faces[ids[hit]]
    normals = (np.asarray(mesh.vertex_normals)[face_vertices] * bary[hit, :, None]).sum(axis=1)
    normals = normals @ np.asarray(camera["R"]).T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    light = np.array([-.35, -.6, -1.])
    light /= np.linalg.norm(light)
    brightness = .30 + .70 * np.maximum(normals @ light, 0.)
    image = np.full((height, width, 3), background, np.uint8)
    image[hit] = np.clip(brightness[:, None] * [174, 182, 195], 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def font(size):
    for path in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
                 "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def compose(rows, headings, subtitles, angles, destination, title, notes):
    panel_width, panel_height = 900, 675
    header, row_label, footer = 150, 42, 125
    width = panel_width * len(headings)
    height = header + len(rows) * (panel_height + row_label) + footer
    image = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.text((24, 12), title, font=font(31), fill=(25, 30, 36))
    for col, (heading, subtitle) in enumerate(zip(headings, subtitles)):
        draw.text((col * panel_width + 24, 64), heading, font=font(29), fill=(25, 30, 36))
        draw.text((col * panel_width + 24, 107), subtitle, font=font(20), fill=(82, 88, 96))
    for row_index, row in enumerate(rows):
        y = header + row_index * (panel_height + row_label)
        draw.text((24, y + 4), f"相对方位 {angles[row_index]}° · 相同低仰角 2°", font=font(24), fill=(35, 40, 46))
        for col, render_image in enumerate(row):
            image.paste(render_image.resize((panel_width, panel_height), Image.Resampling.LANCZOS),
                        (col * panel_width, y + row_label))
    y = height - footer + 14
    for note in notes:
        draw.text((24, y), note, font=font(22), fill=(68, 74, 82))
        y += 33
    image.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-fit-dir", type=Path, required=True)
    parser.add_argument("--old-texture-dir", type=Path, required=True)
    parser.add_argument("--new-texture-dir", type=Path, required=True)
    parser.add_argument("--new-version", default="v4", help="Display/file label for the new textured candidate")
    parser.add_argument("--weighted-texture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    old_gray = trimesh.load(args.old_fit_dir / "canonical_mesh.glb", force="mesh", process=False)
    old_texture = load_textured(args.old_texture_dir)
    new_texture = load_textured(args.new_texture_dir)
    old_manifest = json.loads((args.old_texture_dir / "manifest.json").read_text())
    new_manifest = json.loads((args.new_texture_dir / "manifest.json").read_text())
    source_key = lambda m: sorted((s["id"], s["source_sha256"], s["rgb_sha256"]) for s in m["sources"])
    if source_key(old_manifest) != source_key(new_manifest) or len(source_key(new_manifest)) != 8:
        raise RuntimeError("Expected both textured candidates to use the same eight photographs")
    common_vertices = np.concatenate([old_gray.vertices, old_texture[0].vertices, new_texture[0].vertices])
    angles = [30, 90, 180]
    width, height, background = 1200, 900, (235, 237, 240)
    record = {
        "method": "same_camera_cpu_zbuffer_pipeline_comparison",
        "script_sha256": sha256(__file__),
        "raster_sha256": sha256(Path(__file__).with_name("raster.py")),
        "renderer_module_sha256": sha256(Path(__file__).with_name("bake_texture.py")),
        "relative_azimuths_deg": angles, "elevation_deg": 2,
        "individual_size_wh": [width, height], "background_rgb": list(background),
        "framing": "Identical camera R/t/focal for every column in a row; common maximum radius across all three meshes; no per-view bounding-box fit",
        "canonical_length": "Both model families were normalized to longitudinal extent 2; new proxy uses global X*0.80, Y*0.90, Z*1.0",
        "appearance": "Gray: smooth vertex normals, camera-space Lambert light [-.35,-.6,-1], ambient .30. Textures: unlit photographed colors with baked reflections and illumination.",
        "shared_eight_photographs": source_key(new_manifest),
        "limitations": [
            "Same relative angles are defined in each approximate canonical vehicle frame, not a recovered ground-truth camera coordinate system.",
            "Old and new candidates differ in geometry, proportion adjustment, fitted cameras, visibility filtering and texture selection; not a single-factor experiment.",
            "Phototexture columns reuse all eight input photos; no held-out identity or training-benefit validation.",
            "Low elevation avoids poorly observed top surfaces in this comparison; it does not establish novel high-elevation quality.",
        ],
        "inputs": {}, "render_cameras": [], "outputs": {},
    }
    for name, path in {
        "old_canonical_mesh": args.old_fit_dir / "canonical_mesh.glb",
        "old_textured_arrays": args.old_texture_dir / "textured_mesh.npz",
        "old_texture": args.old_texture_dir / "texture.png",
        "old_texture_manifest": args.old_texture_dir / "manifest.json",
        "new_textured_arrays": args.new_texture_dir / "textured_mesh.npz",
        "new_texture": args.new_texture_dir / "texture.png",
        "new_texture_manifest": args.new_texture_dir / "manifest.json",
    }.items():
        record["inputs"][name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    rows = []
    weighted_rows = []
    weighted = load_textured(args.weighted_texture_dir) if args.weighted_texture_dir else None
    for angle in angles:
        camera = camera_for_angle(common_vertices, angle, elevation=2, width=width, height=height)
        record["render_cameras"].append({"angle": angle, "camera": camera})
        gray = render_gray(old_gray, camera, width, height, background)
        old_image, _, _ = render_textured(*old_texture, camera, width, height, background)
        new_image, _, _ = render_textured(*new_texture, camera, width, height, background)
        row = [gray, old_image, new_image]
        rows.append(row)
        for name, image in zip(("old_gray", "old_texture_v2", f"mv_texture_{args.new_version}"), row):
            image.save(output / f"{name}_az_{angle:03d}_el002.png")
        if weighted:
            weighted_image, _, _ = render_textured(*weighted, camera, width, height, background)
            weighted_image.save(output / f"mv_texture_v3_az_{angle:03d}_el002.png")
            weighted_rows.append([weighted_image, new_image])
    headings = ["旧单图模型：灰模", "旧几何 + 真实照片纹理 v2", f"新多视图几何 + 真实照片纹理 {args.new_version}"]
    subtitles = ["原网格统一车体坐标；法线光照", "与灰模同一几何；八张照片投影", "统一比例调整；八张照片投影；最佳来源选择"]
    notes = ["每行使用完全相同的相机距离、焦距与相对方位，统一低仰角 2°。两个纹理列均使用同八张实拍。",
             "新版同时改变几何、比例、相机拟合与烘焙算法，不能将改善归因于单一因素。纹理含原照片光照与反射。",
             "低角度展示不代表高角度质量；仍有接缝、局部错位和未观测面的近邻补色，尚未通过独立车型保真验证。"]
    compose(rows, headings, subtitles, angles, output / "comparison_3x3.png", "3D 方案对照 · 同相机重新渲染", notes)
    for angle, row in zip(angles, rows):
        compose([row], headings, subtitles, [angle], output / f"comparison_az_{angle:03d}.png",
                "3D 方案对照 · 同相机重新渲染", notes)
    if weighted:
        record["inputs"]["weighted_texture_manifest"] = {
            "path": str((args.weighted_texture_dir / "manifest.json").resolve()),
            "sha256": sha256(args.weighted_texture_dir / "manifest.json")}
        record["inputs"]["weighted_texture"] = {
            "path": str((args.weighted_texture_dir / "texture.png").resolve()),
            "sha256": sha256(args.weighted_texture_dir / "texture.png")}
        compose(weighted_rows, ["v3：多来源加权混合", "v4：每个纹素选择最佳来源"],
                ["同一新几何、比例与相机拟合", "同八张照片；锐度与接缝之间的取舍"], angles,
                output / "texture_v3_v4_comparison.png", "纹理选择对照 · 同模型同相机",
                ["两列使用相同八张照片及同一模型家族；每行相机参数完全相同，仰角均为 2°。",
                 "最佳来源选择能减少多图平均产生的重影，也可能强化来源切换接缝；局部几何/相机错位仍保留。",
                 "只用于目视对照，尚未进行关键点重投影或独立细粒度车型保真验证。"])
    for path in sorted(output.glob("*.png")):
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            record["outputs"][path.name] = {"size_wh": list(image.size), "sha256": sha256(path)}
    (output / "manifest.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"Wrote {len(record['outputs'])} validated PNGs and manifest to {output}")


if __name__ == "__main__":
    main()
