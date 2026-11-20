"""Phase 0 探路实验 2：单张 front 照片经 Hunyuan3D-2.1 生成 3D 几何，多角度渲染与真实照片对照。

目的（可行性探路，不是正式方法模块）：
1. 用宝马 5 系 front_001（用户前景图）生成 3D 网格。
2. 输出 8 个方位角的软件渲染图（无 GL 依赖），重点对照 render 90°/180° 与真实 side_004 / rear_005 照片。
3. 记录耗时、显存与网格规模，评估路线 A 的可行性与细粒度损失量级。

纹理（hy3dpaint）依赖 CUDA 12 自定义算子编译，本机驱动不支持，本轮只做几何。

用法（项目根目录执行）：
CUDA_VISIBLE_DEVICES=1 /nas/wyh/.conda/envs/car-3d-hunyuan/bin/python \
  08_3d_probe/image_to_3d_probe/run_probe.py \
  --output-dir 08_3d_probe/image_to_3d_probe/outputs/bmw5_front_v1
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

HY3DSHAPE_ROOT = "/nas/wyh/3d_tools/Hunyuan3D-2.1/hy3dshape"
sys.path.insert(0, HY3DSHAPE_ROOT)

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: E402
from hy3dshape.rembg import BackgroundRemover  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "Hunyuan3D-2.1")
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "test_images", "宝马5系", "front",
                             "5_2026_255841_151515_001.jpg")

REAL_REFERENCES = {  # 渲染角 → 真实对照照片（同一辆车）
    "000": ("front/5_2026_255841_151515_002.jpg", "真实 front（_002，同簇不同张）"),
    "090": ("side/5_2026_255841_151515_004.jpg", "真实 side（_004）"),
    "180": ("rear/5_2026_255841_151515_005.jpg", "真实 rear（_005）"),
    "270": ("side/5_2026_255841_151515_008.jpg", "真实另一侧 side（_008）"),
}
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def render_views(verts, faces, az_deg, el_deg=5.0, size=1024, bg=(245, 245, 245)):
    """正交投影 + 画家算法 + Lambert 光照的软渲染，无 GL 依赖。"""
    v = verts @ rot_y(np.radians(az_deg)) @ rot_x(np.radians(el_deg))
    tri = v[faces]                                   # (F,3,3)
    z = tri[:, :, 2].mean(axis=1)
    order = np.argsort(z)                            # 远的先画
    tri = tri[order]

    # 光照：法线 · 光线（光从相机左上方来）
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.4, 0.5, 1.0])
    light /= np.linalg.norm(light)
    lam = np.clip(n @ light, 0, 1)
    base = np.array([168, 172, 180])                 # 中性车漆灰
    shade = (base[None, :] * (0.35 + 0.65 * lam)[:, None]).astype(np.uint8)

    x, y = tri[:, :, 0], tri[:, :, 1]
    span = max(x.max() - x.min(), y.max() - y.min()) * 1.05
    cx, cy = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    k = size / span
    px = ((x - cx) * k + size / 2).astype(np.int32)
    py = (size / 2 - (y - cy) * k).astype(np.int32)  # 图像 y 向下

    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    for i in range(len(tri)):
        draw.polygon(list(zip(px[i], py[i])), fill=tuple(int(c) for c in shade[i]))
    return img


def decimate(verts, faces, max_faces=40000):
    """pymeshlab 二次抽取降面，渲染用。"""
    try:
        import pymeshlab
        m = pymeshlab.Mesh(verts.astype(np.float64), faces.astype(np.int64))
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=max_faces)
        out = ms.current_mesh()
        return out.vertex_matrix().astype(np.float32), out.face_matrix().astype(np.int64)
    except Exception as e:  # 渲染降面失败不阻断探路
        print(f"[warn] 降面失败，直接渲染原始网格: {e}")
        return verts, faces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise SystemExit(f"输出目录非空：{args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"缺少模型目录：{args.model_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    record = {
        "probe": "hunyuan3d_bmw5_front_v1",
        "device": device,
        "torch": torch.__version__,
        "input": {"path": args.input, "sha256": sha256_file(args.input)},
        "params": {"steps": args.steps, "octree_resolution": args.octree_resolution,
                   "seed": args.seed, "dtype": "fp16 ckpt / pipeline 默认"},
        "model_dir": args.model_dir,
        "note": "仅几何生成；纹理 hy3dpaint 需 CUDA 12 编译，本机驱动不支持，未运行",
    }

    print(f"[1/5] 加载管线 {args.model_dir}")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_dir)
    # 注意：该管线的 .to() 无返回值（返回 None），不能链式赋值；
    # from_pretrained(device='cuda') 内部已完成设备与精度搬移。

    print("[2/5] 读取前景图并去底")
    image = Image.open(args.input)
    if "A" not in image.mode:
        rembg = BackgroundRemover()
        image = rembg(image)
    image.save(os.path.join(args.output_dir, "foreground_rgba.png"))

    print(f"[3/5] 几何生成（steps={args.steps}, octree={args.octree_resolution}, seed={args.seed}）")
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mesh = pipe(image=image, num_inference_steps=args.steps,
                octree_resolution=args.octree_resolution)[0]
    torch.cuda.synchronize()
    record["gen_seconds"] = round(time.time() - t0, 3)
    record["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    record["mesh"] = {"vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
    print(f"      生成 {record['gen_seconds']}s, 峰值显存 {record['peak_gpu_gib']} GiB, "
          f"{len(mesh.vertices)} 顶点 / {len(mesh.faces)} 面")
    mesh.export(os.path.join(args.output_dir, "mesh.glb"))
    del pipe
    torch.cuda.empty_cache()

    print("[4/5] 8 方位软渲染")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64)
    v = v - v.mean(axis=0)
    vd, fd = decimate(v, f)
    render_dir = os.path.join(args.output_dir, "renders")
    os.makedirs(render_dir, exist_ok=True)
    for az in AZIMUTHS:
        render_views(vd, fd, az).save(os.path.join(render_dir, f"render_{az:03d}.png"))
        print(f"      render_{az:03d}.png")

    print("[5/5] 真实对照与报告")
    comp_dir = os.path.join(args.output_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)
    rows = []
    for az_str, (rel, label) in REAL_REFERENCES.items():
        src = os.path.join(PROJECT_ROOT, "test_images", "宝马5系", rel)
        dst = os.path.join(comp_dir, f"real_{az_str}.jpg")
        shutil.copyfile(src, dst)
        rows.append(
            f'<div class="pair"><div><img src="../renders/render_{az_str}.png">'
            f'<div class="cap">生成 3D 渲染 {az_str}°</div></div>'
            f'<div><img src="real_{az_str}.jpg"><div class="cap">{label}</div></div></div>')
    with open(os.path.join(args.output_dir, "index.html"), "w", encoding="utf-8") as fh:
        fh.write("<html><head><meta charset='utf-8'><style>"
                 "body{font-family:sans-serif;background:#222;color:#eee}"
                 ".pair{display:flex;gap:12px;margin:18px 0}"
                 ".pair img{width:48%;height:auto}"
                 ".cap{text-align:center;margin-top:4px}</style></head><body>"
                 "<h2>Hunyuan3D-2.1 单图生成 vs 真实多角度照片（同一辆车）</h2>"
                 + "".join(rows) + "</body></html>")

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    print("完成，输出目录:", args.output_dir)
    print("下一步: 打开 index.html 对比渲染与真实照片的车身比例/轮廓")


if __name__ == "__main__":
    main()
