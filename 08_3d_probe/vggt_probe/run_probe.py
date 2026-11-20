"""Phase 0 探路实验：用 VGGT 对宝马 5 系的 8 张多角度真实照片做前馈重建。

目的（可行性探路，不是正式方法模块）：
1. 检查相机位姿估计是否合理：front / side / rear 三簇之间的视角夹角应接近 90° / 180°。
2. 检查对齐点云质量，保存 PLY 供人工查看。
3. 记录深度图与运行成本（耗时 / 显存），为路线 B/C 的几何供给能力提供证据。

用法（项目根目录执行）：
python 08_3d_probe/vggt_probe/run_probe.py --output-dir 08_3d_probe/vggt_probe/outputs/bmw5_v1
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

VGGT_REPO = "/nas/wyh/3d_tools/VGGT"
sys.path.insert(0, VGGT_REPO)

from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "VGGT-1B", "model.pt")

# 固定输入顺序：角度簇内按尾号，front -> side -> rear
INPUT_FILES = [
    "front/5_2026_255841_151515_001.jpg",
    "front/5_2026_255841_151515_002.jpg",
    "front/5_2026_255841_151515_003.jpg",
    "side/5_2026_255841_151515_004.jpg",
    "side/5_2026_255841_151515_008.jpg",
    "rear/5_2026_255841_151515_005.jpg",
    "rear/5_2026_255841_151515_006.jpg",
    "rear/5_2026_255841_151515_007.jpg",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_ply(path, points, colors):
    n = len(points)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "end_header\n").encode())
        data = np.zeros(n, dtype=[("xyz", "<f4", 3), ("rgb", "u1", 3)])
        data["xyz"] = points.astype(np.float32)
        data["rgb"] = np.clip(colors, 0, 255).astype(np.uint8)
        data.tofile(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=os.path.join(PROJECT_ROOT, "test_images", "宝马5系"))
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise SystemExit(f"输出目录非空：{args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    files = [os.path.join(args.input_root, rel) for rel in INPUT_FILES]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise SystemExit(f"缺少输入文件：{missing}")
    if not os.path.isfile(args.weights):
        raise SystemExit(f"缺少模型权重：{args.weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    record = {
        "probe": "vggt_bmw5_v1",
        "device": device,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "weights": {
            "path": args.weights,
            "sha256": sha256_file(args.weights),
        },
        "inputs": [{"file": os.path.basename(f), "sha256": sha256_file(f)} for f in files],
    }

    print(f"[1/5] 加载权重 {args.weights}")
    model = VGGT()
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    print("[2/5] 预处理 8 张输入图（pad 模式，保留完整画面）")
    images = load_and_preprocess_images(files, mode="pad").to(device)  # (S,3,H,W)
    record["input_tensor_shape"] = list(images.shape)
    print("      输入张量:", tuple(images.shape))

    print("[3/5] VGGT 前馈推理")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        preds = model(images)
    torch.cuda.synchronize()
    infer_sec = time.time() - t0
    record["infer_seconds"] = round(infer_sec, 3)
    record["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
    record["prediction_keys"] = sorted(preds.keys())
    print(f"      推理 {infer_sec:.1f}s, 峰值显存 {record['peak_gpu_gib']} GiB")

    print("[4/5] 位姿分析：相机中心方位角与光轴夹角")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
    ext = extrinsic.float().squeeze(0).cpu().numpy()     # (S,3,4) world->cam [R|t]
    R, t = ext[:, :, :3], ext[:, :, 3]
    centers = -np.einsum("sij,sj->si", R, t)             # 相机中心 -R^T t
    forward = R[:, 2, :]                                 # 光轴世界方向 R^T e_z

    def angle_deg(u, v):
        c = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    # 光轴两两夹角：front vs side ≈ 90°，front vs rear ≈ 180°
    pairwise_axis = {}
    names = ["front_001", "front_002", "front_003", "side_004", "side_008",
             "rear_005", "rear_006", "rear_007"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairwise_axis[f"{names[i]}~{names[j]}"] = round(angle_deg(forward[i], forward[j]), 1)

    record["pose_analysis"] = {
        "pairwise_optical_axis_deg": pairwise_axis,
        "expected": "front~side ≈ 90, front~rear ≈ 180, 簇内 ≈ 小角度",
        "camera_centers": {n: [round(float(x), 3) for x in c] for n, c in zip(names, centers)},
        "note_azimuth": "世界系任意，方位角绝对值无意义，仅簇间距有意义",
    }
    print("      光轴夹角抽样:",
          {k: v for k, v in pairwise_axis.items() if k.startswith(("front_001~", "side_004~"))})

    print("[5/5] 导出点云 PLY / 深度图 / 报告")
    point_map = preds["world_points"].float().squeeze(0).cpu().numpy()  # (S,H,W,3) 对齐世界系
    point_conf = preds.get("world_points_conf")
    if point_conf is not None:
        point_conf = point_conf.float().squeeze(0).cpu().numpy()     # (S,H,W)
    depth = preds["depth"].float().squeeze(0).cpu().numpy()           # (S,H,W)
    if depth.ndim == 4:
        depth = depth[..., 0]
    rgb = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    pts = point_map.reshape(-1, 3)
    cols = rgb.reshape(-1, 3)
    keep = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 1e4).all(axis=1)
    if point_conf is not None:
        keep &= (point_conf.reshape(-1) > 1.5)  # 世界点置信度过滤
    step = max(1, keep.sum() // 400_000)
    save_ply(os.path.join(args.output_dir, "pointcloud_full.ply"),
             pts[keep][::step], cols[keep][::step])

    # 中心 60% 区域（近似车身）单独一份，方便查看车体本身
    S, H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    core = ((ys > H * 0.2) & (ys < H * 0.8) & (xs > W * 0.2) & (xs < W * 0.8))
    core_mask = np.broadcast_to(core, point_map.shape[:3]).reshape(-1) & keep
    save_ply(os.path.join(args.output_dir, "pointcloud_core.ply"),
             pts[core_mask][::step], cols[core_mask][::step])

    from PIL import Image
    dmin, dmax = np.percentile(depth[depth > 0], [2, 98])
    for i, name in enumerate(names):
        dn = np.clip((depth[i] - dmin) / max(dmax - dmin, 1e-6), 0, 1)
        Image.fromarray((dn * 255).astype(np.uint8)).save(
            os.path.join(args.output_dir, f"depth_{name}.png"))
        Image.fromarray(rgb[i]).save(
            os.path.join(args.output_dir, f"input_{name}.png"))

    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    print("完成，输出目录:", args.output_dir)
    print("下一步: 用 meshlab/在线 PLY 查看器检查 pointcloud_core.ply 的车体形状")


if __name__ == "__main__":
    main()
