# 宝马多视图建模与早期 3D 探路

## 当前发布模型

[宝马 5 系模型、局部修复与导图](bmw5/README.md) · [下载 GLB](bmw5/bmw5.glb) · [街景融合](../07_background_replacement/bmw_street/README.md)

当前资产沿接缝修复 v3 继续调整了日光玻璃、车顶、前脸照片细节和后备厢盖。以下按实验时间保留早期路线与结果。

## 2026-11-17：反光改进实验 v6

针对照片倒影固定在车身上的问题，新增 [分部件材质实验](photo_texture/MATERIALS.md)。
此轮材质实验结果为 `photo_texture/outputs/bmw5_material_v6_smooth/material_mesh.glb`：
保持 v5 几何和 UV，清理车漆/玻璃的固定照片外观，使用近似 PBR 材质计算动态反光，
保留灯组、格栅、轮毂等照片细节，并修复、平滑着色法线。
完整 GLB 与轻量交互预览均已实际浏览器验证；真实材质、内饰和几何精度尚未恢复。
旧 v5 与初版 v6 保留为对照。

## 2026-11-17：新增可连续渲染的照片贴图版本

已新增 [四张真实视图重新生成几何](multiview_probe/README.md) 和
[相机拟合、真实照片纹理烘焙与导出](photo_texture/README.md)。当前候选是
`photo_texture/outputs/bmw5_mv_textured_v5/textured_mesh.glb`：四图生成车体，
统一比例调整，用八张实拍照片形成固定 UV 纹理，再由同一资产渲染连续角度。
原始输入和之前的输出均保留，未覆盖。

这版的低机位水平环绕比灰模更接近原车外观；玻璃、车顶、反光、接缝和
未观测区域仍有缺陷。轮廓分数是输入拟合度，不是独立几何精度、车型保真或
分类训练收益。下方保留早期探路记录，路线判断与本轮对照结果见上述新文档。

状态：Phase 0 已完成（2026-11-16）。本目录是**探路实验**，不是正式方法模块；结论用于决定 3D 主线（路线 A/B/C）选型。

## 背景

现有 7 个增强方法全部是 2D 的，视角（viewpoint）是唯一补不上的增强维度。3D 探路回答两个问题：

1. **探路 1（多视图重建）**：dongchedi 每辆车自带 front/side/rear 真实照片，能否用前馈重建（VGGT）得到真车的几何/位姿？→ 支撑路线 C（重建真车渲染新视角）和路线 B（3D 作生成控制信号）。
2. **探路 2（单图生成 3D）**：一张 front 照片经 Hunyuan3D-2.1 生成 3D 几何，多角度渲染与同车真实照片对照，细粒度损失多大？→ 支撑路线 A（3D 资产 + 任意渲染）。

试验车：`test_images/宝马5系/`（黑色车牌号 `5_2026_255841_151515`，8 张 4464×2976 影棚目录照片）。

## 探路 1 结果（2026-11-16，已完成）

脚本：[vggt_probe/run_probe.py](vggt_probe/run_probe.py)；输出：`vggt_probe/outputs/bmw5_v1/`。

**结论：VGGT 对这类影棚目录照片的联合配准失败；单视图单目深度可用。**

- 8 图光轴两两夹角全部 2°–10°（front~side 应 ≈90°、front~rear 应 ≈180°），相机中心聚成一团（跨度 0.28），对齐点云俯视图无车形。
- 阳性对照：同一脚本跑 VGGT 官方 `examples/llff_fern`（20 张真实多视图）位姿正常 → 排除代码 bug。
- 单目深度图（每图独立）质量可用：车身/背景/地板分层清晰，见 `depth_*.png`。
- 原因：影棚目录照片每角度独立打光/修图（可能合成背景），无真实视差；白车+浅背景纹理弱；~90° 基线重叠少。
- 成本：8 图推理 2.1 秒，峰值显存 9.1 GiB（A800）。

对路线的影响：路线 C 单独做主线基础薄弱；**补救路径**——用目录自带的角度标签作位姿先验 + 每视图单目深度拼粗车体几何，绕过联合配准，可作为路线 B 的几何来源。

## 探路 2 结果（2026-11-16，已完成）

脚本：[image_to_3d_probe/run_probe.py](image_to_3d_probe/run_probe.py)；输出：`image_to_3d_probe/outputs/bmw5_front_v1/`。

**结论：单图→3D 几何生成可用且成本低；宏观形状正确，细粒度判别细节丢失。**

- 输入 front_001（rembg 去底），Hunyuan3D-2.1 几何生成 **36.1 秒 / 峰值 7.6 GiB**，输出 291,315 顶点 / 582,848 面 mesh.glb。
- 剪影定量对照（bbox 等比归一后取翻转最优）：渲染 90° vs 真实 side_004 **IoU 0.862**（宽高比 3.07 vs 3.41）；渲染 180° vs 真实 rear_005 **IoU 0.805**（1.96 vs 2.27）。剪影叠加图 `comparisons/silhouette_*.png`。
- 目视：整车比例、轴距、舱线、轮毂位置正确；格栅、灯组内部、车标等判别细节为平滑近似，无真实纹理（纹理模块需 CUDA 12 编译，本机驱动不支持）。
- 判定：该路线单独作为**身份保真增强**不合格（细粒度标签会注入噪声）；作为**几何代理/控制信号/剪影级视角增强**合格。

实现要点（复跑避免踩坑）：

- 该管线 `.to()` 无返回值，不能 `pipe = pipe.to(...)` 链式赋值（返回 None 导致 `'NoneType' object is not callable`）；`from_pretrained(device='cuda')` 已完成搬移。
- rembg 首次运行从 GitHub 下载 u2net.onnx 会以 ~20KB/s 龟速卡死；需用镜像预置 `~/.u2net/u2net.onnx`（md5 60024c5c889badc19c04ad937298a77b）。
- 渲染为 NumPy/PIL 画家算法软渲染（服务器无 GL），渲染图是 RGB 无 alpha，剪影提取需按背景色阈值而非 alpha。
- 依赖补漏：hy3dshape 隐式依赖 `timm`（根 requirements 有、子模块清单无）。

## Phase 0 总结论与路线建议（2026-11-16）

| 路线 | 探路证据 | 判定 |
| --- | --- | --- |
| A. 单图→3D 资产→任意渲染 | 36s/7.6GiB；剪影 IoU 0.81–0.86；细节丢失 | 工具合格，身份保真不合格 |
| B. 3D 作控制信号 + 生成式新视角编辑 | 未直接测；但 A 证明可廉价获得新视角几何，且已有 Qwen 双参考基础设施 | **推荐主线**，风险在身份保持 |
| C. 多视图重建真车 | VGGT 联合配准在目录照片上崩溃；单目深度可用 | 单独做主线薄弱；作几何补充可行 |

建议的 Phase 1（路线 B 最小闭环，零新基础设施）：沿用方法 06/07 已验证的 **Qwen 双参考模式**——输入车辆原图 + Hunyuan3D 渲染的目标视角灰度图（几何参考），提示词引导"同一辆车按此视角重绘"；用真实 side/rear 照片做对照验收。通过后再评估接入 Fun-Controlnet-Union 的深度/法线硬控制。

## 环境与资源记录

- VGGT：`/nas/wyh/.conda/envs/car-3d-probe`（Python 3.12 + torch 2.7.1+cu118 + numpy 1.26.1）；代码 `/nas/wyh/3d_tools/VGGT`；权重 `models/VGGT-1B/model.pt`（4.68 GB，仅取 model.pt，跳过同内容 safetensors）。
- 网络要点：huggingface.co 直连不通，用 `HF_ENDPOINT=https://hf-mirror.com` + `HF_HUB_ENABLE_HF_TRANSFER=1`（单线程约 1.7 MB/s → 多线程约 15 MB/s）；GitHub 用 gh-proxy.com 镜像下载 tarball，git clone 直连会被掐断。
- GPU：4×A800-80GB，VGGT 探路用 GPU 2，Hunyuan3D 探路计划用 GPU 1。
