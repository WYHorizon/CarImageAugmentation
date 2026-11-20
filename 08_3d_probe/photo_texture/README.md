# 多视图照片贴图与连续视角导出

当前发布模型与可复跑的视角导出见[宝马 5 系成品说明](../bmw5/README.md)。接缝修复 v3 及其后续版本从下文的 v5 贴图分支继续开发。

**历史材质实验：已增加 [v6 分部件材质与动态反光实验](MATERIALS.md)。** 该实验可查看
`outputs/bmw5_material_v6_smooth/material_mesh.glb` 与该目录的
`full_glb_renders/full_pbr_az_030.png`。它清理车漆、玻璃和橡胶区域的固定照片外观，
用近似材质重新生成反光，并保留关键照片细节。v5 作为旧贴图对照保留；下文记录其原始流程。

本模块把同一辆宝马 5 系的照片投影到一个固定 3D 网格，生成**一份几何、一张共享 UV 纹理**。之后改变相机即可连续观察或导出不同角度；不会为每个角度重新生成一辆车。它提供可供 3D 查看器加载的 GLB，以及每隔 30° 的 12 张示例图片。

当前目标是可旋转、可重复导图的车辆外观代理。结果仍有几何误差、接缝和未观测表面的补色，尚未达到经验证的照片级真车重建。已有背景没有重建；纹理包含原照片的光照、反射和透过玻璃可见的内容，不能当作可任意重打光的 PBR 材质或纯反照率。

**早期照片贴图实验选择的主输出为 `outputs/bmw5_mv_textured_v5/`。** 模型：`outputs/bmw5_mv_textured_v5/textured_mesh.glb`；前侧示例：`outputs/bmw5_mv_textured_v5/renders/az_030.png`；车尾示例：`outputs/bmw5_mv_textured_v5/renders/az_180.png`；检查记录：`outputs/bmw5_mv_textured_v5/verification.json`。本轮目视比较中，v5 的前格栅、机盖前端和车尾重影/碎块较 v4 减少，挡风玻璃、屋顶与两侧反光接缝仍存在；这是一辆车上的候选选择，未进行严格身份或照片级质量验证。

## 实现与输入

1. `prepare_inputs.py` 用本地 u2net 的 CPU provider 处理八张原照片，工作尺寸为 1536×1024。原照片不变；保留原始 alpha、最终 alpha、二值 mask 与几何拟合用的主体最大连通域。仅修补小型封闭孔洞，元数据记录修补像素、缩放、bbox 和哈希。
2. 几何可使用旧单图 Hunyuan3D-2.1 网格，或[四张真实照片条件的 Hunyuan3D-2mv 网格](../multiview_probe/README.md)。生成网格是几何候选，没有从照片唯一恢复真实尺寸。
3. `fit_cameras.py` 统一网格坐标，拟合各输入照片的透视相机。世界坐标为 +Y 向上、+Z 车头；相机外参为 world-to-camera，+X 向右、+Y 向下、+Z 向前。几何适配使用显式整体 XYZ 比例，不优化局部表面。
4. `bake_texture.py` 生成共享 UV，结合遮挡、表面朝向和前景 mask 选择照片颜色。当前可比较加权平均与单一最佳来源；区域来源先验另列为宝马专项候选。未通过观测判定的纹理点使用 3D 空间最近已观测表面的颜色，单独保留覆盖 mask。
5. `raster.py` 提供 CPU z-buffer、近远裁剪和透视正确重心插值。导出的 GLB 使用 `KHR_materials_unlit`，避免把已包含照明的照片纹理再次叠加漫反射光照。

| ID | 目视方向 | 名义角度 |
| --- | --- | --- |
| `front_002` | 正前 | 0° |
| `front_001` | 左前 | 45° |
| `side_004` | 车辆左侧，车头朝画面左 | 90° |
| `rear_005` | 左后 | 135° |
| `rear_006` | 正后 | 180° |
| `rear_007` | 右后 | 225° |
| `side_008` | 车辆右侧，车头朝画面右 | 270° |
| `front_003` | 右前 | 315° |

这些方向是拟合初始化的类别先验，不是已标定相机角度。正前是 `_002`、正后是 `_006`；不能把目录内的任意 front/rear 图当作严格 0°/180°。

## 几何候选与取舍

本轮日期：2026-11-17。所有 IoU 都在**参与拟合的八张输入 mask**上计算，并允许每张照片独立拟合相机，不代表独立测试准确率、真实 3D 精度或车型识别收益。

| 几何候选 | 相机拟合目录 | 平均输入剪影 IoU | 本轮用途 |
| --- | --- | ---: | --- |
| 旧单图网格 | `outputs/bmw5_oldmesh_fit_v2/` | 0.95468 | 保留旧几何贴图对照 |
| 新四视图网格，未调比例 | `outputs/bmw5_mv_fit_v2/` | 0.95014 | 检查多视图生成的原始形状；未自动优于旧网格 |
| 新四视图网格，XYZ 比例 0.8/0.9/1.0 | `outputs/bmw5_mv_proportion_fit_v2/` | 0.96127 | 当前贴图实验采用的几何 |
| 同一比例，额外要求相机仰角至少 5° | `outputs/bmw5_mv_proportion_elev_fit_v2/` | 0.91512 | 诊断对照，未采用 |

0.8/0.9/1.0 是根据输入外观选择的整体代理比例，未依据实测车长、车宽、车高标定。新网格提供了四张真实照片的几何条件，调整后输入轮廓拟合略有改善；这不足以宣称新模型在细粒度身份或未见视角上全面优于旧模型。

纹理实验均保留，不覆盖旧结果：

| 输出目录 | 纹理策略 | 主要取舍 |
| --- | --- | --- |
| `outputs/bmw5_oldmesh_textured_v2/` | 旧几何的照片贴图 | 用于判断改善来自几何还是贴图 |
| `outputs/bmw5_mv_textured_v3/` | `weighted`，按权重混色 | 接缝较平滑，但配准误差可能产生重复灯线、轮毂重影 |
| `outputs/bmw5_mv_textured_v4/` | `best`，每个 texel 选择一个最佳来源 | 减少平均重影，来源切换处可能出现接缝或碎块；不代表全面优于 v3 |
| `outputs/bmw5_mv_textured_v5/` | `best` + `vehicle_regions`，本轮主输出 | 宝马专项区域来源先验：前脸/后脸/侧面优先对应真实照片，仍须通过遮挡检查；用于减少法线噪声导致的零碎来源切换 |

`vehicle_regions` 给指定区域对应照片的来源权重乘以 1000，不改变原照片，也不能修正不准确的几何或相机；它不是通用车辆部件识别器。候选选择需要结合整车、关键部件及接缝目视检查，不能只看输入 IoU。

## 文件与验收范围

每个完成的贴图输出包含：

- `textured_mesh.glb`：可由兼容 glTF 2.0 的查看器旋转观察的固定纹理模型。
- `textured_mesh.npz`：与 GLB 对应的顶点、三角面、UV 数组。
- `texture.png`：2048×2048 的照片外观纹理。
- `observed_texel_mask.png`：通过近似可见性和前景检查的 texel 为白色；黑色还包括 UV 图集空白，须结合来源 ID 区分。
- `texture_source_ids.png`：0 至 N−1 对应 `manifest.json` 的来源顺序，254 表示未观测补色，255 表示 UV 图集之外。`weighted` 模式记录最大权重来源，实际颜色由多个来源混合；`best` 模式记录实际选中来源。
- `renders/az_000.png` 至 `az_330.png`：12 张 1200×900 环绕图，以及 `render_cameras.json` 中的完整相机参数。
- `reprojections/`：回到输入相机的八张渲染，仅用于检查输入对齐，不是留出测试。
- `bake_script_snapshot.py`、`manifest.json`：当次脚本、输入/几何哈希与参数。
- `verification.json`：独立完整性校验结果；通过不等于视觉质量或车型保真验收通过。

环绕相机使用固定距离、固定焦距，默认仰角 **2°**，更接近本组照片的低机位覆盖。提高到 12° 或继续向上旋转会暴露车顶等覆盖不足区域；低机位展示不能作为高视角质量的证明。交互查看器可以改变相机位置，但未观测表面的真实性不会因此增加。

v4 与 v5 的通过可见性判定的有效 UV texel 比例均为 **54.66%**；剩余约 45.34% 采用最近表面补色。这个比例按 UV 像素统计，不是整车表面积覆盖率、真实三维观测率或准确率。最近点补色可能跨越邻近但语义不同的表面，在车顶、车底、轮胎内侧等位置尤其需要检查。

v5 的独立完整性检查已通过：23 个产物 PNG（12 张环绕、8 张输入重投影、3 张贴图诊断）、64 个准备阶段 PNG、8 张原始照片哈希，以及 GLB/NPZ/标准化网格、UV、纹理和相机一致性。详见主输出中的 `verification.json`。

## 复跑

以下命令在项目根目录执行，采用新的 `_repeat` 目录；生成、拟合和贴图步骤都要求输出目录为空。复用现有 `/nas/wyh/.conda/envs/car-3d-hunyuan` 环境。准备、相机拟合、贴图、验证为 CPU 流程；只有多视图几何生成使用 GPU。GPU 编号应结合实际占用选择；当前多视图脚本限定物理 GPU 2。

```bash
PROBE_PY=/nas/wyh/.conda/envs/car-3d-hunyuan/bin/python

OMP_NUM_THREADS=4 "$PROBE_PY" 08_3d_probe/photo_texture/prepare_inputs.py \
  --output-dir 08_3d_probe/photo_texture/outputs/bmw5_inputs_repeat

CUDA_VISIBLE_DEVICES=2 "$PROBE_PY" 08_3d_probe/multiview_probe/run_multiview.py \
  --prepared-root 08_3d_probe/photo_texture/outputs/bmw5_inputs_repeat \
  --output-dir 08_3d_probe/multiview_probe/outputs/bmw5_mv_repeat \
  --seed 42 --steps 50 --octree-resolution 512

"$PROBE_PY" 08_3d_probe/photo_texture/fit_cameras.py \
  --mesh 08_3d_probe/multiview_probe/outputs/bmw5_mv_repeat/mesh.glb \
  --prepared-root 08_3d_probe/photo_texture/outputs/bmw5_inputs_repeat \
  --output-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_fit_repeat \
  --axis-scale 0.8 0.9 1.0 --output-faces 80000

"$PROBE_PY" 08_3d_probe/photo_texture/bake_texture.py \
  --fit-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_fit_repeat \
  --prepared-root 08_3d_probe/photo_texture/outputs/bmw5_inputs_repeat \
  --output-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_textured_repeat \
  --texture-size 2048 --angle-power 8 --blend-mode best --source-policy vehicle_regions

"$PROBE_PY" 08_3d_probe/photo_texture/verify_outputs.py \
  --output-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_textured_repeat \
  --fit-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_fit_repeat \
  --prepared-root 08_3d_probe/photo_texture/outputs/bmw5_inputs_repeat
```

只复跑照片贴图时，可直接使用已有 `outputs/bmw5_mv_proportion_fit_v2/` 与 `outputs/bmw5_inputs_v2/`，为新的贴图指定空目录。比较 v4 策略使用 `--blend-mode best --source-policy angle`；比较 v3 策略使用 `--blend-mode weighted --source-policy angle`。旧网格对照将 `--mesh` 换为 `08_3d_probe/image_to_3d_probe/outputs/bmw5_front_v1/mesh.glb`，并去掉 `--axis-scale`。

独立验证器不导入贴图或光栅实现、不运行模型；它检查原图和清单哈希、PNG 解码、canonical/NPZ/GLB 几何、glTF UV 原点、内嵌纹理、unlit 扩展、来源 mask、相机旋转及环绕距离。文件检查与输入拟合指标不能替代未见视角、关键部件和下游识别测试。生成资产位于被 Git 忽略的 `outputs/`，需要保留对应目录才能加载模型或查看诊断图。
