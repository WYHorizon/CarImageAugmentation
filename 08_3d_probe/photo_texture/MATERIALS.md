# 分部件材质实验 v6

v5 将照片 RGB 烘焙到固定 UV，并用 unlit 显示。车漆上的环境倒影、阴影，以及透过车窗看到的背景因此固定在表面。v6 的目标是验证：在保留同一几何和关键照片细节的情况下，能否让车漆、玻璃的反光随相机和环境光变化。

这是一版**显式材质近似**，不是从八张照片恢复真实反照率、真实玻璃、内饰或实测 BRDF，也不是已经通过车型识别评估的训练数据。

## 表示与取舍

- 使用可审查的人工多边形区分车漆、玻璃、橡胶、轮毂/镀铬、受保护细节。标注只适用于本组宝马照片，不是通用分割模型。
- 语义通过拟合相机映射到现有 UV。继承 v5 的实际照片来源，并把已观测与未观测推断分别记录；推断区域不能作为真实观测。
- 车漆采用深色固定底色、粗糙度和清漆层；玻璃用不透明深色反光材质近似；橡胶用较粗糙的深色材质。颜色与材质参数是明确先验，没有声称自动估计了原车物理属性。
- 灯组、格栅、徽标、门把手、窄门缝、轮毂等保留照片 RGB。它们仍包含拍摄时的照明或反射，这是保护身份细节与去光照程度之间的取舍。
- 两层相同几何使用互补的 alpha MASK：一层 PBR，一层 unlit 照片细节。每个表面位置仅显示一层，不把照片中已有的照明再叠加到新车漆上，也不依赖自定义 shader。
- v5 在 UV 拆点后重算法线，导致同一位置的法线跨 UV 岛不连续。v6 在临时精确 XYZ 焊接拓扑上计算角度加权法线，再映回 UV 顶点。顶点位置、三角面、UV 完全保持不变；没有通过几何形变掩盖误差。
- 初版 PBR 暴露明显三角波纹，因此最终候选额外采用车长 2% 半径的、面积加权和方向门控的法线滤波（2 轮）。轻量预览从完整网格采样该法线场，不在简化网格上重新估计。此操作只平滑着色法线，没有修复实际凹凸几何。

GLB 使用 `KHR_materials_clearcoat` 和 `KHR_materials_unlit`。标准 glTF 查看器可加载；没有环境光照的查看器不会呈现本实验棚拍效果。玻璃未启用 transmission，模型没有恢复完整内饰。

## 代码与输出

- `material_regions.py`：人工部件标注、可见区域映射、未观测区域推断及诊断图。
- `material_normals.py`：保留几何与 UV 的跨接缝法线修复。
- `build_materials.py`：生成 2048×2048 材质贴图和 GLB；清理区域不再包含原照片的 RGB 图案。
- `material_preview_template.html`、`pack_material_preview.py`：生成新旧材质可切换的紧凑交互预览，使用已有 15,000 面预览几何。
- `render_material_preview.js`：用实际 WebGL 渲染导出环绕图、转动环境光的对照图并检查拖拽与窄屏。
- `verify_materials.py`：独立检查文件、几何、法线、保护区 RGB、GLB 材质和渲染变化。

**主资产目录为 `outputs/bmw5_material_v6_smooth/`**，初版 `outputs/bmw5_material_v6/` 与原 `outputs/bmw5_mv_textured_v5/` 都保留。

主资产包括：

| 文件 | 含义 |
| --- | --- |
| `material_mesh.glb` | 原 79,996 面几何，两层互补材质，内嵌贴图 |
| `mesh.npz` | 顶点、三角面、UV 和修复后的法线 |
| `basecolor.png` | 近似车漆、玻璃、橡胶的 RGBA 底色；不等于真实反照率 |
| `photo_details.png` | 原照片 RGB 与互补 alpha，只显示保护区 |
| `material_controls.png` | 线性数据，R=clearcoat，G=roughness，B=metalness |
| `material_ids.png` | 0 未知照片保底、1 车漆、2 玻璃近似、3 橡胶、4 轮毂/镀铬照片、5 受保护照片细节 |
| `atlas_valid_mask.png` | 有效 UV 图集覆盖 |
| `manifest.json` | 先验参数、输入哈希、法线修复统计、限制 |
| `preview_manifest.json` | 预览缩减与压缩记录 |
| `renders_v1/comparison.png` | 同一轻量网格、同一相机的新旧材质四角度对照 |
| `renders_v1/browser_verification.json` | 12 角度导出、光照变化、拖拽和 320px 宽度验证 |
| `full_glb_renders/full_pbr_az_030.png` | 标准 GLTFLoader 直接加载完整 GLB 的渲染 |
| `full_glb_renders/full_glb_verification.json` | 完整 GLB 的实际浏览器加载、纹理/材质/alpha 检查 |
| `material_verification.json`、`visual_review.json` | 独立资产校验与视觉判断 |

轻量预览使用 15,000 面几何、768² 压缩照片和无损语义 ID；完整 GLB 为 79,996 面、2048² PNG。保护区逐像素保留的结论只适用于完整纹理；预览存在 JPEG、降采样、几何简化和法线量化误差。预览和完整模型都已实际渲染，完整资产的灯组、轮毂等细节更清楚。

## 复跑

在项目根目录运行。Python 复用现有 CPU 环境，未运行新的生成模型、未更改共享 GPU 环境。输出使用新的 `_repeat` 目录。

```bash
MATERIAL_PY=/nas/wyh/.conda/envs/car-3d-hunyuan/bin/python

"$MATERIAL_PY" 08_3d_probe/photo_texture/material_regions.py \
  --prepared-root 08_3d_probe/photo_texture/outputs/bmw5_inputs_v2 \
  --fit-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_proportion_fit_v2 \
  --textured-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_textured_v5 \
  --output-dir 08_3d_probe/photo_texture/outputs/material_regions_repeat

"$MATERIAL_PY" 08_3d_probe/photo_texture/build_materials.py \
  --source-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_textured_v5 \
  --material-ids 08_3d_probe/photo_texture/outputs/material_regions_repeat/inferred_material_ids.png \
  --valid-mask 08_3d_probe/photo_texture/outputs/material_regions_repeat/atlas_valid_mask.png \
  --normal-radius-fraction 0.02 \
  --output-dir 08_3d_probe/photo_texture/outputs/bmw5_material_repeat

"$MATERIAL_PY" 08_3d_probe/photo_texture/verify_materials.py \
  --source-dir 08_3d_probe/photo_texture/outputs/bmw5_mv_textured_v5 \
  --output-dir 08_3d_probe/photo_texture/outputs/bmw5_material_repeat \
  --protected-mask 08_3d_probe/photo_texture/outputs/material_regions_repeat/protected_mask.png
```

浏览器脚本需要 Node、`three@0.160.1`、`esbuild`、`playwright` 与 Chromium。本机测试工具位于 `/tmp/car-3d-viewer-check/`，未加入项目根依赖；复用时设置 `NODE_PATH=/tmp/car-3d-viewer-check/node_modules` 与 `PLAYWRIGHT_BROWSERS_PATH=/tmp/car-3d-viewer-check/browsers`。依赖在其他机器上需要重新安装。

```bash
node 08_3d_probe/photo_texture/render_full_material_glb.js \
  /absolute/path/to/bmw5_material_repeat
```

`material_preview_bundle_entry.js` 是紧凑 Three 打包入口，可用 esbuild 的 `bundle/minify/format=iife` 选项生成 JS。`pack_material_preview.py --help` 提供完整资源参数；其模板与数据全部内嵌，预览片段限制在 1 MB 以下。

## 验收含义

必须分别确认：

1. 原照片与 v5 资产哈希未变，完整资产的顶点、面、UV 未变。
2. 保护区域在纹理层面逐字节保留原 RGB，PBR/照片层 alpha 互补。
3. 同 XYZ 的法线相同；GLB 内 PBR 与 unlit 扩展分配正确，图像及 UV 对应正确。
4. 固定相机、背景与曝光，转动环境光时，新材质渲染发生变化，旧 unlit 照片渲染保持不变。
5. 目视检查车门倒影是否被清除，灯组、格栅、车标、门缝和轮毂是否仍在；并检查车顶、车尾与斜角。

反光随条件变化只验证表示与渲染机制。纯色底色可能损失真实漆面差异；几何曲面、轮胎、相机拟合和人工区域边界仍有误差。新材质会暴露原先被照片纹理遮盖的几何凹凸。不能仅凭文件校验或动态高光宣称整体逼真度、真实材质或车型身份已恢复。

本轮两条浏览器路径均通过：轻量预览与标准 GLTFLoader 完整资产都能渲染；固定相机改变环境方向，旧照片模式逐像素不变，轻量 PBR 图有 58,262 个像素的最大 RGB 通道变化超过 10/255。该数字是机制检查，不是画质评分。目视确认旧车门倒影与玻璃背景被替换，宽缓波纹、侧窗高光不平、轮廓及局部照片边界仍可见。
