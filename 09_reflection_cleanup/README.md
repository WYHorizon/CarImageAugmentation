# Qwen 去反光与固定网格纹理替换

当前发布的 3D 资产包含后续的玻璃、车顶与后备厢盖修复。[最终模型及各阶段说明](../08_3d_probe/bmw5/README.md)

## 多角度轻度处理与 v5 纹理替换

在已展示的左侧轻度结果基础上，对另外七张原始工作分辨率照片分别运行 Qwen，再仅在审核过的车漆区域混入 45% 模型结果。左侧 `side_004` 直接复用已接受的 raw、保护区域和轻度合成结果。其他图片采用相同 seed=42、40 步、CFG=1；提示词将“车门和前后翼子板漆面”扩展为“车身漆面”。这仍是独立单图处理，没有联合优化多视图颜色或反光。

- 二维输出：`outputs/bmw5_multiview_qwen_mild_v1/<view_id>/`，包含 `source.png`、`raw.png`、`clean.png` 和对照。
- 逐图保护区域：`outputs/bmw5_multiview_masks_v1/`。灯窗、格栅、牌照、车标、轮胎轮毂等采用原图像素；精细门缝和把手另做窄保护边界。
- 替换后的模型：`outputs/bmw5_v5_qwen_mild_texture_v1/textured_mesh.glb`。
- 同机位渲染及对照保存在模型输出目录中；每张原照片的前后对照保存在二维输出目录。

使用的基础模型严格为 `08_3d_probe/photo_texture/outputs/bmw5_mv_textured_v5`。不运行几何生成、相机拟合、UV 展开或重新选图。`replace_frozen_v5_texture.py` 读取原 `texture_source_ids.png` 的逐纹理像素来源，固定投影采样坐标、未观测区域补色来源与 UV 边缘填充；用原照片重放必须与旧纹理逐像素一致，才允许换入处理后的照片。重放记录位于 `outputs/v5_frozen_texture_replay_v1/`。

GLB 保留旧二进制内容，并追加新的 PNG，仅改内嵌图片引用及必要的长度信息。网格、三角面、UV、法线、节点变换和 unlit 材质保持不变。旧 PNG 字节仍在文件内，因此新 GLB 体积稍大。交互预览复用 v5 已有简化几何，并以同一相机切换两版纹理；完整模型的几何不简化。

最终检查通过：117 个已有文件哈希不变，GLB 几何/UV/法线二进制不变，12 个同机位渲染的外轮廓逐像素一致。详见模型输出目录的 `texture_only_verification.json`、`independent_render_shape_check.json` 与 `visual_review.json`。实际浏览器验证了旋转、保持机位切换纹理、1200×900 导图和窄屏布局。目视侧面倒影有所减弱，原有倒影残留、来源接缝及车顶覆盖问题仍存在。

主要脚本依次为 `run_multiview_qwen.py`、`prepare_multiview_masks.py`、`compose_multiview.py`、`replace_frozen_v5_texture.py`。这些仍是本组照片的实验脚本；保护坐标、原图核对和冻结资产要求不能直接套用到其他车辆。

## 多角度重度对照

重度运行使用 `run_multiview_qwen.py --variant strong`：七张照片从相同原始工作图出发，以 CFG=2.5、明确的负面提示词、seed=42、40 步重新生成；左侧复用单图实验已经展示的 `bmw5_side_qwen_v2/strong`。不在轻度结果上继续编辑。混合权重为 0.85，保护蒙版与轻度逐字节相同；这是合成权重，不是反光去除率。

重度二维结果保存到 `outputs/bmw5_multiview_qwen_strong_v1/`，模型保存到 `outputs/bmw5_v5_qwen_strong_texture_v1/`。继续使用同一个冻结的 `v5_frozen_texture_replay_v1`，仅替换 GLB 内嵌纹理，不调整网格、法线、UV、相机、材质或来源选择。轻度与旧侧面输出保留，额外冻结记录位于 `outputs/bmw5_strong_baseline_v1/`。重度通常会降低漆面光泽及曲面明暗层次，需与原版、轻度同机位比较。

重度最终资产检查通过：117 个原资产及额外冻结的 267 个既有文件不变，12 个实际渲染的外轮廓和背景逐像素一致。侧面三档对照为模型目录的 `comparison_three_az_090.png`，统一裁剪、无缩放调色；图中的重度版明显减少车门倒影，但漆面更平、更哑光。`texture_only_verification.json` 保存独立核对结果。最后的 `rear_007` 曾在推理中断后单独补跑，主 manifest 的 `continuation` 保存了参数一致性和原始输出导入来源。

## 单张侧面照片实验记录

本实验只处理二维图片，不加载或修改 3D 网格、UV、相机、材质或已有贴图。输入为宝马 5 系左侧原照片 `test_images/宝马5系/side/5_2026_255841_151515_004.jpg`。

目标是减轻车漆上可辨认的环境倒影，同时保留车身形状、自然明暗和车型细节。不是物理反照率恢复；大模型生成的车漆内容仍可能出现变化，需要目视比较。

## 运行方式

复用本地 `models/Qwen-Image-2.1`，固定 revision `d26bb61231c349cf6b7896fa83353113880e1ba3`。使用现有 `/nas/wyh/.conda/envs/car-weather-qwen21` 环境，无新增权重下载或共享环境升级。

原图 4464×2976 等比缩小至 1536×1024，没有非等比变形或裁切。每个候选均从同一原图出发，seed=42、40 步、KV cache 开启；一次进程共享模型加载，后一版不以前一版为输入。

首轮 `bmw5_side_qwen_v1` 使用两个长提示词、CFG=1，但目视没有有效去除倒影且出现细碎纹理，已标为不采用。第二轮 `bmw5_side_qwen_v2` 使用相同的简短直接提示词：原始候选 A 为 CFG=1，候选 B 为 CFG=2.5 并使用记录在 manifest 中的负面提示词。本地管线只有在 CFG>1 且提供负面提示词时才启用真正的 CFG 分支；CFG 是条件引导，不等于编辑强度。

最终轻度版在允许编辑的车漆区域混入 45% 候选 A，较强版混入 85% 候选 B，并叠加边界羽化。该比例是后处理混合权重，**不是反光去除率或 Qwen 原生 strength 参数**。目的是让两档形成实际可比较的差别，并缓解纯模型输出偏平、偏哑光的问题。

现有 Qwen 管线不提供 `mask_image` 或 `strength`。提示词不能保证结构、位置或未编辑区域不变，因此保存两类结果：

- `mild/raw.png`、`strong/raw.png`：模型原始输出，仅移除可追溯的输入 padding（本次为零）；未配准、扭曲、回贴、调色或锐化。
- `mild/clean.png`、`strong/clean.png`：在可编辑车漆区域采用对应 Qwen 输出，其他区域逐像素复制工作分辨率原图。是“Qwen 编辑 + 原图细节保护”的结果，不是纯模型输出。

允许编辑区域来自已有人工车漆标注，并额外保护可确认的机盖边线、保险杠接缝与装饰。最终使用 `bmw5_side_masks_v2`：外轮廓及灯窗、轮胎等边界向内退让 8px，再向内羽化 8px；内部门缝、把手和饰条改用精细轮廓，不额外退让，只做 3px 羽化，避免宽保护区把旧倒影一并保留。门缝、把手、车窗、灯组、轮毂、轮胎和外轮廓均在保护区域；没有利用通用高频滤波猜测真实细节。

羽化混合在 8-bit sRGB 中完成，alpha=0 的位置显式复制原图。不会根据模型结果拉伸或移动原图，也不会对全图进行色彩匹配。纹理变化是否改善反光仍须目视，像素保护本身不是画质保证。

## 文件

主输出：`outputs/bmw5_side_qwen_v2/`。失败首轮与初次较宽保护边界的合成结果均保留，后者位于主输出的 `initial_mask_review/`。

| 文件 | 内容 |
| --- | --- |
| `source.png` | 原图在工作分辨率的版本 |
| `mild/raw.png`、`strong/raw.png` | 两次独立 Qwen 原始结果 |
| `mild/clean.png`、`strong/clean.png` | 加入原图细节保护的候选 |
| `comparison.png` | 原图、轻度、较强的整车与车门局部对照 |
| `comparison_stacked.png` | 上下排列的完整车辆对照，适合窄屏 |
| `raw_comparison.png` | 无像素保护的原始模型结果对照 |
| `edit_alpha.png`、`protected_mask.png` | 编辑透明度与原图锁定区域 |
| `mild/difference.png`、`strong/difference.png` | 与原图的最大 RGB 通道差异，未归一化增强 |
| `manifest.json`、`compositing.json` | 推理参数、提示词、输入哈希、耗时与合成过程 |
| `verification.json` | 原图、保护区和已有 3D 资产的独立核对 |
| `visual_review.json` | 最终整车、车门局部与保护边界的目视复核 |

最终复核：两版均完整保留 1,363,155 个锁定像素，其中精细保护区域为 139,632 个像素；15 个已有 3D 资产文件的哈希未变。轻度版保留较多原有光泽与倒影；较强版明显减少车门环境倒影，但车漆更哑光、表面明暗更平。两版可用于比较，不应把较强版视为已恢复真实车漆或已适合直接贴回 3D。

## 复跑

在项目根目录执行，输出路径使用新的 `_repeat` 目录。GPU 编号应先核查占用；本次使用空闲物理 GPU 2。

```bash
QWEN_PY=/nas/wyh/.conda/envs/car-weather-qwen21/bin/python

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=8 "$QWEN_PY" \
  09_reflection_cleanup/run_qwen.py \
  --output-dir 09_reflection_cleanup/outputs/bmw5_side_qwen_repeat \
  --prompts-dir 09_reflection_cleanup/prompts/direct_v2 \
  --strong-cfg-scale 2.5 \
  --strong-negative-prompt '漆面上清晰的车辆倒影、建筑倒影、杂乱条纹、碳纤维纹理、布纹、网格纹、划痕、重影、车身变形、车色变浅。'

"$QWEN_PY" 09_reflection_cleanup/build_comparison.py \
  --output-dir 09_reflection_cleanup/outputs/bmw5_side_qwen_repeat \
  --mask-dir 09_reflection_cleanup/outputs/bmw5_side_masks_v2 \
  --mild-mix 0.45 --strong-mix 0.85 --accept-near-opaque-rgba
```

`prepare_mask.py --help` 提供重新生成二维编辑区域的参数。合成脚本会核对当前原图与标注所用图像逐像素一致，尺寸或坐标不匹配时直接停止。

Qwen 输出为 RGBA，本轮 alpha 接近不透明。经检查后，合成直接采用其存储的 RGB，原 RGBA 保留；未将其叠加到白底，也未把模型 alpha 当作编辑区域。`--accept-near-opaque-rgba` 只接受全图 alpha 最小值不低于 249/255 的候选，具体范围记录在 `compositing.json`。

验收时同时看 raw 和 clean：raw 用于观察模型是否漂移，clean 用于核对最终细节保护。不能将修改幅度或高频减少直接解释为去反光质量；更强版本也不一定更好。原图保留区、羽化边界和车身自然明暗都需要检查。

## 单张前挡风玻璃背景清理

本次只处理原始 `front_002` 照片的二维示例，不贴回 3D。目标是清理透过车厢看到的卷帘门横纹与黑孔，保留车身、窗框、贴纸和能辨认的真实内饰。输入继续使用与 v5 对应的原始 `source_rgb.png`，不是轻度或重度去反光照片。

`run_windshield_qwen.py` 用相同本地 Qwen 版本，以 seed=42、40 步、CFG=2.5 生成一次候选；完整提示词、负面提示词和原始 RGBA 保存于 `outputs/bmw5_front_glass_qwen_v1/`。模型原始结果会连带改变内饰，因此还需 `prepare_windshield_mask.py` 的逐图背景标注和 `compose_windshield.py` 的原图细节保护。这里只在玻璃内允许编辑的背景区域使用 Qwen RGB，其他像素复制原图，没有改变输入比例、几何或机位。

首次合成虽去掉了横纹和黑孔，但保护范围包含亮背景边缘，造成明显青白色轮廓，已标为不采用。`outputs/bmw5_front_glass_qwen_v2/` 复用同一份模型输出，只修订合成边界；不是第二次 Qwen 推理。`prepare_windshield_mask_v2.py` 使用逐图手工轮廓与原图颜色辅助定位，把确定背景完整替换，仅在混合边缘像素过渡，不再向背景内侧羽化。对应蒙版、对照、独立像素和既有资产验证均随结果保留。浅灰色车内包覆物按真实内饰保留；生成的低对比背景只是外观修复，不代表恢复了真实隐蔽车厢，也不解决真实玻璃的透射或视角变化。
