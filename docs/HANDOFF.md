# 汽车细粒度天气增强项目：会话交接

[文档首页](README.md)

> 文档整理说明（2026-11-09）：项目文档已统一迁入 `docs/`，本页路径和目录布局已同步调整；以下环境与实验进度仍保留原交接时的快照，不表示实时状态。

## 成像质量 README 分组展示（2026-11-12）

用户反馈成像质量九列总览过小，希望减少每行图片数量。按“原图＋两种效果”改为四组三列：两档失焦、运动与粗噪声、两档低分辨率、两档 JPEG。四辆车全部保留，根 README 和方法文档同步展示，原九列总览改为链接入口。相同页面宽度下，三列单图显示宽度约为九列的三倍，代价是页面更长。

新增 `docs/assets/imaging-quality-strong-{defocus,motion-noise,resolution,compression}.png`，均为 1200×1448，从原 3600×1448 PNG 直接按列重排，12 列逐像素核对一致；未重新运行增强或改变图像比例、强度。未将此前移除的基础组和失败案例重新放回根 README。

## 小米 YU7 README 展示调整（2026-11-12）

根据用户的页面截图，根 README 移除小米 YU7 四部件细节拼图及其直达链接，保留成图、整图与车头放大对比。树枝／骑行者两张成图改为普通 Markdown 图片上下独立排列，方法文档同步采用相同布局，避免双列表格挤压图片比例。原 PNG 均为 1536×1152（4:3），本轮仅调整文档展示；部件核查文件与方法文档中的资料链接保留。下方发布记录描述此前页面状态。

## 小米 YU7 两类遮挡发布（2026-11-12）

用户在查看两版样图后要求加入 README、同步文档并推送。本次发布包含两张生成 PNG、三张对比 JPEG，保存为 `docs/assets/qwen-yu7-*`；根 README 直接展示树枝／骑行者成图、三列整图、车头放大和四部件对比。图片与上轮输出逐字节一致，未重新生成。

同步更新方法文档、文档索引与小米 YU7 复跑命令；新增的三份配置、两份布局参考及来源记录、Qwen 预览标题修正一并纳入本次发布。完整输出、模型和环境缓存仍由 Git 忽略。下方“仅保留本地、未提交或推送”描述的是发布前的历史状态。

## 小米 YU7 两类遮挡生成与本地预览记录（2026-11-12）

用户要求绿色车辆分别做树枝和自行车遮挡，注意比例并确保实际覆盖车体，先查看效果。使用 `test_images/小米YU7.jpg` 和原有两类素材，新增 `branches_yu7.json`、`qwen_branches_yu7_v1.json`、`qwen_cyclist_yu7_v1.json` 与两份 `yu7_layout_reference.png/.json`。两张最终样图都由服务器 Qwen-Image-2.1 生成，1536×1152、seed 42、40 步，当前仅保留本地，未提交或推送。

- 树枝输出 `06_local_occlusion/outputs/qwen_branches_yu7_v1/edited.png`；CPU 布局 `branches_yu7_v1` 真实遮挡16.10%，此值不代表 Qwen 成图覆盖率。
- 骑行者输出 `06_local_occlusion/outputs/qwen_cyclist_yu7_v1/edited.png`；布局964×1000、左上(-80,320)，9 px水平模糊参考。实际头顶更高、轮廓略放大且运动感更明显，仍确实遮住机盖与前脸，轮弧部分可见、接地点画外。
- 综合整图、车头放大和部件图在 `06_local_occlusion/outputs/yu7_occlusion_review_v1/`，`index.html` 可本地浏览，`build_review.py` 可重建展示图，`verification.json` 保存检查记录。36 PNG 哈希／解码和24裁剪逐像素验证通过，输入与代码快照匹配，GPU 2 已释放。
- 共用推理入口仅修正硬编码 Audi 的预览标题为当前场景名称；CPU 抠图与推理算法保持。已更新方法文档，根 README 仍使用此前展示，待用户看过本轮效果。
- 目视确认两类素材覆盖车头，比例可作为预览；未遮挡灯具、轮毂／轮胎、车牌小字存在生成重绘，不宣称像素保真、精确遮挡率或训练收益。

## 局部遮挡发布与 README 整理（2026-11-12）

用户要求推送目前完成的工作，并从根 README 移除已被替代、明确效果不佳的历史案例。本次发布范围包含 `06_local_occlusion/` 的 CPU 合成、素材准备、本地 Qwen 双参考融合、各轮配置与输入素材，以及文档和精选对比图。完整生成目录、模型权重和环境缓存继续由 Git 忽略；下方各段“未提交／未推送”为当时的历史状态。

- 根 README 移除历史强光组、变化偏弱的基础光照、成像质量基础组及旧版强度比较，只展示当前方法效果；详细方法文档和历史素材保留。
- Qwen 首页展示更新为 `qwen_cyclist_foreground_scale_v2`，当前整图与部件预览另存为 `docs/assets/qwen-cyclist-foreground-scale-example.png` 和 `qwen-cyclist-foreground-scale-parts.jpg`，可在 GitHub 直接浏览。
- 首页复跑命令显式指定当前配置，移除指向未入库 `outputs/` 的骑行者图片链接。保留模型可能重绘部件、训练收益尚未验证等适用范围说明。
- CPU 实现与 29 项测试通过的生成快照一致；本轮整理未改动增强算法或重新运行 GPU 推理。

## 当前版本：增大近景骑行者比例（2026-11-11）

用户指出此前近景人物仍偏小，提供两张大人物前景参考，要求主要修改人与汽车的比例。最新配置为 `06_local_occlusion/configs/qwen_cyclist_foreground_scale_v2.json`，输出为 `outputs/qwen_cyclist_foreground_scale_v2/edited.png`，`previews/scale_comparison.png` 对照此前较小的 `motion_boundary_v2`。

布局参考改为素材 964×1000、左上 `(-170,340)`，相对此前线性放大约40%，保持9 px轻横向模糊；原车图、模型、seed 42和40步保持。首轮 scale_v1 被模型过度放大且丢失轮子，记录为草稿。scale_v2 强调参考已为最终尺度，目视与独立复核确认头顶高于车顶、前轮上弧可见、接地点画外，车头遮挡有效；用户尚未验收。

推理41.048秒、全流程55.086秒、峰值已分配显存48.324 GiB。18管线PNG、12裁剪及输入／实现快照核对通过；布局是视觉参考引导，不是标定的透视或像素锁定。配置、参考、结果和文档保留本地，未提交或push。

## 当前版本：减轻骑行者模糊（2026-11-11）

用户认可 `qwen_cyclist_motion_boundary_v1` 整体布局与遮挡，但要求减轻人物过度模糊。新配置为 `06_local_occlusion/configs/qwen_cyclist_motion_boundary_v2.json`，结果为 `outputs/qwen_cyclist_motion_boundary_v2/edited.png`；`previews/blur_comparison.png` 提供上一版／新一版对比。

布局参考仍为素材 690×716、位置 `(60,560)`，从原素材重新制作，将水平模糊核 25→9 px，提示词改为轻微短拖影。双参考、1536×1152、seed 42、40 步均保持。独立复核确认模糊减轻、细节更可辨，但仍有明显速度感；位置、大小与遮挡关系基本稳定，部分车轮保留、接地点画外。用户尚未验收本版。

推理 41.115 秒、全流程 56.042 秒、峰值已分配显存 48.364 GiB；18 管线 PNG、12 原生裁剪、输入与代码快照检查通过，另有一张旧新对比图。生成图未作后期锐化或贴回，车型保真仍有局限。配置与说明保留本地，未提交或 push。

## 当前骑行者要求：部分车轮、明显遮挡和横向拖影（2026-11-11）

用户否定初版比例及浅裁切，又明确纠正过度隐藏自行车的方向：应参考其骑行遮挡图，保留部分车轮、车架和腿部，只隐藏轮胎接地部分；增加人物和自行车横向运动模糊，必须对汽车有明显遮挡。**不要再沿用“全部裤腿／车轮出画”作为验收标准。** `boundary_v2`–`v5` 是中间尝试，未选作当前结果。

最新样例位于 `06_local_occlusion/outputs/qwen_cyclist_motion_boundary_v1/`，独立配置为 `configs/qwen_cyclist_motion_boundary_v1.json`。原素材缩到 690×716，施加 25 px 水平模糊并放到工作图 `(60,560)`，形成带棋盘的位置参考；Qwen 以原图及该参考去底融合，输出不再作像素贴回、额外取景裁切或后加模糊。详见[当前方法](06_local_occlusion.md#部分车轮可见与运动遮挡2026-11-11)。

- 两轮上半部和部分车架可见，接地点画外；人物遮住部分车头、格栅、机盖，横向拖影明显，汽车仍清晰。助手及独立复核认为符合本轮展示目标，待用户评估；运动模糊比用户参考略强。
- 1536×1152、seed 42、40 步；推理 41.124 秒，全流程 56.379 秒，峰值已分配／预留显存 48.349／57.898 GiB。18 PNG、12 裁剪及原图／素材／实现哈希核对通过，GPU 2 当时恢复 0 MiB。
- 布局仍由参考图和提示词软控制，无精确遮挡 mask；车灯、轮毂有重绘，严格车型保真未通过。新增配置、参考图与文档未提交或 push。

## 边界进入的 Qwen 骑行者融合接续（2026-11-11）

用户给出按素材类别从边界进入的遮挡规则，并要求用已有 Qwen 再融合骑行者。新增 `06_local_occlusion/configs/qwen_cyclist_boundary_v1.json`，复用已有双参考入口；原图和素材与首轮相同，生成参数仍为 1536×1152、seed 42、40 步。完整方法及复跑命令见 [边界进入融合](06_local_occlusion.md#边界进入的骑行者融合2026-11-11)。

- 结果在 `06_local_occlusion/outputs/qwen_cyclist_boundary_v1/`。骑行者从左下进入，两只自行车轮底部出画，后轮左侧也被裁切；头部、上身及大部分汽车侧身可见。原始生成 PNG、整图及部件对照、日志与校验记录均保留。
- 本次边界位置通过提示词引导，配置中的意图框没有作为几何控制输入。实际头部位置与提示词目标有偏差；没有实现自动布局引擎、精确 mask、局部像素锁定或事后原车像素贴回。
- 加载／推理／全流程为 8.679／41.135／55.449 秒，峰值已分配／预留显存为 48.351／57.559 GiB；18 PNG 和 12 张原生裁剪核验通过，输入及脚本快照一致。GPU 2 在结束后恢复 0 MiB，本轮仅新增配置，未重跑 CPU 单元测试。
- 车灯、轮毂及轮胎仍有重绘，严格车型保真未通过，待用户评估视觉效果。新增配置和文档均留在本地，未提交或 push。

## Qwen 骑行者双参考融合接续（2026-11-11）

用户明确要求在服务器上使用已有 Qwen，将骑行者素材融合进车辆原图。新增独立 `06_local_occlusion/run_qwen.py` 与 `configs/qwen_cyclist_fusion.json`，默认 CPU 枝叶合成、素材去底和配置保持原样。方法与复跑命令见[本地 Qwen 双参考融合](06_local_occlusion.md#本地-qwen-双参考融合)。

- 两张参考直接输入 `QwenImage21Pipeline`：奥迪原图和用户 771×800 RGB 骑行者素材。跳过已在人物高光／细辐条上失败的专用 `light-checker` 去底，由模型理解棋盘背景并进行前景融合。
- 使用现有 `/nas/wyh/.conda/envs/car-weather-qwen21` 和 `models/Qwen-Image-2.1`，离线运行；首轮配置为 1536×1152、seed 42、40 步、BF16、CFG 1.0、KV cache 开启。GPU 2 仅是本次检查后选用的设备，后续需重新查询占用。
- 管线无 `mask_image`／`strength`，未接入局部像素锁定。不能声明未遮挡原车逐像素不变，也不能将生成 alpha 当成骑行者 mask。原始生成图单独保留，不用原车像素贴回修饰结果。
- 29 项测试属于此前 CPU 路径，不代表 Qwen 融合视觉或保真验收。生成结果仍需对照车灯、格栅、轮毂与骑行者细节。
- `qwen_cyclist_fusion_v1` 已生成一张 1536×1152 结果：加载 8.169 秒、推理 41.060 秒、总计 54.657 秒；峰值已分配／预留显存为 48.286／55.900 GiB。40 步完成，18 PNG 哈希／解码检查与 12 张原生裁剪逐像素核验通过；输入哈希未变，脚本及辅助代码快照一致，GPU 2 当时已恢复 0 MiB。
- 目视可见骑行者、自行车、轮内透景和接地阴影，未把素材棋盘带入场景；人物浅色高光不再被规则去底误删。但车辆灯内纹理、轮毂和轮胎细节有重绘和平滑，包括未被遮挡的后轮，骑行者本身也是生成结果。当前单图融合已完成，待用户评估；严格车型保真暂未通过。README 与方法文档保存整图及部件精选；原始输出、文件核验和 `visual_review.json` 保留在运行目录。
- 原始输出 RGBA 的 alpha 范围为 179–255，约 2.8465% 像素低于 255，低于 250 约 0.00345%；预览合到白底显示，原始文件不改。alpha 不是骑行者遮挡 mask。旧 CPU 实现与枝叶运行快照哈希保持一致，旧天气推理脚本未改；本轮未提交或 push。

## 局部遮挡单图演示接续（2026-11-11）

构图与尺度方法已提交并推送到 `main`，提交为 [`0b66121`](https://github.com/WYHorizon/CarImageAugmentation/commit/0b6612105ab91a4d8bd7234d0b04b2ee0cb6b7b2)。用户随后要求先规划局部遮挡，讨论通用素材引擎后提供一张枝叶图，明确要求“写代码，然后用这张图做一个看看效果”。本轮范围为通用代码加 **一张奥迪 A6L 枝叶遮挡演示**，不把此前设想的四类两档 32 张计划写成已完成。

- 新模块 `06_local_occlusion/` 对应清单方法 5，CPU NumPy/Pillow 运行，无需大模型或 GPU。入口、参数及素材复用见[局部遮挡方法](06_local_occlusion.md)。
- 用户素材为 1000×707 RGB，白灰棋盘是实际图像内容。保留 `assets/branches_user/source.png`，通过显式 `light-checker` 专用去底得到 `rgba.png`，`material.json` 记录参数、来源与哈希。
- 普通透明素材走通用 RGBA 合成；缩放、旋转和模糊使用预乘 alpha，颜色调整仅作用于前景。当前为单层、固定可配置位置，没有自动搜索摆放、类别专用绘制、自动接地或添加阴影。
- 浅色中性棋盘去底是专用算法，不适用于任意白底图或带白色主体的素材；已有有效 alpha 的输入默认保留。用户提供的绿色枝叶与原图冬季背景存在季节差异，没有通过全局重绘改变原车与场景。
- alpha ≥ 128 计为不透明遮挡，另记录 alpha 加权覆盖；车辆统计使用已有轮廓，部件使用矩形 ROI，后者包含部分背景。未提供车辆轮廓时标记车辆统计不可用。
- 单图 CLI 支持 `--input`、`--asset`、`--preset`、`--scenes`、`--output-dir`、`--check-config`；换素材主要增加 RGBA 和配置。未知图片自动使用整图对照，不能沿用样例车辆的部件坐标。
- 当前 29 项测试通过（2.674 秒）：11 项核心、8 项素材、6 项管线、4 项报告。正式 `branches_demo_v1` 运行耗时 13.015 秒，27 个 PNG 校验通过；所有输入哈希未变，alpha 为零区域与原图像素逐值相同。完整报告位于 `06_local_occlusion/outputs/branches_demo_v1/report/index.html`；README 展示整图双列与四部件双列精选。代码、素材、图和文档本轮保留本地，未提交或 push。
- 奥迪车辆轮廓不透明遮挡率约 17.2598%，alpha 加权覆盖约 17.6755%；车灯 ROI 分别为 30.2332%／31.4442%，格栅为 32.5625%／34.2571%，前后轮均为 0%。已目视检查整图和部件，无棋盘残留或明显亮边；该检查由助手完成，不代表用户验收或训练收益验证。
- 16 张原生／224 裁剪逐像素核对，28 个 HTML 本地链接有效，输出和代码哈希与最终快照一致；本轮额外的 `visual_review.json` 保存助手检查记录，运行清单仍保留 `complete_pending_visual_review`，未伪装为用户已验收。

## 构图与尺度方法接续（2026-11-11）

用户批准按方案实现八种明显构图变化，用四张原图生成 32 张样图，并同步整图及对应关键部件报告。新模块为 `05_composition_scale/`，对应增强清单中的方法 3，CPU NumPy/Pillow 运行，无需大模型。参数及运行方式见[构图与尺度增强](05_composition_scale.md)。

- 八种预设为 1.30 倍近景、1.60 倍紧裁剪、1.20 倍四方向偏置取景，以及 ±12° 画面倾斜；旋转自动放大裁去空角，本批约 1.257 倍。
- 原图独立输入，保持输出画布尺寸，图像只做一次 BICUBIC 仿射重采样；不扩背景、反射补边或合成真实新视角。
- 车辆轮廓与部件四角随同一矩阵变换，按可见区域裁部件；截断比例和不可见状态必须在报告中保留。短边 224 先缩完整图再裁映射后的区域。
- 完整主体预设与允许截断预设分开记录；变化强度以车辆占比、中心位移和倾斜角判断，不能沿用成像质量模块的固定部件坐标或主要依靠 MAE。
- `composition_v1` 已完成 32 张 3840×2880 主图、32 张全有效 mask、144 张原生部件裁剪和 25 张预览，共 233 个 PNG 校验通过；无完全出框部件。CPU 全流程耗时 149.836 秒，Python 3.10.12／NumPy 2.2.6／Pillow 12.1.0，四张原图哈希未变，当前生产代码与运行快照一致。
- 144 张原生部件裁剪与对应主图的变换后可见框 crop 逐像素一致，图库 232 个本地链接全部有效。
- 27 项测试通过（12.157 秒）：16 项几何核心、5 项管线、6 项报告。完整主体档均保留 100%；紧裁剪主体保留 94.750%–95.855%，后轮区域最低 57.112%；左右偏置主体分别保留 99.805%–99.959%、98.501%–99.607%。
- 近景车框宽度占比为 88.400%–89.570%；左右中心位移约 −10%／+10% 画宽，上下位移分别约 −8%／+12% 画高；±12° 旋转实际倍率为 1.257109，包含 2 像素采样安全边距。
- 已目视检查四车总览、宝马全部原生部件及精选整图／取景框／部件。变化明显，车身比例保持；宝马紧裁剪及右偏的后轮保留率约为 59.2%／88.9%。部件适配固定格子后可能弱化尺度差异，整图用于判断构图，原生裁剪与保留率用于判断局部损失。
- README 与方法文档纳入四张精选 PNG：四车总览、宝马单车九宫格、同顺序的取景框九宫格和四种预设的宝马四部件对比。图库仍提供全部四车八效果和 224 诊断。
- 本轮 Git 发布范围包含构图增强代码、配置、测试、文档及四张精选预览 PNG；完整 `outputs/` 继续保留在本地，可按文档命令重新生成。此前方法 2 已推送至 `main`，提交为 `6cc718d`；旧方法实现和输出保留。

## 方法 2 提交与通用图片说明（2026-11-11）

用户已要求将方法 2 提交到 GitHub。本次提交包含基础／强档代码与配置、32 项测试、文档及四张精选 PNG；完整 `outputs/`、模型和缓存仍由 Git 忽略。下方“暂留本地”描述是此前审图阶段的历史状态。

已在 README 和方法文档补充自定义图片的两种入口：`apply_quality` 可直接处理任意非空 uint8 RGB 图片，无需场景标注；现有 `run.py` 的 `--scenes` 用于报告，整图批处理可只配置 `full_image: [0,0,1,1]`。核心不依赖样例文件名或车型，CLI 当前不递归扫描目录。模糊随输入短边缩放，分辨率／颗粒网格使用绝对目标短边，换图后仍需结合最终输入尺寸选择强度。

发布前额外用未知文件名、257×385 竖图与 403×269 横图做全图报告烟雾检查，未配置车型或车辆轮廓；八种强档共 16 张结果成功，输入尺寸与原文件保持不变。生产代码哈希仍与通过 32 项测试的强档运行快照一致。

## 成像质量强档接续（2026-11-11）

用户反馈方法 2 除 defocus 外其他效果肉眼不明显，要求继续增大强度。本轮新增 `--profile strong`，配置位于 `04_imaging_quality/configs/strong_presets.json`；默认仍为 basic，原 `quality_v1` 及其配置、预览保留。当前强档输出目录为 `quality_strong_v1`，参数及命令见[方法文档](04_imaging_quality.md)。

- 两档失焦沿用原参数；当前四车分辨率下，运动长度由 27 px 增至 90 px，角度仍为 15°。
- 噪声由逐像素独立噪声改为可选粗颗粒残差场，`peak=24`、`read_sigma=0.025`、网格短边 384；仅随机残差放大后加回原图信号，不对原图信号做降采样。
- 低分辨率短边改为 160／80，JPEG quality 改为 5／1；JPEG 继续在原尺寸实际编码，保留真实 JPEG 和精确解码 PNG。
- `quality_strong_v1` 已完成 32 张 3840×2880 主图，CPU 全流程耗时 219.800 秒；201 个 PNG 与 8 个实际 JPEG 校验通过，四张原图哈希未变，32 项测试通过（9.460 秒）。保留的两档失焦共 8 张 PNG 与 basic 逐字节相同，当前代码与运行快照哈希一致。
- 在短边 224 下，四车平均 MAE：运动 4.068→11.085、噪声 1.101→10.485、短边 720→160 为 0.130→5.477、360→80 为 0.515→10.740、JPEG 35→5 为 1.016→6.318、12→1 为 3.031→8.458。值均按每档结果与原图比较，表示缩小后可见差异增大。
- 已目视检查四车总览、本田 224 部件和奥迪旧新精选：拖影和粗噪声在整图可见，80 比 160 更丢细节，JPEG 5／1 色阶断裂与灰化明显。没有将不同退化描述为等效强度。
- README 当前展示转向强档，新增四车总览和奥迪四行三列原图／旧版／强档精选，保留 basic 历史总览与车灯裁剪；更强效果用于可见退化与压力样例，分类训练收益尚未验证。
- 方法 2 的代码、文档和精选图仍保留本地；已推送提交仍是上一轮光照方法 `7854c33`。

## 成像质量方法接续（2026-11-10）

用户要求先将上一轮光照方法提交 GitHub，再构建增强清单中的方法 2。光照与改进眩光已提交并推送至 `main`，提交为 [`7854c33`](https://github.com/WYHorizon/CarImageAugmentation/commit/7854c339f1454a59646f0fc293ca0bba138034de)，远端版本已核对。完整生成输出和模型未纳入提交。

新方法独立放在 `04_imaging_quality/`；目录号表示项目建立顺序，对应清单中的“成像质量增强”。使用 CPU NumPy/Pillow，无需大模型。参数、运行命令及边界见 [成像质量增强](04_imaging_quality.md)。

- 八种独立预设为两档失焦、运动模糊、近似传感器噪声、短边 720／360 和 JPEG quality 35／12；`quality_v1` 已完成四车 32 张 3840×2880 主图，CPU 全流程耗时 218.660 秒。
- 201 个 PNG 校验通过，包含 32 主图、144 原生部件裁剪、17 预览和 8 实际小图；另有 8 个实际 JPEG，解码像素与对应 PNG 一致。原图哈希未变化，完整 27 项测试通过。
- 低分辨率结果恢复原尺寸便于对照，同时保留真实小图；JPEG 在原始尺寸编码，保留真实文件及逐像素相同的解码 PNG，不暗中叠加低分辨率。
- 噪声使用基础种子、原图 SHA256 与预设 ID 稳定派生的随机种子；报告全部使用 PNG，含原生同坐标部件裁剪和先缩完整图到短边 224 的诊断。
- 已检查四车总览与格栅、奥迪原生车灯和 224 诊断；README 加入总览及未缩放的奥迪车灯精选 PNG，完整图库含四车四部件全部八种效果。模糊、颗粒和压缩损伤可见，未进行正式训练数据验收。
- 短边 720／360 在缩至 224 后的四车平均 MAE 仅 0.130／0.515，噪声从原尺寸 8.727 降至 1.101；下一轮应根据实际分类输入尺寸设有效分辨率与噪声尺度，不能从原生裁剪差异直接推断训练收益。
- 本轮方法 2 代码、文档和精选图暂留本地供用户审图；已推送的内容是上一轮光照方法。完整输出继续由 Git 忽略。

## 眩光改进接续（2026-11-10）

用户认可旧强光组的过曝与高光光晕，但认为阳光眩光、镜头眩光及冷暖逆光显假。当前范围改为 `--profile refined`：保留前两种的参数与实现，重做阳光眩光和镜头眩光，暂不生成逆光。旧 `basic_v1` 与 `strong_v1` 保留作历史对照。

新入口为 `03_lighting_augmentation/configs/refined_presets.json` 与 `lighting/glare.py`。新眩光使用画外入射位置、空间变化光幕与柔和鬼影，不生成画内太阳光球和固定星芒。位置按图人工设定并记录，不宣称由单图恢复真实太阳方向；没有使用实拍眩光模板、标定镜头或生成模型。运行方法见 [光照与光学退化增强](03_lighting_augmentation.md)。

- `refined_v1` 已完成 16 张 3840×2880 主图、80 张原生部件裁剪、15 张 JPEG 预览，CPU 全流程耗时 118.286 秒；[完整图库](../03_lighting_augmentation/outputs/refined_v1/report/index.html)可查看四种效果。
- 36 项测试通过；96 个 PNG 校验通过，原图哈希未变。独立核对 80 张裁剪、132 个图库链接、112 个媒体与 HTML 哈希。
- 前两种效果的 8 张 PNG 与 `strong_v1` 逐字节一致，当前批次没有逆光输出。新版阳光眩光的整图平均绝对码值差为 19.33–42.11，镜头眩光为 13.42–22.92，均按 0–255 码值尺度统计。
- README 展示新精选总览 `docs/assets/lighting-refined-overview.jpg`，旧强光组折叠为历史对照。未开展实拍成对真实性验证或分类训练评估。

## 上一轮强光增强接续（2026-11-10）

用户认为基础组变化过轻，明确希望增强图与原图有较大差别，包含高光过曝、光晕、镜头眩光和逆光。方法扩展为 `--profile basic`／`--profile strong`：前者保留六种轻量光度变换，后者提供六种强光预设。两组均使用 CPU NumPy/Pillow，无需大模型；强光效果为二维成像近似，不重建真实阴影或材质反射。

强光参数入口为 `03_lighting_augmentation/configs/strong_presets.json`，运行命令见 [光照与光学退化增强](03_lighting_augmentation.md)。不同批次使用新输出目录；原始 `basic_v1` 保留作为轻量对照。

- `strong_v1` 已完成 24 张 3840×2880 增强图、112 张原生部件裁剪与 16 张 JPEG 预览；[完整图库](../03_lighting_augmentation/outputs/strong_v1/report/index.html)可查看整图和关键部件。
- 全流程耗时 176.565 秒，136 个 PNG 校验通过，原图哈希未变；该轮 26 项自动测试通过。
- 强光组整图平均绝对 RGB 码值变化为 27.32–78.12，基础组为 1.69–14.93；差异增大不等于训练收益已经验证。
- 参数包括 +1.5 EV 过曝、亮部扩散、光幕／鬼影、−1.7 EV 暖／冷逆光近似。新增光源没有真实遮挡计算，原阴影与反射保留；冷光预设不可宣称准确模拟自然阳光。
- README 已加入精选强光总览 `docs/assets/lighting-strong-overview.jpg`。完整输出继续由 Git 忽略，基础组结果保留；尚未进行分类训练或正式样本验收。

## 基础光照接续状态（2026-11-10）

新增基础光照方法 `03_lighting_augmentation/`，详见 [基础光照增强](03_lighting_augmentation.md)。基于 NumPy/Pillow 的近似线性 RGB 光度变换，CPU 执行；六个预设为曝光 ±0.3 EV、固定枢轴对比度 0.9/1.1、暖/冷白平衡偏移。没有使用生成模型，不包含局部阴影或真实场景重光照。

- 已从共享四张原图生成 24 张 3840×2880 样图，位于 `03_lighting_augmentation/outputs/basic_v1/`。
- [本地完整图库](../03_lighting_augmentation/outputs/basic_v1/report/index.html)包含整图、四个部件、原生裁剪和短边 224 分辨率诊断；112 张部件裁剪加上 24 张主图，共 136 个 PNG 检查通过，另有 16 张 JPEG 预览。
- 13 项数值与管线测试通过；原图哈希未改变，增强主图尺寸和哈希通过；全流程耗时 155.614 秒。
- 六种变换没有随机性，不使用 seed；每个预设直接作用于原图。增益/曲线不重绘结构，但新增截断和近黑像素仍会损失可见信息。
- 宝马在增曝光、高对比度下车辆区新增高光截断分别约 20.013%、17.933%，不得记录为“全部样图无损合格”。已保存指标摘要和视觉观察，尚未进行分类训练。
- 根 README、文档索引和方法文档已同步，精选总览位于 `docs/assets/lighting-basic-overview.jpg`；完整输出仍由 Git 忽略。

## 上一阶段接续状态（2026-11-09）

用户最新明确希望以**细粒度车型识别**为下游任务发表论文，并表示已有**大约几万张图片、几千个车系**的数据集。数据位置、标签层级、真实恶劣天气覆盖及划分尚未核查。新增 [论文选题与最小验证方案](research_proposal.md)：推荐验证判别部件保真的天气增强，明确相关工作重复风险及可见性校准、真实天气分类评测要求；这是研究假设，尚未实现或证明新颖性与收益。

已查到 Alibaba PAI 发布的 [Qwen-Image-2.1-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union)，可保留 Qwen 2.1 底模研究结构控制。公开 VideoX-Fun 有匹配推理实现；当前项目使用的普通 `QwenImage21Pipeline` 不能直接加一个 `controlnet` 参数完成接入。此项仅做公开资料/源码核查，尚未下载控制权重、适配本地脚本或验证运行。

Qwen-Image-2.1 的依赖安装、权重完整性核查和离线加载均已通过，现已完成四辆车的四类天气样图，包括追加的降雪与部分积雪。当前入口为 [Qwen 编辑方法与全部结果](02_qwen_image_edit.md)，项目实时目录见 [文档首页](README.md)。以下历史快照中的“尚未部署”和“当前请求”不再表示当前状态。

- 环境：`/nas/wyh/.conda/envs/car-weather-qwen21`；共享权重：`models/Qwen-Image-2.1`；方法代码：`02_qwen_image_edit/`。
- 缓存及临时文件使用 `/nas/wyh/.cache/` 下已有目录，推理离线执行；未修改全局 shell 缓存配置。
- 首张采用 seed 42、40 步、2400×1792 画布，去边框输出 2388×1791。完整记录在 `02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/`；生成耗时 116.23 秒，峰值已分配显存 65.47 GiB。
- 初次运行图片已保存，但中文 manifest 写入编码失败；修复 UTF-8 读写后，同配置重跑成功，生成图哈希完全一致。两次记录都保留，详见方法文档。
- 用户随后要求生成其他天气和车辆：已完成奥迪 A6L、宝马 4 系、小米 YU7、本田 CR-V × 雨后阴天、中雨、薄雾，共 12 个组合；新增 11 张、复用首张奥迪雨后图。批次目录 `02_qwen_image_edit/outputs/weather_batch_v1/`，浏览入口为 [完整结果页面](../02_qwen_image_edit/outputs/weather_batch_v1/report/index.html)。
- 新增图同为 seed 42、40 步及首轮尺寸；批次耗时 28 分 14 秒，单张生成约 116.6 秒，峰值已分配显存最高 65.50 GiB。12 个组合、192 个 PNG 文件检查通过，四张原图未改动，GPU 2 已释放。
- 三类天气变化有效；雨后和中雨组颗粒感较强，薄雾组明显平滑一些。但轮毂标识、灯内细节、小字等仍有重绘损失，**本轮没有判定严格细粒度保真合格样本**，也未做训练评估。
- 后续按用户要求新增四张降雪与部分积雪图，位于 `02_qwen_image_edit/outputs/snow_batch_v1/`；已完成并通过 64 个 PNG 校验，原图未改动。当前 [四车四天气完整图库](../02_qwen_image_edit/outputs/weather_batch_v2/report/index.html) 共 16 张、256 个 PNG 校验通过，复用旧图而未重跑。
- 雪景有地面雪斑和车顶薄雪，仍露出部分路面及车漆，但雪粒偏密且细节仍有重绘；小米额外生成了原图没有的前景黄色标线，已记入视觉检查，未修图掩盖。雪景批次耗时 10 分 9 秒，生成子进程均已结束；不应根据过去的释放记录判断 GPU 当前空闲。
- 用户要求一步一步推进。Qwen 样图扩展已完成；用于追查细节损失的“保持原图不变”同配置重建对照仍未执行，不应记为已经通过的诊断。
- 用户曾询问 **SDXL + ControlNet + IP-Adapter** 的实施步骤，随后明确“如果没 Qwen 好，就不要试”。由于没有本项目证据证明该组合优于 Qwen，**目前暂不推进 SDXL 部署**；[实施说明](03_sdxl_controlnet_ipadapter.md) 仅保留为备选。已完成的只是接口导入检查，尚未创建 SDXL 环境、补装 OpenCV、下载权重或生成图片。后续优先围绕现有 Qwen 结果定位保真问题，候选的不变编辑诊断仍未执行。

## 原交接快照（2026-11-08）

更新时间：2026-11-08 16:08 UTC 左右（北京时间 2026-11-09 00:08 左右）。

项目根目录：`/nas/wyh/CarImageAugmentation`。

本文供新会话直接接续。内容依据本轮对话、现有代码与文档，以及交接时的文件、依赖和 GPU 检查整理。**当前进度：纯物理基线已完成三轮样图实验；下一步选用 Qwen-Image-2.1 进行原图条件编辑。独立 Conda 环境及 PyTorch 已装好，CUDA/BF16 实际运算通过；生成模型依赖、权重和推理代码尚未部署。**

## 1. 用户目标与协作约定

用户希望从汽车原图生成更多天气变化图，供模型训练使用。核心要求是兼顾汽车细粒度特征和真实环境，不只是叠加雨丝。

需要重点保护车灯轮廓及内部灯带、格栅布局、车标、轮毂辐条、车身比例与折线、窗框，以及基本漆色。允许随天气改变车漆、玻璃、镀铬的照明、反射和湿润外观，但不能改变零件设计或车型身份。

工作方式已经明确：

- 总目录用于试验多种方法，每种方法放入独立、命名清楚的子目录；代码、配置、依赖清单、测试、结果均归属该方法；说明文档统一放入 `docs/`，并标明对应方法。
- 原图共享，禁止覆盖；大模型权重可以共享，避免每种方法重复下载。
- 用户要求安装过程**一个步骤一个步骤来**。检查上一步后再给下一步，不要一次自动安装全部依赖、下载大模型或批量开跑。
- 现有天气结果全部由本地 Python 代码运行生成，没有调用图像生成服务或模型。后续也应明确区分代码运行结果、生成模型结果和方案设想。
- 不承诺“细粒度绝对无损”。没有重绘不代表细节仍可辨，像素约束通过也不代表手工轮廓完全准确或训练性能一定提高。

当前请求是生成这份交接文档，不包含继续安装、下载权重或运行生成模型。

## 2. 目录与资料入口（文档路径已更新）

```text
/nas/wyh/CarImageAugmentation/
├── docs/
│   ├── README.md
│   ├── HANDOFF.md
│   ├── 01_physics_based.md
│   ├── 01_physics_results.md
│   └── generative_weather_plan.md
├── test_images/
│   ├── 奥迪A6L.jpg
│   ├── 宝马4系.jpg
│   ├── 小米YU7.jpg
│   └── 本田CR-V.jpg
└── 01_physics_based/
    ├── requirements.txt
    ├── .gitignore
    ├── run.py
    ├── compare_runs.py
    ├── configs/
    │   ├── scenes.json
    │   └── presets.json
    ├── src/
    │   ├── __init__.py
    │   ├── physics_weather.py
    │   └── scene_weather.py
    ├── tests/
    │   ├── test_physics.py
    │   └── test_scene_weather.py
    └── outputs/
        ├── demo/
        ├── rain_intensity_v2/
        └── overcast_wet_v3/
```

四张原图均为 **3840 × 2880**。早期对话曾用过其他奥迪文件名，目前以这里的实际文件名为准。

优先阅读：

- [总目录说明](README.md)
- [物理方法说明](01_physics_based.md)
- [三轮结果及局限](01_physics_results.md)
- [生成模型调研与实施方案](generative_weather_plan.md)

交接时未发现项目或上级目录适用的 `AGENTS.md`，根目录未见 `.git`。接手时如果文件状态发生变化，应重新检查。

以下目录**只是规划，交接时均不存在**：

```text
models/                         # 共享模型权重
02_qwen_image_edit/             # 首选生成式编辑方法
03_sdxl_controlnet_ipadapter/   # 结构控制对照
04_intrinsic_weather/           # 按需追加的本征空间路线
```

这种目录布局已经与用户达成一致，适合比较多种方法。Conda 环境放在项目外，便于独立维护；未来正式数据集可共享于根目录的 `datasets/`。

## 3. 已完成工作的推进过程

| 阶段 | 用户观察与要求 | 已完成工作 |
| --- | --- | --- |
| 第一轮 | 先做纯物理，检查是否保留细节 | 实现雾、小雨、人工近似场景几何，生成 12 张图 |
| 第二轮 | 下雨不明显，希望有中雨或大雨 | 增加中雨、大雨，生成三档降雨共 12 张图 |
| 第三轮 | 雨丝明显，但地面干、照明仍像晴天 | 增加阴天照明、阴影补偿、湿地与反射近似，生成 12 张图 |
| 方法评估 | 车身是否被锁住？物理方法局限较大 | 明确车体只做受限光度变换，认可需要生成模型处理更复杂的反射与融合 |
| 方案调研 | 多搜集模型与 ControlNet 等借鉴方法 | 已写生成模型调研文档，选 Qwen-Image-2.1 为首选，SDXL 控制路线为对照 |
| 环境准备 | 逐步安装，先选环境方案 | 用户建立独立 Conda 环境，随后装好 PyTorch；交接时完成当前状态检查 |

没有进行训练评测，也没有运行任何生成模型。现有四张照片用于方法与画质验证，不能证明训练收益。

## 4. 纯物理基线的实现

依赖是 NumPy 和 Pillow；无需 GPU。这里的“纯物理”包含解析成像与物理启发的工程近似，没有学习式深度、扩散编辑或车体重绘，不是标定完整的物理逆渲染系统。

### 4.1 雾与雨：`src/physics_weather.py`

- 将 sRGB 转到近似线性 RGB，完成作用后再转回输出。
- 人工背景深度层、近似相机地平面和车辆深度形成 2.5D 几何，再把相机 Z 深度换为沿视线距离。
- 雾采用 `I = J*t + A*(1-t)`，`t = exp(-beta*d)`，远处衰减更多。
- 雨滴在相机视锥中采样，按距离投影尺寸，根据下落速度、风速、曝光时间确定雨丝长度与方向；加入高斯截面、近处失焦和深度遮挡。
- 雨滴正常经过车辆前方，没有为车灯、格栅等区域人为开孔。
- 中、大雨使用可超过显示白色的有效线性辐射亮度，记录最终显示截断比例。

人工几何未经实测标定。小雨、中雨、大雨是**视觉强度档位**，不能当作真实 mm/h 雨量标签。

### 4.2 阴天与湿地：`src/scene_weather.py`

执行顺序为：原图 → 阴天照明近似 → 湿路面 → 空气散射 → 雨滴。

- 通过低频照明增益、人工标注的投影/接触阴影、未遮阴路面参考和原始环境光比例，近似降低直射光观感。
- 车辆 RGB 只乘以有界的逐像素 RGB 增益，没有对车辆原始 RGB 做模糊、形变或重采样。低频滤波用于控制场。
- 湿地只写入车辆保护区之外；采用空气—水 Fresnel（折射率 1.333）、两次透射近似 `(1-F)^2`、漫反射衰减和粗糙反射。
- 地平面坐标中的连续噪声控制非均匀水膜；人工天空区域作阴天外观近似。
- 反射来自同视角可见车辆/背景的 2.5D 代理，不能恢复真实镜面视角中不可见的表面。
- 细化轮胎、车底和车身轮廓，减少湿地边缘留下的多边形干地光圈。

对用户“是不是车体一点没动”的准确回答：**不是完全没动，车体明暗和色调发生了受限变化；车体结构和原有反射内容基本保留。** 最新结果实际增益约 0.740–0.759，主要接近整体压暗，无法把晴天反射真正改为阴天反射。这是后续引入生成编辑的主要原因。

### 4.3 配置与运行行为

- `configs/scenes.json`：按原图文件名索引人工配置；含车辆轮廓、相机、背景层、部件区域及 `weather_geometry`。
- `weather_geometry`：细化车体轮廓、地面、投影、接触阴影、天空、路面亮度参考、轮胎反射接地点等。
- 坐标对应 EXIF 方向校正后的图像，归一化到 `[0,1]`。
- 新原图必须补同名场景配置，否则程序报错；不能假定任意图片自动适配。
- 默认输入为共享 `test_images/`，默认输出为 `outputs/demo/`；默认路径相对脚本定位，自定义相对路径相对执行时的工作目录。
- 拒绝覆盖非空输出目录，也拒绝向原图目录内写结果。
- `--count` 增加雨天随机变体；雾是确定性变换，增加 count 不会产生新的雾形态。
- 种子由基础种子、原图文件名、天气名和变体序号稳定派生。
- 训练候选输出为原尺寸 PNG；缩放图只用于预览。

### 4.4 天气预设

| 预设 | 含义 |
| --- | --- |
| `fog_light` / `fog_medium` | 薄雾 / 较明显的雾，beta 分别为 0.012 / 0.026 |
| `rain_light` / `rain_medium` / `rain_heavy` | 三档降雨，保留原始照明与地面状态 |
| `overcast_wet` | 阴天照明与湿地，无降水；内部沿用 fog 分支且 beta=0 |
| `rain_overcast_wet_medium` | 阴天、湿地与中等视觉降雨 |
| `rain_overcast_wet_heavy` | 阴天、湿地与较强视觉降雨 |

旧降雨参数：

| 参数 | light | medium | heavy |
| --- | --- | --- | --- |
| 雾散射 beta | 0.0005 | 0.003 | 0.008 |
| 雨滴密度假设值 /m³ | 6 | 20 | 55 |
| 曝光秒数 | 0.004 | 0.008 | 0.012 |
| 雨滴直径 mm | 1–2 | 1.2–2.7 | 1.5–3.5 |
| 显式采样深度 m | 默认 40 | 18 | 20 |
| 线性 RGB 辐射亮度 | [0.9, 0.94, 0.98] | [3, 3.2, 3.4] | [6, 6.3, 6.6] |

阴天湿地预设共用：曝光 0.70、车体增益界限 [0.74, 1.08]、白平衡 [0.96, 1, 1.035]、剩余直射光比例 0.08、湿润度 0.92、漫反射保留 0.68、反射强度 0.88。组合降雨调低了雨滴辐射亮度，完整参数以 JSON 和各轮快照为准。

## 5. 已有结果、检查与查看入口

| 输出目录 | 内容 | 训练候选数量 |
| --- | --- | --- |
| `01_physics_based/outputs/demo/` | 四张原图 × 薄雾/较明显雾/小雨 | 12 |
| `01_physics_based/outputs/rain_intensity_v2/` | 四张原图 × 小雨/中雨/大雨 | 12 |
| `01_physics_based/outputs/overcast_wet_v3/` | 四张原图 × 阴天湿地/湿地中雨/湿地大雨 | 12 |

合计 36 个候选 PNG 文件；不同实验间存在重复的小雨结果，不代表 36 个独立天气变化。原有实验曾验证第二轮小雨与首轮小雨逐字节相同。

每轮包含 `images/`、`previews/`、`diagnostics/`、`config_snapshot/` 和 `manifest.json`。清单记录原图/输出哈希、实现哈希、依赖版本、参数、种子、显示截断比例和部件可见度等；新阶段额外记录车体增益与阶段隔离误差。

建议先看最新结果：

- [阴天湿地总览](../01_physics_based/outputs/overcast_wet_v3/previews/overview.jpg)
- [新旧降雨总览](../01_physics_based/outputs/overcast_wet_v3/previews/before_after_overview.jpg)
- [奥迪新旧对比](../01_physics_based/outputs/overcast_wet_v3/previews/奥迪A6L_before_after.jpg)
- [小米新旧对比](../01_physics_based/outputs/overcast_wet_v3/previews/小米YU7_before_after.jpg)
- [本田新旧对比](../01_physics_based/outputs/overcast_wet_v3/previews/本田CR-V_before_after.jpg)
- [奥迪轮胎接地对比](../01_physics_based/outputs/overcast_wet_v3/previews/奥迪A6L_contact_comparison.jpg)
- [宝马关键部件](../01_physics_based/outputs/overcast_wet_v3/previews/宝马4系_details.jpg)
- [最新 manifest](../01_physics_based/outputs/overcast_wet_v3/manifest.json)

此前完成实验时验证：

- 最新 17 项单元测试通过，覆盖雾不变量、距离关系、雨滴遮挡与复现、HDR、输入保护、命令行记录、水 Fresnel、车辆区域隔离、细密图案不模糊、零湿润恒等、湿地随机性范围及非法参数。
- 第二轮最坏显示截断比例：小雨 0%、中雨约 0.130%、大雨约 2.078%；关键车辆区最低散射透射率约为 0.996、0.975、0.936。
- 第三轮 12 张图截断比例均为 0；车体乘法关系最大误差为 0，湿地层在手工车辆保护区内最大改动为 0。
- 第三轮实际车体增益约 0.740–0.759，路面阴影补偿最大增益约 3.771。

**本次交接另行复核**：三轮清单各 12 条，四张当前原图与三轮记录的源图哈希均匹配；第三轮 12 个输出 PNG 的哈希均匹配。单元测试本次未重跑，17 项通过是已有实验记录。历史结果使用 NumPy 2.2.6 / Pillow 12.1.0。

仍有的局限：晴天方向性反射/阴影残留、路面反射视角近似、局部干地边缘、水膜未经凹陷地形求解。没有真实恢复湿车漆水珠、积水、落地水花或镜头雨滴。零误差只证明指定像素操作遵守约束，不是车型保真认证。

## 6. 物理基线复现命令

下面命令使用原有基础环境，不要误认为新建的 Qwen 环境就是历史物理实验环境。选择未存在或为空的新输出目录。

```bash
cd /nas/wyh/CarImageAugmentation

/root/miniconda3/bin/python 01_physics_based/run.py \
  --weather rain_light rain_medium rain_heavy \
  --output-dir 01_physics_based/outputs/rain_new_run

/root/miniconda3/bin/python 01_physics_based/run.py \
  --weather overcast_wet rain_overcast_wet_medium rain_overcast_wet_heavy \
  --output-dir 01_physics_based/outputs/overcast_wet_new_run

/root/miniconda3/bin/python 01_physics_based/compare_runs.py \
  --before 01_physics_based/outputs/rain_intensity_v2 \
  --after 01_physics_based/outputs/overcast_wet_new_run
```

单元测试：

```bash
cd /nas/wyh/CarImageAugmentation/01_physics_based
/root/miniconda3/bin/python -m unittest discover -s tests -v
```

现有 `outputs/demo/` 非空，直接不带参数运行默认命令会被保护机制拒绝；这是预期行为。

## 7. 为什么转向生成模型

用户认为物理方法局限明显，这一判断与现有结果相符。单张已成像照片缺少真实材质、完整光照和不可见几何，仅靠乘法、散射和可见表面反射代理，很难改变车漆/玻璃中的晴天环境反射。

下一步允许生成模型修改照明与反射，同时用原图参考、结构控制和逐部件验收保护车型。避免整车硬遮罩后贴回原车，否则会重新出现“环境变了、车仍像原天气”的问题。

保留已有物理模块：生成模型先做阴天、湿地和融合；合格后再加可控雨层。物理雨不能修复生成阶段已经改错的灯带或轮毂。

## 8. 已确定的模型路线

完整比较、来源与参数见 [生成模型方案](generative_weather_plan.md)。以下均是待验证的实验选择，不是四张原图上的实测质量排名。

### 8.1 首选：Qwen-Image-2.1

- 模型：`Qwen/Qwen-Image-2.1`。
- 管线：`QwenImage21Pipeline`。
- 主输入：未经增强的原图；不要把处理多轮的灰雾图当作唯一身份参考。
- 起点：完整模型、BF16、batch=1、40 步，按此前核查的官方基线不使用 CFG。先不引入量化、Turbo/Lightning 或生成式超分。
- 第一种目标：“雨后阴天、道路湿润、没有正在下落的雨丝”，便于发现部件错误；另设干燥阴天用于区分照明和湿地变化。
- 原图单参考起步；身份漂移时再比较“原图 + 完整车辆裁剪参考”，明确两图用途，防止拼贴、多车或构图改变。
- 不要把 `Qwen-Image-Edit-2511` 与 2.1 混用。前者使用 `QwenImageEditPlusPipeline`，接口与参数含义不同。
- 官方材料中的标记/掩码式局部编辑，不等于 SDXL 的硬 `mask_image` 接口；安装后先核查实际实现。

初始提示词（尚未运行）：

> 将这张照片编辑为刚下过雨的阴天场景。保持原机位、构图、车辆位置与比例，保持同一辆车及其基本漆色。让天空、建筑、车漆和玻璃受到一致的阴天漫射光照，移除明显直射阳光观感；道路湿润，有符合粗糙路面材质的克制反光。允许车漆和玻璃中的环境反射随天气改变。精确保留车灯内部灯带、格栅布局、车标、轮毂辐条、车身折线及窗框。画面中没有正在下落的雨丝，不增加积雪，不增加车辆或物体。

提示词是目标描述，不构成硬保真约束。

### 8.2 对照：SDXL + Canny ControlNet + IP-Adapter Plus

```text
base: stabilityai/stable-diffusion-xl-base-1.0
canny: diffusers/controlnet-canny-sdxl-1.0
depth_optional: diffusers/controlnet-depth-sdxl-1.0
adapter: h94/IP-Adapter
adapter_subfolder: sdxl_models
adapter_weight: ip-adapter-plus_sdxl_vit-h.safetensors
image_encoder: h94/IP-Adapter / models/image_encoder（ViT-H）
pipeline: StableDiffusionXLControlNetImg2ImgPipeline
```

先只用 Canny + IP-Adapter，布局/深度漂移时再加 Depth。Canny 应保留车灯、格栅、轮毂和车身结构，减少车漆/玻璃里的旧环境反射边缘。高频不一定是部件；锁住所有边缘可能同时锁住晴天反射。

建议起点，**未实测**：40 步、CFG 5.5、img2img strength 0.40、Canny 0.70、IP-Adapter 0.50、可选 Depth 0.40；种子 `[42, 123, 2026]`。之后按失败方式单独调整，不一开始全参数搜索。

ControlNet 是底模相关的控制模块。SDXL ControlNet 不能直接接入 Qwen-Image-2.1，IP-Adapter、PnP、MasaCtrl 也存在架构适配边界。

### 8.3 后续候选

- **IntrinsicWeather**：天气专用逆渲染/正向渲染路线。组件包括 `stabilityai/stable-diffusion-3.5-medium`、`facebook/dinov2-base`、`GilgameshYX/InverseRenderer-1024`、`GilgameshYX/ForwardRenderer`。先检查原天气重建是否损伤细节，再编辑天气；预测本征图不是真实材质测量。官方部分示例缩为正方形，必须改为等比例映射。
- **IC-Light**：重光照诊断；**FLUX Kontext / FLUX.2**：替代编辑底模。按首轮失败情况再部署。
- **SaSPA**：借鉴细粒度分类的结构、主体参考与筛选思路；**Diff-Mix**：后续数据集级微调参考；**PnP/MasaCtrl**：可参考的特征保持机制。
- **VDC** 需要天气效果参考/配对条件；**WeatherWeaver** 偏视频，当前四张静态图暂不优先。

这些方法不需要全部组合，也没有全部下载或运行。

## 9. 首选模型要下载什么

下列信息来自此前会话对官方仓库元数据的核查，**本地尚未下载权重**；继续安装时应核对可访问性和实际接口。

| 项目 | 记录 |
| --- | --- |
| 仓库 | `Qwen/Qwen-Image-2.1` |
| 已核查 revision | `d26bb61231c349cf6b7896fa83353113880e1ba3` |
| 仓库总大小 | 约 33.1 GB（十进制） |
| `transformer/` | 约 14.23 GB，2 个 safetensors 分片 |
| `text_encoder/` | 约 17.53 GB，4 个 safetensors 分片 |
| `vae/` | 约 1.35 GB，1 个 safetensors 文件 |
| `processor/` | 约 16 MB，含 tokenizer 和图像处理配置 |
| 其他必需文件 | scheduler、model_index、分片索引及配置等 |
| 可排除 | `assets/*` 展示资源 |
| 建议本地目录 | `/nas/wyh/CarImageAugmentation/models/Qwen-Image-2.1` |

仓库已包含需要的文本/视觉条件组件，不需要另外再单独下载一个完整 Qwen-VL。此前核查的组件类型包括 `Qwen3VLProcessor`、`Qwen3VLForConditionalGeneration`、`QwenImage21Transformer2DModel`、`AutoencoderKLQwenImage21` 和 `FlowMatchEulerDiscreteScheduler`。

当时模型卡采用 Qwen Research 许可，Edit-2511 则标注 Apache-2.0；正式使用以所选 revision 的许可文件为准。此前仓库访问检查显示未 gated，不保证后续状态不变。

到下载步骤时可使用下面命令；**这是后续命令，不是本轮已执行操作**：

```bash
hf download Qwen/Qwen-Image-2.1 \
  --revision d26bb61231c349cf6b7896fa83353113880e1ba3 \
  --local-dir /nas/wyh/CarImageAugmentation/models/Qwen-Image-2.1 \
  --exclude "assets/*"
```

权重加环境、缓存和结果此前粗估预留约 80 GB，实际下载前检查目标磁盘空间。33 GB 磁盘权重不等于推理峰值显存。

## 10. 硬件与 Conda 环境的最新状态

### 10.1 GPU

交接检查：4 张 `NVIDIA A800-SXM4-80GB`，每张总显存 81251 MiB，驱动 `470.161.03`。当时物理 GPU 2 空闲约 80 GB，其他卡有占用。空闲状态随时会变，开跑前重新 `nvidia-smi`。

虽然驱动较旧，**新环境中的 cu118 PyTorch 已完成实际 GPU 运算**。不要仅根据驱动版本就判定当前环境不可用，也不要直接更换共享机器驱动或盲装 CUDA 12/13 轮子。后续模型/加速库如果另有需求，应逐项验证。

### 10.2 Conda 路径约定

用户环境根目录 `/nas/wyh/.conda` 中有 `envs/` 和 `pkgs/`；它不只是一个环境文件夹。当前新环境实际路径：

```text
/nas/wyh/.conda/envs/car-weather-qwen21
```

环境创建命令此前由用户执行：

```bash
conda create -p /nas/wyh/.conda/envs/car-weather-qwen21 python=3.12 -y
conda activate /nas/wyh/.conda/envs/car-weather-qwen21
```

全局 `.condarc` 配置的 `envs_dirs` 是 `/nas/.conda/envs`，`pkgs_dirs` 是 `/nas/.conda/pkgs`，并不是 `/nas/wyh/.conda/...`。因此优先用 `-p` 或绝对路径，不要假定 `conda create -n` 会建在用户希望的位置。不要修改其他现有环境，例如 `dreamcar`。

### 10.3 新环境实测包状态

下表是本次交接现场检查结果，覆盖了此前“还没有 torch”的旧状态：

| 项目 | 当前状态 |
| --- | --- |
| Python | 3.12.16 |
| Python 路径 | `/nas/wyh/.conda/envs/car-weather-qwen21/bin/python` |
| pip | 26.2.2 |
| torch | **2.7.1+cu118，已安装** |
| torchvision | **0.22.1+cu118，已安装** |
| Pillow | 13.0.0 |
| diffusers | 未安装 |
| transformers | 未安装 |
| accelerate | 未安装 |
| safetensors | 未安装 |
| huggingface_hub | 未安装 |
| `pip check` | `No broken requirements found.` |

此前提供给用户的 PyTorch 安装命令是：

```bash
/nas/wyh/.conda/envs/car-weather-qwen21/bin/python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

**该步骤现在已经完成，不要重复要求用户安装 PyTorch。**

本次在物理 GPU 2 上实际执行了新环境的 CUDA/BF16 测试：

```bash
CUDA_VISIBLE_DEVICES=2 /nas/wyh/.conda/envs/car-weather-qwen21/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.is_bf16_supported())
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
print(y[0, 0].item())
PY
```

结果：torch `2.7.1+cu118`，运行时 CUDA `11.8`，CUDA 可用，BF16 支持为 True，运算返回 `16.0`。这证明基础 GPU 路径可用，尚未证明 Qwen 管线能运行。

### 10.4 原有基础环境

`/root/miniconda3/bin/python`：Python 3.10.12、NumPy 2.2.6、Pillow 12.1.0；此前检查 torch 为 2.7.0+cu118。物理实验在这个基础环境完成，不要为了生成模型直接全局升级它。

## 11. 新会话从哪里继续

**下一步是安装并验证支持 Qwen-Image-2.1 的 Diffusers、Transformers 和配套依赖，随后才下载模型。** 继续遵守用户“一次一步”的节奏。

建议接续顺序：

1. 阅读本文件和生成模型方案；确认用户是否又完成了安装，避免重复步骤。
2. 核查官方 Qwen-Image-2.1 模型卡与对应推理实现，确定兼容版本。此前核查记录为 PyTorch >=2.4、Transformers >=5.17，Diffusers 需要包含 `QwenImage21Pipeline` 的源码版本；模型索引记录过 `0.37.0.dev0`，不能据此假定任意稳定版可用。
3. 给出当前这一步的依赖安装命令，在目标 Conda 环境操作。配套包括 `accelerate`、`safetensors`、`huggingface_hub`；Pillow 已安装。固定可用 Diffusers 提交，避免无记录地跟随 main。检查依赖解析是否会替换已验证的 cu118 torch。
4. 安装后验证 `QwenImage21Pipeline` 及相关 Transformers 类可导入，运行 `pip check`；记录完整包版本和源码提交。
5. 用户推进到下载步骤后，再检查空间并下载上述固定 revision；保存清单，避免重复下载。
6. 在 `02_qwen_image_edit/` 内建立最小推理代码、配置、依赖锁定、输出和 manifest，先跑通一张图与一个种子。
7. 检查输出尺寸、车灯/格栅/轮毂和接地关系后，再扩到四张图三种子；再建设 SDXL 对照。

还没核实的内容包括：生成依赖组合的实际兼容性、模型加载与峰值显存、每图耗时、编辑保真率、局部编辑接口和最终训练收益。不能将拟定参数写成已经验证的最佳配置。

如果下一条用户只问“下一步装什么”，直接从生成依赖步骤接续，先明确 PyTorch/CUDA 已检查通过，不重新解释整个研究过程。

## 12. 首轮生成实验与验收设计

### 12.1 范围和数量

总计划：两条路线 × 四张原图 × 一个天气目标 × 三个种子 = **24 张候选**，另加处理/重建诊断。实际先完成 Qwen 单图，再逐步扩展，不要求一次部署两条路线。

首个天气目标是雨后阴天、湿地、无雨丝。先检查基础编辑，之后再增加降水，避免颗粒遮盖结构错误。

### 12.2 分辨率与处理损失

- 原图 3840×2880。不要拉伸车体；保留等比例缩放、padding、裁切及逆映射元数据。
- 此前调研记录 Qwen 2.1 的一个高分辨率画布为 2400×1792，它与原图不是严格相同宽高比。先核对管线支持尺寸，再等比例填充和去边框。
- SDXL 首轮使用约 1MP、保持宽高比的工作画布。
- 同时保存“工作分辨率原图”和生成图，区分缩放损失与模型编辑损失。
- Qwen 做保持原图不变的编辑诊断；SDXL 做 VAE 重建；IntrinsicWeather 做源天气重建。这些是各自漂移诊断，不是等价算法。
- 把 1K/2K 输出放大为 4K 不能证明细节被保留；首轮不使用生成式超分。

### 12.3 三个独立维度

| 维度 | 重点 |
| --- | --- |
| 部件保真 | 灯带、格栅、车标、辐条、车身比例/折线、窗框、漆色类别；可见增删或替换判失败 |
| 天气融合 | 反射、光照、投影、路面粗糙反光、轮胎接地是否协调 |
| 生产成本 | 全部候选、合格数量/比例、失败类型、每张合格图时间与显存 |

逐图保留原图、未经贴回修补的输出、整车/部件/接地裁剪、提示词、参数、种子、权重 revision、依赖、耗时和显存。不只展示最好看的结果；修补前后分开统计。

CLIP、DINO、分类器置信度、SSIM、LPIPS 可辅助筛选，不能认证灯带或格栅细节。自动阈值需要人工合格/失败样本校准。

### 12.4 接物理雨和训练验证

只有无雨丝的环境编辑通过验收后，才叠加现有物理雨。需要按生成图画布映射几何与掩码并重新检查，不能直接把原图配置无条件套上。分别记录“生成环境”和“生成环境 + 物理雨”。

扩充数据后，按原始照片及同次拍摄组划分训练/验证/测试，同一原图的所有衍生图留在同一集合。保持车型和天气采样均衡，以独立真实天气测试集验证收益，同时关注晴天表现和易混车型错误。现有四张图的合成观感不能替代这一评测。

## 13. 官方来源与后续核查入口

更完整的论文/代码链接已在研究文档中，本文件保留接续部署最需要的入口：

- [Qwen-Image-2.1 模型仓库](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [QwenImage21Pipeline 文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage21)
- [Diffusers 官方代码](https://github.com/huggingface/diffusers)
- [PyTorch 2.7.1 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/#v271)
- [SDXL ControlNet 文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl)
- [IP-Adapter 作者仓库](https://github.com/tencent-ailab/IP-Adapter)
- [IntrinsicWeather 作者仓库](https://github.com/YixinZhu042/IntrinsicWeather)

旧研究文档的环境描述是调研时快照；如果与本文件的现场检查冲突，以本文件第 10 节以及接手时的实际检查为准。当前最关键的状态变化是：**新环境的 PyTorch 已安装且 CUDA/BF16 运算通过，生成依赖与权重仍未部署。**
