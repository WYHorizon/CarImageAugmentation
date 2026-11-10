# 汽车细粒度天气增强：生成模型调研与实施方案

[文档首页](README.md) · [会话交接](HANDOFF.md)

调研日期：2026-11-08。任务是从原始汽车照片生成训练增强图，保持车型/年款/配置相关结构，改善车辆、反射、照明、地面与天气的一致性。

本次完成官方论文、项目、模型卡和部分推理代码核查，没有下载生成模型权重、安装依赖或运行生成模型。下面的排序是实验选型判断，不是当前四张图上的实测排名。参数表中明确标为“建议起点”的数值也不是论文最优参数。

## 1. 选型结论

第一批部署两条路线：

1. **Qwen-Image-2.1 原图条件编辑**：作为直接编辑、允许车漆/玻璃反射改变的主实验入口。
2. **SDXL + Canny ControlNet + IP-Adapter Plus**：作为可显式调节结构和参考外观约束的对照。先用 Canny，必要时增加 Depth；不一开始堆满所有控制器。

第三条针对性路线是 **IntrinsicWeather**：先验证逆渲染和源天气重建，再决定是否批量编辑。IC-Light 可做车辆重光照诊断；FLUX 可作替代底模；这些都不必与第一批同时部署。

最终候选流程：原图条件生成阴天/湿润场景 → 车辆部件验收 → 可控物理雨层。物理雨只能增加降水多样性，不能修复生成阶段已经改错的部件。

## 2. 本机条件与实施约束

本机检查到 4 张 NVIDIA A800-SXM4-80GB。检查时其中一张卡空闲约 80GB；这是瞬时状态，实际开跑前重新选择空闲卡。现有环境是 Python 3.10、PyTorch 2.7.0+cu118，尚无 diffusers、transformers、accelerate、safetensors、OpenCV。

建议各方法建立独立环境，模型权重共享缓存。优先使用单张 80GB 卡、batch=1；BF16/FP16 按具体模型选择，实测峰值显存，必要时 CPU offload。显卡数量和总显存不能直接视为一张可用大显卡的容量。

不预报每张图耗时或显存峰值，因为还未运行。先建立能完整跑通一张图的环境，再冻结依赖与权重版本。

## 3. 模型与成熟度比较

| 模型/路线 | 已核实的能力或资源 | 适合本任务的角色 | 主要限制 |
| --- | --- | --- | --- |
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 开放权重；视觉生成部分 7B；原图/多图参考、局部编辑提示；官方 Diffusers 接口 `QwenImage21Pipeline` | 首选直接编辑入口 | 不是汽车部件精确保真器；局部掩码提示不等于像素硬约束；模型卡标记 Qwen Research 许可 |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | `QwenImageEditPlusPipeline`；多图编辑；模型卡强调一致性改善；Apache-2.0 | 2.1 的版本对照或替代 | 与 2.1 接口、参数、条件机制不同；不能共用一套参数含义 |
| [SDXL + ControlNet](https://huggingface.co/docs/diffusers/main/en/api/pipelines/controlnet_sdxl) + [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter) | 现成 SDXL Canny/Depth 权重及图像参考适配器；作者提供组合示例 | 结构约束对照、参数消融 | 原始 SDXL 分辨率与细节上限有限；控制器过强会把旧反射/阴影一起留下 |
| [IntrinsicWeather](https://github.com/YixinZhu042/IntrinsicWeather) | 官方逆渲染、正向渲染代码和权重；基于 SD3.5 Medium；有 512/1024 逆渲染模型 | 天气专用、材质/几何条件路线 | 驾驶场景到车辆特写的迁移待验证；预测的本征图可能已经丢失灯带和格栅细节 |
| [IC-Light](https://github.com/lllyasviel/IC-Light) | 前景条件及前景+背景条件重光照 | 判断生成式重光照能改善多少融合感 | 本身不是完整天气/湿地编辑系统；公开示例大量为人像，汽车保真须实测 |
| [FLUX.1 Kontext dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | 12B 指令编辑、参考对象一致性、开放权重 | 直接编辑替代底模 | 下载需接受访问条件；没有汽车小部件保证 |
| [FLUX.2 dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) | 32B 生成/编辑、多参考图；[官方推理仓库](https://github.com/black-forest-labs/flux2) | 后续质量对照 | 模型及文本编码器资源开销更大，不列为第一批；仍需实际比较保真 |

Qwen-Image-2.0 也有官方发布与服务，但本次已核实、可直接定位的本地部署候选是上表中的 2.1 和 Edit-2511，不把一个新闻/论文名称直接当作可下载检查点。

## 4. ControlNet 等方法分别解决什么

**ControlNet 是控制模块，不是可以独立生成目标图的通用底模。** 不同底模需要适配的控制权重。SDXL Canny ControlNet 不能直接接到 Qwen-Image-2.1；IP-Adapter、MasaCtrl、PnP 也存在架构和实现适配边界。

| 方法 | 核心作用 | 本任务的用法 |
| --- | --- | --- |
| [ControlNet](https://arxiv.org/abs/2302.05543) | 用边缘、深度等空间条件限制生成 | 约束车体轮廓、部件轮廓、场景布局 |
| [IP-Adapter Plus](https://huggingface.co/h94/IP-Adapter) | 用原图视觉特征作为参考条件；Plus 使用 patch 特征 | 用原始车辆裁剪补充外观线索，选普通 Plus，非人脸 FaceID 版本 |
| [PnP Diffusion Features](https://pnp-diffusion.github.io/) | 向生成过程注入原图空间特征与注意力信息 | 若 SDXL 结构漂移明显，作为后续特征保持思路；不是 Qwen 即插即用插件 |
| [MasaCtrl](https://github.com/TencentARC/MasaCtrl) | 从源图查询相关内容/纹理的互自注意力控制 | 备选一致性机制；作者有 SDXL 实现，但组合前仍需兼容性测试 |
| [Blended Latent Diffusion](https://arxiv.org/abs/2206.02779) | 局部编辑及潜空间重建误差处理 | 借鉴区域融合；认识到“不编辑某区域”不自动等于精确重建 |
| [SaSPA](https://eyalmichaeli.github.io/SaSPA-Aug/) | 边缘+主体表示+过滤的细粒度分类增强 | 借鉴结构与身份双条件，以及训练集评测设计；其同类参考图策略不保证保留本张照片每个零件 |
| [Diff-Mix](https://github.com/Zhicaiwww/Diff-Mix) | 针对分类数据微调与跨类别增强 | 借鉴后续数据集级适配；跨类混合不是当前保持原车型的天气编辑目标 |
| [VDC](https://github.com/omarAlezaby/VDC) | 从有/无效果的配对视觉示例提取编辑条件 | 若有真实配对天气参考，可研究视觉天气条件；当前没有这类配对数据，不是第一批 |
| [WeatherWeaver](https://arxiv.org/abs/2505.00704) | 视频天气生成/移除 | 将来处理汽车视频再考虑，当前四张静态图无需引入视频管线 |
| [Alterbute](https://talreiss.github.io/alterbute/) | 按细粒度对象身份编辑材质等属性 | 借鉴“身份与可变外观分开”的训练设计；项目也允许形状变化，不能照搬其身份宽容度 |

并非要把所有方法组合起来。第一批只部署最少必要模块，观察真实失败方式后再加控制。

## 5. 路线 A：Qwen-Image-2.1

### 输入与参数

- 主输入是未增强原图，不把多轮物理处理后的灰雾图作为唯一身份参考。
- 初始配置仅用一张原图；有漂移时另设“原图+车辆裁剪参考”配置，明确图 1 决定布局，图 2 仅提供车辆细节，防止生成拼贴或多辆车。
- 使用完整模型、BF16、batch=1、固定种子。第一轮不加 Turbo/Lightning，不先引入量化作为混杂因素。
- [官方 Diffusers 文档](https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage21)建议的基线是 40 步、无 CFG。这里先按该起点测试，勿把 SDXL 的 CFG/strength 参数直接套用到此接口。
- [官方模型卡](https://huggingface.co/Qwen/Qwen-Image-2.1)要求相应版本的 Transformers 和包含该管线的 Diffusers；安装时核实 `QwenImage21Pipeline` 可导入，再锁定提交和依赖。不要在现有物理方法环境里直接升级全部包。

当前拟用接口：`QwenImage21Pipeline.from_pretrained`，以 `image` 提供原图、`prompt` 提供目标、`num_inference_steps=40` 与显式生成器种子采样。这里未提供一个假装已经验证过的批量脚本。

### 目标描述

第一种目标先用“雨后阴天，无正在下落的雨丝”。建议提示词：

> 将这张照片编辑为刚下过雨的阴天场景。保持原机位、构图、车辆位置与比例，保持同一辆车及其基本漆色。让天空、建筑、车漆和玻璃受到一致的阴天漫射光照，移除明显直射阳光观感；道路湿润，有符合粗糙路面材质的克制反光。允许车漆和玻璃中的环境反射随天气改变。精确保留车灯内部灯带、格栅布局、车标、轮毂辐条、车身折线及窗框。画面中没有正在下落的雨丝，不增加积雪，不增加车辆或物体。

第二种目标是“干燥阴天”，用于区分重光照与湿地修改。初次不要用浓雾、暴雨遮住结构错误。上述提示是本项目拟定的实验指令，不是保证保真的技术约束。

### 局部编辑

2.1 的官方资料描述了参考图、绘制标记或掩码式局部编辑能力。先核实所选推理实现具体怎样接收这些条件；不能假设存在与 SDXL 相同的 `mask_image` 或 ControlNet 参数。

对车漆和玻璃开放反射与光照变化；车灯、格栅、轮毂优先保持形状与内部布局，但仍允许随环境改变明暗。模型输出需经过部件验收。首轮评分以未经贴回原车/零件修补的输出为准，修补需求另计。

## 6. 路线 B：SDXL + ControlNet + IP-Adapter Plus

> 最新决策（2026-11-09）：用户要求没有优于 Qwen 的依据就不试。当前没有本项目证据证明此组合更优，暂不推进部署；本节与后面的双路线实验矩阵保留为备选规划。最新状态见 [SDXL 实施说明](03_sdxl_controlnet_ipadapter.md)。

具体模型组件：

```text
base: stabilityai/stable-diffusion-xl-base-1.0
canny_control: diffusers/controlnet-canny-sdxl-1.0
depth_control_optional: diffusers/controlnet-depth-sdxl-1.0
reference_adapter: h94/IP-Adapter
adapter_subfolder: sdxl_models
adapter_weight: ip-adapter-plus_sdxl_vit-h.safetensors
image_encoder: h94/IP-Adapter / models/image_encoder (ViT-H)
pipeline: StableDiffusionXLControlNetImg2ImgPipeline
```

依据：[Canny 模型卡](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)、[Depth 模型卡](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0)、[IP-Adapter 权重说明](https://huggingface.co/h94/IP-Adapter)、[作者组合示例](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter_sdxl_controlnet_demo.ipynb)。需在部署时检查当前管线与 Plus 权重的图像编码器加载兼容性。

操作顺序：

1. 原图等比例缩放/填充至工作画布；从原图生成结构控制图。
2. 对 Canny 图作区域处理：保留外轮廓、门缝、窗框及灯/格栅/轮毂结构；减少车漆与玻璃内部树枝、建筑反射边缘。反射本身也有高频，不能把“高频”全部当作车型结构。
3. 以原图启动 img2img；以完整车辆裁剪作为 IP-Adapter 参考。检查 CLIP 图像预处理是否裁掉车头/车尾，使用有留边的车辆参考。
4. 初始只用 Canny + IP-Adapter；整体深度/布局漂移时才加 Depth。反光玻璃的深度估计需人工检查。
5. 同时允许车身和环境编辑，输出后做部件对比。整车完全硬遮罩再贴回原图，会重现当前融合问题。

**本项目建议起点，尚未实测：**

```yaml
steps: 40
cfg: 5.5
img2img_strength: 0.40
canny_scale: 0.70
ip_adapter_scale: 0.50
depth_scale_if_enabled: 0.40
seeds: [42, 123, 2026]
```

第一轮不做全笛卡尔参数搜索。根据结果再单独比较 strength=0.25/0.40/0.55；过强的 Canny/IP 可能留下旧天气，过弱则可能改错车型。降低控制和保持结构之间需要实际权衡。

## 7. 路线 C：IntrinsicWeather

官方资料：[项目说明](https://yixinzhu042.github.io/IntrinsicWeather/)、[代码及权重操作](https://github.com/YixinZhu042/IntrinsicWeather)。

所需组件包括：

```text
stabilityai/stable-diffusion-3.5-medium
facebook/dinov2-base
GilgameshYX/InverseRenderer-1024
GilgameshYX/ForwardRenderer
```

操作：原图 → 预测 albedo/normal/roughness/metallic/irradiance → 保存所有中间图 → 用源天气条件重建原图 → 检查部件结构 → 再按 overcast/rainy 条件渲染。

必须先做源天气重建检查。如果没有改变天气时就丢掉灯带、轮毂结构，不应将后续输出自动视为合格增强图。预测的材质与法线是模型估计，不是真实测量。

作者公开示例中，输入和本征图有直接缩为正方形的步骤：[逆渲染预处理](https://raw.githubusercontent.com/YixinZhu042/IntrinsicWeather/main/gradio_inverse_demo.py)、[正向渲染预处理](https://raw.githubusercontent.com/YixinZhu042/IntrinsicWeather/main/gradio_forward_demo.py)。本项目的 4:3 图必须等比例填充并记录逆变换，避免把车体拉伸。其 512/1024 本征图分辨率也是车辆特写细节的风险。

第一轮复用预训练检查点，不需要下载整个 WeatherSynthetic 数据集，也不训练新网络。官方环境采用 Python 3.12；该管线与直接图像编辑的环境和运行流程分开。

## 8. 最小实验与验收

### 阶段 0：建立处理损失基线

- 统一方向、颜色模式、画布映射，保存 scale/padding/crop 元数据。
- 同时保留“工作分辨率的原图”与生成图对照，分开判断缩放损失和生成修改。
- Qwen 做一次“保持原图不变”的编辑诊断；SDXL 检查 VAE 重建；IntrinsicWeather 检查源天气重建。它们只是测量各自固有漂移，不能直接当作公平的相同算法测试。

### 阶段 1：先筛掉结构不可靠的路线

两条首批路线 × 四张原图 × 一个目标（雨后阴天、无雨丝）× 三个种子 = **24 张候选图**。加上处理/重建基线，另行统计总成本。

逐图保留：原图、未经修补输出、车辆裁剪、关键部件裁剪、轮胎接地处、参数、权重 revision、依赖版本、种子、耗时和峰值显存。不要只展示每组最佳图。

### 阶段 2：针对问题追加实验

- 第一轮融合好但车型结构漂移：在 SDXL 路线增加局部结构控制/特征保持；在 Qwen 路线比较单图与车辆裁剪参考。
- 通用模型普遍保不住结构：追加 IntrinsicWeather 的材质/几何约束路线，先通过重建检查。
- 两条都保真但外观变化不足：减弱把源反射锁住的条件，不把整辆原车复制回生成环境。

### 三个独立验收维度

1. **部件保真**：灯带轮廓与内部线条、格栅布局、轮毂辐条、车身折线、窗框、漆色类别。肉眼可见的部件增删或替换判为结构失败。
2. **天气融合**：车漆/玻璃反射是否符合环境、明暗与投影是否协调、道路粗糙度和反光是否自然、接地关系是否合理。
3. **生产成本**：候选数、通过数、通过率、每张合格图耗时与显存。标记“结构错误”“天气变化不足”“接触/反射不自然”等失败类型。

CLIP/DINO/分类器置信度只用于辅助筛选；它们不能认证格栅横条、灯带或轮毂细节。SSIM/LPIPS 也不能把正常重光照和结构错误简单区分。自动阈值需以人工标注的合格/失真样本校准，不预设一个相似度数字作保证。

### 阶段 3：恢复可控降雨并验证训练收益

先验收不带雨丝的生成图，再接已有物理雨模块，按生成图画布映射深度和掩码。分别记录“生成式环境”“生成式环境+物理雨”，定位真正有效的变化。

之后扩到多车型、多视角、多车色，按原始照片/拍摄组划分训练验证测试，所有衍生图跟随原图分组；用独立真实天气集评估，兼顾晴天性能和易混车型错误。四张样图不能证明训练收益。

## 9. 分辨率与细粒度的实际边界

输入是 3840×2880，生成模型不一定在同一分辨率工作。Qwen-2.1 官方列出 4:3 的高分辨率画布为 2400×1792，使用时仍需正确填充/裁去边框，不能把它当作与原图严格同宽高比。SDXL 首轮使用约 1MP、保持宽高比的工作画布，并在原生工作分辨率与共同评测分辨率分别对照。

不能将 1K/2K 输出放大回 4K，就声称原始细节完整。生成式超分也可能凭空改出灯带与辐条，不列入第一轮。若任务最终需要真实 4K 零件信息，应将分辨率本身作为额外约束，后续研究局部高分辨率处理与全局光照一致性。

## 10. 目录规划与下一步交付

以下是规划，不代表已经部署：

```text
CarImageAugmentation/
├── test_images/                       # 共享原图
├── models/                            # 共享权重缓存，实际部署时再建
├── docs/                            # 项目说明与各方法文档
│   └── generative_weather_plan.md
├── 01_physics_based/                  # 已完成的物理基线
├── 02_qwen_image_edit/                # 原图条件生成式编辑
├── 03_sdxl_controlnet_ipadapter/       # 可控扩散对照
└── 04_intrinsic_weather/              # 本征空间天气路线，按需追加
```

各方法目录保存自己的配置、依赖清单、批处理代码、控制图、输出和 manifest；环境说明、方法说明与评估报告统一放在 `docs/`，并标明对应方法；大权重和原图共享。第一轮交付应是 24 张候选 + 重建基线 + 统一部件/整图对比 + 失败统计，再根据通过率决定下一条路线。
