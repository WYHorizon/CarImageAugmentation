# 02 — Qwen-Image-2.1 原图条件天气编辑

[文档首页](README.md) · [生成模型方案](generative_weather_plan.md)

本方法使用本地 Qwen-Image-2.1 权重编辑天气。首轮只处理奥迪 A6L 一张原图，目标为“雨后阴天、湿路面、没有正在下落的雨丝”。允许生成模型改变车辆照明和反射；灯带、格栅、车标、轮毂等部件是否保持，需要查看输出逐项验收。

## 实现与环境

- [推理脚本](../02_qwen_image_edit/run.py)：离线加载、等比例缩放与补边、生成、记录和预览。
- [首轮配置](../02_qwen_image_edit/configs/audi_overcast_wet_seed42.json)：输入、提示词、种子、画布和推理参数。
- [主要依赖](../02_qwen_image_edit/requirements.txt)：已验证环境的核心包版本；每次运行另存完整包版本快照。
- Python 环境：`/nas/wyh/.conda/envs/car-weather-qwen21`。
- 模型目录：`/nas/wyh/CarImageAugmentation/models/Qwen-Image-2.1`。
- 权重 revision：`d26bb61231c349cf6b7896fa83353113880e1ba3`。

权重已在本会话完成检查：27 个文件共 33,131,616,761 字节，7 个 safetensors 文件的 SHA256 与仓库一致，其余文件 Git blob 哈希及分片索引通过。BF16 离线加载在物理 GPU 2（A800 80GB）通过；单纯加载耗时 9.5 秒，PyTorch 峰值已分配显存约 30.23 GiB。这不是生成阶段的显存指标。

脚本在进程内将 HF、Torch、pip、XDG 缓存和临时目录指向 `/nas/wyh/.cache/` 下对应目录，启用离线模式。无需修改全局 shell 配置。

## 首轮参数与分辨率

| 项目 | 设置 |
| --- | --- |
| 输入 | 未增强的 `test_images/奥迪A6L.jpg`，3840×2880 |
| 权重精度 | 完整 BF16；不量化 |
| 种子 / 步数 / batch | 42 / 40 / 1 |
| CFG | `true_cfg_scale=1.0`，不使用负提示词 |
| 条件 KV 缓存 | `use_kv_cache=true` |
| 注意力 | 默认 `QwenImage21AttnProcessor`；不启用编译或 Flex Attention |
| VAE tiling | 首轮关闭 |
| 生成画布 | 2400×1792 |
| 原图内容缩放 | 2388×1791，严格保持 4:3，缩放系数 0.621875 |
| 补边 | 左 6、上 0、右 6、下 1 像素，边缘复制 |
| 去边框输出 | 2388×1791，未放大回 4K |

管线的 `output_resolution` 同时影响参考图内部缩放。本轮设为 2074，并在运行前用已安装管线的 `calculate_dimensions` 验证参考画布仍为 2400×1792，避免仅指定输出尺寸却让参考图再次缩小或改变比例。

## 运行

先用 `nvidia-smi` 检查 GPU 空闲情况。以下 GPU 2 是本轮使用的卡号，后续运行需要重新检查。

```bash
cd /nas/wyh/CarImageAugmentation
CUDA_VISIBLE_DEVICES=2 /nas/wyh/.conda/envs/car-weather-qwen21/bin/python \
  02_qwen_image_edit/run.py \
  --config 02_qwen_image_edit/configs/audi_overcast_wet_seed42.json \
  --output-dir 02_qwen_image_edit/outputs/audi_overcast_wet_seed42_repeat01
```

输出目录必须不存在；脚本拒绝覆盖已有实验，也拒绝写入原图目录或模型目录。`--check-config` 仅检查输入与坐标映射，不加载模型权重、不生成结果。

## 输出与验收

每次运行保存 `config.json`、`manifest.json`、`run_snapshot.py`、`environment_freeze.txt`、`scene_reference.json`、`progress.json`、`crop_regions.json`，以及：

- `source_canvas.png`：实际传给管线的补边参考图。
- `source_working.png`：等比例缩小的原图内容，供同尺寸比较。
- `edited_canvas.png`：管线未经修补的原始输出，保留其原始颜色模式。
- `edited.png`：只按已记录位置去掉补边；不贴回原车或零件、不超分。
- `previews/`：整图、整车、四个部件与轮胎接地对比。
- `crops/`：原图和输出在同一坐标的原生分辨率 PNG 裁剪。

预览若遇到 RGBA 输出，只为显示而合成到白底；原始 RGBA 文件单独保留，manifest 记录 alpha 范围及非不透明像素比例。

manifest 记录输入/输出哈希、代码与管线实现哈希、权重下载 revision、调度器配置、参数、GPU、依赖、加载及推理耗时、PyTorch 峰值已分配/预留显存。逐步检查 latent 是否出现非有限值。运行失败也保留 manifest 和错误信息。

成功运行后的状态为 `complete_pending_visual_review`，不代表车型保真验收通过。首轮不进行批量扩展；先检查车灯灯带、格栅、车标、轮毂、车身比例、反射、路面和接地关系，并区分工作分辨率缩放损失与模型重绘造成的变化。

## 首张结果（2026-11-09）

已完成奥迪 A6L 的单图编辑。完整运行记录位于 `02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/`：

- [整图对比](../02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/previews/overview.jpg)：左为缩至工作分辨率的原图，右为生成结果。
- [生成图 PNG](../02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/edited.png)：2388×1791，仅去掉补边。
- [车灯、格栅和轮毂对比](../02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/previews/details.jpg)。
- [运行记录](../02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/manifest.json)、[文件检查](../02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/verification.json)、[视觉检查](../02_qwen_image_edit/outputs/audi_overcast_wet_seed42_2k_retry01/review.json)。

| 项目 | 实测结果 |
| --- | --- |
| 模型加载 | 8.42 秒 |
| 生成（含解码） | 116.23 秒 |
| 全流程（含加载、保存及预览） | 135.99 秒 |
| PyTorch 峰值已分配显存 | 65.47 GiB |
| PyTorch 峰值预留显存 | 76.27 GiB |
| 推理检查 | 40 步全部完成，逐步 latent 有限值检查通过 |
| 文件检查 | 16 个 PNG 的哈希及解码检查通过，原图哈希未变，画布与去边框尺寸符合配置 |

显存峰值包含 VAE 解码，明显高于单纯加载和去噪阶段。程序退出后，物理 GPU 2 显存占用已恢复为 0 MiB。以上仅是这张图、本环境和这组设置的实测，不代表其他分辨率的预算。

视觉检查结论：**天气编辑生效，但本张暂不通过严格细粒度保真验收，也不据此扩展训练集。**

- 阴天天空、整体漫射光、湿路面及车辆下方反射已出现，原图的强烈日照阴影明显减弱。
- 车辆整体位置、外形和基本漆色大体保持；车灯外轮廓、格栅和四环标识仍可辨认。
- 车漆、玻璃和镀铬表面的细密颗粒感明显增强，局部灯内纹理有重绘痕迹；前轮中心标识变形、辨识度下降，轮胎字样及轮毂细部也有损失。没有据此断言辐条数量改变。
- 上述判断对照的是同尺寸原图，不把单纯缩小分辨率造成的损失归因于编辑。未进行训练评估，也未使用原车像素贴回掩盖变化。

原始输出为 RGBA，alpha 范围 251–255，约 7.54% 的像素 alpha 小于 255；预览按前述规则合成到白底。原始输出与显示用预览均保留。

### 技术重跑说明

首次运行 `audi_overcast_wet_seed42_2k/` 已完成生成与图片保存，但写入含中文的最终 manifest 时，进程默认编码为 ASCII，发生 `UnicodeEncodeError`。当时的图片、日志、步进记录及代码快照已保留，并用 `recovery.json` 和恢复说明版 `manifest.json` 标明记录失败；未补造丢失的完整耗时和解码显存数据。

已将文本读写统一为显式 UTF-8，JSON 改为同目录临时文件写入后替换，并在预览制作前保存生成指标。使用同一原图、提示词、种子和参数重跑一次，完成全部记录。两次的输入画布、工作原图、生成画布及去边框生成图 SHA256 均完全一致，因此这是一次技术重跑，不是从多个种子中筛选样图。上表数据全部来自成功重跑。

建议下一步只做一张“保持原图不变”的同种子、同画布对照，检查基础重建是否也产生颗粒感和零件漂移，再决定调整天气提示词或其他参数。本轮尚未运行该对照。

## 多车辆、多天气扩展（2026-11-09，已完成）

用户随后要求继续生成其他天气及其他车辆。本轮按该请求扩展样图，暂未执行前述不变重建对照。

- 车辆：奥迪 A6L、宝马 4 系、小米 YU7、本田 CR-V。
- 天气：雨后阴天无雨丝、中等视觉强度降雨、薄雾，共 12 个组合；“中雨”“薄雾”是提示词描述，不是标定气象数据。
- 已有奥迪雨后图复用，新增 11 张。均直接读取 `test_images/` 原图，不连续编辑已生成图片。
- 全部使用 seed 42、40 步、2400×1792 画布、BF16、CFG 1.0、KV cache 开启、VAE tiling 关闭；原图缩放、补边及去边框规则沿用首轮。
- 雨后阴天沿用原提示词。中雨提示词要求可见但不过密的细雨丝、轻微雨幕及湿路面；薄雾提示词要求远景对比度下降、近车清晰、路面干燥。新提示词还要求表面平滑、保留标识和文字，这些仍是软约束，不保证部件或像素不变。

配置入口为 [批次配置](../02_qwen_image_edit/configs/weather_batch_v1.json)，具体提示词在其引用的各单图 JSON 中。执行器为 [run_batch.py](../02_qwen_image_edit/run_batch.py)，汇总工具为 [summarize_batch.py](../02_qwen_image_edit/summarize_batch.py)。单图推理代码仅把预览天气标题改为读取配置字段，生成参数与管线调用方式未改。

本轮通过独立后台会话串行执行；批次目录为 `02_qwen_image_edit/outputs/weather_batch_v1/`。`batch.log` 是队列日志，`batch_status.json` 记录当前任务和完成数，`logs/` 保存逐图日志，`runs/` 保存各图完整输出。缓存及临时文件仍指向 `/nas/wyh/.cache/`。

查看进度：

```bash
cd /nas/wyh/CarImageAugmentation
cat 02_qwen_image_edit/outputs/weather_batch_v1/batch_status.json
tail -f 02_qwen_image_edit/outputs/weather_batch_v1/batch.log
```

生成后刷新汇总：

```bash
/nas/wyh/.conda/envs/car-weather-qwen21/bin/python \
  02_qwen_image_edit/summarize_batch.py \
  --batch-config 02_qwen_image_edit/configs/weather_batch_v1.json \
  --output-dir 02_qwen_image_edit/outputs/weather_batch_v1/report
```

完整 PNG、各部件对比及逐图 manifest 保存在各自输出目录。生成成功本身不表示细粒度保真或训练有效性通过。

### 结果入口

- [完整浏览页面](../02_qwen_image_edit/outputs/weather_batch_v1/report/index.html)：每辆车的横向对比，可打开生成 PNG、零件对比和逐图记录。
- [四车三天气总览](../02_qwen_image_edit/outputs/weather_batch_v1/report/overview.jpg)：每行一辆车，每列依次为原图、雨后阴天、中雨、薄雾。
- 分车查看：[奥迪 A6L](../02_qwen_image_edit/outputs/weather_batch_v1/report/audi_comparison.jpg) · [宝马 4 系](../02_qwen_image_edit/outputs/weather_batch_v1/report/bmw_comparison.jpg) · [小米 YU7](../02_qwen_image_edit/outputs/weather_batch_v1/report/xiaomi_comparison.jpg) · [本田 CR-V](../02_qwen_image_edit/outputs/weather_batch_v1/report/honda_comparison.jpg)。
- [完整文件校验](../02_qwen_image_edit/outputs/weather_batch_v1/report/summary.json) · [运行指标](../02_qwen_image_edit/outputs/weather_batch_v1/report/metrics.json)。
- 逐图视觉记录：[雨后阴天](../02_qwen_image_edit/outputs/weather_batch_v1/report/visual_review_overcast.json) · [中雨](../02_qwen_image_edit/outputs/weather_batch_v1/report/visual_review_rain.json) · [薄雾](../02_qwen_image_edit/outputs/weather_batch_v1/report/visual_review_fog.json)。

### 执行与检查结果

| 项目 | 结果 |
| --- | --- |
| 组合完成数 | 12/12；本轮新增 11 张，复用首轮奥迪雨后图 1 张 |
| 新增图尺寸 | 均为 2388×1791，未超分、未贴回原车零件 |
| 批次实际耗时 | 28 分 14 秒，包含串行启动、模型加载、生成、保存及校验；不含最终整理报告 |
| 新增图生成耗时 | 每张 116.32–116.91 秒，平均 116.59 秒；11 张生成时间合计 21 分 23 秒 |
| 峰值已分配显存 | 65.47–65.50 GiB |
| 峰值预留显存 | 75.54–76.27 GiB |
| 文件检查 | 12/12 通过；共 192 个 PNG 的哈希、格式及解码检查通过，主图尺寸符合映射 |
| 输入保护 | 四张原图 SHA256 均未改变 |
| 推理状态 | 新增 11 张均完成 40 步，未触发 latent 非有限值检查，无失败重跑 |
| 批次结束 | 后台队列已完成，GPU 2 占用恢复 0 MiB |

三类天气均在四辆车上产生了可辨识的变化。雨后图有阴天光照、湿路面与反射；中雨图增加了细雨丝和轻微雨幕；薄雾图有背景雾化，近车仍清楚，路面保持干燥，未见明显误加雨雪。

薄雾组的车漆和玻璃比雨后/中雨组明显平滑，密集颗粒感减轻。这只是当前样图的比较结果：不同天气提示词并不相同，尚不能据此单独确定颗粒的成因。

所有结果仍作为**待筛选的天气候选样图**，本轮未判定任何一张通过严格细粒度保真验收。大体车型、位置和基本颜色保持，但各图仍有程度不一的灯内纹理简化、轮毂中心标识重绘、车牌小字或轮胎文字变化；雨丝和雾气还会降低可见度，不能把所有清晰度下降都归因于结构改变。没有依据预览断言辐条数量改变，也没有进行训练效果评估。

本轮按用户要求完成其他天气和车辆的样图扩展，没有新增种子搜索或进一步训练。后续如要追查细节损失，仍可使用前述“保持原图不变”的单张对照；该对照尚未执行。

## 降雪与部分积雪补充（2026-11-09，已完成）

按用户要求，四辆车各新增一张“正在下雪、局部浅积雪”结果。全部从对应未增强原图开始，沿用 seed 42、40 步、2400×1792 画布、BF16 等设置，去边框输出仍为 2388×1791；没有覆盖旧实验。

提示词要求地面和路边出现不连续薄雪、保留部分裸露路面，车顶及机盖少量积雪，大部分漆色和关键部件仍可见。具体提示词和原图路径见 [雪景批次配置](../02_qwen_image_edit/configs/snow_batch_v1.json) 引用的各单图 JSON。仍是原图条件生成式编辑，积雪和雪花都由本地 Qwen 生成，没有另叠程序化雪层。

### 查看结果

- [四辆车的雪景页面](../02_qwen_image_edit/outputs/snow_batch_v1/report/index.html)：原图/雪景对比、原生 PNG、零件放大图及运行记录。
- [雪景对比总览](../02_qwen_image_edit/outputs/snow_batch_v1/report/overview.jpg)：左侧原图，右侧雪景；从上到下为奥迪、宝马、小米、本田。
- [四车四天气完整图库](../02_qwen_image_edit/outputs/weather_batch_v2/report/index.html) · [完整总览](../02_qwen_image_edit/outputs/weather_batch_v2/report/overview.jpg)：依次为原图、雨后阴天、中雨、薄雾、降雪与部分积雪。
- [雪景文件校验](../02_qwen_image_edit/outputs/snow_batch_v1/report/summary.json) · [视觉检查记录](../02_qwen_image_edit/outputs/snow_batch_v1/report/visual_review_snow.json) · [运行指标](../02_qwen_image_edit/outputs/snow_batch_v1/report/metrics.json)。

雪景数据保存在 `02_qwen_image_edit/outputs/snow_batch_v1/`，合并图库保存在 `02_qwen_image_edit/outputs/weather_batch_v2/report/`。合并图库通过 [16 组合配置](../02_qwen_image_edit/configs/weather_batch_v2.json) 引用已有 12 张和新增 4 张，没有重复生成旧天气。

### 检查结论

- 四张均有可见降雪、地面不连续浅雪和车顶薄雪，仍可见裸露路面与基本车色；机盖覆雪程度各不相同。车灯、格栅和轮毂没有被整片积雪埋住，但雪粒偏密，存在额外遮挡。
- 原图对照显示，小米雪景在前景**新增了一条原图没有的斜向黄色路面标线**，属于背景布局漂移，未做后期修除。
- 各车仍存在轮毂中心标识、车牌小字、灯内或轮胎细节的重绘与清晰度损失。天气目标已在视觉上实现，但四张仍作为候选图，未判定严格细粒度保真合格，也没有训练评估。
- 雪景 4/4 完成 40 步，64 个 PNG 的哈希、格式及完整解码检查通过，四张源图哈希未变。合并图库 16/16 通过检查，共核验 256 个 PNG。
- 本次生成队列耗时 10 分 9 秒；单张生成 116.57–116.72 秒，平均 116.63 秒；峰值已分配显存 65.55 GiB、峰值预留显存 76.40 GiB。四个生成子进程均已结束。这是本次任务的运行记录，不表示复查时 GPU 仍为空闲。

刷新雪景及合并图库可分别调用 `summarize_batch.py`，使用 `configs/snow_batch_v1.json` 与 `configs/weather_batch_v2.json`，输出到上述各自的 `report/` 目录。
