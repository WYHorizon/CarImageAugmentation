# CarImageAugmentation

面向**细粒度车型识别**的汽车天气增强实验项目。目标是从车辆原图生成雨、雾、阴天湿路、降雪等天气变化，同时尽量保留车灯、格栅、车标、轮毂、车身折线和基本漆色，为后续分类训练与鲁棒性评测提供候选数据。

目前实现了程序化物理天气基线和本地 Qwen-Image-2.1 原图条件编辑。项目处于方法探索与样图验证阶段，**尚未完成分类训练或证明增强带来的识别收益**。

## 方法与当前进展

| 方法 | 实现方式 | 已完成的实验 | 主要限制 |
| --- | --- | --- | --- |
| 物理天气增强 | NumPy/Pillow、人工场景几何、雾散射、雨滴投影、阴天照明与湿路面近似 | 四辆车、三轮实验，共 36 张输出，部分跨批次重复 | 无车辆重绘，但依赖人工配置，晴天反射和照明难以充分改变 |
| Qwen 天气编辑 | 本地 Qwen-Image-2.1，以原图和天气提示词为条件 | 四辆车 × 阴天湿路／中雨／薄雾／降雪与部分积雪，共 16 个组合 | 天气变化有效，灯内细节、轮毂标识等仍有重绘损失 |

共享样图为奥迪 A6L、宝马 4 系、小米 YU7、本田 CR-V，原图均为 3840×2880。现有 Qwen 样图尚未被判定通过严格细粒度保真验收；文件校验通过不代表车型保真或训练有效性通过。

Qwen 当前未接入 ControlNet，场景配置中的部件区域用于裁剪对照，不作为生成时的结构约束。SDXL + ControlNet + IP-Adapter 暂缓部署，判别部件保真控制仍是待验证的研究方案。

## 目录结构

```text
CarImageAugmentation/
├── README.md                         # 项目介绍与运行入口
├── docs/                             # 方法说明、实验记录与研究方案
│   ├── README.md                     # 完整文档索引
│   ├── 01_physics_based.md
│   ├── 01_physics_results.md
│   ├── 02_qwen_image_edit.md
│   ├── 03_sdxl_controlnet_ipadapter.md
│   ├── generative_weather_plan.md
│   ├── research_proposal.md
│   └── HANDOFF.md
├── test_images/                      # 四张共享原图，各方法只读取
├── models/                           # 本地模型权重，不纳入 Git
│   └── Qwen-Image-2.1/
├── 01_physics_based/
│   ├── configs/                      # scenes.json 场景几何、presets.json 天气参数
│   ├── src/                          # 雨雾成像、阴天照明与湿路面实现
│   ├── tests/                        # 成像、复现与车辆保护区测试
│   ├── run.py                        # 物理增强入口
│   ├── compare_runs.py               # 雨天与阴天湿地实验对比
│   ├── requirements.txt
│   └── outputs/                      # 本地生成结果，不纳入 Git
└── 02_qwen_image_edit/
    ├── configs/                      # 单图提示词与参数、批次任务清单
    ├── run.py                        # 单图离线推理与部件预览
    ├── run_batch.py                  # 串行批处理、日志与已有结果校验
    ├── summarize_batch.py            # 文件校验、HTML 图库与总览
    ├── requirements.txt
    └── outputs/                      # 本地生成结果，不纳入 Git
```

仓库包含代码、配置、文档和测试原图。`models/`、各方法的 `outputs/`、Python 缓存与虚拟环境由 `.gitignore` 排除，克隆后需另行准备权重并生成结果。详细文档中的历史结果图片链接仅在对应本地输出存在时可用。

## 获取项目

```bash
git clone https://github.com/WYHorizon/CarImageAugmentation.git
cd CarImageAugmentation
```

以下命令均从**项目根目录**执行，面向 Linux/Bash。两个方法使用独立 Python 环境；已有本地项目可直接进入项目根目录，无需再次克隆。

## 运行物理天气增强

### 1. 准备环境

使用 Python 3.10 或更高版本，仅需 NumPy 和 Pillow，无需 GPU。

```bash
python3 -m venv 01_physics_based/.venv
source 01_physics_based/.venv/bin/activate
python -m pip install -r 01_physics_based/requirements.txt
```

### 2. 生成样图

下面读取 `test_images/` 中的四张原图，生成薄雾、较明显雾和小雨，共 12 张原分辨率 PNG：

```bash
python 01_physics_based/run.py \
  --weather fog_light fog_medium rain_light \
  --output-dir 01_physics_based/outputs/demo_run
```

生成阴天湿地、湿地中雨与湿地大雨：

```bash
python 01_physics_based/run.py \
  --weather overcast_wet rain_overcast_wet_medium rain_overcast_wet_heavy \
  --output-dir 01_physics_based/outputs/wet_run
```

| 参数／配置 | 用途 |
| --- | --- |
| `--input-dir` | 原图目录，默认 `test_images/` |
| `--weather` | 可指定多个天气预设 |
| `--count` / `--seed` | 每种雨天的随机变体数／基础种子，默认 1／42；雾为确定性变换 |
| `--scenes` / `--presets` | 自定义场景配置／天气预设文件 |
| `--output-dir` | 输出目录，必须不存在或为空，且不能位于输入目录内 |

支持的预设为 `fog_light`、`fog_medium`、`rain_light`、`rain_medium`、`rain_heavy`、`overcast_wet`、`rain_overcast_wet_medium`、`rain_overcast_wet_heavy`。雨量名称表示视觉强度，未按真实 mm/h 标定。

输出目录包含 `images/`、`previews/`、`diagnostics/`、`config_snapshot/` 和 `manifest.json`，可先查看 `previews/overview.jpg` 与部件对比图。

### 3. 运行测试

```bash
python -m unittest discover -s 01_physics_based/tests -v
```

新增图片时，必须在 [scenes.json](01_physics_based/configs/scenes.json) 中添加同名条目，配置车辆轮廓、近似深度、相机和部件区域；阴天湿地还需对应的 `weather_geometry`。坐标基于 EXIF 方向校正后的图像，归一化到 `[0,1]`。当前没有自动适配任意新图的几何估计流程。

## 运行 Qwen 天气编辑

### 1. 环境与权重

已验证环境使用 Python 3.12、PyTorch 2.7.1+cu118、Diffusers 0.42.1 和 Transformers 5.22.2，完整核心版本见 [requirements.txt](02_qwen_image_edit/requirements.txt)。需要支持 BF16 的 NVIDIA CUDA GPU。

原实验在 A800 80GB 上运行：2400×1792 画布、40 步、seed 42，单张生成约 116–117 秒，峰值已分配显存约 65.5 GiB、预留显存最高约 76.4 GiB。这些是该配置的实测值，其他硬件与参数需要单独验证。

已有服务器环境可直接激活：

```bash
conda activate /nas/wyh/.conda/envs/car-weather-qwen21
```

在新机器上，也可用独立 Python 3.12 虚拟环境安装记录的依赖：

```bash
python3.12 -m venv 02_qwen_image_edit/.venv
source 02_qwen_image_edit/.venv/bin/activate
python -m pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r 02_qwen_image_edit/requirements.txt
```

从 [Qwen 模型仓库](https://huggingface.co/Qwen/Qwen-Image-2.1) 准备指定 revision 的完整权重（约 33 GB）：

```bash
hf download Qwen/Qwen-Image-2.1 \
  --revision d26bb61231c349cf6b7896fa83353113880e1ba3 \
  --local-dir models/Qwen-Image-2.1
```

推理脚本只做离线加载，不会自动下载模型。保留下载工具生成的 `models/Qwen-Image-2.1/.cache/huggingface/download/` 元数据；脚本会读取其中的 `model_index.json.metadata`，核对 revision 与单图配置是否一致。

迁移到新机器前还需检查两个本地路径：

- [run.py](02_qwen_image_edit/run.py) 的 `CACHE` 当前固定为 `/nas/wyh/.cache`。确保该目录及代码指定的缓存、临时子目录存在且可写，或将 `CACHE` 改为本机路径；脚本会覆盖相应环境变量。
- 预览使用 `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`，需保证该字体存在，或调整代码中的字体路径。

### 2. 检查配置并运行单图

先检查空闲 GPU 和输入配置。下例中的 GPU `0` 只是示例，应按实际空闲设备选择：

```bash
nvidia-smi

python 02_qwen_image_edit/run.py \
  --config 02_qwen_image_edit/configs/audi_overcast_wet_seed42.json \
  --output-dir 02_qwen_image_edit/outputs/audi_overcast_wet_run \
  --check-config
```

`--check-config` 不加载模型权重、不生成图片，但仍需要已安装的 Diffusers、原图、场景配置及上述模型下载元数据。该命令不会创建输出目录。

检查通过后，用同一配置生成：

```bash
CUDA_VISIBLE_DEVICES=0 python 02_qwen_image_edit/run.py \
  --config 02_qwen_image_edit/configs/audi_overcast_wet_seed42.json \
  --output-dir 02_qwen_image_edit/outputs/audi_overcast_wet_run
```

输出目录必须尚不存在，不能写入原图或模型目录。重复实验请更换目录名。配置文件中的 `input_image`、`model_dir` 和 `scene_config` 相对于项目根目录解析；命令行中的相对路径按执行目录解析。

默认配置将 3840×2880 原图等比例缩至 2388×1791，再补边为 2400×1792 输入画布；输出仅去除已记录的补边，不放大回 4K。主要文件包括：

| 输出 | 内容 |
| --- | --- |
| `source_working.png` / `source_canvas.png` | 同尺寸对照原图／实际输入画布 |
| `edited.png` / `edited_canvas.png` | 去边框结果／模型原始画布输出 |
| `previews/` / `crops/` | 整图、整车及部件对比／原生分辨率裁剪 |
| `manifest.json` / `progress.json` | 参数、哈希、耗时、显存与运行状态／逐步进度 |
| `config.json` / `run_snapshot.py` / `environment_freeze.txt` | 本次配置、代码和依赖快照 |

### 3. 批量生成与汇总

可先使用没有旧结果复用依赖的四车雪景批次：

```bash
python 02_qwen_image_edit/run_batch.py \
  --batch-config 02_qwen_image_edit/configs/snow_batch_v1.json \
  --output-dir 02_qwen_image_edit/outputs/snow_batch_v1 \
  --gpu 0

python 02_qwen_image_edit/summarize_batch.py \
  --batch-config 02_qwen_image_edit/configs/snow_batch_v1.json \
  --output-dir 02_qwen_image_edit/outputs/snow_batch_v1/report
```

批处理串行执行，记录 `batch_status.json` 和 `logs/`，对已有结果校验配置与哈希后复用。当前执行器要求指定 GPU 至少有 **80000 MiB 空闲显存**，最多等待 30 分钟；这是现有 2K 配置的门槛，不会自动适配小显存设备。

批处理的 `--output-dir` 只决定批次日志与状态目录，必须位于 `02_qwen_image_edit/outputs/` 内；**各图实际输出位置由批次 JSON 中的 `jobs[].output_dir` 决定**。新实验应复制批次配置，更新 `batch_id` 和各图输出目录；不依赖旧结果时将相应 `reuse` 设为 `false`。

`weather_batch_v1.json` 要求复用首张奥迪雨后结果，`weather_batch_v2.json` 要求复用已有的 12 个雨／雾结果。刚克隆仓库时这些输出不存在，不能直接将历史批次作为全新生成任务运行；需要先准备其引用的结果，或按上述规则制作新批次配置。

汇总后打开本地 `02_qwen_image_edit/outputs/snow_batch_v1/report/index.html` 查看图库，`summary.json` 记录文件检查结果。汇总工具不执行视觉保真验收。

## 实验与数据约定

- 原图集中保存在 `test_images/`，各方法共享读取，实验使用独立输出目录。
- 新增 Qwen 输入也需在场景配置中添加同名条目，用于整车和部件对照。当前提示词中的“保持细节”是软约束，需逐图检查生成结果。
- `complete_pending_visual_review` 表示生成与记录完成，仍需检查车型特征、天气效果及背景布局变化。
- 训练、验证、测试应按原始照片及拍摄组划分，同一原图的所有增强版本留在同一集合，避免数据泄漏。
- 四张展示原图只用于方法验证。下游收益需要独立真实天气测试集及分类训练实验支持。

## 详细文档

| 文档 | 内容 |
| --- | --- |
| [文档索引](docs/README.md) | 全部说明与目录约定 |
| [物理方法](docs/01_physics_based.md) · [实验结果](docs/01_physics_results.md) | 成像模型、参数、保真边界与三轮记录 |
| [Qwen 编辑方法](docs/02_qwen_image_edit.md) | 单图与批次结果、资源消耗、视觉检查 |
| [生成模型方案](docs/generative_weather_plan.md) | 候选路线与实验验收流程 |
| [SDXL 备选路线](docs/03_sdxl_controlnet_ipadapter.md) | 暂缓实施的结构与参考条件方案 |
| [论文研究方案](docs/research_proposal.md) | 判别部件保真候选、评测协议与最小验证计划 |
| [项目交接记录](docs/HANDOFF.md) | 历史进展与环境快照，以各段标注时间为准 |
