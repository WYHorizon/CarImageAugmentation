# 汽车细粒度图像增强实验

项目根目录是 `/nas/wyh/CarImageAugmentation`，用于比较汽车天气与光照增强方法。项目说明统一放在 `docs/`；每种方法的代码、依赖、配置、测试和输出保留在自己的方法目录中。

## 文档索引

| 文档 | 内容 |
| --- | --- |
| [会话交接](HANDOFF.md) | 项目目标、协作约定、历史进展和接续说明；环境状态以文内标注的交接时间为准 |
| [纯物理方法说明](01_physics_based.md) | 成像方法、参数、运行方式及保真边界 |
| [纯物理实验结果](01_physics_results.md) | 三轮实验记录、结果图片入口及局限 |
| [Qwen 编辑方法](02_qwen_image_edit.md) | 本地环境、单图/批次推理、四车四天气结果（含降雪与部分积雪）、显存实测及保真检查 |
| [光照与光学退化增强](03_lighting_augmentation.md) | CPU 过曝、光晕与改进眩光，四种当前预设共 16 张、整图／关键部件对比；保留基础组与旧强光组历史记录 |
| [SDXL 结构与参考条件对照](03_sdxl_controlnet_ipadapter.md) | 暂缓的备选路线；仅完成方案及接口核查，尚未部署权重 |
| [生成模型方案](generative_weather_plan.md) | Qwen、SDXL 等模型选择、结构控制与实验验收流程 |
| [论文选题与最小验证方案](research_proposal.md) | 相关工作重复风险、判别部件保真天气增强候选、真实分类评测与逐步验证；尚未实现新算法 |

## 项目目录

```text
CarImageAugmentation/
├── docs/                     # 项目说明文档的统一入口
│   ├── README.md
│   ├── HANDOFF.md
│   ├── 01_physics_based.md
│   ├── 01_physics_results.md
│   ├── 02_qwen_image_edit.md
│   ├── 03_lighting_augmentation.md
│   ├── 03_sdxl_controlnet_ipadapter.md
│   ├── generative_weather_plan.md
│   └── research_proposal.md
├── test_images/              # 共享原图，各方法只读取
├── models/                   # 共享模型权重及仓库自带文件
├── 01_physics_based/         # 纯物理成像与程序化天气增强
│   ├── configs/
│   ├── src/
│   ├── tests/
│   ├── outputs/
│   ├── run.py
│   ├── compare_runs.py
│   └── requirements.txt
├── 02_qwen_image_edit/       # Qwen 原图条件天气编辑
│   ├── configs/
│   ├── outputs/
│   ├── run.py
│   ├── run_batch.py
│   ├── summarize_batch.py
│   └── requirements.txt
├── 03_lighting_augmentation/ # 基础光度与强光学效果，无生成模型
│   ├── configs/
│   ├── lighting/
│   ├── tests/
│   ├── outputs/
│   ├── run.py
│   └── requirements.txt
└── ...                       # 后续方法各自新增目录
```

这种布局适合比较多种方法：方法内部独立运行，生成结果不会混在一起，共享原图又能保证各方法使用相同输入。不要在各方法目录中重复拷贝数据集；后续如果需要正式训练集，可以再增加共享 `datasets/`，并按原始照片/拍摄组统一划分训练、验证与测试。

文档中的命令以项目根目录或各方法代码目录为工作目录，具体见命令前的说明；不要因为文档位于 `docs/` 就在该目录直接运行脚本。`requirements.txt` 是方法的依赖清单，继续与代码放在一起。模型仓库自带的模型卡、许可和配置随权重保留。
