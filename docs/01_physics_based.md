# 01 — 纯物理天气增强

[文档首页](README.md) · [实验结果](01_physics_results.md)

代码目录：`/nas/wyh/CarImageAugmentation/01_physics_based`。本页中的配置和输出相对路径以该代码目录为基准。

本方法作为汽车细粒度训练数据增强的第一条基线。实现依赖 NumPy 和 Pillow，使用原图像素、人工近似场景几何和解析成像模型，不使用扩散模型、学习式深度估计或车辆重绘。

当前支持 `fog_light`（薄雾）、`fog_medium`（较明显的雾），以及 `rain_light`、`rain_medium`、`rain_heavy` 三档降雨。另有 `overcast_wet`、`rain_overcast_wet_medium`、`rain_overcast_wet_heavy`，分别组合阴天湿地、阴天湿地中雨、阴天湿地大雨。所有训练候选输出保持 EXIF 方向校正后的原图分辨率，使用 PNG 保存。缩放只用于预览页。

## 目录与运行

```text
01_physics_based/
├── configs/
│   ├── scenes.json          # 四张样图的人工轮廓、近似深度、相机和关键部件区域
│   └── presets.json         # 天气参数
├── src/physics_weather.py   # 线性光转换、近似深度、雾、雨和可见度诊断
├── src/scene_weather.py     # 阴天照明比例、水膜 Fresnel 与路面反射近似
├── tests/test_physics.py    # 成像不变量、随机复现、遮挡等测试
├── run.py                  # 批处理、参数快照、预览和清单
├── compare_runs.py         # 新旧降雨版本与轮胎接地处对比
├── requirements.txt
└── outputs/                # 此方法的所有实验输出
```

默认读取代码目录上一级共享的 `test_images/`。从项目根目录运行：

```bash
cd /nas/wyh/CarImageAugmentation
python 01_physics_based/run.py
```

或从本方法目录运行：

```bash
cd /nas/wyh/CarImageAugmentation/01_physics_based
python run.py
python -m unittest discover -s tests -v
```

程序默认生成 4 张原图 × 3 种天气，共 12 张原分辨率图片。已经有结果时，指定一个新的实验目录，不会覆盖已有结果：

```bash
python run.py --weather fog_light fog_medium --output-dir outputs/fog_comparison_02
python run.py --weather rain_light --count 3 --seed 123 --output-dir outputs/rain_seed123
python run.py --weather rain_light rain_medium rain_heavy --output-dir outputs/rain_intensity_v2
python run.py --weather overcast_wet rain_overcast_wet_medium rain_overcast_wet_heavy --output-dir outputs/overcast_wet_v3
python compare_runs.py --before outputs/rain_intensity_v2 --after outputs/overcast_wet_v3
```

原有物理实验环境已安装依赖。新环境可在代码目录执行 `python -m pip install -r requirements.txt`。`--input-dir`、`--output-dir`、`--scenes`、`--presets` 支持自定义路径；默认路径与执行命令时的工作目录无关。自定义相对路径按当前工作目录解析。

## 成像方法

### 雾

先按 sRGB 标准传递函数转换到近似线性 RGB，施加单次散射成像模型：

```text
I(x) = J(x) * t(x) + A * [1 - t(x)]
t(x) = exp[-beta * d(x)]
```

`J` 为输入图像，`A` 为配置的环境光，`d` 为近似的沿视线距离。场景先用相机空间 Z 表达，再按相机射线方向转换为距离。远背景更易退对比，近车保留更多细节。

深度来自三部分：背景分层、针孔相机下的近似地平面、手工车辆轮廓内的平滑深度。相机视场角、相机高度、背景距离均为假设值。车辆轮廓边缘作少量羽化，粗背景层使用更宽的平滑过渡，减少人工分层边界成为训练线索的风险。

来源参考：[Foggy Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)。本项目是简化的独立实现，没有复现其双目深度处理管线。

### 降雨

在相机视锥内按空间密度随机采样雨滴，用针孔投影确定尺寸和位置；用下落速度、风速与曝光时间确定雨丝长度、方向。高斯截面近似光学模糊，曝光占空比约束不透明度，再按场景深度逐像素判断遮挡。雨滴可以正常经过车辆前方，不为关键部件开孔。

雨滴以透射率混合进入线性 RGB。近处雨滴较宽、较模糊，远处更细。小雨默认显式采样至 40 米；中雨、大雨分别采样至 18、20 米，将计算集中在可分辨雨滴上，远处未解析的降水由散射项近似。

中雨、大雨增加雨滴空间密度、直径、曝光长度及风速扰动，并增加远景散射。雨滴的有效辐射亮度允许超过显示白色的 1.0，以近似日照下雨滴反射、折射的高亮；最终保存时执行显示范围截断，并在清单中记录超出显示范围的像素比例。该亮度是可调的近似参数，并未从真实 HDR 环境恢复。仅将雨滴混合系数变大，会把雨丝当作不透明白线，因此保留曝光占空比约束，单独设置辐射亮度。

三个名称代表**视觉增强强度**，不能当作气象学中以 mm/h 定义的小雨、中雨、大雨。对应配置如下：

| 预设 | 粒子密度（假设值 /m³） | 曝光时间 | 雨滴直径 | 显式采样最远距离 | 散射 beta |
| --- | --- | --- | --- | --- | --- |
| rain_light | 6 | 1/250 s | 1–2 mm | 40 m | 0.0005 |
| rain_medium | 20 | 1/125 s | 1.2–2.7 mm | 18 m | 0.003 |
| rain_heavy | 55 | 0.012 s | 1.5–3.5 mm | 20 m | 0.008 |

来源参考：[Rain rendering for evaluating and improving robustness to bad weather](https://arxiv.org/abs/2009.03683)。当前实现未复现其精确雨滴折射、环境光积分或数据驱动场景变换。

### 阴天照明与湿路面

`surface_weather` 是可选处理阶段，旧雾、雨预设不会自动启用。处理顺序为：原图 → 受限阴天照明 → 湿路面 → 空气散射 → 雨滴。

阴天处理使用低频亮度场估计照明增益，并通过人工投影区域、未遮阴路面参考区域及假设的原始环境光比例，近似消减路面直射光投影。轮胎接触遮蔽与车底遮蔽保留。天空在人工支持区域内作灰色漫射照明近似，暗树枝仍保留源像素。这里的照明分离是启发式估计，不是从单图精确恢复太阳、材质和环境光。

车辆区域仅执行 `J_vehicle' = gain_RGB(x) * J_vehicle`，默认逐通道增益限制在 **[0.74, 1.08]**。低频滤波只用于控制场，车辆 RGB 本身不做模糊、形变或重采样。模型不会重新设计灯带、格栅或轮毂；增益下限限制额外压暗的程度，但不等于最终判别信息绝对无损。

湿地层在车辆保护区外计算。水膜模型采用空气—水界面的 Fresnel 方程（默认水折射率 1.333）、两次透射近似 `(1-F)^2`、漫反射衰减，以及粗糙的环境反射。湿润度和水膜反射连贯程度由近似地平面坐标上的连续噪声控制，形成非均匀湿润变化。没有把整张地面替换成镜面。

反射内容由原图中的背景平面代理与车辆可见表面代理提供；车辆代理沿人工轮胎接地点投射到路面，并作粗糙模糊。这是 **2.5D 可见表面反射近似**，没有恢复真实镜面视角中不可见的车底、车体背面或建筑。反射重采样仅写入路面。

湿材质外观依据：[Rendering of Wet Materials](https://graphics.stanford.edu/~henrik/papers/egwr99/)。Fresnel 公式依据：[Physically Based Rendering](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)。本项目的照明估计、人工几何与水膜分布是独立的简化工程实现，并非这两项工作的完整复现。

清单额外记录车辆实际增益范围、车辆 RGB 乘法关系的最大误差、湿地层对车辆保护区的最大改动。诊断图展示照明增益、路面掩码、水膜覆盖、Fresnel 反射率及车辆路面反射权重。零误差仅证明相应阶段遵守像素操作约束，不能证明人工轮廓准确或模型分类性能不下降。

## 保真范围与真实性边界

- 原图像素位置和车辆部件纹理作为输入保持固定，增强过程没有几何变形、裁切、车辆生成或超分辨率重建。
- 雾会降低对比度，雨丝会遮挡像素，因此“没有重画部件”不等于“每个细节仍然可辨”。需要结合关键区域预览评估。
- 人工多边形是近似轮廓，不是精确分割；背景也只有少量距离层，树枝、栏杆、其他汽车的精细深度并不完整。请检查 `diagnostics/` 的几何图。
- 原图通常经过相机色调处理，并非线性 RAW；逆 sRGB 是实用近似。原图已经存在的雾霾也未移除，因此这里表示**追加散射**，不能据此标注绝对能见度。
- 旧的 `rain_light/medium/heavy` 保留原有照明及地面状态，可理解为原场景中突然出现降水。新的 `rain_overcast_wet_*` 增加阴天照明与湿地近似。车辆原有方向性反射、高光以及场景中未标出的阴影不能全部消除，仍可能有晴天残留。
- 当前没有重建湿润车漆上的细密水滴、真实凹陷积水、镜头雨滴或落地水花。程序化水膜与可见表面反射不等于完整的持续阴雨物理场景。
- 场景距离、雨滴密度、曝光及雨滴辐射亮度未标定，参数不是实测雨量。当前雨滴采用简化辐射混合，并非完整水滴折射渲染。
- 当前不处理积雪与强重打光，因为它们会涉及覆盖判别部件或从单图估计照明、材质的额外误差。

## 输出和复现

```text
outputs/<实验名>/
├── images/                 # 原分辨率训练候选 PNG
├── previews/
│   ├── overview.jpg        # 原图与三种天气的总览
│   ├── *_comparison.jpg    # 每张原图单独一行对比
│   └── *_details.jpg       # 灯、格栅/前脸、前后轮同位置裁剪对比
├── diagnostics/
│   ├── *_geometry.jpg      # 人工车辆轮廓、关键区域与近似深度图
│   └── *_surface.jpg       # 阴天/湿地阶段的照明及水膜控制场
├── config_snapshot/        # 本次使用的完整配置
└── manifest.json           # 图像哈希、实现哈希、依赖版本、种子和部件诊断
```

雨天变体种子由基础种子、原图文件名、天气名和变体序号确定，不受其他图像处理顺序影响。在同一实现与依赖环境下可复现。雾是确定性变换，`--count` 只增加雨天变体；要增加雾多样性，应新建不同 beta/A 的预设，避免生成重复数据。

清单记录关键矩形区域与车辆轮廓交集内的雾透射率和雨遮挡强度，同时记录交集占比，避免将框内的背景误当成车辆部件。它们只是诊断值，尚未通过人工数据校准为合格阈值；预览和清单不等于车型保真认证。

## 增加图像与训练使用

新图必须在 `configs/scenes.json` 中增加同名条目。所有坐标均相对于 **EXIF 方向校正后的图像** 归一化到 `[0,1]`：

- `subject_polygon`：车辆轮廓；`subject_depth_m`：从轮廓最左至最右的近似 Z 深度。
- `layers`：按顺序叠加的背景区域；没有前景深度时不能假定新图自动适配。
- `ground_start_y`、`camera.horizon_y`：地面出现位置与地平线位置；前者应大于后者。
- `regions`：关键部件矩形 `[左, 上, 右, 下]`，用于预览与诊断，不影响雨雪是否穿过该区域。
- `weather_geometry`：新阶段使用的路面、投影、接触阴影、晴天路面亮度参考、天空支持区域、轮胎接地点及细化车辆轮廓。需逐图校正。`vehicle_outline` 比原始深度轮廓细化了轮胎和车底边界，防止把车底路面一起当作车辆而留下干地边缘。

新增图像没有配置时程序会明确报错。批量数据的人工标注成本是此基线的一项限制；未来可单独增加深度估计方法，但应作为另一项实验或明确记录为混合方法。

先按原图及同次拍摄组划分训练/验证/测试，再将原图与全部增强版本留在同一集合。天气与车型均衡采样。评测需要包含独立真实天气图片；当前 4 张图仅用于方法和画质验证，尚未证明训练收益。
