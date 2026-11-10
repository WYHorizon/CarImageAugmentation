# 03 — SDXL + ControlNet + IP-Adapter 对照路线

[文档首页](README.md) · [总体实验计划](generative_weather_plan.md) · [Qwen 已有结果](02_qwen_image_edit.md)

## 当前状态（2026-11-09）

**当前暂不推进部署。** 用户进一步明确：如果没有比 Qwen 更好的理由，就不要试 SDXL。目前没有本项目实测证据证明 SDXL + ControlNet + IP-Adapter 优于 Qwen；此前将其列为候选，是为了考察结构条件的作用，而不是已经确认底模或最终效果更好。下面的步骤保留为备选方案，不作为当前执行队列。

目前仅完成方案与接口核查，尚未创建独立 SDXL 环境、安装补充依赖、下载权重或执行 SDXL 生成。前面的 16 张天气图均属于 Qwen，不应记入本方法。

在现有 `car-weather-qwen21` 环境中，以下组件已实际导入成功：`StableDiffusionXLControlNetImg2ImgPipeline`、`ControlNetModel`、`AutoencoderKL`、`CLIPVisionModelWithProjection`、`CLIPImageProcessor`、`CLIPTextModel`。管线具有 `load_ip_adapter` 方法和 `image`、`control_image`、`ip_adapter_image` 参数。这只证明接口可用，完整权重加载及联合推理仍待测试。

当前已验证环境的核心版本为 Torch 2.7.1+cu118、Diffusers 0.42.1、Transformers 5.22.2、Accelerate 1.15.0。未安装 `opencv-python` 或 `opencv-python-headless`；项目 `models/` 当前仅有 Qwen 权重。在检查的 `/nas/wyh/.cache/huggingface/` 中未发现可复用的 SDXL 组件；尚未穷举其他环境和磁盘目录。

## 方法与组件

三个条件共同引导生成：原图作为 img2img 起点；Canny 边缘图提供空间结构；完整车辆裁剪提供参考外观。提示词描述目标天气。ControlNet 和 IP-Adapter 都是生成条件，不是零件或像素不变的硬保证。

| 组件 | 用途 | 计划使用 |
| --- | --- | --- |
| SDXL Base | 根据原图和天气文字生成画面 | `stabilityai/stable-diffusion-xl-base-1.0` |
| Canny ControlNet | 引导车体、车灯、格栅、轮毂及场景主要轮廓 | `diffusers/controlnet-canny-sdxl-1.0` |
| IP-Adapter Plus SDXL | 从车辆参考图提供外观特征 | `h94/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors` |
| CLIP ViT-H 图像编码器 | 编码上述车辆参考图 | `h94/IP-Adapter/models/image_encoder` |

Plus 的 ViT-H 权重应配套 `models/image_encoder`，不能误用该仓库 `sdxl_models/image_encoder` 中的 bigG 编码器。依据：[Diffusers IP-Adapter 用法](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter)、[IP-Adapter 模型说明](https://huggingface.co/h94/IP-Adapter)。

主管线为 `StableDiffusionXLControlNetImg2ImgPipeline`。依据：[SDXL ControlNet 管线文档](https://huggingface.co/docs/diffusers/api/pipelines/controlnet_sdxl)、[Canny 权重说明](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0)。首轮使用预训练权重，不训练 LoRA；Depth ControlNet 留给整体深度/布局问题仍明显时再评估。

## 分步执行

### 1. 准备独立环境

计划环境为 `/nas/wyh/.conda/envs/car-weather-sdxl`。优先复用或克隆已验证环境中的 Torch、Diffusers 等核心依赖，在新环境中核实并补装合适版本的 `opencv-python-headless`，完成 `pip check`、管线及 CLIP 类导入、Canny 小样检查。正式安装前再检查 conda/pip 缓存及 OpenCV 的 NumPy 版本约束。

每个进程显式设置 pip、HF、Torch、XDG 缓存及 TMPDIR/TMP/TEMP，继续使用 `/nas/wyh/.cache/` 下对应已有目录。方法代码、配置和输出归入计划目录 `03_sdxl_controlnet_ipadapter/`；说明仍集中在 `docs/`。本步骤完成后再进入模型下载。

### 2. 下载并检查必要组件

权重存入共享 `models/`，按仓库选择必要的推理权重、配置、分词器和图像编码器文件，避免整仓下载无关的 SD1.5、FaceID 或其他变体。记录 revision、文件大小和哈希，检查本地离线加载和 IP-Adapter 与编码器的配套关系。

此时才实测加载与生成显存，不能用 Qwen 的显存数字估算 SDXL 联合管线，也不能根据旧记录假定 GPU 2 仍空闲。

### 3. 准备一张奥迪输入及两种条件

- 原图等比例缩至拟定的 1152×864（严格 4:3，约 1MP），记录缩放映射并保存工作原图。这是本项目首测尺寸，不是已验证的最优尺寸。
- 从原图提取 Canny，保存原始边缘图和实际控制图供检查。保留车体外轮廓、门缝、窗框、车灯、格栅、轮毂及背景主要布局；对车漆和玻璃内部的树枝、建筑反射边缘适度减弱，避免把旧天气反射一同固定住。
- 使用完整车辆裁剪作为 IP-Adapter 参考，保留边缘空间并等比例补成正方形；保存编码器实际看到的预处理图，确认车头、车尾和车轮未被中心裁切掉。

### 4. 检查基础损失，再做首张天气编辑

先检查 SDXL VAE 编码—解码结果与工作原图，记录其自身带来的细节损失及有限值情况。随后只用奥迪、seed 42、雨后阴天无雨丝，运行完整组合，检查光照、湿地、反射与细节。后处理仅做已记录的几何变换，不贴回原车或零件，不放大后声称恢复了细节。

项目建议起点，尚未实测：

| 参数 | 初始值 |
| --- | --- |
| 调度步数 `num_inference_steps` | 40 |
| 图生图改动强度 `strength` | 0.40 |
| 文本引导 `guidance_scale` | 5.5 |
| 边缘条件 `controlnet_conditioning_scale` | 0.70 |
| 图像参考 `ip_adapter_scale` | 0.50 |
| 随机种子 | 42 |

已检查本地管线的 `get_timesteps`：在标准调度设置下，40 步配合 strength 0.40 只使用末段约 16 步去噪；运行时应记录实际调用步数和调度器。这个“40”不能直接当成与 Qwen 的 40 步计算量相同。`strength` 的具体含义也不能套到当前 Qwen 管线上。

### 5. 用结果决定下一项对照

先将 SDXL 原生工作分辨率结果与同尺寸原图比较；再把已有 Qwen 输出缩至相同尺寸比较，同时分别保留两种方法的原生输出。共同尺寸下好看，不表示原生 2K/4K 细节得到了恢复。

优先检查灯带、格栅、轮毂中心标识与辐条、车身折线、窗框、车牌文字，以及新增标线或背景物体。天气方面检查旧阳光/反射是否残留，湿地与接地关系是否自然。

若需要确定各条件的贡献，再对同一图、种子、天气做 SDXL img2img、SDXL + Canny、SDXL + Canny + IP-Adapter 的条件增减对照。结构变化太大时再单独调低 strength 或加强结构条件；天气变化不足时单独减少保留旧外观的约束。每次改一个因素，不直接遍历所有参数组合。

首张通过初步检查后，才扩展到四辆车及更多种子/天气。若始终无法兼顾天气变化和细节，记录失败边界，再考虑计划中的后续路线。
