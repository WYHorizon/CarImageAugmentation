# 宝马 5 系真实四视图几何实验

本实验使用同一辆车的四张真实照片，经前景分割后输入官方 **Hunyuan3D-2mv**。这与旧实验的“单张前侧照片生成其余几何”不同；生成结果仍是无纹理的几何候选，不能单凭运行成功认定车型细节保真。

## 输入与方向

| 官方槽 | 照片 | 目视确认 |
| --- | --- | --- |
| front / 0 | `front/5_2026_255841_151515_002.jpg` | 正前 |
| left / 1 | `side/5_2026_255841_151515_004.jpg` | 车头朝画面左 |
| back / 2 | `rear/5_2026_255841_151515_006.jpg` | 正后 |
| right / 3 | `side/5_2026_255841_151515_008.jpg` | 车头朝画面右 |

左右语义对照了官方 `assets/example_mv_images/1/left.png`，没有翻转照片。前景来自 `../photo_texture/outputs/bmw5_inputs_v2/`，由 CPU U2Net 分割；原始照片不修改。官方处理器会对每张前景独立居中并等比放入正方形，最终张量为 `[1,4,3,512,512]`。这些方向标签并不是已标定的真实相机位姿。

## 已完成运行

2026-11-17 的成功结果位于 `outputs/bmw5_mv_seed42_v2/`：

- 官方标准 `hunyuan3d-dit-v2-mv` fp16，seed 42，50 步，octree 512，guidance 5.0。
- 仅使用物理 GPU 2；生成与网格提取 61.172 秒，峰值 PyTorch 已分配显存 6.618 GiB。耗时不包含模型加载、下载和前景分割。
- `mesh.glb`：608,181 顶点、1,216,502 三角面，29,195,472 字节，导出顶点法线；没有纹理。
- 有限顶点、合法索引、绕序一致；不是闭合流形。14 个连通分量中，主体包含 1,213,966 面；其余可作为浮点清理候选。
- 目视灰模可确认 +Z 为车头、−Z 为车尾，Y 为上；长轴 PCA 与 Z 轴的偏角约 0.023°。
- 原始几何仍有比例与细节误差，不能把这一运行记录视作比旧方法更高保真的量化证明。相机拟合、真实照片贴图与图像导出由 `../photo_texture/` 继续处理。

`manifest.json` 记录模型、代码、原照片、前景图的 SHA256，以及实际参数和成本。`inputs/` 保留四张使用过的 RGBA 副本及官方预处理后的模型输入，`geometry_checks.json` 保存资产检查，`environment.json` 保存运行依赖版本。

`outputs/bmw5_mv_seed42_v1/` 是首次加载格式选择失败的记录：官方新代码默认寻找 safetensors，而本次下载为 ckpt；脚本已显式传入 `use_safetensors=False`，失败记录保留，未覆盖。

## 复跑

复用现有 `/nas/wyh/.conda/envs/car-3d-hunyuan` 环境（Python 3.10.22、torch 2.5.1+cu118），没有修改该环境的依赖。代码单独下载到 `/nas/wyh/3d_tools/Hunyuan3D-2`，没有修改旧的 Hunyuan3D-2.1 代码和输出。

在项目根目录运行，输出目录须为空：

```bash
CUDA_VISIBLE_DEVICES=2 /nas/wyh/.conda/envs/car-3d-hunyuan/bin/python \
  08_3d_probe/multiview_probe/run_multiview.py \
  --output-dir 08_3d_probe/multiview_probe/outputs/bmw5_mv_seed42_repeat \
  --seed 42 --steps 50 --octree-resolution 512
```

脚本在导入 torch 前限制为物理 GPU 2，推理使用本地权重并关闭 Hugging Face 联网加载。该脚本针对本次四张宝马照片的探路实验，输入槽配置定义在 `INPUTS` 中。

模型来自 [官方 Hunyuan3D-2mv](https://huggingface.co/tencent/Hunyuan3D-2mv)，固定 revision `3a761b539b29fe4ff64714813aa9560fd66f5de0`。权重路径为 `models/Hunyuan3D-2mv/hunyuan3d-dit-v2-mv/model.fp16.ckpt`，SHA256 为 `361a2e90a8560bc09ed17a0b2ecda7a7d25015d1b7ed74965c547c8e30ce852d`，已与官方 LFS oid 核对。接口与方向依据 [官方多视图示例](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/shape_gen_multiview.py) 及本地预处理器源码；权重、许可证和通知保存在模型目录。
