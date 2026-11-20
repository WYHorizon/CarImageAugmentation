# 宝马街景融合

| 街景 | 原始背景 | 宝马 5 系 GLB 选角度 | 宝马 525Li 环绕图选帧 |
| --- | --- | --- | --- |
| 商铺街，1080 × 811 | [背景](../assets/backgrounds/04_neighborhood_shopping_street.png) | [前景近景成图](../../docs/assets/bmw5/shopping-glb.png) | [第 04 帧](../../bmw_img/bmw-2026-525li-carbon-black-360/04.webp) · [成图](../../docs/assets/bmw5/shopping-frame.png) |
| 居民街，800 × 533 | [背景](../assets/backgrounds/05_residential_street.png) | [成图](../../docs/assets/bmw5/residential-glb.png) | [第 06 帧](../../bmw_img/bmw-2026-525li-carbon-black-360/06.webp) · [成图](../../docs/assets/bmw5/residential-frame.png) |
| 麓山南路，1024 × 1365 | [背景](../assets/backgrounds/06_tree_lined_avenue.png) | [成图](../../docs/assets/bmw5/lushan-south-road-glb.png) | [第 02 帧](../../bmw_img/bmw-2026-525li-carbon-black-360/02.webp) · [成图](../../docs/assets/bmw5/lushan-south-road-frame.png) |

GLB 使用八张棚拍照片制作的[已修复宝马 5 系](../../08_3d_probe/bmw5/README.md)。另一列取自独立的 2026 款 525Li 碳黑色环绕素材，共 36 个 WebP，解码后有 20 个不同画面。两列均按路面透视放车，建立接地阴影，再用本地 Qwen-Image-2.1 做双参考光照融合。车牌、车标等局部细节用输入保护，车与阴影以外使用原街景。两种车辆素材在窗户、轮毂和原始照明上都有差异，这组图用于比较实际观感。

## 复跑商铺街的模型版本

从仓库根目录运行。使用记录的透明渲染、GLB 和 Qwen 原始输出，可以在 CPU 上重做几何投影阴影、图像配准、车牌等细节保护和最终合成。

```bash
python -m pip install numpy pillow opencv-python-headless
python 07_background_replacement/bmw_street/replay.py /tmp/bmw-shopping.png --save-layout /tmp/bmw-layout.png
```

要从 GLB 重新生成车辆像素，先安装[渲染依赖](../../08_3d_probe/bmw5/README.md)，再用相同相机跑完整示例。

```bash
node 08_3d_probe/bmw5/render.cjs /tmp/bmw-views --street
python 07_background_replacement/bmw_street/replay.py /tmp/bmw-from-glb.png --render /tmp/bmw-views/shopping-2x.png
```

`--save-layout` 保存的是送入模型的已摆放街景。历史推理以它和空街景为双参考，seed 42、40 步。可通过 `--edit` 替换已裁回原尺寸的 Qwen 输出。复跑默认读取随示例保存的那次推理，所需模型权重独立准备。[机位](camera.json) · [推理参数与完整提示词](qwen_settings.json) · [已发布成图](../../docs/assets/bmw5/shopping-glb.png)
