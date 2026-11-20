# 宝马 5 系可旋转模型

[下载 GLB](bmw5.glb) · [三个视角](../../docs/assets/bmw5/three-views.jpg) · [同机位修复对比](../../docs/assets/bmw5/repairs.jpg) · [街景融合与复跑](../../07_background_replacement/bmw_street/README.md)

四张正前、左侧、正后、右侧照片输入 Hunyuan3D-2mv 生成几何。随后用八张照片拟合相机并生成共享纹理。格栅、车灯、展示牌和可见边线尽量沿用照片。车顶上表面、玻璃遮蔽后的区域，以及后备厢盖破碎的贴图，根据邻近表面和可见照片修整。玻璃以不透视的浅青灰色模拟日光观感。贴图仍包含拍摄时的照明，高机位和遮蔽区域的真实形状缺少独立测量。[建模输入](../../test_images/宝马5系) · [几何生成](../multiview_probe/README.md) · [相机拟合及照片贴图](../photo_texture/README.md)

| 版本 | 本次采用的内容 |
| --- | --- |
| `bmw5_mv_textured_v5` | 四图几何和按车身区域选来源的八图纹理 |
| `bmw5_v5_seam_refined_v2` → `bmw5_v5_hood_refined_v3` | 修接缝、前翼子板和机盖。这是后续局部修复使用的 v3 基线。早期加权混色另有一个 `bmw5_mv_textured_v3` |
| `bmw5_v3_daylight_glass_v1` | 在玻璃试验、遮蔽玻璃及棚光版本之后，选用更浅的日光玻璃 |
| `bmw5_v3_roof_refined_v1` → `bmw5_v3_photo_detail_v1` | 局部调整车顶形面、天线周围及车顶颜色，并补回照片可见的车头细节 |
| `bmw5_v3_trunk_refined_v1` | 当前发布资产。只将参考照片引导的局部重绘转回后备厢盖漆面，保留玻璃、尾灯、标识及牌照 |

各部位可打开原尺寸对照：[机盖](../../docs/assets/bmw5/hood.png) · [玻璃](../../docs/assets/bmw5/glass.jpg) · [车顶](../../docs/assets/bmw5/roof.jpg) · [后备厢盖](../../docs/assets/bmw5/trunk.jpg)。Qwen 轻度和重度去反光、v6 分部件材质另有实验记录。[去反光](../../09_reflection_cleanup/README.md) · [v6 材质](../photo_texture/MATERIALS.md)

## 导出新视角

下载的 GLB 可用 Blender 或兼容 glTF 2.0 的查看器打开。从仓库根目录执行下列命令可复用发布预览使用的日光。Node 依赖也可以安装在独立目录后通过 `NODE_PATH` 指定。

```bash
npm install --prefix 08_3d_probe/bmw5 --no-save --no-package-lock three@0.160.1 esbuild@0.28.2 playwright@1.65.1
08_3d_probe/bmw5/node_modules/.bin/playwright install chromium
node 08_3d_probe/bmw5/render.cjs /tmp/bmw-views
node 08_3d_probe/bmw5/render.cjs /tmp/bmw-views --angle 65
```

默认导出方位角 32°、90°、148° 的透明 PNG。角度 0° 对应车头，默认仰角为 8°。`--street` 使用商铺街记录的透视机位。模型的 SHA-256 是 `5b47e4d438b856f59aee020da14d75f79bd410d3b2309feb5732ecd2571e6d02`。[模型核验记录](verification.json)

[逐版修复脚本](../photo_texture/recipes/)保留实验原有阶段名。脚本的输入还包括各阶段的模型、贴图、蒙版、拟合相机和局部重绘素材。需要重放历史步骤时，将相应阶段脚本放回 `08_3d_probe/photo_texture/outputs/<阶段名>/`，并准备其引用的上游产物。上面的新视角导出和商铺街示例可直接用仓库提供的最终 GLB。
