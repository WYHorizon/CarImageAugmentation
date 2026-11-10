# 首轮样图结果

[文档首页](README.md) · [方法说明](01_physics_based.md)

本页中的输出相对路径以 `/nas/wyh/CarImageAugmentation/01_physics_based` 为基准；图片和清单链接可直接打开。

输入是共享 `test_images/` 中的奥迪 A6L、宝马 4 系、小米 YU7、本田 CR-V，共 4 张图。

输出位于 `outputs/demo/`：每张图分别生成薄雾、较明显的雾和短暂小雨，共 **12 张 3840 × 2880 PNG**。

- [整图对比](../01_physics_based/outputs/demo/previews/overview.jpg)：各行分别为四辆车，各列为原图、薄雾、较明显的雾、小雨。
- [奥迪关键部件](../01_physics_based/outputs/demo/previews/奥迪A6L_details.jpg)
- [宝马关键部件](../01_physics_based/outputs/demo/previews/宝马4系_details.jpg)
- [小米关键部件](../01_physics_based/outputs/demo/previews/小米YU7_details.jpg)
- [本田关键部件](../01_physics_based/outputs/demo/previews/本田CR-V_details.jpg)
- [复现清单与部件诊断](../01_physics_based/outputs/demo/manifest.json)

10 项自动检查通过，覆盖零雾强度下逐字节还原、雾的对比度关系、距离衰减、场景深度有效性、雨滴遮挡、随机种子复现、输入不可变性、命令行输出记录及已有结果保护。另核对全部输出的尺寸、PNG 完整性、源图与输出哈希、配置快照和实现哈希。

当前关键矩形与车辆轮廓交集中的最低雾透射率：薄雾约 **0.905**，较明显的雾约 **0.805**。小雨中关键区域最高雨层混合系数约 **0.060**。这些数值描述本次光学作用强度，不是车型保真或训练有效性的合格阈值。

预览检查后对背景近似深度层增加了平滑过渡，减少了明显的多边形边界。雾的变化比小雨更明显；小雨在缩小后的图像中仍较克制。黑色车体的结构和纹理来源是原图，但雾会抬高黑位、降低细节对比度。

这一轮验证了基线可运行和可复现，尚未进行模型训练评测。场景深度没有标定，小雨保留原有照明和地面状态，结果只能作为近似物理增强候选。

## 降雨强度补充实验

结果位于 `outputs/rain_intensity_v2/`。对四张原图分别运行 `rain_light`、`rain_medium`、`rain_heavy`，共 12 张原分辨率 PNG；其中中雨和大雨是新增的 8 张结果。

- [四张图的降雨总览](../01_physics_based/outputs/rain_intensity_v2/previews/overview.jpg)
- [奥迪单行对比：原图 / 小雨 / 中雨 / 大雨](../01_physics_based/outputs/rain_intensity_v2/previews/奥迪A6L_comparison.jpg)
- [奥迪关键部件对比](../01_physics_based/outputs/rain_intensity_v2/previews/奥迪A6L_details.jpg)
- [参数、种子和诊断清单](../01_physics_based/outputs/rain_intensity_v2/manifest.json)

全部图片由本地 Python 代码运行得到，没有调用图像生成模型。新增档位调整雨滴空间密度、尺寸、曝光长度、风速扰动、有效辐射亮度和散射强度。中雨与大雨的名称表示视觉增强档位，未标定为气象学雨量。

观察：中雨雨丝可见，关键部件遮挡较少；大雨雨丝及雨幕明显，局部辨识信息被更多雨丝覆盖。车辆没有重新绘制；视觉差异来自原图上的散射和雨滴作用。大雨仍保留原图干地与晴天阴影，因此整体有合成感，不宜当作持续阴雨的完整再现。

建议将中雨作为优先试训候选，大雨作为少量困难样本，再以独立真实天气测试集验证收益。这是基于样图的选择建议，尚无训练实验支持。

新增的 HDR 雨滴亮度在显示转换时可能截断高亮。四张图中最高的超显示范围像素比例：小雨 0%，中雨约 0.130%，大雨约 2.078%。相应的关键车辆区域最低散射透射率约为 0.996、0.975、0.936。局部遮挡及高亮变化仍需结合部件放大图审查，不能仅凭未改变几何结构认定所有判别信息完整。

12 项自动测试通过，包括新增的三档雨滴有效信号强度、HDR 成像与显示转换检查。另核验新结果的尺寸、PNG 完整性、图像哈希、实现哈希；四张小雨 PNG 与首轮结果逐字节相同，原始图片未变。

复现命令（从本方法目录执行，替换为新的输出目录）：

```bash
cd /nas/wyh/CarImageAugmentation/01_physics_based
python run.py --weather rain_light rain_medium rain_heavy --output-dir outputs/rain_intensity_v3
```

## 阴天照明与湿地补充实验

结果位于 `outputs/overcast_wet_v3/`，含四张原图 × 三种状态，共 **12 张 3840 × 2880 PNG**：

- `overcast_wet`：阴天照明与湿地近似，不含降水，便于单独检查环境变化。
- `rain_overcast_wet_medium`：环境变化 + 中等视觉降雨。
- `rain_overcast_wet_heavy`：环境变化 + 较强视觉降雨。

查看结果：

- [四辆车的新旧版本总览](../01_physics_based/outputs/overcast_wet_v3/previews/before_after_overview.jpg)
- [奥迪：原图 / 旧中雨 / 新中雨 / 新大雨](../01_physics_based/outputs/overcast_wet_v3/previews/奥迪A6L_before_after.jpg)
- [小米：原图 / 旧中雨 / 新中雨 / 新大雨](../01_physics_based/outputs/overcast_wet_v3/previews/小米YU7_before_after.jpg)
- [奥迪轮胎接地处对比](../01_physics_based/outputs/overcast_wet_v3/previews/奥迪A6L_contact_comparison.jpg)
- [宝马关键部件对比](../01_physics_based/outputs/overcast_wet_v3/previews/宝马4系_details.jpg)
- [参数与阶段隔离诊断](../01_physics_based/outputs/overcast_wet_v3/manifest.json)

全部通过本地 NumPy/Pillow 代码生成，没有使用生成模型或学习式深度、重光照模型。新增方法包含人工近似几何、照明增益估计、路面阴影补偿、空气—水 Fresnel、水膜漫反射衰减、地面坐标中的非均匀水膜及可见表面的粗糙反射代理。属于物理启发的确定性工程近似，照明分解、水膜位置及几何代理没有经过真实场景标定。

实际检查结果：

- 原图、输出 PNG、配置快照和实现哈希核对通过，输出尺寸保持 3840 × 2880。
- 车辆重光照仅按原像素乘以 RGB 增益，本次实际逐通道增益范围约为 **0.740–0.759**；配置允许范围为 [0.74, 1.08]。
- 重光照阶段的车辆乘法关系最大误差为 **0**，湿地层对车辆保护区的最大像素改动为 **0**。这两个诊断作用于手工轮廓定义的区域，不能认证轮廓本身百分之百准确。
- 本次 12 张图片的线性 RGB 超出显示范围像素比例均为 **0**。
- **17 项测试通过**，新增检查包括水 Fresnel 极限、湿地层与车辆隔离、细密明暗图案不被模糊、湿润度为零的恒等关系、湿地随机性仅作用于路面，以及非法参数拒绝。

从预览看，地面由偏亮、偏暖的干地变为更暗、略有环境反射的湿地；晴天投影对比有所减弱，接触遮蔽保留。细化了轮胎和车底轮廓，减少初版湿地层在车旁留下的多边形干地边界。

当前效果仍较克制，部分晴天方向性反射和阴影有残留。路面反射使用同视角可见表面，无法代替真实镜面视角；非均匀水膜也未根据真实凹陷地形求解。车辆没有重绘或形变，但照明、量化、雨滴遮挡仍会影响细粒度线索的可见度，不能据此承诺绝对无损或训练精度不下降。

复现（从本方法目录执行，选择新的输出目录）：

```bash
cd /nas/wyh/CarImageAugmentation/01_physics_based
python run.py --weather overcast_wet rain_overcast_wet_medium rain_overcast_wet_heavy --output-dir outputs/overcast_wet_v4
python compare_runs.py --before outputs/rain_intensity_v2 --after outputs/overcast_wet_v4
```
