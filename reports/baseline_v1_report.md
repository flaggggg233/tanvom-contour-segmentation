# TANVOM contour segmentation baseline-v1

|指标|结果|
|---|---:|
|训练样本|18880|
|验证样本|2664|
|模型参数量|14,328,209|
|最佳 epoch|2|
|最佳 validation Dice|0.949115|
|最佳 validation IoU|0.903158|
|Precision|0.948303|
|Recall|0.949929|
|F1|0.949115|
|最终 validation loss|0.108082|
|训练时间（秒）|22180.4|
|最终 batch size|2|
|OOM|False|
|GPU 峰值显存（MB）|2136|

本次仅训练合并的 ISOM 101/102 contour geometry。没有进行超参数调优、测试时增强、拓扑后处理或矢量化。

需要先人工检查 visualizations/best_tiles 和 visualizations/worst_tiles，才能决定是否进入 topology/vectorization 阶段。
