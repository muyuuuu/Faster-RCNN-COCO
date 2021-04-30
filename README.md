# 目标检测快速训练与推理框架

每次做目标检测的任务都重头开坑，这次干脆基于 `torchvision` 写一个针对 `COCO` 数据集的通用框架。包括：

- 加载数据
- 训练
- 保存与加载模型
- 推理

写一个通用的训练、推理框架，下次直接用。

在这里下载 [mini-coco](https://github.com/chongruo/tiny-coco) 数据集，可以测试。