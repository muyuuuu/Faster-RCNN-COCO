# 目标检测快速训练与推理框架

每次做目标检测的任务都重头开坑，这次干脆基于 `torchvision` 写一个针对 `COCO` 数据集的通用框架。包括：

- 加载数据
- 训练
- 保存与加载模型
- 推理

写一个通用的训练、推理框架，下次直接用。注意，这里只是调库，并非自己重头写神经网络。

**注意：**

- 目前只针对 `COCO` 数据集。
- 训练注意增加类别，多一个背景类。

在这里下载 [mini-coco](https://github.com/chongruo/tiny-coco) 数据集，可以测试。

---

- `preprocess` 文件夹，这俩代码都是单独执行的
  - `mean-std.py`，计算样本三通道的均值与方差，用于数据标准化
  - `reconstruct-anno.py`，重构 `json`，支持多个 `batchsize`
- `model` 文件夹
  - `data_helper.py`，加载数据
  - `engine.py`，训练与推理的具体过程
  - `eval.py`，推理程序，`python eval.py` 执行
  - `model.py`，模型
  - `train.py`，训练程序，`python train.py` 执行
  - `utils.py`，保存、加载模型和写入日志