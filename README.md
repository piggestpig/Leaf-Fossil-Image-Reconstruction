# Foreseeing the Invisible: Amodal Reconstruction of Leaf Fossil Images

[![arXiv](https://img.shields.io/badge/arXiv-2608.04423-red)](https://arxiv.org/abs/2608.04423v1)
[![Demo](https://img.shields.io/badge/%F0%9F%8C%90-Online%20Demo-blue)](https://piggestpig.github.io/Leaf-Fossil-Image-Reconstruction/)

[中文](./README.md) | [English](./README_EN.md)

⭐这是一个基于DINOv3的树叶化石图像重建项目，包含合成数据集、训练代码、权重、网页部署等内容，存放在不同的分支中

## 快速开始
4bit量化模型已经部署到网页上了！[点击查看在线演示](https://piggestpig.github.io/Leaf-Fossil-Image-Reconstruction/)

需要Edge/Chrome/Safari等支持WebGPU的浏览器，下载约300MB模型后使用本地算力推理

## 下载和训练

点击页面上方的 `<> Code` -> `Download ZIP`下载内容
- 首先安装 [requirements.txt](./requirements.txt) 中的依赖库。我们重新整理了代码，支持不同平台和python版本，只要安装成功即可运行。
- 论文里是在AutoDL服务器上使用RTX Pro 6000训练的，环境为Ubuntu 22.04 / Python 3.12 /  CUDA 13.0
- 实测Windows+Intel核显也能运行，注意一定要安装**正确的pytorch版本**，比如Intel这边用 pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

### · main
- 你正在查看就是主分支，这里有简化版训练/推理代码，总共约75MB。
- [train.py](./train.py): 需要下载`DINOv3_weights分支`里的初始权重，放置在`weights`文件夹中，然后运行此脚本训练模型。使用方法和超参数写在文件开头。
- [predict.py](./predict.py): 训练后运行此脚本推理图片；或者直接下载`AmodalDINO_weights分支`的预训练权重，放置在`weights`文件夹中，然后运行此脚本进行推理。使用方法写在文件开头。
- [dataset文件夹](./dataset) 存放了1%的数据集用于快速测试。完整数据集位于`dataset分支`。

### · [AmodalDINO_weights](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/AmodalDINO_weights)
- 完整的fp32预训练权重，约1.2GB，通过分卷压缩上传到github上

### · [DINOv3_weights](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/DINOv3_weights)
- DINOv3 ViT-L/16的官方初始权重，约1.1GB，通过分卷压缩上传到github上

### · [blender](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/blender)
- 用于合成树叶化石图片的原素材 + Blender 5.2工程文件 + 后处理python脚本，约700MB

### · [dataset](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/dataset)
- 完整的树叶化石数据集，包含 10000组train + 1000组val + 9组test，约4.2GB

### · [web-deployment](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/web-deployment)
- 网页部署代码和量化模型，约400MB，其中AmodalDINO被切分成3个文件上传github

## 图片展示

![Fig.1](https://arxiv.org/html/2608.04423v1/figures/real_teaser.jpg)  
图1：论文封面

<br>

![Fig.2](https://arxiv.org/html/2608.04423v1/figures/pipeline_overview.jpg)  
图2：完整管线

<br>

![Fig.3](https://arxiv.org/html/2608.04423v1/figures/synthetic_samples.jpg)  
图3：合成的树叶化石数据集

<br>

![Fig.4](https://arxiv.org/html/2608.04423v1/figures/panels_083188.jpg)  
图4：输入RGB化石图像，得到4个分割头的结果 (visible leaf, amodal leaf, amodal vein, detail vein)

<br>

![Fig.5](https://arxiv.org/html/2608.04423v1/figures/ablation.jpg)  
图5：Ablation test, 用控制变量法研究两个关键的设计要点

<br>

![Fig.6](https://arxiv.org/html/2608.04423v1/figures/kins_qual.jpg)  
图6：用RGB输入+visible/amodal分割头在KINS数据集上进行训练，得到Baseline的可视化结果

<br>

![Fig.7](https://arxiv.org/html/2608.04423v1/figures/real_fossils.jpg)  
图7：另外3张在真实化石上预测的结果

<br>

![Fig.8](https://arxiv.org/html/2608.04423v1/figures/browser_ui.jpg)  
图8：网页部署

<br>

![Fig.9](https://arxiv.org/html/2608.04423v1/figures/failure_instances.jpg)  
图9：失败案例，一块化石有多片树叶

更多内容正在整理中
