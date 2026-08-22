# Foreseeing the Invisible: Amodal Reconstruction of Leaf Fossil Images

[![arXiv](https://img.shields.io/badge/arXiv-2608.04423-red)](https://arxiv.org/abs/2608.04423v1)
[![Demo](https://img.shields.io/badge/%F0%9F%8C%90-Online%20Demo-blue)](https://piggestpig.github.io/Leaf-Fossil-Image-Reconstruction/)

[中文](./README.md) | [English](./README_EN.md)

⭐ This is a DINOv3-based leaf-fossil image reconstruction project. The synthetic dataset, training code, weights, and web deployment live on separate branches.

## Quick start

A 4-bit quantized model is already deployed in the browser. [Open the online demo](https://piggestpig.github.io/Leaf-Fossil-Image-Reconstruction/)

Use a WebGPU-capable browser such as Edge, Chrome, or Safari. It will download around 300 MB models and run inference on your device.

## Download and training

Click `<> Code` → `Download ZIP` at the top of the page.
- Install the packages in [requirements.txt](./requirements.txt) first. The code was reorganized to work across platforms and Python versions; if the install succeeds, it should run.
- The paper was trained on an AutoDL server with an RTX PRO 6000, Ubuntu 22.04 / Python 3.12 / CUDA 13.0.
- It also runs on Windows with Intel integrated graphics. Install the **correct PyTorch build**—for Intel, for example: `pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu`

### · main

- You are on the main branch: a compact training / inference codebase, about 75 MB in total.
- [train.py](./train.py): download the initial weights from the `DINOv3_weights` branch, put them in the `weights` folder, then run this script. Usage and hyperparameters are at the top of the file.
- [predict.py](./predict.py): run this after training, or download the pretrained weights from the `AmodalDINO_weights` branch, put them in `weights`, then run inference. Usage is at the top of the file.
- The [dataset](./dataset) folder contains 1% of the data for a quick test. The full dataset is on the `dataset` branch.

### · [AmodalDINO_weights](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/AmodalDINO_weights)

- Full fp32 pretrained weights, about 1.2 GB, uploaded as split zip volumes.

### · [DINOv3_weights](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/DINOv3_weights)

- Official DINOv3 ViT-L/16 initialization weights, about 1.1 GB, uploaded as split zip volumes.

### · [blender](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/blender)

- Source assets for synthesizing leaf-fossil images, plus a Blender 5.2 project and post-processing Python scripts, about 700 MB.

### · [dataset](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/dataset)

- Full leaf-fossil dataset: 10,000 train / 1,000 val / 9 test images, about 4.2 GB.

### · [web-deployment](https://github.com/PiggestPig/Leaf-Fossil-Image-Reconstruction/tree/web-deployment)

- Web deployment code and the quantized model, about 400 MB. AmodalDINO is split into 3 files for GitHub.

## Figures

![Fig.1](https://arxiv.org/html/2608.04423v1/figures/real_teaser.jpg)  
Figure 1: Paper teaser.

<br>

![Fig.2](https://arxiv.org/html/2608.04423v1/figures/pipeline_overview.jpg)  
Figure 2: Full pipeline.

<br>

![Fig.3](https://arxiv.org/html/2608.04423v1/figures/synthetic_samples.jpg)  
Figure 3: Synthetic leaf-fossil dataset.

<br>

![Fig.4](https://arxiv.org/html/2608.04423v1/figures/panels_083188.jpg)  
Figure 4: From an RGB fossil image, four heads: visible leaf, amodal leaf, amodal vein, and detail vein.

<br>

![Fig.5](https://arxiv.org/html/2608.04423v1/figures/ablation.jpg)  
Figure 5: Ablation of the two key design choices.

<br>

![Fig.6](https://arxiv.org/html/2608.04423v1/figures/kins_qual.jpg)  
Figure 6: Qualitative baseline on KINS, trained with RGB input and visible / amodal heads.

<br>

![Fig.7](https://arxiv.org/html/2608.04423v1/figures/real_fossils.jpg)  
Figure 7: Three more predictions on real fossils.

<br>

![Fig.8](https://arxiv.org/html/2608.04423v1/figures/browser_ui.jpg)  
Figure 8: In-browser deployment.

<br>

![Fig.9](https://arxiv.org/html/2608.04423v1/figures/failure_instances.jpg)  
Figure 9: Failure cases, where one fossil contains multiple leaves.

More content is being prepared.
