# CLIP 在细粒度图像分类上的零样本评估与适配实验

## 项目背景

本项目旨在评估 OpenAI CLIP 模型在细粒度图像分类任务（CUB-200-2011 鸟类数据集）上的零样本能力，并与传统监督学习模型（ResNet-18）进行对比。进一步探索通过线性 probe 对 CLIP 图像特征进行轻量适配的有效性。

该实验受 **Benchmarking Large Vision-Language Models on Fine-Grained Image Tasks** (ICLR 2026) 等工作的启发，重点关注视觉-语言模型在细粒度识别中的局限性。

## 数据集

使用 [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) 数据集，包含 200 类鸟类，训练集 5994 张，测试集 5794 张。

## 方法

### 1. 零样本分类 (Zero-shot)
- 模型：CLIP ViT-B/32
- 文本 prompt：`"a photo of a {class_name}, a type of bird."`
- 通过计算图像与文本特征的余弦相似度进行分类。

### 2. 线性 Probe (Linear Probe)
- 使用 CLIP 图像编码器提取测试集和训练集特征（归一化）
- 训练一个多类逻辑回归分类器（scikit-learn）
- 评估测试集准确率

### 3. 监督基线 (Supervised Baseline)
- 模型：ResNet-18（ImageNet 预训练）
- 微调 10 个 epoch，使用标准数据增强
- 评估测试集准确率

## 结果

| 方法 | 准确率（ResNet训练十轮） |
|------|--------------|
| CLIP 零样本 | 47.20% |
| CLIP 线性 Probe | 60.10% |
| ResNet-18 微调 | 68.83% |

## 项目结构
```
clip-finegrained-eval/
├── README.md
├── src/
│   ├── clip-finegrained-eval.py
└── results/
    └── test_result.png
```

## 作者
- 姓名：赵术让
- 学校：东南大学计算机科学与技术学院
- 联系方式: 479005476@qq.com
