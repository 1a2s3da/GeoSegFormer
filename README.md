# 植物叶片病斑分割项目 (RBP-UNet on PlantSeg v3)

基于"区域-边界-原型一致性协同学习"的复杂场景植物病害多类语义分割。

## 项目结构

```
plant_disease_seg/
├── README.md
├── requirements.txt
├── data/
│   └── plantsegv3/                (PlantSeg v3 数据集, 11458 图, 115 类)
│       ├── images/{train,val,test}
│       ├── annotations/{train,val,test}
│       ├── annotation_{train,val,test}.json
│       └── Metadatav2.csv
├── docs/
│   └── 方法论.pdf                  (模型设计文档)
├── pretrained/
│   └── resnet50.pth                (ImageNet 预训练 ResNet50, 项目内自带)
├── code/
│   ├── main.py                     (统一入口)
│   ├── models/
│   │   ├── rbp_unet.py             (完整 RBP-UNet, 方法论实现)
│   │   ├── baseline_unet.py        (ResNet50-UNet baseline)
│   │   └── losses.py               (Region/Boundary/Cons/Proto 损失)
│   ├── data/
│   │   └── plantseg.py             (Dataset + 数据增强 + 距离场计算)
│   ├── train/
│   │   ├── train_baseline.py
│   │   └── train_rbp.py
│   ├── evaluation/
│   │   └── eval_test.py
│   └── utils/
│       └── metrics.py              (mIoU / mAcc / aAcc)
└── runs/                           (训练产出)
```

## 任务

- 数据集: PlantSeg v3
- 任务类型: 多类语义分割 (115 类: 1 背景 + 114 病害)
- 数据规模: 7916 train / 1247 val / 2295 test
- 评估指标: mIoU, mAcc

## Baseline 对比 (摘自 PlantSeg 官方)

| Method | Encoder | mIoU | mAcc |
|---|---|---|---|
| DeepLabV3 | ResNet50 | 17.24 | 37.95 |
| DeepLabV3 | ResNet101 | 20.72 | 40.63 |
| DeepLabV3+ | ResNet50 | 25.08 | 40.66 |
| DeepLabV3+ | ResNet101 | 27.18 | 42.29 |
| SAN | ViT-B/16 | 34.79 | 50.19 |
| SAN | ViT-L/14 | 36.91 | 52.81 |
| SegNext | MSCAN-L | 44.52 | 59.95 |

## 方法 (RBP-UNet)

按方法论 PDF 实现, 4 个核心模块:

1. **ResNet50-UNet 主干**: 5 层编码 + 4 层 U-Net 解码
2. **多尺度特征增强 (SAMRE)**: 1 个普通 Conv + 2 个 dilated Conv 用 Softmax 门控动态加权融合
3. **区域-边界双向交互**: 区域分支 (多类 softmax) + 边界距离场分支 (Tanh) + 注意力交互
4. **三原型一致性**: 在共享特征空间构建"病斑内部 / 边界 / 背景"原型, NT-Xent 对比损失
5. **联合损失**: L = L_region (CE+Dice+FT) + λ1·L_boundary (SmoothL1+Dice) + λ2·L_cons (Sobel) + λ3·L_proto

## 安装

```bash
cd plant_disease_seg
pip install -r requirements.txt
```

ResNet50 / Swin-T / Swin-B / MiT-B3 预训练权重都在 `pretrained/` 下 (共 ~700 MB), 不会再去下载.

## 训练 / 评估

```bash
cd plant_disease_seg

python3 code/main.py train-baseline --epochs 30 --bs 16
python3 code/main.py train-rbp --epochs 40 --bs 12
python3 code/main.py eval-baseline
python3 code/main.py eval-rbp
python3 code/main.py pipeline
```

## 备注

- 项目根目录 = README.md 所在目录, 代码自动定位
- 输入图像 resize 到 256x256 训练 (与 baselines 主流配置接近)
- 训练数据增强: hflip / vflip / colorjitter
- 距离场为 signed normalized distance (in [-1, 1]), max_dist=20 像素
