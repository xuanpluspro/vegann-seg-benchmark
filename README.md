# VegAnn Segmentation Benchmark

基于 PyTorch + SMP 的植物/背景二分类对比实验框架。输入 RGB，默认 512×512，BCE + Dice，AdamW，余弦学习率，CUDA 混合精度。包含 10 组模型配置；不包含数据集或权重。

## 1. AutoDL 安装

使用已安装 PyTorch、torchvision 且版本相互匹配的环境（建议 PyTorch 2.4 或以上）。

```bash
git clone https://github.com/xuanpluspro/vegann-seg-benchmark.git
cd vegann-seg-benchmark
pip install -r requirements.txt
```

首次使用预训练权重需要服务器能访问下载源；下载失败应检查网络或上传本地权重，不能把失败当作从零训练成功。SMP 固定为 0.5.0。

## 2. 数据格式

`--data-root` 指向 VOC2012 本身，不是 ImageSets：

```text
VOC2012/
  JPEGImages/               # 原图，保留原名
  SegmentationClass/       # 同名 PNG 掩膜
  ImageSets/
    Segmentation/
      train.txt
      val.txt
      test.txt
```

也支持直接位于 `ImageSets/train.txt` 等的文件。每行一个无扩展名的图片 ID。无需 XML 检测标注。Windows 路径示例：`D:\VegAnn-Parquet\exchange\VOCdevkit-4\VOC2012`；上传服务器后改成实际 Linux 路径。

必须明确选择掩膜模式，避免静默误读：

- `--mask-encoding voc`：0 背景、1 植物、255 忽略。支持 VOC 调色板 PNG，直接读索引值。
- `--mask-encoding binary255`：0 背景、255 植物，没有忽略标签。

RGB 彩色掩膜会报错，需要先转换为单通道类别标签。数据检查拒绝缺图、异常标签、重复 ID、训练/验证/测试重叠。图像缩放使用双线性，掩膜使用最近邻；训练只做同步水平/垂直翻转。

## 3. 先跑一组

以下服务器路径是假设上传位置，请按实际路径调整：

```bash
python train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --arch Unet --encoder resnet34 --epochs 100 --batch-size 8 --output outputs/unet_resnet34_seed42
```

先用 `--epochs 2 --workers 0` 在单独输出目录验证流程，再开始正式训练。显存不足时降低 batch size。没有独立测试集时显式加 `--test-split none`，输出只包含验证指标，不能称为测试结果。支持自定义 `--train-split`、`--val-split`、`--test-split`。

## 4. 批量训练

```bash
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --epochs 100 --batch-size 8
```

默认顺序跑 `experiments.json` 的 10 组：Unet、UnetPlusPlus、MAnet、FPN、PSPNet、DeepLabV3Plus（ResNet34）；Unet、DeepLabV3Plus（MobileNetV2）；Segformer（MiT-B0、MiT-B2）。Segformer 的 MiT 编码器使用自己的预训练权重，预训练条件应在论文中说明。SMP 的 Unet 是使用所选骨干的变体，并非原始从零训练 U-Net。

```bash
# 三次随机种子重复实验
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --seeds 42 43 44
# 中断后恢复；保持原来的训练参数和 epochs 不变
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --resume
```

每个实验启动独立 Python 进程，失败后记录退出码并继续下一组。看 `outputs/<实验>/console.log`；结果随每次完成写入 `outputs/results.csv`。重新运行已有实验时必须使用 `--resume` 或新输出目录。可编辑 experiments.json 精简模型或设置每组的 `lr`；不应声称一种学习率对所有模型最优。

## 5. 输出与评估

每组保存 `config.json`、`history.csv`、`last.pth`、`best.pth`、`metrics.json`。最佳权重仅按验证集两类 mIoU 选择，最后评估独立测试集。预测阈值固定 0.5，不使用测试集调阈值。

指标从整个数据集累计混淆矩阵计算：前景 IoU、背景 IoU、两类 mIoU、前景 Dice、Precision、Recall、Pixel Accuracy。分母为零的指标为 null；mIoU 对有定义类别平均。参数量包含编码器、解码器和输出层。框架暂未包含 FLOPs、推理延迟或边界指标，请勿将其与已测指标混淆。

```bash
python train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --arch Unet --encoder resnet34 --output outputs/unet_resnet34_seed42 --eval-only
```

恢复和独立评估需匹配原始配置（包括 epochs、batch size、seed）；训练权重是自己生成的可信文件才可载入。`--weights none` 从零训练；默认 ImageNet 骨干预训练，编码器全量参与训练。

## 实验建议

所有模型使用相同数据划分、尺寸、指标及预算；先单种子筛选，再多种子报告均值和标准差。使用已有 VOC 划分，不自动随机重分数据。若对照 VegAnn 官方五套划分，分别生成对应清单并分开保存输出。实验日志和数据不提交 Git。
