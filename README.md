# VegAnn Segmentation Benchmark

基于 PyTorch + SMP 的植物/背景二分类对比实验框架。输入 RGB，默认 512×512，BCE + Dice，AdamW，余弦学习率，CUDA 混合精度。包含 10 组模型配置；不包含数据集或权重。

## 1. AutoDL 安装

使用已安装 PyTorch、torchvision 且版本相互匹配的环境（建议 PyTorch 2.4 或以上）。

```bash
git clone https://github.com/xuanpluspro/vegann-seg-benchmark.git
cd vegann-seg-benchmark
pip install -r requirements.txt
```

默认 `--weights none`，所有模型随机初始化，从零训练，不下载预训练权重。仅显式传入 `--weights imagenet` 才启用骨干预训练。SMP 固定为 0.5.0。

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

- `--mask-encoding voc`：默认 0 植物、1 背景、255 忽略，与上传的 metrics.py 一致；若实际掩膜是 1 植物、0 背景，必须加 `--foreground-class 1`。支持 VOC 调色板 PNG，直接读索引值。
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

默认顺序跑 `experiments.json` 的 10 组：Unet、UnetPlusPlus、MAnet、FPN、PSPNet、DeepLabV3Plus（ResNet34）；Unet、DeepLabV3Plus（MobileNetV2）；Segformer（MiT-B0、MiT-B2）。所有骨干（包括 Segformer 的 MiT）默认随机初始化，无预训练下载。SMP 的 Unet 是使用所选骨干的变体，并非原始从零训练 U-Net。

```bash
# 三次随机种子重复实验
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --seeds 42 43 44
# 中断后恢复；保持原来的训练参数和 epochs 不变
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --resume
```

每个实验启动独立 Python 进程，失败后记录退出码并继续下一组。看 `outputs/<实验>/console.log`；结果随每次完成写入 `outputs/results.csv`。重新运行已有实验时必须使用 `--resume` 或新输出目录。可编辑 experiments.json 精简模型或设置每组的 `lr`；不应声称一种学习率对所有模型最优。

## 5. 输出与评估

每组保存 `config.json`、`history.csv`、`last.pth`、`best.pth`、`metrics.json`。最佳权重仅按验证集两类 mIoU 选择，最后评估独立测试集。预测阈值固定 0.5，不使用测试集调阈值。

评估直接使用用户提供的 metrics.py（原文件保留）：foreground_iou、background_iou、miou、dice、precision、recall、specificity、accuracy、hd95、assd、boundary_f1。区域指标累计整个数据集混淆矩阵；HD95、ASSD、Boundary F1 按有效图片平均；空掩膜约定沿用原文件。HD95/ASSD 单位是评估尺寸下的像素，Boundary F1 默认容差 2 像素（--boundary-tolerance 可调整）。训练输出仍是植物概率，评估时统一转换为 0 植物、1 背景、255 忽略。参数量包含完整模型，计算量和速度由 benchmark.py 自动统计。边界距离计算会增加验证耗时。

```bash
python train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --arch Unet --encoder resnet34 --output outputs/unet_resnet34_seed42 --eval-only
```

恢复和独立评估需匹配原始配置（包括 epochs、batch size、seed）；训练权重是自己生成的可信文件才可载入。`--weights none` 从零训练；默认从零训练，全部参数参与训练。更改指标版本、前景标签或预训练模式后应使用新输出目录，避免与旧实验混用。

## 实验建议

所有模型使用相同数据划分、尺寸、指标及预算；先单种子筛选，再多种子报告均值和标准差。使用已有 VOC 划分，不自动随机重分数据。若对照 VegAnn 官方五套划分，分别生成对应清单并分开保存输出。实验日志和数据不提交 Git。

## 自动复杂度和速度测试

每组加载最佳权重，完成精度评估后，自动保存 benchmark.json，并把 params_m、gflops、counted_gflops、flops_complete、latency_ms、fps 写入 results.csv。

- 输入为 batch=1、RGB、--size × --size（默认 512×512），与训练 batch size 无关。
- 参数量 params_m 单位百万，统计完整网络。
- GFLOPs 使用 fvcore 原生口径：一次乘加计 1 次运算。不要与乘加计 2 次的表格直接比较。
- fvcore 未统计的算子会记录在 unsupported_ops；存在未支持算子时 gflops 留空，counted_gflops 仅为已统计部分，不能当作完整计算量。即使 flops_complete 为 true，也遵循 fvcore 默认忽略某些操作的估计口径。
- FPS/延迟使用 eager PyTorch FP32，关闭 autocast 和 TF32，无 TensorRT、无 torch.compile。默认预热 50 次，测量 200 次，CUDA Event 计时并同步。
- latency_ms 是单图 GPU 前向平均耗时；fps=1000/latency_ms，为纯模型前向的等效 FPS。不包含读图、预处理、CPU/GPU 传输、sigmoid/阈值和指标计算，不代表应用端到端吞吐。
- 没有 CUDA 时不生成 GPU FPS；避免与 CPU 结果混用。元数据记录 GPU、PyTorch/CUDA/cuDNN 版本、输入形状、精度和计时范围。测速期间避免同卡同时训练其他任务。
- --benchmark-warmup 和 --benchmark-iterations 可修改次数；所有模型必须一致。

GPU 完整训练及测速需在 AutoDL 环境验证。本仓库构建环境仅完成静态和局部逻辑检查，没有实测性能数字。

## 修改记录

所有仓库改动都通过 GitHub commit 保存，可在仓库 Commits 查看逐次修改及文件 diff。此次加入 benchmark.py，并更新训练、批量汇总和依赖。每个实验的 config.json、history.csv、metrics.json、benchmark.json 是实验记录，与 Git 修改记录分别保存。
