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

默认按 0 背景、1 植物读取，无需额外指定前景类别或掩膜模式：

- `--mask-encoding voc`（默认）：0 背景、1 植物、255 忽略。支持单通道或 VOC 调色板 PNG，直接读类别索引；这里的 voc 是掩膜编码方式，不是模型。旧数据若是 0 植物、1 背景，可显式传入 `--foreground-class 0`。
- `--mask-encoding binary255`：0 背景、255 植物，没有忽略标签。

RGB 彩色掩膜会报错，需要先转换为单通道类别标签。数据检查拒绝缺图、异常标签、重复 ID、训练/验证/测试重叠。图像缩放使用双线性，掩膜使用最近邻；训练只做同步水平/垂直翻转。

## 3. 先跑一组

以下服务器路径是假设上传位置，请按实际路径调整：

```bash
python train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --arch Unet --encoder resnet34 --epochs 30 --batch-size 32 --output outputs/unet_resnet34_seed42
```

先用 `--epochs 2 --workers 0` 在单独输出目录验证流程，再开始正式训练。显存不足时降低 batch size。只有 train.txt 和 val.txt、没有独立 test.txt 时显式加 `--test-split none`，输出只包含验证指标，不能称为测试结果。支持自定义 `--train-split`、`--val-split`、`--test-split`。

## 4. 批量训练

```bash
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --epochs 30 --batch-size 32
```

默认顺序跑 `experiments.json` 的 10 组：Unet、UnetPlusPlus、MAnet、FPN、PSPNet、DeepLabV3Plus（ResNet34）；Unet、DeepLabV3Plus（MobileNetV2）；Segformer（MiT-B0、MiT-B2）。所有骨干（包括 Segformer 的 MiT）默认随机初始化，无预训练下载。SMP 的 Unet 是使用所选骨干的变体，并非原始从零训练 U-Net。

```bash
# 三次随机种子重复实验
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --seeds 42 43 44
# 中断后恢复；保持原来的训练参数和 epochs 不变
python batch_train.py --data-root /root/autodl-tmp/data/VOC2012 --mask-encoding voc --legacy-output --resume
```

每个实验启动独立 Python 进程，失败后记录退出码并继续下一组。看 `outputs/<实验>/console.log`；结果随每次完成写入 `outputs/results.csv`。重新运行已有实验时必须使用 `--resume` 或新输出目录。可编辑 experiments.json 精简模型或设置每组的 `lr`；不应声称一种学习率对所有模型最优。

## 5. 输出与评估

每组保存 `config.json`、`history.csv`、`last.pth`、`best.pth`、`metrics.json`。最佳权重仅按验证集两类 mIoU 选择，最后评估独立测试集。预测阈值固定 0.5，不使用测试集调阈值。

评估使用共享的 bimseg-segmentation-v2 协议，详见文末“统一指标与结果格式”。二分类输出概率按 0.5 阈值转为 1 前景、0 背景；忽略像素不参与评价。边界距离计算会增加验证耗时。


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

默认训练参数：batch size=32、epochs=30（用户指定）。批量训练未传入参数时，每组也使用此默认值。测速 batch size 仍为 1。实际显存占用因模型而异，OOM 时可通过 --batch-size 降低训练批量。

## 本地预训练权重（AutoDL 离线训练）

10 个实验共用 4 个 ImageNet 骨干权重：resnet34（6组）、mobilenet_v2（2组）、mit_b0（1组）、mit_b2（1组）。
下载的是骨干权重，分割解码器随机初始化；训练时全部参数参与优化。

在网络正常的电脑下载本仓库 download_pretrained.py，运行：
```bash
python download_pretrained.py --output pretrained
```
下载脚本仅依赖 Python 标准库，不要求电脑安装 PyTorch 或有 GPU。来源使用 SMP 0.5.0 官方指定的 Hugging Face 仓库及固定 revision。
manifest.json 保存来源、文件大小和 SHA256；完整成功的文件重跑时校验并跳过，未完成的文件重新下载。

把整个 pretrained 文件夹上传到 /root/autodl-tmp/benchmark/pretrained，然后在 AutoDL 执行：
```bash
cd /root/autodl-tmp/benchmark
python batch_train.py --data-root /root/autodl-tmp/benchmark/data/VOCdevkit-4 --weights local --pretrained-dir /root/autodl-tmp/benchmark/pretrained --epochs 30 --batch-size 32 --output-root outputs/imagenet_local
```
没有 test.txt 时追加 --test-split none。local 模式创建模型时明确禁用自动下载，使用 safetensors 加载并严格检查 encoder 参数。缺文件或参数不匹配直接报错，不回退联网。
预训练实验使用新输出目录；不要用 --resume 接续原先从零训练的实验。恢复本地预训练实验时加 --resume，直接恢复实验 checkpoint，不需要重新读取预训练文件。

## 输出目录分类管理

新版批量训练默认采用 outputs/<数据集>/<实验名称>/<模型_seed>/，results.csv 位于该实验目录。
--dataset-name 指定数据集名称，缺省使用 data-root 的文件夹名；--run-name 指定实验名称，缺省自动生成预训练方式、epoch、batch size及时间组成的名称。调学习率等参数时使用不同实验名称，实际完整参数保存在各模型 config.json 中。

```bash
python batch_train.py --data-root /root/autodl-tmp/benchmark/data/VOCdevkit-4 --weights local --pretrained-dir pretrained --epochs 30 --batch-size 32 --dataset-name VegAnn --run-name local_ep30_bs32_lr1e-4
```

恢复新版实验：重复相同参数和 dataset-name/run-name，再加 --resume。恢复时不允许省略 run-name，防止误创建新实验。
旧版已保存的平铺目录保持原位置，恢复时加 --legacy-output --resume，并沿用旧 output-root。
不要在训练过程中移动输出文件夹：旧 checkpoint 内记录模型输出路径，移动还可能影响正在写入的日志。新版仅改变后续实验的组织方式。



## 统一指标与结果格式（bimseg-segmentation-v2）

三个仓库使用内容完全一致的 `segmentation_metrics.py`。区域指标先累计整个评估集的混淆矩阵，行=真实类别、列=预测类别；不对逐图 IoU/Dice 求平均。

| 标准字段 | 计算口径 |
|---|---|
| `miou` / `macc` | 全部类别（包含背景）的 IoU / Recall 宏平均；所有任务按验证集 mIoU 选择最佳权重 |
| `foreground_miou` | 仅前景各类别 IoU 的宏平均；二分类即前景 IoU |
| `union_foreground_iou` | 合并所有非背景类别后的 IoU，不能反映前景内部错分 |
| `dice` / `precision` / `recall` | 全数据集统计，前景类别宏平均 |
| `dice_all` | 包含背景的 Dice 宏平均，作为补充，不能与 `dice` 混用 |
| `pixacc` / `fwiou` | 总体像素准确率 / 按真实像素频率加权的 IoU |
| `specificity` | 前景并集的特异度，即背景 Recall |
| `boundary_f1` / `hd95` / `assd` | 逐图逐前景类别计算，先在图内对有效前景类别平均，再对有效图片平均；另存每类结果 |

区域指标分母为零时记录 JSON `null` / CSV 空单元格，宏平均排除该项。边界类别在预测和真实标签中都为空时不参与平均；仅一方为空时 BF1=0、HD95/ASSD=图像对角线。整幅图没有任何有效前景类别对时不参与边界均值；整个数据集都没有时为 null。Boundary F1 默认容差 2 像素，HD95/ASSD 单位为评估尺寸下的像素；其他分数为 0–1。忽略像素不参与统计；将其屏蔽可能在忽略区边缘产生轮廓，比较时必须使用相同有效像素和分辨率。

每次最终评估统一写出：
- `metrics.json`：完整结果、每类指标、混淆矩阵和协议版本。
- `results.csv`：固定字段，每行对应 val 或 test。区域 `precision` 与测速 `precision_mode` 分开，避免覆盖。
- `class_metrics.csv`：每个数据划分、每个类别一行，含 IoU、Dice、Precision、Recall、像素数及边界结果；背景边界列留空。
- `confusion_matrix.csv`：长表 `split,true_class,predicted_class,pixels`。
- 开启复杂度测试时另有 `benchmark.json`：完整网络参数量、GFLOPs、FP32 单图前向延迟/FPS等。

批量仓库的实验根目录 `results.csv` 也使用相同列，每个模型/种子/划分一行，以 `name`、`seed`、`split` 区分，不再使用 `test_` / `val_` 前缀的宽表。模型子目录同样保存上述四个文件。缺失或失败数据留空，失败行记录非零 `status`。

旧 JSON 中的兼容字段暂时保留，标准 CSV 只使用上表字段。`plant-seg-lab` 的 `Foreground IoU` 现在对应 `foreground_miou`，原合并值改称 `Union Foreground IoU`；`nclassbenchmark` 的旧 `mdice` 仍是含背景均值，论文推荐取标准 `dice`（等于 `mdice_fg`）。各仓库的损失函数、模型结构、数据增强和预训练设置不因这次指标统一而改变。

旧权重可以重新评价，无需重新训练；旧 CSV 不会自动变为新口径。使用对应原始训练参数在两个 benchmark 仓库追加 `--eval-only`，或在 plant-seg-lab 使用 `summarize.py` / `evaluate_transfer.py` 重新生成结果。不要混合旧边界指标和 v2 指标。统计代码的验证命令：`python -m unittest discover -s tests -p 'test_metric_*.py' -v`（仅需 NumPy/SciPy，无需 GPU）。
