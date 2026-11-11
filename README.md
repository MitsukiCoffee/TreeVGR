# TreeVGR 评测复现

本仓库基于 [作者源码](https://github.com/Haochen-Wang409/TreeVGR)，复现论文表 2、表 3 中 Qwen2.5-VL-7B-Instruct 与 TreeVGR-7B 的结果。

## 原作者示例与论文的差异

作者已在原 README 中说明：inference_treebench.py 的结果与论文略有差异，论文主要使用 VLMEvalKit 评测。原示例的 TreeBench 准确率为 49.38%、mIoU 为 43.3；论文表 2 为 50.4%、44.0。

本仓库新增的 evaluation 脚本使用 VLMEvalKit 的模型配置、提示词、图像处理和评分实现，对照论文表 2、表 3。原示例保留在根目录。

## 复现范围

模型权重：Qwen/Qwen2.5-VL-7B-Instruct、HaochenWang/TreeVGR-7B。

| 测试集 | 基座总分 | TreeVGR 总分 | 同时统计 |
| --- | ---: | ---: | --- |
| TreeBench | 37.0 | 50.4 | 10 类准确率，TreeVGR mIoU 44.0 |
| V* Bench | 74.3 | 91.1 | 属性、空间关系 |
| HR-Bench-4K | 72.1 | 77.1 | Single、Cross |
| HR-Bench-8K | 68.8 | 73.1 | Single、Cross |

完整参考分数位于 evaluation/protocol.json。V* 和 HR-Bench 使用公开的 exact_matching 评分路径。

## 服务器快速运行

在已配置 CUDA 的 Python 环境中，从仓库根目录执行：

~~~bash
python evaluation/prepare_toolkit.py
pip install -e evaluation/vendor/VLMEvalKit
pip install -r evaluation/requirements.txt
pip install flash-attn --no-build-isolation

CUDA_VISIBLE_DEVICES=0 python evaluation/run.py
python evaluation/summarize.py
~~~

默认依次运行两个模型在四个测试集上的评测；权重和数据自动下载，结果保存到 evaluation/outputs。准备脚本首次获取指定版本 VLMEvalKit，已有目录直接复用。

按测试集运行：

~~~bash
python evaluation/eval_treebench.py
python evaluation/eval_vstar.py
python evaluation/eval_hrbench_4k.py
python evaluation/eval_hrbench_8k.py
~~~

常用参数：

| 参数 | 用途 |
| --- | --- |
| --model base / treevgr | 只评测一个模型，默认两个都跑 |
| --model-path /models/TreeVGR-7B | 指定本地权重，配合 --model 使用 |
| --data-root /data/LMUData | 指定数据缓存目录 |
| --output /data/results | 指定结果目录 |
| --mode infer / eval | 只推理 / 用已有预测重新评分，默认两步连续执行 |
| --limit 10 | 只跑前 10 条，结果另存到 smoke 目录；小样本推理可搭配 --mode infer |

中断后重跑同一命令会继续已有记录。更换权重或评测设置时，使用新的 --output 目录。

## 新增脚本

| 文件 | 用途 |
| --- | --- |
| evaluation/prepare_toolkit.py | 获取 VLMEvalKit |
| evaluation/run.py | 通用入口，调度模型与测试集，保存预测并调用官方评分 |
| evaluation/eval_treebench.py | TreeBench 入口 |
| evaluation/eval_vstar.py | V* Bench 入口 |
| evaluation/eval_hrbench_4k.py | HR-Bench-4K 入口 |
| evaluation/eval_hrbench_8k.py | HR-Bench-8K 入口 |
| evaluation/summarize.py | 汇总论文分数、实测分数、偏差和模型提升幅度 |
| evaluation/common.py | 提示调用、原始输出保存和指标整理 |
| evaluation/protocol.json | 模型名、测试集名、默认设置和论文参考值 |

每题保存原始生成文本、提取答案和提示信息，支持断点续跑。scores 目录保存官方评分结果；汇总生成 paper_comparison.csv 和 paper_comparison.md，未完成的项目留空。

论文：[Traceable Evidence Enhanced Visual Grounded Reasoning](https://arxiv.org/abs/2507.07999)。
