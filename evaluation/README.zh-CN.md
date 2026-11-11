# 表 2、表 3 评测复现

本目录评测 Qwen2.5-VL-7B-Instruct 与作者发布的 TreeVGR-7B，覆盖 TreeBench、V*、HR-Bench-4K 和 HR-Bench-8K。不进行训练。目标是核对论文总分、分类分数及相对基座的提升。

## 采用的实现

固定 VLMEvalKit 提交 c247c4d61d92be5643802aa9be5f2bc34d913d2a。模型注册配置、提示词选择、图像处理、生成参数、答案处理和数据集评分均调用该版本的公开实现；不使用根目录 inference_treebench.py 的简易示例。

- 基座注册名 Qwen2.5-VL-7B-Instruct，use_custom_prompt=False。
- TreeVGR 注册名 TreeVGR-7B，保留模型自己的定位推理提示和输出处理。
- 两个注册配置的 min_pixels=1280×28×28，max_pixels=16384×28×28；不另外覆盖。
- TreeBench 沿用官方标签/选项抽取及召回式 mIoU，基座 mIoU 不纳入论文对照。
- V* 和 HR-Bench 明确采用公开的 exact_matching 评分路径，不调用付费评判 API。它包含官方选项匹配规则，并非简单比较整段回答字符串。
- HR-Bench 读取官方按 cycle 分组后 Average 的结果，不自行用逐题平均替换。
- 固定种子 0（每题使用 seed+行号）是本项目为可重复运行设置的默认值，不是论文报告的种子。
- 当前固定版本作为公开实现依据，不声称它就是作者生成论文结果时的提交。表 3 的 V* 基座参考为 74.3，不使用表 8 的 71.2。

## 服务器准备

以下命令均从仓库根目录执行。使用独立 Python 环境；GPU、CUDA 与 FlashAttention 按服务器实际配置安装。准备脚本只获取评测工具，不下载权重。

~~~text
python evaluation/prepare_toolkit.py
python -m pip install -e evaluation/vendor/VLMEvalKit
python -m pip install -r evaluation/requirements.txt
python -m pip install flash-attn --no-build-isolation
~~~

先安装评测工具，再安装本项目依赖以应用作者的 torch、transformers 等限制。若服务器报告依赖冲突，应先解决环境兼容性；本地 CPU 检查不代表这一 GPU 依赖组合已经实测。每次运行会记录实际依赖版本。

模型默认从公开 Hugging Face 仓库下载，记录 snapshot revision 后加载本地快照。可用 --model-path 指定预先下载的目录，必须同时选定 --model base 或 treevgr。本地模型文件会计算 SHA256，首次检查可能较慢。使用 HF_HOME 可指定 Hugging Face 缓存；LMUData 或 --data-root 指定测试数据目录。数据由官方 dataset adapter 获取。

## 按测试集运行

四个入口默认都依次运行两个模型，每个模型独立进程：

~~~text
python evaluation/eval_treebench.py
python evaluation/eval_vstar.py
python evaluation/eval_hrbench_4k.py
python evaluation/eval_hrbench_8k.py
~~~

也可以统一运行八个组合：

~~~text
python evaluation/run.py --dataset all --model all
~~~

先检查计划（不加载依赖、数据或模型），然后校验数据，再跑两条样本：

~~~text
python evaluation/run.py --dry-run
python evaluation/run.py --mode prepare --data-root /data/LMUData
python evaluation/eval_treebench.py --model treevgr --limit 2 --data-root /data/LMUData
~~~

prepare 会取得各数据集、检查必要字段和索引，并逐张检查图片文件头。limit 输出放在 smoke 目录，永远不进入正式论文对照；HR-Bench 或 V* 的极小子集可能缺失分类，导致汇总校验失败。对这两个数据集的小样本测试可使用 --mode infer，只验证推理。

正式运行示例：

~~~text
python evaluation/eval_treebench.py --model base --model-path /models/Qwen2.5-VL-7B-Instruct --data-root /data/LMUData --output /data/treevgr-results
python evaluation/eval_treebench.py --model treevgr --model-path /models/TreeVGR-7B --data-root /data/LMUData --output /data/treevgr-results
~~~

脚本顺序执行，每个任务独立进程。建议通过 CUDA_VISIBLE_DEVICES 为每个进程只暴露一张 GPU；基座适配器使用 device_map=auto，TreeVGR 使用 cuda。多 GPU 服务器可在不同终端指定不同设备，运行互不重叠的模型/数据集组合。不要同时写入同一个 output、数据集、模型目录。

## 恢复与重新评分

每完成一题立即原子保存 records/编号.json。再次执行同一命令会复用已完成题目。失败立即退出并记录 last_failure.json，不将失败伪装成正确完成。模型、数据、代码或关键设置发生变化时，拒绝混用旧记录，请改用新 --output。

只推理：

~~~text
python evaluation/eval_vstar.py --mode infer
~~~

用已保存预测重新评分，不加载模型权重：

~~~text
python evaluation/eval_vstar.py --mode eval
~~~

eval 必须使用原先的 output、data-root、limit 和 seed。每次评分写入新的 scores 子目录，不复用旧评分缓存。

## 结果文件

每个 full/测试集/base 或 treevgr 目录包含：

- manifest.json：数据行数、类别分布、数据源 SHA256、权重版本、注册参数、工具提交、依赖、种子和脚本哈希。
- effective_model.json：模型实际生成参数、系统提示、像素限制和自定义提示开关。
- records/*.json 和 predictions.jsonl：索引、问题、真值、类别、适配器返回答案、后处理前的 decoded 文本、模板和提示消息。模板记录是在 apply_chat_template 返回时截取的，适配器后续追加的前缀以固定适配器实现为准。
- scores/运行编号/：传给官方评分器的预测 XLSX、官方逐题得分/缓存和原始汇总。
- summary.json：转换到 0–100 尺度的总体及分类分数。
- output/logs/：推理与评分的标准输出和错误日志。

TreeVGR 的非 TreeBench 适配器会返回抽取后的答案；脚本通过只读包装 tokenizer.batch_decode 另行保存后处理前文本，不改变生成结果或官方评分输入。

## 对照论文

~~~text
python evaluation/summarize.py --output /data/treevgr-results
~~~

输出 paper_comparison.csv 与 paper_comparison.md，包含论文分数、实测分数、偏差和两个模型之间的提升。默认要求八个组合全部完成；只查看阶段性结果可加 --allow-partial。未运行的结果显示为空，不填 0，不自动判定“复现成功”。

| 测试集 | 基座总分 | TreeVGR 总分 | 额外指标 |
| --- | ---: | ---: | --- |
| TreeBench | 37.0 | 50.4 | 10 类准确率，TreeVGR mIoU 44.0 |
| VStarBench | 74.3 | 91.1 | Attr、Spatial |
| HRBench4K | 72.1 | 77.1 | Single、Cross |
| HRBench8K | 68.8 | 73.1 | Single、Cross |

完整参考数值见 protocol.json。原始资料：[论文](https://arxiv.org/abs/2507.07999)、[作者仓库](https://github.com/Haochen-Wang409/TreeVGR)、[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)。

## 本地检查

~~~text
python -m unittest discover -s evaluation/tests -v
python evaluation/run.py --dry-run
~~~

测试覆盖提示分支、评分尺度、HR 分组平均、断点配置隔离、原始输出保存及模拟推理→评分→恢复流程。模拟测试不验证真实模型分数，不代替服务器小样本和完整 GPU 评测。
