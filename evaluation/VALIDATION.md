# 本地验证记录

日期：2026-11-11。

- python -m unittest discover -s evaluation/tests -v：8 项通过。包括模拟推理、原始文本保存、断点恢复、离线评分、提示优先级和论文结果汇总边界。
- python -m compileall -q evaluation：退出码 0。
- 四个 eval_*.py 的 --dry-run：均生成正确的两个模型任务，共八个组合。
- 从固定 VLMEvalKit 源码直接提取 evaluate_box_iou，验证完全重合返回 1、无预测框返回 0。
- 直接加载固定源码的 report_acc_hrbench，在两个 cycle 的合成样本上验证官方 Average 为 0.75。

上述检查没有加载真实权重、运行 GPU 推理或产生论文复现分数。模拟测试中出现的 100 分仅为测试夹具结果。服务器上仍需安装依赖、运行数据检查与小样本推理，再开展完整评测。
