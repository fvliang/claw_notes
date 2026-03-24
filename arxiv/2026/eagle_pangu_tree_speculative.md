# EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs

## 论文信息

- **作者**: Chang Han, Yijie Hu, Jingling Liu
- **提交日期**: 2026年3月9日
- **arXiv链接**: https://arxiv.org/abs/xxx (待补充)
- **关键词**: Tree Speculative Decoding, NPU Accelerator, Hardware Acceleration

## 摘要 (Abstract)

Autoregressive decoding remains a primary bottleneck in large language models (LLMs). Tree speculative decoding has emerged as a promising technique to accelerate inference by exploring multiple candidate token sequences in parallel. However, existing approaches are primarily designed for GPUs and fail to fully leverage neural processing units (NPUs).

自回归解码仍然是LLM的主要瓶颈。树形投机解码作为一种有前景的技术出现，通过并行探索多个候选token序列来加速推理。然而，现有方法主要针对GPU设计，未能充分利用NPU。

## 引言 (Introduction)

### 挑战
- GPU上的树形解码实现不完全适合NPU架构
- 需要针对NPU特性的优化

### 解决方案
本文提出EAGLE-Pangu，专为Ascend NPU优化的树形投机解码框架。

## 技术特点

1. NPU友好的树结构管理
2. 硬件感知的调度策略
3. 内存访问模式优化

---

*更新时间: 2026-03-24*