# DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72

## 论文信息

- **标题**: DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72
- **作者**: Wanqian Li, Jintao Peng, Zongfei Jing, Tianyu Zhang, Ze Long, Xianjie Qiao, Xiaoming Chen, Dongxu Yang, Kefeng Duan, June Yang
- **来源**: arXiv
- **arXiv ID**: 2604.00925 (待确认)
- **日期**: 2026年4月2日
- **主题**: LLM Inference, Distributed Systems

## 摘要 (Abstract)

### English
Large language model (LLM) inference increasingly depends on multi-GPU execution, yet existing inference parallelization strategies require layer-wise inter-rank synchronization, making end-to-end performance sensitive to workload imbalance. We present DWDP (Distributed Weight Data Parallelism), an inference parallelization strategy that preserves data-parallelism across the entire LLM and eliminates inter-layer synchronization. DWDP distributes model weights across GPU memory hierarchy and performs weight updates asynchronously using high-bandwidth NVLink interconnects. Our evaluation on a 72-GPU NVL72 cluster demonstrates that DWDP achieves up to 2.1x speedup over state-of-the-art tensor parallelism while maintaining identical output quality.

### 中文
大型语言模型推理越来越依赖于多GPU执行，但现有的推理并行化策略需要层间跨GPU同步，使得端到端性能对工作负载不平衡非常敏感。我们提出了DWDP（分布式权重数据并行），这是一种保持整个LLM数据并行性的推理并行化策略，并消除了层间同步。DWDP将模型权重分布在GPU内存层次结构中，并使用高带宽NVLink互连异步执行权重更新。我们在72-GPU NVL72集群上的评估表明，DWDP相比最新的张量并行化获得了高达2.1倍的加速，同时保持相同的输出质量。

## 引言 (Introduction)

### English
The rapid growth in size and capability of large language models (LLMs) has led to increasing demand for efficient inference systems. LLM inference consists of two phases: prefill (processing input tokens) and decode (generating output tokens). While prefill is compute-bound, decode is memory-bandwidth-bound due to autoregressive token generation.

Current parallelization strategies for LLM inference include:
- Tensor parallelism: Splits model layers across GPUs
- Pipeline parallelism: Stages different layers on different GPUs
- Data parallelism: Replicates model on multiple GPUs

However, these approaches suffer from synchronization overhead and load imbalance, especially in heterogeneous GPU clusters.

### 中文
大型语言模型（LLM）的规模和能力快速增长，对高效推理系统的需求也在增加。LLM推理包括两个阶段：预填充（处理输入token）和解码（生成输出token）。预填充是计算密集型的，而解码由于自回归token生成是内存带宽密集型的。

目前LLM推理的并行化策略包括：
- 张量并行：将模型层分割到多个GPU
- 流水线并行：在不同GPU上放置不同层
- 数据并行：在多个GPU上复制模型

然而，这些方法存在同步开销和负载不平衡的问题，特别是在异构GPU集群中。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 需要补充完整的GitHub链接和博客内容