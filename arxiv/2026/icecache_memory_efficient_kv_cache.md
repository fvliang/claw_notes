# IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs

**arXiv**: 2604.10539
**链接**: https://arxiv.org/abs/2604.10539
**作者**: Yuzhen Mao et al.
**会议**: arXiv 2026
**主题**: llm_serving / KV Cache Optimization
**项目页面**: https://yuzhenmao.github.io/IceCache/

## 摘要 (Abstract)

Key-Value (KV) cache plays a crucial role in accelerating inference in large language models (LLMs) by storing intermediate attention states and avoiding redundant computation during autoregressive generation. However, its memory footprint scales linearly with sequence length, often leading to severe memory bottlenecks on resource-constrained hardware. Prior work has explored offloading KV cache to the CPU while retaining only a subset on the GPU, but these approaches often rely on imprecise token selection and suffer performance degradation in long-generation tasks such as chain-of-thought reasoning. In this paper, we propose a novel KV cache management strategy, IceCache, which integrates semantic token clustering with PagedAttention. By organizing semantically related tokens into contiguous memory regions managed by a hierarchical, dynamically updatable data structure, our method enables more efficient token selection and better utilization of memory bandwidth during CPU-GPU transfers. Experimental results on LongBench show that, with a 256-token budget, IceCache maintains 99% of the original accuracy achieved by the full KV cache model. Moreover, compared to other offloading-based methods, IceCache attains competitive or even superior latency and accuracy while using only 25% of the KV cache token budget, demonstrating its effectiveness in long-sequence scenarios.

## 摘要 (中文)

KV cache 在加速 LLM 推理中起关键作用，通过存储中间注意力状态避免自回归生成中的冗余计算。然而其内存占用与序列长度线性增长，在资源受限硬件上常导致严重的内存瓶颈。之前的工作探索了将 KV cache 卸载到 CPU 同时仅在 GPU 上保留子集，但这些方法通常依赖不精确的 token 选择，在长生成任务（如 chain-of-thought 推理）中导致性能下降。本文提出 IceCache，一种新的 KV cache 管理策略，集成了语义 token 聚类与 PagedAttention。通过将语义相关的 token 组织到由分层、动态可更新数据结构管理的连续内存区域中，该方法实现了更高效的 token 选择和更好的 CPU-GPU 传输内存带宽利用。在 LongBench 上的实验表明，仅用 256-token 预算，IceCache 保持了完整 KV cache 模型 99% 的原始精度。与其它卸载方法相比，仅用 25% 的 KV cache token 预算即可获得相当甚至更优的延迟和精度。

## 关键贡献

1. 语义 token 聻类 + PagedAttention 的 KV cache 管理策略
2. 分层、动态可更新的数据结构用于内存区域管理
3. 256-token 预预算下保持 99% 精度，25% 预算下性能优越