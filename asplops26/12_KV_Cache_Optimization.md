# KV Cache Optimization Strategies for Scalable and Efficient LLM Inference

**论文链接**: [arXiv:2603.20397](https://arxiv.org/abs/2603.20397)

**作者**: Yichun Xu, Navjot K. Khaira, Tejinder Singh

**提交日期**: 2026年3月20日

---

## Abstract (摘要)

The key-value (KV) cache is a foundational optimization in Transformer-based large language models (LLMs), eliminating redundant recomputation of past token representations during autoregressive generation. However, its memory footprint scales linearly with context length, imposing critical bottlenecks on GPU memory capacity, memory bandwidth, and inference throughput as production LLMs push context windows from thousands to millions of tokens. Efficient KV cache management has thus become a first-order challenge for scalable LLM deployment. This paper provides a systematic review of recent KV cache optimization techniques, organizing them into five principal directions: cache eviction, cache compression, hybrid memory solutions, novel attention mechanisms, and combination strategies. For each category we analyze the underlying mechanisms, deployment trade-offs, and empirical performance across memory reduction, throughput, and model accuracy metrics. We further map techniques to seven practical deployment scenarios, including long-context single requests, high-throughput datacenter serving, edge devices, multi-turn conversations, and accuracy-critical reasoning, providing actionable guidance for practitioners selecting among competing approaches. Our analysis reveals that no single technique dominates across all settings; instead, the optimal strategy depends on context length, hardware constraints, and workload characteristics, pointing toward adaptive, multi-stage optimization pipelines as a promising direction for future research.

---

键值（KV）缓存是Transformer基础的大型语言模型（LLM）中的基础优化，消除了自回归生成过程中过去token表示的冗余重新计算。然而，其内存占用随上下文长度线性增长，随着生产LLM将上下文窗口从数千token扩展到数百万token，在GPU内存容量、内存带宽和推理吞吐量方面造成关键瓶颈。因此，高效的KV缓存管理已成为可扩展LLM部署的首要挑战。本文对最近的KV缓存优化技术进行了系统综述，将其组织为五个主要方向：缓存淘汰、缓存压缩、混合内存解决方案、新颖的注意力机制和组合策略。对于每个类别，我们分析了底层机制、部署权衡以及内存减少、吞吐量和模型精度指标方面的实证性能。我们进一步将技术映射到七个实际部署场景，包括长上下文单请求、高吞吐量数据中心服务、边缘设备、多轮对话和精度关键推理，为从业者在竞争方法中选择提供可操作指导。我们的分析表明，没有单一技术在所有设置中都占主导地位；相反，最佳策略取决于上下文长度、硬件约束和工作负载特征，指向自适应多阶段优化管道作为未来研究的有前途方向。

---

## 主要内容

本文对KV缓存优化技术进行了系统综述，分为五大类：

1. **Cache Eviction（缓存淘汰）**: 选择性移除不重要的KV缓存
2. **Cache Compression（缓存压缩）**: 压缩KV缓存以减少内存占用
3. **Hybrid Memory Solutions（混合内存解决方案）**: 结合CPU/GPU内存
4. **Novel Attention Mechanisms（新型注意力机制）**: 改进的注意力实现
5. **Combination Strategies（组合策略）**: 结合多种优化技术

---

## 部署场景映射

本文还分析了7个实际部署场景：
- 长上下文单请求
- 高吞吐量数据中心服务
- 边缘设备
- 多轮对话
- 精度关键推理

---

## 关键发现

- 没有单一技术在所有场景中都最优
- 最佳策略取决于：上下文长度、硬件约束、工作负载特征
- 自适应多阶段优化管道是未来研究方向

---

## 关键词

KV Cache, LLM Inference, Memory Optimization, Transformer, System Optimization