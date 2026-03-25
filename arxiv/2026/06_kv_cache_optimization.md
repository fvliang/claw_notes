# KV Cache Optimization Strategies for Scalable and Efficient LLM Inference

## 论文信息

- **原文链接**: https://arxiv.org/abs/2603.10536
- **作者**: Yichun Xu, Navjot K. Khaira, Tejinder Singh
- **年份**: 2026
- **来源**: arXiv

## 摘要 (Abstract)

The key-value (KV) cache is a foundational optimization in Transformer-based large language models (LLMs), eliminating redundant recomputation of past token representations during autoregressive generation. However, its memory footprint scales linearly with context length, imposing critical bottlenecks on GPU memory capacity, memory bandwidth, and inference throughput. In this survey, we systematically categorize and analyze KV cache optimization strategies across three primary dimensions: (1) cache compression techniques that reduce memory footprint, (2) cache eviction policies that intelligently manage cache capacity, and (3) cache-aware serving systems that optimize end-to-end inference performance. We examine the trade-offs between memory savings, computational overhead, and output quality for each category of methods. Furthermore, we discuss open challenges and promising research directions for future KV cache optimization.

## 摘要 (中文)

键值（KV）缓存是Transformer基大型语言模型（LLM）的基础优化，它消除了自回归生成过程中过去token表示的冗余重新计算。然而，其内存占用随上下文长度线性增长，对GPU内存容量、内存带宽和推理吞吐量造成关键瓶颈。在本综述中，我们系统地将KV缓存优化策略分为三个主要维度进行分类和分析：（1）减少内存占用的缓存压缩技术，（2）智能管理缓存容量的缓存驱逐策略，以及（3）优化端到端推理性能的缓存感知服务系统。我们研究了每类方法在内存节省、计算开销和输出质量之间的权衡。此外，我们还讨论了未来KV缓存优化的开放挑战和有前途的研究方向。

## 引言 (Introduction)

KV缓存的核心挑战：
1. 内存占用随上下文长度线性增长
2. GPU内存容量限制
3. 内存带宽瓶颈
4. 推理吞吐量受限

三大优化维度：
1. 缓存压缩技术
2. 缓存驱逐策略
3. 缓存感知服务系统

## GitHub/项目

（待补充）