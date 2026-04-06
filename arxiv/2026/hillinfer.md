# HillInfer: Efficient Long-Context LLM Inference on the Edge with Hierarchical KV Eviction using SmartSSD

**作者**: He Sun, Shinan Liu, Li Li, Mingjun Xiao

**arXiv**: 2602.18750

**年份**: 2026

**会议**: arXiv

**主题**: LLM Serving

## 摘要 (Abstract)

Deploying Large Language Models (LLMs) on memory-constrained AI Personal Computers (AIPCs) enables low-latency, privacy-preserving AI applications. However, the massive memory footprint of long-context LLMs makes it challenging to run them on edge devices with limited GPU memory. Existing approaches either rely on expensive high-bandwidth memory (HBM) or suffer from significant performance degradation when offloading to slower storage. In this paper, we propose HillInfer, a hierarchical KV cache eviction system that leverages SmartSSD -- a novel storage device that integrates an FPGA accelerator directly beside the NAND flash dies. HillInfer performs hierarchical KV cache management: hot tokens are kept in GPU memory, warm tokens are offloaded to SmartSSD's DRAM, and cold tokens are evicted to SmartSSD's NAND flash. Our key insight is that SmartSSD's integrated FPGA can efficiently process KV cache directly in storage, filtering and retrieving only relevant tokens without数据传输瓶颈. We design a hierarchical KV eviction policy that considers token importance, recency, and access patterns to maximize cache hit rates at each level. Experimental results show that HillInfer achieves 2.1× speedup over GPU-only baselines while reducing memory consumption by 85%.

## 摘要中文

在内存受限的AI个人电脑（AIPC）上部署大型语言模型（LLM）能够实现低延迟、隐私保护的AI应用。然而，长上下文LLM的巨大内存占用使得在GPU内存有限的边缘设备上运行它们具有挑战性。现有方法要么依赖昂贵的高带宽内存（HBM），要么在卸载到较慢的存储时遭受显著的性能下降。在本文中，我们提出了HillInfer，一种分层KV缓存驱逐系统，利用SmartSSD——一种在NAND闪存芯片旁边集成FPGA加速器的新型存储设备。HillInfer执行分层KV缓存管理：热令牌保留在GPU内存中，温暖令牌卸载到SmartSSD的DRAM，冷令牌驱逐到SmartSSD的NAND闪存。我们的关键洞察是SmartSSD的集成FPGA可以直接在存储中处理KV缓存，过滤和检索相关令牌而无需数据传输瓶颈。我们设计了一种分层KV驱逐策略，考虑令牌重要性、访问频率和访问模式，以最大化每级的缓存命中率。实验结果表明，HillInfer在降低85%内存消耗的同时，实现了比纯GPU基线2.1倍的加速。

