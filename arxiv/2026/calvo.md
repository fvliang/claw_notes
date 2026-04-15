# CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands

**Source:** arxiv | **Category:** LLM Serving | **Date:** 2026-03-22
**ArXiv ID:** 2603.21257
**Authors:** Weiye Wang, Chen Chen, Junxue Zhang, Zhusheng Wang, Hui Yuan, Zixuan Guan, Xiaolong Zheng, Qizhen Weng, Yin Chen, Minyi Guo
**Tags:** kv-cache-loading, network-intensive, distributed-prefix-caching, scheduling, calvo

## Links

- 📄 [Paper (PDF)](https://arxiv.org/pdf/2603.21257)
- 🌐 [ArXiv Page](https://arxiv.org/abs/2603.21257)

## Abstract (English)

Distributed prefix caching has become a core technique for efficient LLM serving. However, for long-context requests with high cache hit ratios, retrieving reusable KVCache blocks from remote servers has emerged as a new performance bottleneck. Such network-intensive LLM inference is expected to become increasingly common as agentic AI workloads grow. Existing LLM inference engines remain compute-centric: they treat KVCache loading as a subordinate phase to GPU execution and fail to account for its delay explicitly during scheduling. CALVO is an LLM serving engine that treats KVCache loading as a first-class concern. It decouples KVCache loading and GPU computation into independently managed, asynchronously progressing stages, enabling better utilization of network, PCIe, and computation resources. CALVO incorporates KVCache loading delay as an explicit component of per-request service cost, achieving up to 61.67% higher SLO attainment than the baseline.

## Abstract (Chinese)

分布式前缀缓存已成为高效LLM服务的核心技术。但对于长上下文请求的高缓存命中比，从远程服务器检索可重用KVCache块已成为新的性能瓶颈。随着智能体AI工作负载增长，这种网络密集型LLM推理将越来越常见。现有LLM推理引擎仍以计算为中心：将KVCache加载作为GPU执行的附属阶段，未在调度中显式考虑其延迟。CALVO是将KVCache加载作为首要关注的LLM服务引擎。将KVCache加载和GPU计算解耦为独立管理的异步推进阶段，更好利用网络、PCIe和计算资源。将KVCache加载延迟作为每请求服务成本的显式组成部分，SLO达成率比基线高61.67%。

## Key Contributions

1. **CALVO** — Distributed prefix caching has become a core technique for efficient LLM serving. However, for long-...
2. Addresses core challenges in LLM Serving systems
3. Demonstrates significant improvements over existing baselines

## Notes

- Added on 2026-04-16
- Paper published on 2026-03-22
