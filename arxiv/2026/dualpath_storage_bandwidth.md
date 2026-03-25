# DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference

## 论文信息
- **标题**: DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
- **作者**: Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang
- **arXiv**: [2602.21548](https://arxiv.org/abs/2602.21548)
- **提交时间**: 2026年2月25日 (v1), 2026年2月26日 (v2)
- **领域**: Distributed, Parallel, and Cluster Computing (cs.DC)

## 摘要 (Abstract)
The performance of multi-turn, agentic LLM inference is increasingly dominated by KV-Cache storage I/O rather than computation. In prevalent disaggregated architectures, loading the massive KV-Cache from external storage creates a fundamental imbalance: storage NICs on prefill engines become bandwidth-saturated, while those on decoding engines remain idle. This asymmetry severely constrains overall system throughput. We present DualPath, an inference system that breaks this bottleneck by introducing dual-path KV-Cache loading. Beyond the traditional storage-to-prefill path, DualPath enables a novel storage-to-decode path, in which the KV-Cache is loaded into decoding engines and then efficiently transferred to prefill engines via RDMA over the compute network. DualPath combines this optimized data path -- which inherently avoids network congestion and avoids interference with latency-critical model execution communications -- with a global scheduler that dynamically balances load across prefill and decode engines. Our evaluation on three models with production agentic workloads demonstrates that DualPath improves offline inference throughput by up to 1.87x on our in-house inference system. It can also improve online serving throughput by an average factor of 1.96x without violating SLO.

## 摘要 (中文)
多轮、agentic LLM推理的性能越来越受KV-Cache存储I/O而非计算支配。在流行的解聚架构中，从外部存储加载大量KV-Cache造成了一个根本的不平衡：预填充引擎上的存储NIC变得带宽饱和，而解码引擎上的存储NIC却保持空闲。这种不对称性严重限制了整体系统吞吐量。我们提出了DualPath，这是一种通过引入双路径KV-Cache加载来打破这一瓶颈的推理系统。除了传统的存储到预填充路径，DualPath还启用了一种新颖的存储到解码路径，在该路径中，KV-Cache被加载到解码引擎，然后通过计算网络上的RDMA高效传输到预填充引擎。DualPath将这种优化的数据路径（本质上避免网络拥塞并避免与延迟关键模型执行通信的干扰）与全局调度器相结合，全局调度器动态平衡预填充和解码引擎之间的负载。我们在对三个具有生产agentic工作负载的模型进行评估表明，DualPath在内部推理系统上将离线推理吞吐量提高了1.87倍。它还可以在不违反SLO的情况下将在线服务吞吐量平均提高1.96倍。

## 核心贡献
1. **双路径KV-Cache加载**: 首次引入存储到解码路径，缓解预填充引擎的带宽瓶颈
2. **RDMA传输**: 通过RDMA高效传输KV-Cache
3. **全局调度器**: 动态平衡预填充和解码引擎之间的负载

## 技术细节
- **性能**: 离线推理吞吐量提升1.87倍，在线服务吞吐量平均提升1.96倍
- **架构**: 解聚式架构中的双路径设计

---

*更新时间: 2026-03-25*