# TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving

## 论文信息

- **标题**: TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregated LLM Serving
- **作者**: Feng Ren, Ruoyu Qin, Teng Ma, Shangming Cai, Zheng Liu, Chao Lei, Dejiang Zhu, Ke Yang, Zheming Li, Jialei Cui, Weixiao Huang, Yikai Zhao, Yineng Zhang, Hao Wu, Xiang Gao, Yuhao Fu, Jinlei Jiang, Yongwei Wu, Mingxing Zhang
- **来源**: arXiv
- **日期**: 2026年3月31日
- **主题**: LLM Serving, Distributed Systems, Data Movement

## 摘要 (Abstract)

### English
Modern LLM serving systems increasingly adopt disaggregated architectures that separate prefill and decode stages onto different GPU clusters. However, orchestrating diverse interconnects (from multi-rail RDMA to proprietary fabrics such as Multi-Node NVLink and Ascend UB) for efficient data movement across these stages remains a critical challenge. We present TENT, a declarative slice spraying engine that provides high-level abstractions for describing data movement patterns in disaggregated LLM serving. TENT enables automatic optimization of data placement and routing based on workload characteristics and network topology. Our experiments on production clusters show that TENT achieves 40% higher throughput and 3x better tail latency compared to existing approaches.

### 中文
现代LLM服务系统越来越多地采用解聚架构，将预填充和解码阶段分离到不同的GPU集群。然而，有效协调多样化的互联（从多轨RDMA到专有互连如多节点NVLink和Ascend UB）以实现跨阶段的高效数据移动仍然是一个关键挑战。我们提出了TENT，一个声明式切片喷射引擎，为解聚LLM服务中的数据移动模式提供高级抽象。TENT能够根据工作负载特性和网络拓扑自动优化数据放置和路由。我们在生产集群上的实验表明，与现有方法相比，TENT实现了40%的更高吞吐量和3倍的更好尾延迟。

## 引言 (Introduction)

### English
Disaggregated LLM serving has emerged as a promising approach to handle the diverse compute and memory requirements of prefill and decode phases. While this architecture offers flexibility, it introduces new challenges in data movement:

1. **Network heterogeneity**: Different GPU clusters may use different interconnect technologies
2. **Load imbalance**: Prefill and decode workloads have different computational characteristics
3. **Resilience**: Failures in one cluster should not affect the other

TENT addresses these challenges by providing a declarative interface that allows system designers to specify data movement requirements at a high level, while the engine automatically handles optimization.

### 中文
解聚LLM服务已经成为一种有前景的方法来处理预填充和解码阶段的不同计算和内存需求。虽然这种架构提供了灵活性，但它引入了数据移动的新挑战：

1. **网络异构性**：不同的GPU集群可能使用不同的互连技术
2. **负载不平衡**：预填充和解码工作负载具有不同的计算特性
3. **弹性**：一个集群的故障不应影响另一个集群

TENT通过提供声明式接口来解决这些挑战，允许系统设计者在高级别指定数据移动需求，而引擎自动处理优化。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 需要补充完整的GitHub链接和博客内容