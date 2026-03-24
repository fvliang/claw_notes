# Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving

## 论文信息
- **作者**: Moonshot AI
- **会议**: arXiv 2024
- **PDF**: https://github.com/kvcache-ai/Mooncake/blob/main/Mooncake-v1.pdf
- **GitHub**: https://github.com/kvcache-ai/Mooncake
- **日期**: 2024.06

## 摘要 (Abstract)
Mooncake is a KVCache-centric disaggregated architecture for LLM serving developed by Moonshot AI. The core innovation is treating KVCache as a first-class citizen and designing a comprehensive system that optimizes the entire inference pipeline around efficient KVCache management.

The system implements a novel architecture that separates the prefill and decode stages while focusing on maximizing KVCache reuse across requests. Mooncake achieves significantly higher throughput compared to existing systems by leveraging:
- Efficient KVCache transfer between disaggregated stages
- Smart cache management and eviction policies
- Memory-centric scheduling

## 摘要中文
Mooncake是由Moonshot AI开发的以KVCache为中心的解耦架构的LLM服务系统。其核心创新是将KVCache作为一等公民，并设计了一个围绕高效KVCache管理优化整个推理流程的综合系统。

该系统实现了创新的架构，将预填充和解码阶段分离，同时专注于最大化跨请求的KVCache复用。Mooncake通过利用以下方面实现了比现有系统显著更高的吞吐量：
- 解耦阶段之间的高效KVCache传输
- 智能缓存管理和驱逐策略
- 以内存为中心的调度

## 引言 (Introduction)
Traditional LLM serving systems treat KVCache as a byproduct of inference. Mooncake redesigns the entire serving stack with KVCache at the center:

1. **KVCache-centric architecture**: All components optimized for KVCache efficiency
2. **Disaggregated prefill/decode**: Different compute resources for each phase
3. **Cross-request KVCache sharing**: Significant memory savings
4. **Transfer-efficient design**: Minimize KVCache movement overhead

This architecture is particularly effective for scenarios with high cache hit rates and long context lengths.

## GitHub 介绍
Mooncake is a KVCache-centric inference serving system for large language models. It implements a disaggregated architecture with a focus on maximizing KVCache reuse and efficient memory management. The project provides high-throughput LLM serving with significant improvements over existing systems like vLLM.