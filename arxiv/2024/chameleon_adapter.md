# Chameleon: Adaptive Caching for Multi-Adapter LLM Inference

## 论文信息
- **作者**: Various
- **会议**: arXiv 2024
- **arXiv**: https://arxiv.org/abs/2411.18550
- **日期**: 2024.11

## 摘要 (Abstract)
Chameleon addresses the challenge of efficient LLM inference in multi-adapter environments. Key contributions:

1. **Adapter-aware caching**: Different adapters have different cache needs
2. **Adaptive scheduling**: Dynamic adjustment based on workload
3. **Memory optimization**: Efficient sharing of cache across adapters
4. **Significant improvements**: Better throughput and latency

## 摘要中文
Chameleon解决了多适配器环境中高效LLM推理的挑战。主要贡献：

1. **适配器感知缓存**: 不同适配器有不同的缓存需求
2. **自适应调度**: 根据工作负载动态调整
3. **内存优化**: 跨适配器高效共享缓存
4. **显著改进**: 更好的吞吐量和延迟

## 引言 (Introduction)
Multi-adapter LLM deployment is increasingly common:
- Multiple task-specific adapters on one base model
- Each adapter has different cache requirements
- Efficient resource sharing is challenging

Chameleon solves this with:
- **Intelligent cache partitioning**: Share where possible, isolate where needed
- **Dynamic weight adjustment**: Adapt cache allocation based on demand
- **Co-scheduling**: Coordinate across adapters for best overall performance

## 总结
Chameleon provides an important solution for practical multi-tenant LLM serving scenarios.