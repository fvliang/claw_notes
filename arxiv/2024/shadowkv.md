# ShadowKV: KV Cache in Shadows for Long-Context Inference

## 论文信息
- **作者**: Various
- **会议**: arXiv 2024
- **arXiv**: https://arxiv.org/abs/2410.21485
- **日期**: 2024.10

## 摘要 (Abstract)
ShadowKV introduces a novel approach for high-throughput long-context LLM inference. Key innovations:

1. **Shadow cache**: Secondary cache layer for efficiency
2. **Hierarchical caching**: Multi-level cache hierarchy
3. **Selective computation**: Only compute when necessary
4. **3-4x speedup**: Significant improvements for long contexts

## 摘要中文
ShadowKV为高吞吐量长上下文LLM推理引入了新方法。关键创新：

1. **阴影缓存**: 用于效率的二级缓存层
2. **分层缓存**: 多级缓存层次结构
3. **选择性计算**: 仅在必要时计算
4. **3-4倍加速**: 长上下文显著改进

## 引言 (Introduction)
Long-context inference faces unique challenges:
- KV cache becomes prohibitively large
- Memory bandwidth is bottleneck
- Computation waste on redundant tokens

ShadowKV addresses this with:
- **Shadow mechanism**: Lightweight secondary cache tracks what can be reused
- **Efficient recomputation**: On-demand KV regeneration when needed
- **Smart caching policy**: Balances memory vs compute tradeoff

## 总结
ShadowKV provides an elegant solution for practical long-context LLM serving.