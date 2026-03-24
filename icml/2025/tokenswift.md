# TokenSwift: Ultra Long Sequence Generation

## 论文信息
- **作者**: bigai-nlco
- **会议**: ICML 2025
- **GitHub**: https://github.com/bigai-nlco/TokenSwift
- **日期**: 2025

## 摘要 (Abstract)
TokenSwift focuses on lossless acceleration of ultra long sequence generation. The main contributions include:

1. **Hierarchical speculation**: Multi-level draft generation pipeline
2. **Long context optimization**: Specifically designed for 10K+ token sequences
3. **Memory efficiency**: Optimized KVCache management for long contexts
4. **2-3x speedup**: Maintains output quality while significantly accelerating

## 摘要中文
TokenSwift专注于超长序列生成的无损加速。主要贡献包括：

1. **分层投机**: 多级draft生成管道
2. **长上下文优化**: 专为10K+ token序列设计
3. **内存效率**: 针对长上下文的优化KVCache管理
4. **2-3倍加速**: 在显著加速的同时保持输出质量

## 引言 (Introduction)
Ultra long context generation (10K+ tokens) presents unique challenges:
- O(n²) attention complexity becomes prohibitive
- KVCache memory requirements explode
- Autoregressive decoding becomes the bottleneck

TokenSwift addresses these with:
- **Streaming speculation**: Draft tokens generated incrementally
- **Selective attention**: Only relevant context attended to
- **Efficient cache eviction**: Smart management of long context cache

## GitHub 介绍
Official implementation of TokenSwift: Lossless Acceleration of Ultra Long Sequence Generation (ICML 2025). Provides efficient inference for extremely long context scenarios.