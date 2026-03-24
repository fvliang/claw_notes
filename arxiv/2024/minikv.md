# MiniKV: Layer-Discriminative KV Cache

## 论文信息
- **作者**: Various (Microsoft?)
- **会议**: arXiv 2024
- **arXiv**: https://arxiv.org/abs/2411.19092
- **日期**: 2024.11

## 摘要 (Abstract)
MiniKV introduces a 2-bit layer-discriminative KV cache optimization for efficient LLM inference. Key innovations:

1. **Layer-aware quantization**: Different layers get different precision
2. **2-bit compression**: Aggressive KV cache compression
3. **Minimal accuracy loss**: Maintains model quality through careful design
4. **Significant memory savings**: Up to 4x KV cache reduction

## 摘要中文
MiniKV为高效LLM推理引入了2比特层区分性KV缓存优化。关键创新：

1. **层感知量化**: 不同层获得不同精度
2. **2比特压缩**: 激进的KV缓存压缩
3. **最小精度损失**: 通过精心设计保持模型质量
4. **显著的内存节省**: KV缓存减少高达4倍

## 引言 (Introduction)
KV cache is a major memory bottleneck in LLM serving:
- Grows linearly with sequence length
- Becomes prohibitive for long contexts
- Limits batch size and concurrency

MiniKV addresses this with:
- **Layer discrimination**: Not all layers need same precision
- **2-bit quantization**: Aggressive but careful compression
- **Minimal quality impact**: Careful calibration maintains quality

## 总结
MiniKV provides a valuable technique for memory-constrained scenarios while maintaining acceptable model quality.