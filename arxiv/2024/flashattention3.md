# FlashAttention-3: Fast and Accurate Attention

## 论文信息
- **作者**: TriDao et al.
- **会议**: arXiv 2024
- **PDF**: https://tridao.me/publications/flash3/flash3.pdf
- **GitHub**: https://github.com/Dao-AILab/flash-attention
- **日期**: 2024.07

## 摘要 (Abstract)
FlashAttention-3 brings significant improvements to the foundational attention mechanism for LLM inference and training. Key innovations:

1. **Asynchrony**: Overlaps computation and memory operations
2. **Low-precision**: FP8 support with minimal accuracy loss
3. **Hardware optimization**: Better utilization of modern GPU tensor cores
4. **2-3x speedup** over FlashAttention-2

FlashAttention series has become the de-facto standard for efficient attention computation in modern LLM systems.

## 摘要中文
FlashAttention-3为LLM推理和训练的基础注意力机制带来了显著改进。关键创新：

1. **异步性**: 重叠计算和内存操作
2. **低精度**: FP8支持，精度损失最小
3. **硬件优化**: 更好地利用现代GPU张量核心
4. **比FlashAttention-2快2-3倍**

FlashAttention系列已成为现代LLM系统中高效注意力计算的事实标准。

## 引言 (Introduction)
Attention computation is the core bottleneck in Transformer models:

1. **Memory-bound operation**: Standard attention requires O(N²) memory
2. **IO overhead**: Slow HBM access dominates runtime
3. **Limited parallelism**: Difficult to fully utilize GPU resources

FlashAttention-3 addresses these with:
- **Tile-based computation**: Process in smaller blocks
- **Online softmax**: Reduce memory while maintaining accuracy
- **Asynchronous execution**: Hide memory latency
- **FP8 optimization**: Leverage tensor cores more efficiently

## GitHub 介绍
The official implementation of FlashAttention, now at version 3. Provides highly optimized CUDA kernels for attention computation, widely used in LLM training and inference systems including vLLM, Megatron, and many others.