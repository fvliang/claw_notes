# Star-Attention: Efficient LLM Inference over Long Sequences

## 论文信息
- **作者**: NVIDIA
- **会议**: arXiv 2024
- **arXiv**: https://arxiv.org/abs/2411.17116
- **GitHub**: https://github.com/NVIDIA/Star-Attention
- **日期**: 2024.11

## 摘要 (Abstract)
Star-Attention introduces a novel sparse attention pattern for efficient long-context LLM inference. The key innovation is using a star-shaped attention pattern that:

1. **Reduces attention complexity** from O(n²) to O(n)
2. **Maintains model quality** with minimal accuracy loss
3. **Achieves 11x speedup** on long sequence benchmarks

This is achieved by dividing the context into blocks with block-wise attention and using a small set of anchor tokens to maintain global coherence.

## 摘要中文
Star-Attention为高效的长上下文LLM推理引入了一种新的稀疏注意力模式。关键创新是使用星形注意力模式：

1. **将注意力复杂度**从O(n²)降低到O(n)
2. **保持模型质量**，精度损失最小
3. **在长序列基准测试上实现11倍加速**

这是通过将上下文分成块并使用小块注意力和少量锚点token来保持全局一致性实现的。

## 引言 (Introduction)
Long context inference is critical for many applications but faces challenges:

1. **Computational complexity**: Full attention scales quadratically
2. **Memory pressure**: KVCache grows with sequence length
3. **Latency**: Autoregressive generation becomes slow

Star-Attention solves this with:
- **Block-wise attention**: Local blocks attend within themselves
- **Anchor tokens**: Small set of tokens maintain global context
- **Approximate attention**: Trade-off between quality and speed

## GitHub 介绍
NVIDIA's implementation of Star-Attention for efficient long-sequence LLM inference. The repository provides optimized CUDA kernels and integration with popular LLM serving frameworks.