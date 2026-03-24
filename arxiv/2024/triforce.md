# TriForce: Hierarchical Speculative Decoding

## 论文信息
- **作者**: Infini-AI-Lab
- **会议**: COLM 2024
- **GitHub**: https://github.com/Infini-AI-Lab/TriForce
- **日期**: 2024

## 摘要 (Abstract)
TriForce introduces a hierarchical speculative decoding approach for lossless acceleration of long sequence generation. The key innovation is a multi-level draft-verification hierarchy:

1. **Coarse-level draft**: Fast, smaller model generates draft
2. **Medium-level verification**: Intermediate model verifies
3. **Fine-level verification**: Full model confirms final tokens

This hierarchical approach achieves near-linear speedups while maintaining identical output to standard decoding.

## 摘要中文
TriForce为长序列生成引入了分层投机解码方法，实现无损加速。关键创新是多级draft-验证层次结构：

1. **粗粒度draft**: 快速的小模型生成draft
2. **中等粒度验证**: 中间模型验证
3. **细粒度验证**: 完整模型确认最终token

这种方法实现接近线性的加速，同时保持与标准解码相同的输出。

## 引言 (Introduction)
Long-context LLM inference is particularly challenging due to:
1. **Quadratic attention complexity**: O(n²) scaling with sequence length
2. **Memory pressure**: Large KVCache requirements
3. **Latency issues**: Slow autoregressive generation

TriForce addresses these by:
- **Hierarchical speculation**: Multiple levels reduce verification cost
- **Lossless acceleration**: No quality degradation
- **Scalable architecture**: Works with increasing sequence lengths

## GitHub 介绍
Official implementation of TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding (COLM 2024). The system provides efficient long-context inference with significant speedups.