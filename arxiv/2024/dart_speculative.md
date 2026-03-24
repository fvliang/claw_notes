# DART: Diffusion-Inspired Speculative Decoding

## 论文信息
- **作者**: Various
- **会议**: arXiv 2024
- **GitHub**: https://github.com/fvliang/DART
- **日期**: 2024

## 摘要 (Abstract)
DART (Diffusion-inspired Speculative Decoding) brings ideas from diffusion models to speculative decoding for LLM inference. The approach:

1. **Multiple draft candidates**: Generate several possible token sequences
2. **Tree-based verification**: Efficiently verify multiple candidates
3. **Quality-aware selection**: Choose best among multiple drafts
4. **Improved acceptance rates**: Better overall efficiency

## 摘要中文
DART（扩散启发的投机解码）将扩散模型的思想引入LLM推理的投机解码。该方法：

1. **多个draft候选**: 生成多个可能的token序列
2. **基于树的验证**: 高效验证多个候选
3. **质量感知选择**: 在多个draft中选择最佳
4. **改进的接受率**: 更好的整体效率

## 引言 (Introduction)
Standard speculative decoding has limitations:
- Linear draft sequence
- Limited exploration of token space
- Single acceptance path

DART improves by:
- **Branching drafts**: Explore multiple token possibilities
- **Parallel verification**: Verify multiple paths efficiently
- **Best-path selection**: Choose optimal sequence

## GitHub 介绍
Official Implementation of DART (DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference). Provides efficient implementation of diffusion-inspired speculative decoding approach.