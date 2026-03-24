# EAGLE: Early Exit and Speculative Decoding

## 论文信息
- **作者**: Various (SafeAILab)
- **会议**: ICML 2024, EMNLP 2024, NeurIPS 2025
- **GitHub**: https://github.com/SafeAILab/EAGLE
- **日期**: 2024-2025

## 摘要 (Abstract)
EAGLE (Early EXIT and spEculative decoding) is a family of speculative decoding methods that achieve significant speedups for LLM inference. The key innovations include:

1. **EAGLE-1 (ICML'24)**: Uses early-exit techniques to speed up draft token generation
2. **EAGLE-2 (EMNLP'24)**: Improved speculative decoding with better acceptance rates
3. **EAGLE-3 (NeurIPS'25)**: Further optimizations for better performance

EAGLE achieves 2-3x speedup over standard autoregressive decoding while maintaining output quality.

## 摘要中文
EAGLE（早期退出和投机解码）是一系列投机解码方法，可为LLM推理带来显著的加速。关键创新包括：

1. **EAGLE-1 (ICML'24)**: 使用早期退出技术加速draft token生成
2. **EAGLE-2 (EMNLP'24)**: 改进的投机解码，具有更好的接受率
3. **EAGLE-3 (NeurIPS'25)**: 进一步优化以获得更好的性能

EAGLE在保持输出质量的同时，实现了比标准自回归解码快2-3倍的速度。

## 引言 (Introduction)
Speculative decoding uses a smaller "draft" model to predict multiple tokens ahead, which are then verified by the larger "target" model. EAGLE improves upon this by:

1. **Early exit in draft model**: Draft model exits early for confident predictions
2. **Tree-structured verification**: More efficient batch verification of multiple drafts
3. **Improved acceptance rate**: Better selection of draft tokens to verify
4. **Training-free adaptation**: Can be applied to existing models without fine-tuning

## GitHub 介绍
Official implementation of EAGLE-1 (ICML'24), EAGLE-2 (EMNLP'24), and EAGLE-3 (NeurIPS'25) speculative decoding methods. The repository provides efficient implementations that can be integrated with various LLM backends to achieve significant inference speedups.