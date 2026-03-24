# SWIFT: On-the-Fly Self-Speculative Decoding

## 论文信息
- **作者**: Microsoft Research
- **会议**: ICLR 2025
- **GitHub**: https://github.com/hemingkx/SWIFT
- **日期**: 2025

## 摘要 (Abstract)
SWIFT (Speculative Inference With Fast Token-generation) is an on-the-fly self-speculative decoding method that enables efficient LLM inference without pre-trained draft models. Key features:

1. **Self-speculation**: Uses the same model for draft and verify
2. **On-the-fly generation**: No need for separate draft model training
3. **Layer-wise early exit**: Different layers for different confidence levels
4. **Significant speedups**: 1.5-2x speedup with minimal overhead

## 摘要中文
SWIFT（Speculative Inference With Fast Token-generation）是一种即时自投机解码方法，无需预训练的draft模型即可实现高效的LLM推理。关键特性：

1. **自投机**: 使用相同模型进行draft和验证
2. **即时生成**: 无需单独的draft模型训练
3. **分层早期退出**: 不同置信度使用不同层
4. **显著加速**: 1.5-2倍加速，开销极小

## 引言 (Introduction)
SWIFT addresses limitations of prior speculative decoding methods:
- No need for two-model setups (draft + target)
- No training required - works with existing models
- Dynamic layer selection based on token confidence
- Maintains output quality while accelerating

The approach is particularly valuable for:
- Deployment scenarios with limited resources
- Cases where draft model training is impractical
- Applications requiring flexible deployment

## GitHub 介绍
Official implementation of SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration (ICLR 2025). Provides easy-to-integrate code for accelerating LLM inference.