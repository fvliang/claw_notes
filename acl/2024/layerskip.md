# LayerSkip: Early Exit and Self-Speculative Decoding

## 论文信息
- **作者**: Facebook Research
- **会议**: ACL 2024
- **GitHub**: https://github.com/facebookresearch/LayerSkip
- **日期**: 2024

## 摘要 (Abstract)
LayerSkip presents a novel approach to LLM inference that combines early exit with self-speculative decoding. The key insight is to use the same model for both draft generation and verification, enabling:

1. **Early exit at variable layers**: Different tokens exit at different layers based on confidence
2. **Self-speculative decoding**: Model drafts tokens and verifies them in a unified framework
3. **Significant speedups**: Up to 2x speedup on various benchmarks

## 摘要中文
LayerSkip提出了一种新的LLM推理方法，将早期退出与自投机解码相结合。关键在于使用相同的模型进行draft生成和验证，实现：

1. **可变层的早期退出**: 不同token根据置信度在不同层退出
2. **自投机解码**: 模型在统一框架中draft tokens并验证它们
3. **显著加速**: 在各种基准测试上实现高达2倍的加速

## 引言 (Introduction)
Existing speculative decoding approaches typically use two separate models (draft and target). LayerSkip innovates by:

1. **Self-speculation**: A single model handles both draft and verify
2. **Layer-wise early exit**: Confident tokens skip remaining layers
3. **Training for early exit**: Special training recipe to enable early exit capability
4. **End-to-end optimization**: Whole system optimized together

This approach reduces model overhead and improves cache efficiency.

## GitHub 介绍
Official implementation of LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding (ACL 2024). The code includes training scripts for enabling early exit capabilities and inference scripts for self-speculative decoding.