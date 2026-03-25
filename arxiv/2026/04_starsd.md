# StarSD: One-for-Many Speculative Decoding

## 论文信息

- **原文链接**: https://arxiv.org/abs/2601.07710
- **作者**: Junhao He, Feiran You, Hongyang Du
- **年份**: 2026
- **来源**: arXiv

## 摘要 (Abstract)

Speculative decoding has shown great promise in accelerating LLM inference by using a small draft model to propose tokens and a large target model to verify them. However, existing methods typically require a dedicated draft model for each target model, resulting in high memory overhead and training costs. We present StarSD, a one-for-many speculative decoding framework that leverages a single universal draft model to serve multiple target models. StarSD employs a hierarchical verification mechanism that first uses lightweight adapters to quickly filter unlikely tokens, then applies the target model for final verification. Additionally, we propose a knowledge distillation technique that transfers the capabilities of multiple target models into the universal draft model. Our experiments show that StarSD can achieve comparable speedup to dedicated draft models while reducing memory overhead by 70%.

## 摘要 (中文)

投机解码在使用小型draft模型提出token、大型目标模型验证方面显示出加速LLM推理的巨大潜力。然而，现有方法通常需要为每个目标模型配备专用的draft模型，导致高内存开销和训练成本。我们提出了StarSD，一个一对多投机解码框架，它利用单一的通用draft模型来服务多个目标模型。StarSD采用分层验证机制，首先使用轻量级适配器快速过滤不太可能的token，然后使用目标模型进行最终验证。此外，我们提出了一种知识蒸馏技术，将多个目标模型的能力转移到通用draft模型中。我们的实验表明，StarSD可以实现与专用draft模型相当的加速效果，同时减少70%的内存开销。

## 引言 (Introduction)

核心挑战：
1. 每个目标模型需要专用draft模型
2. 内存开销大
3. 训练成本高

StarSD的创新：
- 单一通用draft模型服务多个目标模型
- 分层验证机制
- 知识蒸馏技术
- 减少70%内存开销

## GitHub/项目

（待补充）