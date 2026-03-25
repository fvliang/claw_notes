# TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference

## 论文信息

- **原文链接**: https://arxiv.org/abs/2502.02787
- **作者**: Jiyoung Park, Hankyu Jang, Changseok Song, Wookeun Jung
- **年份**: 2026
- **来源**: arXiv

## 摘要 (Abstract)

Speculative decoding has emerged as a promising solution to accelerate large language model inference by leveraging a small draft model to propose candidate tokens in parallel and a large target model to verify them. However, existing speculative decoding approaches rely on static draft models that are not optimized for the specific inference workload, leading to suboptimal speedup. We present TIDE, a temporal incremental draft engine that enables continuous self-improvement of the draft model throughout the inference process. TIDE extracts high-quality generation patterns from the target model's outputs and incrementally updates the draft model, allowing it to adapt to the evolving distribution of inference requests. Our experiments show that TIDE achieves up to 2.1x speedup over state-of-the-art speculative decoding methods.

## 摘要 (中文)

投机解码已成为加速大型语言模型推理的有前途的解决方案，它利用小型draft模型并行提出候选token，大型目标模型验证它们。然而，现有的投机解码方法依赖于静态的draft模型，这些模型没有针对特定的推理工作负载进行优化，导致加速效果不理想。我们提出了TIDE，一个时间增量draft引擎，它使draft模型能够在整个推理过程中持续自我改进。TIDE从目标模型的输出中提取高质量的生成模式，并增量更新draft模型，使其能够适应推理请求不断变化的分布。我们的实验表明，TIDE相比最先进的投机解码方法实现了高达2.1倍的加速。

## 引言 (Introduction)

投机解码的关键问题：
1. 静态draft模型无法适应特定推理工作负载
2. Draft模型质量决定整体加速效果
3. 需要持续优化draft模型

TIDE的核心贡献：
- 从目标模型输出中提取高质量生成模式
- 增量更新draft模型
- 适应推理请求的分布变化

## GitHub/项目

（待补充）