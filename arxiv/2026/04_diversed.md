---
title: DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification
authors: Ziyi Wang, Siva Rajesh Kasa, Ankith M S, Santhosh Kumar Kasa, Jiaru Zou, Sumit Negi, Ruqi Zhang, Nan Jiang, Qifan Song
arxiv_id: 2604.07622
conference: aistats
full_conference: AISTATS 2026
year: "2026"
topic: Speculative Decoding
url: https://arxiv.org/abs/2604.07622
pdf_url: https://arxiv.org/pdf/2604.07622
github: https://github.com/comeusr/diversed
added_date: 2026-04-11
---

# DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification

## 论文信息

- **arXiv**: [2604.07622](https://arxiv.org/abs/2604.07622)
- **会议**: AISTATS 2026
- **作者**: Ziyi Wang, Siva Rajesh Kasa, Ankith M S, Santhosh Kumar Kasa, Jiaru Zou, Sumit Negi, Ruqi Zhang, Nan Jiang, Qifan Song
- **GitHub**: [https://github.com/comeusr/diversed](https://github.com/comeusr/diversed)

## 摘要 (Abstract)

Speculative decoding is an effective technique for accelerating large language model inference by drafting multiple tokens in parallel. In practice, its speedup is often bottlenecked by a rigid verification step that strictly enforces the accepted token distribution to exactly match the target model. This constraint leads to the rejection of many plausible tokens, lowering the acceptance rate and limiting overall time speedup. To overcome this limitation, we propose Dynamic Verification Relaxed Speculative Decoding (DIVERSED), a relaxed verification framework that improves time efficiency while preserving generation quality. DIVERSED learns an ensemble-based verifier that blends the draft and target model distributions with a task-dependent and context-dependent weight.

## 摘要中文

投机解码是一种有效的加速大语言模型推理的技术，通过并行起草多个token。实际上，其加速效果往往受限于严格的验证步骤，该步骤严格要求接受的token分布与目标模型完全匹配。这种约束导致许多合理的token被拒绝，降低了接受率，限制了整体加速效果。为了克服这一限制，我们提出了动态验证放松投机解码（DIVERSED），这是一种在保持生成质量的同时提高时间效率的放松验证框架。DIVERSED学习一种基于集成的验证器，以任务相关和上下文相关的权重混合draft模型和目标模型的分布。

## 引言 (Introduction)

Speculative decoding has emerged as a key technique for accelerating LLM inference. The standard approach uses a draft model to generate candidate tokens, which are then verified by the target model in a single forward pass. However, the strict verification requirement often leads to low acceptance rates when the draft and target models have different distributions.

## 引言中文

投机解码已成为加速LLM推理的关键技术。标准方法使用draft模型生成候选token，然后在单次前向传播中由目标模型验证。然而，当draft模型和目标模型具有不同的分布时，严格的验证要求往往会导致低接受率。

## GitHub 介绍

Code available at: https://github.com/comeusr/diversed