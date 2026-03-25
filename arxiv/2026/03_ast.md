# When RL Meets Adaptive Speculative Training: A Unified Training-Serving System

## 论文信息

- **原文链接**: https://arxiv.org/abs/2602.02604
- **作者**: Junxiong Wang, Fengxiang Bie, Jisen Li, et al.
- **年份**: 2026
- **来源**: arXiv

## 摘要 (Abstract)

Speculative decoding has emerged as a prominent technique for accelerating large language model (LLM) inference. However, existing works primarily focus on optimizing inference efficiency, overlooking the potential of integrating training and serving. Moreover, the training of speculative decoding models often requires substantial computational resources and time. In this paper, we present an Adaptive Speculative Training (AST) framework that integrates training and serving by dynamically adjusting the training objectives based on inference performance. Specifically, AST utilizes an online feedback mechanism to continuously monitor the inference quality and acceptance rate, and adaptively modifies the training objective to enhance the draft model's capability. We also propose an efficient model update strategy that minimizes the performance degradation during training-serving co-optimization. Experimental results demonstrate that AST achieves significant improvements in both training efficiency and inference throughput.

## 摘要 (中文)

投机解码已成为加速大型语言模型（LLM）推理的突出技术。然而，现有工作主要关注优化推理效率，忽略了训练和服务整合的潜力。此外，投机解码模型的训练通常需要大量计算资源和时间。在本文中，我们提出了自适应投机训练（AST）框架，该框架通过基于推理性能动态调整训练目标来整合训练和服务。具体来说，AST利用在线反馈机制持续监控推理质量和接受率，并自适应地修改训练目标以增强draft模型的能力。我们还提出了一种高效的模型更新策略，最大程度地减少训练-服务协同优化过程中的性能退化。实验结果表明，AST在训练效率和推理吞吐量方面都实现了显著提升。

## 引言 (Introduction)

现有方法的局限性：
1. 训练与服务分离，忽视协同优化潜力
2. 训练需要大量计算资源
3. 无法动态适应推理性能

AST的创新点：
- 训练与服务统一框架
- 基于推理性能的动态训练目标调整
- 在线反馈机制持续优化
- 高效模型更新策略

## GitHub/项目

（待补充）