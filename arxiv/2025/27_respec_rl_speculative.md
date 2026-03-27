# ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems

## 论文信息

- **标题**: ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems
- **作者**: Qiaoling Chen, Zijun Liu, Peng Sun, Shenggui Li, Guoteng Wang, Ziming Liu, Yonggang Wen, Siyuan Feng, Tianwei Zhang
- **arXiv**: [arXiv:2510.26475](https://arxiv.org/abs/2510.26475)
- **提交日期**: 2025年10月30日
- **领域**: Machine Learning (cs.LG), Distributed, Parallel, and Cluster Computing (cs.DC)
- **会议**: arXiv预印本

## 摘要 (Abstract)

Adapting large language models (LLMs) via reinforcement learning (RL) is often bottlenecked by the generation stage, which can consume over 75% of the training time. Speculative decoding (SD) accelerates autoregressive generation in serving systems, but its behavior under RL training remains largely unexplored. We identify three critical gaps that hinder the naive integration of SD into RL systems: diminishing speedups at large batch sizes, drafter staleness under continual actor updates, and drafter-induced policy degradation.

To address these gaps, we present ReSpec, a system that adapts SD to RL through three complementary mechanisms: dynamically tuning SD configurations, evolving the drafter via knowledge distillation, and weighting updates by rollout rewards. On Qwen models (3B--14B), ReSpec achieves up to 4.5x speedup while preserving reward convergence and training stability, providing a practical solution for efficient RL-based LLM adaptation.

## 摘要 (中文)

通过强化学习(RL)适配大型语言模型(LLM)通常受到生成阶段的瓶颈限制，生成阶段可能消耗超过75%的训练时间。投机解码(SD)加速了服务系统中的自回归生成，但其在RL训练下的行为仍然大部分未被探索。我们确定了三个阻碍SD简单集成到RL系统的关键差距：大批量下加速效果减弱、持续actor更新下的草稿模型过时、以及草稿模型导致的策略退化。

为了解决这些差距，我们提出了ReSpec，这是一个通过三种互补机制将SD适配到RL的系统：动态调整SD配置、通过知识蒸馏演进草稿模型、以及根据rollout奖励加权更新。在Qwen模型(3B-14B)上，ReSpec实现了高达4.5倍的加速，同时保持奖励收敛和训练稳定性，为高效的基于RL的LLM适配提供了实用的解决方案。

## 引言 (Introduction)

将LLM通过强化学习适配新任务时面临以下挑战：

1. **生成阶段瓶颈**：RL训练中生成阶段消耗超过75%的训练时间
2. **大批量下加速减弱**：传统SD在大批量推理时加速效果下降
3. **草稿模型过时**：RL中actor持续更新，草稿模型很快过时
4. **策略退化**：草稿模型可能引入偏差，影响策略质量

ReSpec的解决方案：
- **动态SD配置调整**：根据训练阶段调整投机参数
- **知识蒸馏演进草稿**：使草稿模型跟上actor的更新
- **奖励加权更新**：根据rollout质量调整更新

## 实验结果

- 在Qwen模型(3B-14B)上实现最高4.5倍加速
- 保持奖励收敛和训练稳定性

## GitHub

暂无公开GitHub仓库