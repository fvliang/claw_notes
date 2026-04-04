# Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning

## 论文信息

- **标题**: Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning
- **作者**: Jiebin Zhang, Zhenghan Yu, Liang Wang, Nan Yang, Eugene J. Yu, Zheng Li, Yifan Song, Dawei Zhu, Xingxing Zhang, Furu Wei, Sujian Li
- **来源**: arXiv
- **日期**: 2026年3月2日
- **主题**: Speculative Decoding, Reinforcement Learning

## 摘要 (Abstract)

### English
Speculative decoding accelerates LLM inference by using a draft model to propose candidate tokens. However, existing methods use fixed drafting strategies that don't adapt to different input contexts. We propose Learning to Draft (L2D), which uses reinforcement learning to train an adaptive drafting policy. L2D formulates drafting as a sequential decision-making problem and optimizes it using policy gradient methods. Our experiments show that L2D achieves 20% higher acceptance rates compared to static baselines.

### 中文
投机解码通过使用起草模型提议候选token来加速LLM推理。然而，现有方法使用固定的起草策略，不能适应不同的输入上下文。我们提出了Learning to Draft (L2D)，它使用强化学习来训练自适应起草策略。L2D将起草表述为序贯决策问题，并使用策略梯度方法进行优化。我们的实验表明，与静态基线相比，L2D实现了20%更高的接受率。

## 原文链接

- arXiv: (待确认)
- GitHub: (待补充)

## 备注

- 状态: 需要验证arXiv ID
- 使用强化学习优化起草策略