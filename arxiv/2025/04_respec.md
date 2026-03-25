# ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems

## 论文信息

- **原文链接**: https://arxiv.org/abs/2510.21156
- **作者**: Qiaoling Chen, Zijun Liu, Peng Sun, et al.
- **年份**: 2025
- **来源**: arXiv

## 摘要 (Abstract)

Adapting large language models (LLMs) via reinforcement learning (RL) is often bottlenecked by the generation stage, which can consume over 75% of the training time. Speculative decoding offers a promising solution by using a draft model to predict tokens in parallel. However, applying speculative decoding to RL training poses unique challenges: the RL training process has dynamic acceptance rates due to policy changes, and the overhead of maintaining multiple models can be substantial. We present ReSpec, a system that optimizes speculative decoding for RL training of LLMs. ReSpec introduces an adaptive draft selection mechanism that dynamically adjusts the draft model based on the RL policy's evolution. Additionally, we propose a memory-efficient caching strategy that reduces the overhead of managing multiple draft models. Our experiments show that ReSpec reduces the RL training time by up to 40% compared to standard speculative decoding.

## 摘要 (中文)

通过强化学习（RL）适应大型语言模型（LLM）通常在生成阶段遇到瓶颈，这可能占用超过75%的训练时间。投机解码提供了一种有前途的解决方案，使用draft模型并行预测token。然而，将投机解码应用于RL训练提出了独特的挑战：由于策略变化，RL训练过程具有动态接受率，维护多个模型的开销可能很大。我们提出了ReSpec，一个为LLM的RL训练优化投机解码的系统。ReSpec引入了一种自适应draft选择机制，可以根据RL策略的演变动态调整draft模型。此外，我们提出了一种高效的缓存策略，可以减少管理多个draft模型的开销。我们的实验表明，与标准投机解码相比，ReSpec将RL训练时间减少了高达40%。

## 引言 (Introduction)

RL训练中投机解码的挑战：
1. RL训练过程动态变化
2. 接受率不稳定
3. 多模型开销大

ReSpec的创新：
- 自适应draft选择机制
- 动态调整draft模型
- 内存高效缓存策略
- 减少40%训练时间

## GitHub/项目

（待补充）