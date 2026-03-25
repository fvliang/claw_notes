# Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving

## 论文信息

- **原文链接**: https://arxiv.org/abs/2512.02090
- **作者**: Rui Li, Zhaoning Zhang, Libo Zhang, et al.
- **年份**: 2025
- **来源**: arXiv

## 摘要 (Abstract)

Speculative decoding has emerged as a promising technique to accelerate large language model (LLM) inference by leveraging a small draft model to propose candidate tokens and a large target model to verify them in parallel. However, existing speculative decoding methods adopt a fixed verification batch size throughout the inference process, failing to adapt to the dynamically varying acceptance rates and available computational resources. We present Nightjar, a dynamic adaptive speculative decoding system that dynamically adjusts the verification batch size based on runtime inference characteristics to maximize the overall system throughput. Specifically, Nightjar monitors the acceptance rate of speculative tokens in real-time and uses a lightweight predictive model to forecast future acceptance rates under different candidate batch sizes. Then, Nightjar formulates the batch size selection as a utility maximization problem and selects the optimal batch size that maximizes the throughput gain. We implement Nightjar and evaluate it with various LLM serving workloads. Our experiments show that Nightjar achieves 1.19x-1.58x throughput improvement over state-of-the-art speculative decoding systems.

## 摘要 (中文)

投机解码已成为加速大型语言模型（LLM）推理的有前途的技术，它利用小型draft模型提出候选token，大型目标模型并行验证。然而，现有的投机解码方法在整个推理过程中采用固定的验证批大小，无法适应动态变化的接受率和可用计算资源。我们提出了Nightjar，一个动态自适应投机解码系统，它根据运行时推理特征动态调整验证批大小，以最大化整体系统吞吐量。具体来说，Nightjar实时监控投机token的接受率，并使用轻量级预测模型来预测不同候选批大小下的未来接受率。然后，Nightjar将批大小选择表述为效用最大化问题，并选择最大化吞吐量增益的最优批大小。我们实现了Nightjar并使用各种LLM服务工作负载对其进行评估。我们的实验表明，Nightjar相比最先进的投机解码系统实现了1.19倍-1.58倍的吞吐量提升。

## 引言 (Introduction)

投机解码的核心挑战：
1. 固定批大小无法适应动态变化的接受率
2. 计算资源利用率不足
3. 需要实时调整策略

Nightjar提出：
- 实时监控接受率
- 轻量级预测模型预测未来接受率
- 效用最大化选择最优批大小

## GitHub/项目

（待补充）