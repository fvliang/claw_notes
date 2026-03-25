# DSD: A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving

## 论文信息

- **原文链接**: https://arxiv.org/abs/2511.15588
- **作者**: Fengze Yu, Leshu Li, Brad McDanel, Sai Qian Zhang
- **年份**: 2025
- **来源**: arXiv

## 摘要 (Abstract)

Large language model (LLM) inference often suffers from high latency, which limits its practical applicability in real-time applications. Speculative decoding has emerged as a promising technique to reduce inference latency by using a smaller draft model to propose candidate tokens. However, existing speculative decoding solutions are typically designed for single-machine scenarios and cannot fully exploit the potential of distributed computing resources. We present DSD, a distributed speculative decoding solution for edge-cloud agile large model serving. DSD leverages both edge devices and cloud servers to collaboratively perform speculative decoding, dynamically allocating computation based on device capabilities and network conditions. Our experiments show that DSD achieves up to 3x speedup compared to cloud-only serving while maintaining response quality.

## 摘要 (中文)

大型语言模型（LLM）推理经常遭受高延迟的困扰，这限制了其在实时应用中的实际适用性。投机解码已成为一种有前途的技术，通过使用较小的draft模型提出候选token来减少推理延迟。然而，现有的投机解码解决方案通常针对单机器场景设计，无法充分发挥分布式计算资源的潜力。我们提出了DSD，一个用于边缘云敏捷大模型服务的分布式投机解码解决方案。DSD利用边缘设备和云服务器协作执行投机解码，根据设备能力和网络条件动态分配计算。我们的实验表明，与纯云服务相比，DSD实现了高达3倍的加速，同时保持响应质量。

## 引言 (Introduction)

核心挑战：
1. LLM推理延迟高
2. 单机场景无法利用分布式资源
3. 边缘云资源协调困难

DSD的创新：
- 边缘云协作投机解码
- 动态计算分配
- 高达3倍加速
- 保持响应质量

## GitHub/项目

（待补充）