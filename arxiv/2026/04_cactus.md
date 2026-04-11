---
title: Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling
authors: Yongchang Hao, Lili Mou
arxiv_id: 2604.04987
conference: iclr
full_conference: ICLR 2026
year: "2026"
topic: Speculative Decoding
url: https://arxiv.org/abs/2604.04987
pdf_url: https://arxiv.org/pdf/2604.04987
added_date: 2026-04-11
---

# Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling

## 论文信息

- **arXiv**: [2604.04987](https://arxiv.org/abs/2604.04987)
- **会议**: ICLR 2026
- **作者**: Yongchang Hao, Lili Mou

## 摘要 (Abstract)

Speculative sampling (SpS) has been successful in accelerating the decoding throughput of auto-regressive large language models by leveraging smaller draft models. SpS strictly enforces the generated distribution to match that of the verifier LLM. This is unnecessarily restrictive as slight variations of the verifier distribution, such as sampling with top-k or temperature, would also be acceptable. Typical acceptance sampling (TAS) alleviates this issue by accepting more tokens using entropy-based heuristics. However, this approach distorts the verifier distribution, potentially degrading output quality when the verifier encodes critical information. In this work, we formalize the speculative sampling algorithm through the lens of constrained optimization. Based on this formulation, we propose Cactus (constrained acceptance speculative sampling), a method that guarantees controlled divergence from the verifier distribution and increasing acceptance rates.

## 摘要中文

投机采样（SpS）通过利用较小的draft模型成功加速了自回归大语言模型的解码吞吐量。SpS严格要求生成的分布与验证器LLM的分布完全匹配。这种限制过于严格，因为验证器分布的轻微变化（如使用top-k或温度采样）也是可接受的。典型接受采样（TAS）通过使用基于熵的启发式方法接受更多token来缓解这个问题。然而，这种方法会扭曲验证器分布，当验证器编码关键信息时可能会降低输出质量。在这项工作中，我们通过约束优化的视角形式化投机采样算法。基于这种形式化，我们提出了Cactus（约束接受投机采样），一种保证受控偏离验证器分布并提高接受率的方法。

## 引言 (Introduction)

Standard speculative sampling has a strict requirement that accepted tokens exactly match the target distribution. Cactus relaxes this requirement while maintaining quality guarantees.

## 引言中文

标准投机采样严格要求接受的token完全匹配目标分布。Cactus在保持质量保证的同时放宽了这一要求。

## 主要贡献

1. Formalize speculative sampling through constrained optimization
2. Propose Cactus that guarantees controlled divergence from verifier distribution
3. Achieve higher acceptance rates while maintaining output quality
4. Empirical results across wide range of benchmarks confirm effectiveness