# DualDiffusion: A Speculative Decoding Strategy for Masked Diffusion Models

- **Arxiv ID**: 2604.05250
- **Conference**: icml 2026
- **Link**: https://arxiv.org/abs/2604.05250
- **GitHub**: 
- **Tags**: speculative-decoding, masked-diffusion, kv-cache, inference-acceleration

## Abstract (English)

Masked Diffusion Models (MDMs) offer a promising alternative to autoregressive language models by enabling parallel token generation and bidirectional context modeling. However, their inference speed is significantly limited by the inability to cache key-value pairs due to bidirectional attention, requiring O(N²) computations at each generation step. We propose DualDiffusion, a speculative decoding framework for MDMs that combines fast drafter models with slower, more accurate verifier models. By running multiple steps of a lightweight drafter followed by a single verification step, DualDiffusion achieves a superior Pareto frontier between generation steps and accuracy compared to existing approaches. Evaluated on MMLU and GSM8K, DualDiffusion maintains high accuracy while reducing the number of generation steps required.

## Abstract (Chinese)

掩码扩散模型(MDM)通过并行token生成和双向上下文建模，提供了自回归语言模型的有前景替代方案。但其推理速度因双向注意力无法缓存KV对而受限，每步需要O(N²)计算。我们提出了DualDiffusion，一个面向MDM的投机解码框架，将快速drafter模型与更慢但更准确的验证器模型结合。通过运行多步轻量级drafter后进行单步验证，DualDiffusion在生成步数与准确率之间实现了优越的帕累托前沿。在MMLU和GSM8K上的评估表明，DualDiffusion在减少生成步数的同时保持高准确率。

## Introduction (English)

Masked Diffusion Models (MDMs) have emerged as a compelling alternative to autoregressive (AR) language models. Unlike AR models that generate tokens sequentially, MDMs can unmask multiple tokens in parallel while leveraging bidirectional attention. However, MDMs face a critical computational bottleneck: the bidirectional attention mechanism prevents the use of KV caching, requiring O(N²) attention computations per step. We introduce DualDiffusion, a speculative decoding framework designed specifically for masked diffusion models, running multiple unmasking steps using a fast drafter then performing a single verification step with an accurate verifier model.

## Introduction (Chinese)

掩码扩散模型(MDM)已成为自回归(AR)语言模型的引人注目的替代方案。与AR模型顺序生成token不同，MDM可以并行解码多个token并利用双向注意力。然而MDM面临关键计算瓶颈：双向注意力机制阻止了KV缓存的使用，每步需要O(N²)注意力计算。我们引入DualDiffusion，一个专为掩码扩散模型设计的投机解码框架，使用快速drafter运行多步解码后用精确验证器执行单步验证。

## GitHub Introduction

N/A - No GitHub repository found for this paper.

## Blog Content

N/A - No blog post found for this paper.

---
*Auto-collected on 2026-04-21*
