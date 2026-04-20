# The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference

- **Arxiv ID**: 2604.15409
- **Conference**: arxiv 2026
- **Link**: https://arxiv.org/abs/2604.15409
- **GitHub**: 
- **Tags**: kv-cache, fp16, inference-reliability, numerical-divergence

## Abstract (English)

KV caching is a ubiquitous optimization in autoregressive transformer inference, long presumed to be numerically equivalent to cache-free computation. This assumption fails under standard FP16 precision: cache-ON and cache-OFF execution paths employ different floating-point accumulation orderings which, due to FP16 non-associativity, produce a deterministic divergence in decoded token sequences. Across three open-weight models (LLaMA-2-7B, Mistral-7B-v0.3, Gemma-2-2B) evaluated on GSM8K, we observe a 100% token divergence rate across all sampling strategies, including greedy decoding. Controlled FP32 falsification reduces divergence by eight orders of magnitude, confirming FP16 non-associativity as the sole causal driver. Layer-wise drift profiling reveals architecturally predictable propagation patterns: GQA exhibits sharp divergence at the first layer, while Gemma's larger head dimension produces uniform accumulation across all layers.

## Abstract (Chinese)

KV缓存是自回归Transformer推理中的通用优化，长期以来被认为与无缓存计算数值等价。这一假设在标准FP16精度下失效：cache-ON和cache-OFF执行路径采用不同的浮点累积顺序，由于FP16非结合性，在解码token序列中产生确定性偏差。在三个开源模型(LLaMA-2-7B, Mistral-7B-v0.3, Gemma-2-2B)上，我们在所有采样策略下观察到100%的token偏差率。FP32对照实验将偏差减少八个数量级，确认FP16非结合性是唯一因果驱动因素。层级漂移分析揭示了架构可预测的传播模式。

## Introduction (English)

Autoregressive transformer inference is almost always based on the Key-Value (KV) cache: instead of recomputing attention over the entire prefix at every decoding step, previously computed keys and values are stored and retrieved as needed. This technique is typically assumed to be numerically equivalent to cache-free inference. We show this to be untrue in the simplest case of a standard FP16 inference. The cache-ON and cache-OFF execution paths have different layouts and kernel structures, and slightly different floating-point accumulation order. These differences lead to a systematic numerical divergence in the KV tensors written during decoding. This divergence is deterministic and reproducible: for the same input, same model, and same hardware, cache-ON and cache-OFF inference will always compute different output tokens.

## Introduction (Chinese)

自回归Transformer推理几乎总是基于KV缓存：在每一步解码时存储和检索先前计算的键和值，而非重新计算整个前缀的注意力。这项技术通常被认为与无缓存推理数值等价。我们证明这在标准FP16推理的最简单情况下不成立。cache-ON和cache-OFF执行路径具有不同的布局和内核结构，以及略微不同的浮点累积顺序，导致解码期间KV张量中出现系统性数值偏差。这种偏差是确定性的、可复现的。

## GitHub Introduction

N/A - No GitHub repository found for this paper.

## Blog Content

N/A - No blog post found for this paper.

---
*Auto-collected on 2026-04-21*
