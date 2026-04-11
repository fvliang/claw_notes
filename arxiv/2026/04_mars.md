---
title: MARS: Enabling Autoregressive Models Multi-Token Generation
authors: Ziqi Jin, Lei Wang, Ziwei Luo, Aixin Sun
arxiv_id: 2604.07023
conference: arxiv
full_conference: arXiv
year: "2026"
topic: LLM Inference
url: https://arxiv.org/abs/2604.07023
pdf_url: https://arxiv.org/pdf/2604.07023
added_date: 2026-04-11
---

# MARS: Enabling Autoregressive Models Multi-Token Generation

## 论文信息

- **arXiv**: [2604.07023](https://arxiv.org/abs/2604.07023)
- **作者**: Ziqi Jin, Lei Wang, Ziwei Luo, Aixin Sun

## 摘要 (Abstract)

Autoregressive (AR) language models generate text one token at a time, even when consecutive tokens are highly predictable given earlier context. We introduce MARS (Mask AutoRegreSsion), a lightweight fine-tuning method that teaches an instruction-tuned AR model to predict multiple tokens per forward pass. MARS adds no architectural modifications, no extra parameters, and produces a single model that can still be called exactly like the original AR model with no performance degradation. Unlike speculative decoding, which maintains a separate draft model alongside the target, or multi-head approaches such as Medusa, which attach additional prediction heads, MARS requires only continued training on existing instruction data.

## 摘要中文

自回归（AR）语言模型即使在给定先前上下文时连续token高度可预测的情况下，也是一个token一个token地生成文本。我们提出了MARS（Mask AutoRegreSsion），一种轻量级的微调方法，用于教导指令调整后的AR模型在每次前向传播中预测多个token。MARS不需要任何架构修改，不需要额外参数，生成的模型可以像原始AR模型一样调用，且没有性能下降。与投机解码（需要维护单独的draft模型）或Medusa等多头方法（需要附加额外的预测头）不同，MARS只需要在现有指令数据上进行持续训练。

## 引言 (Introduction)

Traditional autoregressive language models generate tokens sequentially, which limits throughput. MARS enables multi-token generation through a simple fine-tuning approach without changing the model architecture.

## 引言中文

传统的自回归语言模型顺序生成token，这限制了吞吐量。MARS通过简单的微调方法实现多token生成，无需改变模型架构。

## 实验结果

When generating one token per forward pass, MARS matches or exceeds the AR baseline on six standard benchmarks. When allowed to accept multiple tokens per step, it maintains baseline-level accuracy while achieving 1.5-1.7x throughput. We further develop a block-level KV caching strategy for batch inference, achieving up to 1.71x wall-clock speedup over AR with KV cache on Qwen2.5-7B.