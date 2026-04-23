# EasySpec: Layer-Parallel Speculative Decoding for Efficient Multi-GPU Utilization

**ArXiv ID:** 2502.02493
**Published:** 2025-02-04
**Authors:** Yize Wu, Ke Gao, Yanjun Wu
**Conference/Venue:** arXiv.org
**URL:** https://arxiv.org/abs/2502.02493
**PDF:** https://arxiv.org/pdf/2502.02493
**GitHub:** Yize-Wu/EasySpec
**Categories:** Computer Science

## Abstract (English)

Speculative decoding is an effective and lossless method for Large Language Model (LLM) inference acceleration. It employs a smaller model to generate a draft token sequence, which is then verified by the original base model. In multi-GPU systems, inference latency can be further reduced through tensor parallelism (TP), while the optimal TP size of the draft model is typically smaller than that of the base model, leading to GPU idling during the drafting stage. We observe that such inefficiency stems from the sequential execution of layers, which is seemingly natural but actually unnecessary. Therefore, we propose EasySpec, a layer-parallel speculation strategy that optimizes the efficiency of multi-GPU utilization. EasySpec breaks the inter-layer data dependencies in the draft model, enabling multiple layers to run simultaneously across multiple devices as'fuzzy'speculation. After each drafting-and-verification iteration, the draft model's key-value cache is calibrated in a single forward pass, preventing long-term fuzzy-error accumulation at minimal additional latency. EasySpec is a training-free and plug-in method. We evaluated EasySpec on several mainstream open-source LLMs, using smaller versions of models from the same series as drafters. The results demonstrate that EasySpec can achieve a peak speedup of 4.17x compared to vanilla decoding, while preserving the original distributions of the base LLMs. Specifically, the drafting stage can be accelerated by up to 1.62x with a maximum speculation accuracy drop of only 7%. The code is available at https://github.com/Yize-Wu/EasySpec.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

https://github.com/Yize-Wu/EasySpec

---
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
