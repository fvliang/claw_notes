# InnerQ: Hardware-aware Tuning-free Quantization of KV Cache for Large Language Models

**ArXiv ID:** 2602.23200
**Published:** 2026-02-26
**Authors:** Sayed Mohammadreza Tayaranian Hosseini, Amir Ardakani, W. Gross
**URL:** https://arxiv.org/abs/2602.23200
**PDF:** https://arxiv.org/pdf/2602.23200
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Reducing the hardware footprint of large language models (LLMs) during decoding is critical for efficient long-sequence generation. A key bottleneck is the key-value (KV) cache, whose size scales with sequence length and easily dominates the memory footprint of the model. Previous work proposed quantization methods that are focused on compressing the KV cache while maintaining its information. We introduce InnerQ, a hardware-aware KV-cache quantization scheme that lowers decode latency without sacrificing accuracy. InnerQ applies group-wise quantization while grouping the cache matrices over their inner dimension. Unlike previous work that group over the outer dimension, InnerQ aligns dequantization with the vector-matrix multiplication and enables scale factor reuse across GPU compute units. This reduces memory accesses and accelerates dequantization, yielding up to $22\%$ speedup over previous work and up to $88\%$ over half-precision vector-matrix multiplication. To preserve fidelity under aggressive compression, InnerQ incorporates (i) hybrid quantization, selecting symmetric or asymmetric quantization per group based on local statistics; (ii) high-precision windows for both the most recent tokens and the attention sink tokens to mitigate outlier leakage; and (iii) per-channel normalization of the key cache, computed once during prefill and folded into the query to avoid runtime overhead. Our evaluation experiments on Llama models shows that InnerQ maintains a few-shot GSM8K performance comparable to non-quantized KV caches and surpasses prior KV cache quantization methods.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

暂无 GitHub 仓库

---
*注: 此文件由自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
