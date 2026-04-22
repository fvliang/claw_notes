# Fast Forward: Accelerating LLM Prefill with Predictive FFN Sparsity

**ArXiv ID:** 2602.00397
**Published:** 2026-01-30
**Authors:** Aayush Gautam, Mukul Gagrani, Junyoung Park, Mingu Lee, Chiris Lott, Narasimha Reddy
**URL:** https://arxiv.org/abs/2602.00397
**PDF:** https://arxiv.org/pdf/2602.00397
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

The prefill stage of large language model (LLM) inference is a key computational bottleneck for long-context workloads. At short-to-moderate context lengths (1K--16K tokens), Feed-Forward Networks (FFNs) dominate this cost, accounting for most of the total FLOPs. Existing FFN sparsification methods, designed for autoregressive decoding, fail to exploit the prefill stage's parallelism and often degrade accuracy. To address this, we introduce FastForward, a predictive sparsity framework that accelerates LLM prefill through block-wise, context-aware FFN sparsity. FastForward combines (1) a lightweight expert predictor to select high-importance neurons per block, (2) an error compensation network to correct sparsity-induced errors, and (3) a layer-wise sparsity scheduler to allocate compute based on token-mixing importance. Across LLaMA and Qwen models up to 8B parameters, FastForward delivers up to 1.45$\times$ compute-bound speedup at 50% FFN sparsity with $<$ 6% accuracy loss compared to the dense baseline on LongBench, substantially reducing Time-to-First-Token (TTFT) for efficient, long-context LLM inference on constrained hardware.

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
