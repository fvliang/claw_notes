# KVO-LLM: Boosting Long-Context Generation Throughput for Batched LLM Inference

## Metadata
- **Authors:** Zhenyu Li, Dongxu Lyu, Gang Wang, Yuzhou Chen, Liyan Chen
- **Conference:** arXiv 2025
- **Topic:** KV Cache
- **arXiv ID:** 
- **Published:** 2025-06-22
- **GitHub:** cuckoo-network/cuckoo

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

With the widespread deployment of long-context large language models (LLMs), efficient and high-quality generation is becoming increasingly important. Modern LLMs employ batching and key-value (KV) cache to improve generation throughput and quality. However, as the context length and batch size rise drastically, the KV cache incurs extreme external memory access (EMA) issues. Recent LLM accelerators face substantial processing element (PE) under-utilization due to the low arithmetic intensity of attention with KV cache, while existing KV cache compression algorithms struggle with hardware inefficiency or significant accuracy degradation. To address these issues, an algorithm-architecture co-optimization, KVO-LLM, is proposed for long-context batched LLM generation. At the algorithm level, we propose a KV cache quantization-aware pruning method that first adopts salient-token-aware quantization and then prunes KV channels and tokens by attention guided pruning based on salient tokens identified during quantization. Achieving substantial savings on hardware overhead, our algorithm reduces the EMA of KV cache over 91% with significant accuracy advantages compared to previous KV cache compression algorithms. At the architecture level, we propose a multi-core jointly optimized accelerator that adopts operator fusion and cross-batch interleaving strategy, maximizing PE and DRAM bandwidth utilization. Compared to the state-of-the-art LLM accelerators, KVO-LLM improves generation throughput by up to $7.32 \times$, and attains $5.52 \sim 8.38 \times$ better energy efficiency.

## 摘要 (中文)

[中文翻译待补充] With the widespread deployment of long-context large language models (LLMs), efficient and high-quality generation is becoming increasingly important. Modern LLMs employ batching and key-value (KV) ca...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

cuckoo-network/cuckoo

---
*Auto-collected on 2026-04-24 evening*
