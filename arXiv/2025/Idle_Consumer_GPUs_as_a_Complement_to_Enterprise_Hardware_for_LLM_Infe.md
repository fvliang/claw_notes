# Idle Consumer GPUs as a Complement to Enterprise Hardware for LLM Inference: Performance, Cost and Carbon Analysis

## Metadata
- **Authors:** ['A. Almeida']
- **Conference:** arXiv 2025
- **Topic:** Quantization
- **arXiv ID:** 
- **Published:** 2025-09-17
- **GitHub:** 

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

We examine the cost-performance landscape of Large Language Model (LLM) inference across two GPU tiers: Nvidia's enterprise-class H100 and the widely available consumer-grade RTX 4090. We benchmark latency, tokens per second, and cost per million tokens for models spanning 1.5 billion to 70 billion parameters, then zoom in on a 14 billion parameter model for a detailed comparison. H100s deliver up to 1.5× throughput and sub-55 ms tail latencies, yet 4090 clusters provide up to 75% lower token cost for batched or latency-tolerant workloads. We describe how quantization (GPTQ/AWQ), modern serving stacks (vLLM, SGLang), and Petals-style distributed execution enable consumer GPUs to push beyond on-board memory limits, with moderate latency tradeoffs. From a sustainability perspective, we show that H100s can be roughly 3.1× more energy-efficient per token, and how this edge narrows on low-carbon grids or when tapping otherwise-idle consumer hardware. We conclude that hybrid routing traffic to enterprise GPUs and to consumer pools can offer a pragmatic blend of performance, cost, and sustainability, based on Service Level Objectives (SLOs). The open benchmarks and cost models aim to guide practitioners in building heterogeneous GPU stacks for scalable, economical, and greener LLM services.

## 摘要 (中文)

[中文翻译待补充] We examine the cost-performance landscape of Large Language Model (LLM) inference across two GPU tiers: Nvidia's enterprise-class H100 and the widely available consumer-grade RTX 4090. We benchmark la...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

[GitHub仓库待搜索补充]

---
*Auto-collected on 2026-04-24*
