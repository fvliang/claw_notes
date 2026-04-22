# TeLLMe: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs

**ArXiv ID:** N/A
**Published:** 2026-02-21
**Authors:** Ye Qiao, Zhiheng Chen, Yifan Zhang, Yian Wang, Sitao Huang
**URL:** https://www.semanticscholar.org/paper/66b136abe6969bd55505353c6c6b76805fa825ce
**PDF:** N/A
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

With the emergence of wearable devices and other embedded systems, deploying large language models (LLMs) on edge platforms becomes an urgent need. However, it is challenging because of their high computational and memory demands. Although recent low-bitwidth quantization methods (e.g., BitNet, DeepSeek) compress weights to as low as 1.58 bits with minimal accuracy loss, edge deployment is still constrained by limited on-chip resources, power budgets, and the often-neglected long latency of the prefill stage. We present TeLLMe, the first table-lookup-based ternary LLM accelerator for low-power edge FPGAs that fully supports both prefill and autoregressive decoding using 1.58-bit weights and 8-bit activations. TeLLMe incorporates our proposed novel techniques including (1) a table-lookup-based ternary matrix multiplication (TLMM) engine utilizing grouped activations and online precomputation for low resource utilization and high throughput; (2) a fine-grained URAM-based weight buffer management scheme supporting weight loading from global memory and compute engine weight access; (3) a streaming dataflow architecture that fuses floating-point element-wise operations with linear computations to hide latency; (4) a reversed-reordered prefill stage attention with fused attention operation for high memory efficiency; and (5) a resource-efficient specialized decoding stage attention. Under a 5W power budget, TeLLMe delivers up to 25 tokens/s decoding throughput and 0.45s to 0.96s Time-to-First-Token (TTFT) for 64–128 token prompts, marking a significant energy-efficiency advancement in LLM inference on edge FPGAs.

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
