# TARDIS: A GPU-Centric KV Cache Service for Efficient LLM Inference

## Metadata
- **Authors:** Yifan Hu, Shi Qiu, Jianqin Yan, Hao Chen, Xintao Wang
- **Conference:** arXiv 2025
- **Topic:** KV Cache
- **arXiv ID:** 
- **Published:** 2025-10-11
- **GitHub:** 

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Key-value (KV) cache is a crucial optimization for large language model (LLM) serving, particularly in long-context inference scenarios. While existing KV stores suffer from a fundamental mismatch between the CPU-centric KV cache storage and the GPU-accelerated computation: (1) the low-parallelism CPUs cannot satisfy the fine-grained, highly parallel I/O demand, and (2) the synchronization between CPUs and GPUs introduced by layerwise asynchronous KV operations prevents the adoption of low-level optimizations (such as CUDA Graph). This paper presents TARDIS, a GPU-centric KV cache service for efficient long-context LLM inference. Inspired by file-to-memory mapping (mmap) supported by traditional CPU file systems, the key idea of TARDIS is to map KVs directly onto GPU's high-bandwidth memory (HBM) via modern GPU file systems, so that GPU kernels can access KVs without CPU intervention thus achieving high-concurrency, fine-grained KV storage/retrieval. At the core of TARDIS is a GPU-driven KV store (called GStore), which leverages GeminiFS to allow GPUs to directly access KVs on NVMe SSDs. Based on GStore, TARDIS designs an on-GPU scheduler that can adaptively schedule KV requests onto HBM/SSDs, and enables asynchronous, layer-wise token swapping through CUDA Graph, overlapping computation and KV cache access without CPU synchronization overhead. Evaluation reveals that TARDIS significantly outperforms state-of-the-art CPU-centric KV cache designs for LLM inference, boosting serving throughput by up to 20.52% while maintaining performance within 2.04% of an ideal in-memory KV cache.

## 摘要 (中文)

[中文翻译待补充] Key-value (KV) cache is a crucial optimization for large language model (LLM) serving, particularly in long-context inference scenarios. While existing KV stores suffer from a fundamental mismatch bet...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

[GitHub仓库待搜索补充]

---
*Auto-collected on 2026-04-26 evening*
