# Coruscant: Co-Designing GPU Kernel and Sparse Tensor Core to Advocate Unstructured Sparsity in Efficient LLM Inference

## Metadata
- **Authors:** Donghyeon Joo, Helya Hosseini, Ramyad Hadidi, Bahar Asgari
- **Conference:** arXiv 2025
- **Topic:** Quantization
- **arXiv ID:** 
- **Published:** 2025-10-17
- **GitHub:** 0xSero/turboquant

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

In the era of large language models (LLMs) and long-context generation, model compression techniques such as pruning, quantization, and distillation offer effective ways to reduce memory usage. Among them, pruning is constrained by the difficulty of exploiting unstructured sparsity on modern hardware. Consequently, LLM pruning is often restricted to structured patterns for hardware efficiency, although unstructured sparsity offers better accuracy retention at higher sparsity. To bridge this gap between the full potential of pruning and efficiency, we propose the Coruscant GPU SpMM kernel that leverages a bitmap-based sparse format for reduced memory footprint inside GPU memory and reduced latency of memory-bound matrix multiplications in LLM inference. This is achieved by transferring the compressed matrix tiles to GPU processors and decompressing them locally for tensor core execution. We see further optimization opportunity in microarchitecture-level and propose Coruscant Sparse Tensor Core, which computes directly on the compressed format without decompression by integrating a bitmap decoder. Coruscant kernel achieves up to 2 × speedup over cuBLAS and 1.48 × over Flash-LLM. With Coruscant Sparse Tensor Core, the speedup reaches 2.75 × over cuBLAS. Most importantly, Coruscant serves as an ideal solution for state-of-the-art LLM pruning methods by significantly reducing the memory footprint and accelerating SpMM on sparsity range 30% to 70%, enabling exploration of diverse sparsity patterns and pruning strategies.

## 摘要 (中文)

[中文翻译待补充] In the era of large language models (LLMs) and long-context generation, model compression techniques such as pruning, quantization, and distillation offer effective ways to reduce memory usage. Among ...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

0xSero/turboquant

---
*Auto-collected on 2026-04-24 evening*
