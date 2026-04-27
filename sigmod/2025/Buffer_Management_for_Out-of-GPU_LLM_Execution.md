# Buffer Management for Out-of-GPU LLM Execution

## Metadata
- **Authors:** Jiashen Cao, Joy Arulraj, Hyesoon Kim
- **Conference:** SIGMOD 2025
- **Topic:** Distributed Inference
- **arXiv ID:** 
- **Published:** 2025-06-22
- **GitHub:** ome-projects/ome

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

The rapid advancement of large language models (LLMs) has caused their parameter sizes to grow beyond the memory capacity of a single GPU. Although distributed inference across multiple GPUs is a solution in enterprise settings, it remains inaccessible for most non-commercial users. Thus, there is a growing demand to run LLMs on a single GPU when the model does not fit entirely in GPU memory. A common approach is to offload parts of the model from the GPU to the CPU during inference. However, repeatedly transferring parameters between these devices incurs significant overhead. To address this challenge, we propose a new buffer management policy, LIRS-M, which maximizes buffer hits and minimizes data transfer. Experimental results show that our approach achieves a 2.0× speedup compared to StoA offloading techniques while delivering robust buffer-hit performance.

## 摘要 (中文)

[中文翻译待补充] The rapid advancement of large language models (LLMs) has caused their parameter sizes to grow beyond the memory capacity of a single GPU. Although distributed inference across multiple GPUs is a solu...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

ome-projects/ome

---
*Auto-collected on 2026-04-28 morning*
