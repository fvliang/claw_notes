# GPU-Centric Memory Tiering for LLM Serving With NVIDIA Grace Hopper Superchip

## Metadata
- **Authors:** ['Woohyun Choi', 'Jinwoo Jeong', 'Hanhwi Jang', 'Jeongseob Ahn']
- **Conference:** arXiv 2025
- **Topic:** Quantization
- **arXiv ID:** 
- **Published:** 2025-01-01
- **GitHub:** dipampaul17/KVSplit

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

This study investigates the performance of serving large language models (LLMs) with a focus on the high-bandwidth interconnect between GPU and CPU using a real NVIDIA Grace Hopper Superchip. This architecture features a GPU-centric memory tiering system, comprising a performance tier with GPU memory and a capacity tier with host memory. We revisit a conventional pipelined execution for LLM inference, utilizing host memory connected via NVLink alongside GPU memory. For the Llama-3.1 8B base (FP16) model, such a GPU-centric tiered memory system meets the target latency requirements for both prefill and decoding while improving throughput compared to the in-memory case, where all model weights are maintained in GPU memory. However, even with NVLink-connected CPU memory, meeting latency constraints for large models like the 70B and 405B FP16 models remains challenging. To address this, we explore the efficacy of model quantization (e.g., AWQ) along with the pipelined execution. Our evaluation reveals that the model quantization makes the pipelined execution a viable solution for serving large models. For the Llama-3.1 70B and 405B AWQ models, we show that the pipelined execution achieves 1.6× and 2.9× throughput improvement, respectively, compared to the in-memory only case, while meeting the latency constraint.

## 摘要 (中文)

[中文翻译待补充] This study investigates the performance of serving large language models (LLMs) with a focus on the high-bandwidth interconnect between GPU and CPU using a real NVIDIA Grace Hopper Superchip. This arc...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

dipampaul17/KVSplit

---
*Auto-collected on 2026-04-24*
