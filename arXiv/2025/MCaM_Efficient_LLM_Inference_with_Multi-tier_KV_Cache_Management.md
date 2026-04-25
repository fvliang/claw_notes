# MCaM : Efficient LLM Inference with Multi-tier KV Cache Management

## Metadata
- **Authors:** Kexin Chu, Zixu Shen, Shengxun Cheng, Dawei Xiang, Ziqin Liu
- **Conference:** arXiv 2025
- **Topic:** KV Cache
- **arXiv ID:** 
- **Published:** 2025-07-21
- **GitHub:** vllm-project/vllm

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

The KV cache in current LLM serving system is primarily used to accelerate processing within a single request and is aggressively deleted once the response is generated. However, in scenarios like virtual assistants and multi-turn conversations, the KV cache can be reused across requests, which can dramatically reduce computation costs and improve serving latency. Caching historical tokens, however, significantly increases memory requirements. Furthermore, existing serving systems treat the request scheduler and KV cache separately, despite their tight coupling.MCaM is a multi-tier cache system that enables the KV cache reuse and sharing across requests. It leverages DRAM as slow- tier memory for storing the KV cache of historical prompts. To efficiently utilize fast-tier Hign Bandwidth Memory(HBM) on GPU, we co-designed the KV cache manager and scheduler to coordinate request scheduling and token placement across tiers. To hide the reload time, MCaM employs a pipeline prefetcher that overlaps communication and computation. Additionally, MCaM incorporates a quality-aware sparsification algorithm to heterogeneously compress the KV cache in each layer. This approach not only reduces data transfer size but also decreases the overall KV cache size. To remove data offloading from a request’s critical path, we designed an asynchronous offload engine that swaps data from HBM to DRAM in the background. Our experiments show that MCaM can reduce TTFT by up to 69% and improve prompt prefilling throughput by 3.3X. It can also reduce the end-to-end latency of LLM inference by up to 58% when request length increase to 4096 tokens.

## 摘要 (中文)

[中文翻译待补充] The KV cache in current LLM serving system is primarily used to accelerate processing within a single request and is aggressively deleted once the response is generated. However, in scenarios like vir...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

vllm-project/vllm

---
*Auto-collected on 2026-04-25*
