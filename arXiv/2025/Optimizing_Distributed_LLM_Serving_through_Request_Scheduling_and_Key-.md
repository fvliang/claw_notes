# Optimizing Distributed LLM Serving through Request Scheduling and Key-Value Cache Sharing

## Metadata
- **Authors:** Hongye Jiang, Mu Wang, Su Yao, Cui Ting, Ziwei Li
- **Conference:** arXiv 2025
- **Topic:** KV Cache
- **arXiv ID:** 
- **Published:** 2025-12-14
- **GitHub:** vllm-project/vllm

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

The widespread deployment of Large Language Models (LLMs) is often constrained by the significant computational and memory demands of the inference process. A critical bottleneck in distributed serving systems arises from the redundant processing of requests that share common prefixes, such as system prompts or few-shot examples. Traditional contentagnostic load balancers fail to exploit these redundancies, leading to inefficient resource utilization and increased latency. This paper introduces a dynamic, prefix-aware request scheduling system designed to optimize distributed LLM serving. Our approach intelligently routes incoming requests to specific GPU workers by analyzing prompt content and matching it with the resident Key-Value (KV) caches across the cluster. By colocating requests with shared prefixes, our system maximizes KV cache reuse, minimizes expensive prefill computations, and enables more efficient batched attention operations at the worker level. We implemented and evaluated this scheduler on a 12 -node GPU cluster using a real-world chatbot workload. The results demonstrate the profound impact of content-aware scheduling: our system increased the aggregate prefill throughput by over 144 % and reduced the median Time-to-First-Token by over 40 % compared to a conventional Round-Robin policy. These performance gains are a direct result of an 82 % relative increase in the prefix cache hit rate, validating our approach as a highly effective and cost-efficient strategy for enhancing the throughput and responsiveness of large-scale LLM services.

## 摘要 (中文)

[中文翻译待补充] The widespread deployment of Large Language Models (LLMs) is often constrained by the significant computational and memory demands of the inference process. A critical bottleneck in distributed servin...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

vllm-project/vllm

---
*Auto-collected on 2026-04-25*
