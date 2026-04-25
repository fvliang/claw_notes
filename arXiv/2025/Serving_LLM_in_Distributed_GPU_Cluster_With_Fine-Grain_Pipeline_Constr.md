# Serving LLM in Distributed GPU Cluster With Fine-Grain Pipeline Constraints

## Metadata
- **Authors:** Yanying Lin, Shijie Peng, Shuaipeng Wu, Yanbo Li, Chengzhi Lu
- **Conference:** arXiv 2025
- **Topic:** Inference Scheduling
- **arXiv ID:** 
- **Published:** 2025-09-01
- **GitHub:** vllm-project/vllm

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

As Large Language Models (LLMs) continue to advance, their parameter sizes are growing exponentially—far outpacing hardware capabilities. This widening gap necessitates distributed computing through pipeline parallelism for efficient inference. However, the uneven distribution of requests across pipeline stages creates significant performance bottlenecks in real-world deployments. To address this challenge, we present Planck, a performance optimization framework specifically designed for distributed LLM inference. Planck implements fine-grained control through two key mechanisms: a progressive SLO allocation strategy that dynamically adjusts time constraints based on workload patterns, and stage-specific performance controllers that prevent bottlenecks before they cascade through the system. By intelligently balancing resources across pipeline stages, Planck effectively eliminates queue buildup—essentially preventing traffic congestion before it forms. Evaluation using diverse workloads in real cloud environments demonstrates that Planck reduces P99 tail latency by up to 18% and decreases the longest queue lengths by as much as 47.8% across pipeline stages, significantly improving both system responsiveness and resource utilization.

## 摘要 (中文)

[中文翻译待补充] As Large Language Models (LLMs) continue to advance, their parameter sizes are growing exponentially—far outpacing hardware capabilities. This widening gap necessitates distributed computing through p...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

vllm-project/vllm

---
*Auto-collected on 2026-04-25*
