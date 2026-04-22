# Context-Aware Autoscaling for Cost-Efficient Large Language Model Inference With Prefix Cache Integration

**ArXiv ID:** N/A
**Published:** N/A
**Authors:** Seyed Hossein Ahmadpanah, A. Sahafi, S. H. Erfani
**URL:** https://www.semanticscholar.org/paper/62b577ec192e84f2bb2143ea1e43c23dc81bf3e3
**PDF:** N/A
**GitHub:** 暂无
**Fields:** N/A

## Abstract (English)

Although granular resource management has been made possible by the architectural shift to Prefill-Decode (PD) disaggregation in Large Language Model (LLM) serving, it is still difficult to maintain strict Service Level Objectives (SLOs) under bursty traffic. Modern autoscaling frameworks, like TokenScale, balance resources using fine-grained “velocity” metrics, but they have a significant drawback: cache blindness. These systems over-provision during spikes in high-locality traffic because they model instance throughput as a static hardware constant, which ignores the enormous effective throughput gains offered by prefix caching (e.g., RadixAttention). Furthermore, they often overlook the cost-efficiency opportunities inherent in heterogeneous GPU clusters. We present AdaptiveScale, a context-aware autoscaling framework that connects resource elasticity and memory locality. Three new mechanisms are suggested by AdaptiveScale: 1) Effective Token Velocity ( $V_{eff}$ ), a dynamic metric that incorporates real-time Radix-tree telemetry to amplify scaling logic based on context reuse; 2) Heterogeneity-Aware Tiered Scheduling, which optimizes convertible decoder selection by pinning cost-effective nodes (like L40S) to decoding while routing prefill bursts to compute-dense GPUs (like H100); and 3) Elastic State Preservation, a mechanism that uses RDMA to zero-copy offload KV-cache states during role-switching, thereby removing the latency penalties associated with cache eviction. AdaptiveScale was implemented on vLLM, and it was assessed using production traces from OpenAI and Azure. Our results demonstrate that AdaptiveScale reduces GPU operational costs by 28% compared to state-of-the-art velocity-based scalers in high-locality workloads, while consistently maintaining 99% SLO attainment.

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
