# RAPID-Serve: Resource-efficient and Accelerated P/D Intra-GPU Disaggregation

**ArXiv ID:** 2601.11822
**Published:** 2026-01-16
**Authors:** Amna Masood, Pratishtha Gaur, N. Jayasena
**URL:** https://arxiv.org/abs/2601.11822
**PDF:** https://arxiv.org/pdf/2601.11822
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Two widely adopted techniques for LLM inference serving systems today are hybrid batching and disaggregated serving. A hybrid batch combines prefill and decode tokens of different requests in the same batch to improve resource utilization and throughput at the cost of increased latency per token. In contrast, disaggregated serving decouples compute-bound prefill and bandwidth-bound decode phases to optimize for service level objectives (SLOs) at the cost of resource under-utilization and KV-cache transfer overheads. To address the limitations of these techniques, we propose RAPID-Serve: a technique to concurrently execute prefill and decode on the same GPU(s) to meet latency SLOs while maintaining high throughput and efficient resource utilization. Furthermore, we propose Adaptive Resource Management for runtime compute resource allocation, optionally leveraging CU masking (a fine-grained Compute Unit partitioning feature on AMD Instinct\textsuperscript{TM} GPUs). RAPID-Serve provides up to 4.1x (average 1.7x) unconstrained throughput improvement and 32x and higher (average 4.9x) throughput improvement under SLO constraints, showing it as an effective strategy compared to the state-of-the-art approaches, particularly in resource-constrained environments.

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
