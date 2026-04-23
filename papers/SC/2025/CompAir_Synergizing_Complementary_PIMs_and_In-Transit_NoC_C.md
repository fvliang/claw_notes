# CompAir: Synergizing Complementary PIMs and In-Transit NoC Computation for Efficient LLM Acceleration

**ArXiv ID:** 2509.13710
**Published:** 2025-09-17
**Authors:** Hongyi Li, Songchen Ma, Huanyu Qu, Weihao Zhang, Jia Chen, Junfeng Lin, Fengbin Tu, Rong Zhao
**Conference/Venue:** arXiv.org
**URL:** https://arxiv.org/abs/2509.13710
**PDF:** https://arxiv.org/pdf/2509.13710
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

The rapid advancement of Large Language Models (LLMs) has revolutionized various aspects of human life, yet their immense computational and energy demands pose significant challenges for efficient inference. The memory wall, the growing processor-memory speed disparity, remains a critical bottleneck for LLM. Process-In-Memory (PIM) architectures overcome limitations by co-locating compute units with memory, leveraging 5-20$\times$ higher internal bandwidth and enabling greater energy efficiency than GPUs. However, existing PIMs struggle to balance flexibility, performance, and cost-efficiency for LLMs'dynamic memory-compute patterns and operator diversity. DRAM-PIM suffers from inter-bank communication overhead despite its vector parallelism. SRAM-PIM offers sub-10ns latency for matrix operation but is constrained by limited capacity. This work introduces CompAir, a novel PIM architecture that integrates DRAM-PIM and SRAM-PIM with hybrid bonding, enabling efficient linear computations while unlocking multi-granularity data pathways. We further develop CompAir-NoC, an advanced network-on-chip with an embedded arithmetic logic unit that performs non-linear operations during data movement, simultaneously reducing communication overhead and area cost. Finally, we develop a hierarchical Instruction Set Architecture that ensures both flexibility and programmability of the hybrid PIM. Experimental results demonstrate that CompAir achieves 1.83-7.98$\times$ prefill and 1.95-6.28$\times$ decode improvement over the current state-of-the-art fully PIM architecture. Compared to the hybrid A100 and HBM-PIM system, CompAir achieves 3.52$\times$ energy consumption reduction with comparable throughput. This work represents the first systematic exploration of hybrid DRAM-PIM and SRAM-PIM architectures with in-network computation capabilities, offering a high-efficiency solution for LLM.

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
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
