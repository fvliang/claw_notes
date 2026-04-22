# UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training

**ArXiv ID:** 2604.19241v1
**Published:** 2026-04-21
**Authors:** Size Zheng, Xuegui Zheng, Li-wen Chang, Jidong Zhai
**URL:** https://arxiv.org/abs/2604.19241v1
**PDF:** https://arxiv.org/pdf/2604.19241v1
**GitHub:** 暂无
**Categories:** cs.DC

## Abstract (English)

The exponential growth in Large Language Model (LLM) parameters has transformed model training into an increasingly resource-intensive endeavor. With the stagnation of Moore's Law and the widening disparity between computation throughput and communication bandwidth, expert parallelism (EP) has emerged as a critical strategy for scaling mixture-of-experts (MoE) models. However, despite numerous proposals for optimizing EP, ranging from communication compression to computation-communication overlap, adoption within production-grade frameworks like Megatron-LM remains conservative. Existing solutions often rely on ad-hoc, complex kernels that lack adaptability across diverse optimization configurations and frequently neglect numerical stability, failing to meet the strict precision requirements of large-scale training.   In this paper, we introduce UniEP, a novel system that unifies diverse EP optimization strategies into a cohesive abstraction. UniEP fuses the MoE communication and computation into MegaKernels, effectively transforming complex architectural tuning into a unified parameter search space for automated adaptability. Crucially, UniEP incorporates a deterministic token ordering mechanism that guarantees numerical consistency with sequential execution, even under aggressive overlap schedules. We evaluate UniEP on GPU clusters equipped with NVIDIA Hopper GPUs. Our results demonstrate that UniEP achieves 1.03$\times$-1.38$\times$ speedups over state-of-the-art work, effectively mitigating communication bottlenecks while maintaining the rigorous accuracy standards required for production LLM training.

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
