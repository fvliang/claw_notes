# CCCL: In-GPU Compression-Coupled Collective Communication

**ArXiv ID:** 2604.17172
**Published:** 2026-04-19
**Authors:** Chon Lam Lao, Zhiying Xu, Zhuang Wang, Ziming Mao, Delong Meng, Jia Zhen, Jun Wu, Ion Stoica, Yida Wang, Yang Zhou
**URL:** https://arxiv.org/abs/2604.17172
**PDF:** https://arxiv.org/pdf/2604.17172
**GitHub:** 暂无
**Categories:** cs.DC, cs.AI

## Abstract (English)

Collective communication incurs significant overhead in LLM workloads. Although overlapping communication with computation in application-level is a common strategy, it often requires substantial code modifications and is impractical for many workloads (e.g., tensor and expert parallelism). We present CCCL, a built-in compression-based collective communication library that supports operations such as allreduce, alltoall, and send/recv without requiring any user-side changes, thereby enabling seamless adoption in existing applications. CCCL tightly fuses compression kernels to minimize memory accesses and integrates with NCCL to eliminate the data coalescing stage, making it fast enough (up to 3x NVLink bandwidth) to sustain communication. Our evaluation shows that CCCL improves end-to-end throughput in vLLM PD disaggregation workloads by up to 10.1% and microbenchmark throughput by up to 30%.

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
