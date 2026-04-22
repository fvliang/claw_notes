# PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies

**ArXiv ID:** 2603.09216
**Published:** 2026-03-10
**Authors:** Sunjung Lee, Sanghoon Cha, Hyeonsu Kim, Seung-Yeon Seo, Yuhwan Ro, Sukhan Lee, Byeongho Kim, Yongjun Park, Kyomin Sohn, Seungwon Lee, Jaehoon Yu
**URL:** https://arxiv.org/abs/2603.09216
**PDF:** https://arxiv.org/pdf/2603.09216
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-intensive decode phase, and the decode phase has been widely recognized as well-suited to processing-in-memory (PIM) in both academia and industry. However, practical PIM-enabled systems face two obstacles between these phases, a memory attribute inconsistency in which prefill favors placing weights in a cacheable region for reuse whereas decode requires weights in a non-cacheable region to reliably trigger PIM, and a weight layout inconsistency between host-friendly and PIM-aware layouts. To address these problems, we introduce \textit{PIM-SHERPA}, a software-only method for efficient on-device LLM inference by resolving PIM memory attribute and layout inconsistencies. PIM-SHERPA provides two approaches, DRAM double buffering (DDB), which keeps a single PIM-aware weights in the non-cacheable region while prefetching the swizzled weights of the next layer into small cacheable buffers, and online weight rearrangement with swizzled memory copy (OWR), which performs the on-demand swizzled memory copy immediately before GEMM. Compared to a baseline PIM emulation system, PIM-SHERPA achieves approximately 47.8 - 49.7\% memory capacity savings while maintaining comparable performance to the theoretical maximum on the Llama 3.2 model. To the best of our knowledge, this is the first work to identify the memory attribute inconsistency and propose effective solutions on product-level PIM-enabled systems.

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
