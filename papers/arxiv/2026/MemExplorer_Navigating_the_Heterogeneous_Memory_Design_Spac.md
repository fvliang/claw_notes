# MemExplorer: Navigating the Heterogeneous Memory Design Space for Agentic Inference NPUs

**ArXiv ID:** 2604.16007
**Published:** 2026-04-17
**Authors:** Haoran Wu, Zeyu Cao, Yao Lai, Binglei Lou, Jiayi Nie, Can Xiao, T. Adeniran, Przemyslaw Forys, Kauser Johar, Catriona R Wright, Junyi Liu, Kai Shi, Nicholas D. Lane, R. Antonova, Jianyi Cheng, Timothy Jones, Aaron Zhao, Robert Mullins
**URL:** https://arxiv.org/abs/2604.16007
**PDF:** https://arxiv.org/pdf/2604.16007
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Emerging agentic LLM workloads are driving rapidly growing demand on both memory capacity and bandwidth, with different phases of inference (e.g., prefill and decode) imposing distinct requirements. Industry is responding by composing heterogeneous accelerators into single interconnected systems, as exemplified by NVIDIA's Vera Rubin platform, where each device brings its own memory architecture. This heterogeneity is further compounded by a widening landscape of available memory technologies: high-density on-chip SRAM, HBM, LPDDR, GDDR, and emerging options such as high-bandwidth flash (HBF), each offering different capacity, bandwidth, and power trade-offs. Identifying the right memory architecture for next-generation inference accelerators requires navigating a vast and rapidly evolving design space, in which the interplay between workload characteristics, NPU design dimensions, and memory system design remains largely underexplored. To address this challenge, we present MemExplorer, a new memory system synthesizer for heterogeneous NPU systems. MemExplorer provides a unified abstraction for modeling diverse memory technologies across different hierarchy levels (e.g., on-chip and off-chip) and automatically determines an efficient heterogeneous memory system together with NPU design choices (e.g., matrix engine size) to balance throughput and power between prefilling and decoding devices in a multi-device NPU system. Experimental results show that, under the same power budget for agentic workloads, MemExplorer achieves up to 2.3x higher energy efficiency than the baseline NPU and 3.23x higher than H100 in the prefill-only setting. Under equivalent performance targets in the decode setting, it further delivers up to 1.93x and 2.72x higher power efficiency over the baseline NPU and H100, respectively.

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
