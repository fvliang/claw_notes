# UniCAIM: A Unified CAM/CIM Architecture with Static-Dynamic KV Cache Pruning for Efficient Long-Context LLM Inference

**ArXiv ID:** 2504.07479
**Published:** 2025-04-10
**Authors:** Weikai Xu, Wenxuan Zeng, Qianqian Huang, Meng Li, Ruei-Hao Huang
**Conference/Venue:** Design Automation Conference
**URL:** https://arxiv.org/abs/2504.07479
**PDF:** https://arxiv.org/pdf/2504.07479
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

Transformer-based large language models (LLMs) have achieved impressive performance in various natural language processing (NLP) applications. However, the high memory and computation cost induced by the KV cache limits the inference efficiency, especially for long input sequences. Compute-in-memory (CIM)-based accelerators have been proposed for LLM acceleration with KV cache pruning. However, as existing accelerators only support static pruning with a fixed pattern or dynamic pruning with primitive implementations, they suffer from either high accuracy degradation or low efficiency. In this paper, we propose a ferroelectric FET (FeFET)-based unified content addressable memory (CAM) and CIM architecture, dubbed as UniCAIM. UniCAIM features simultaneous support for static and dynamic pruning with 3 computation modes: 1) in the CAM mode, UniCAIM enables approximate similarity measurement in $\mathcal{O}(1)$ time for dynamic KV cache pruning with high energy efficiency; 2) in the charge-domain CIM mode, static pruning can be supported based on accumulative similarity score, which is much more flexible compared to fixed patterns; 3) in the current-domain mode, exact attention computation can be conducted with a subset of selected KV cache. We further propose a novel CAM/CIM cell design that leverages the multi-level characteristics of FeFETs for signed multibit storage of the KV cache and in-place attention computation. With extensive experimental results, we demonstrate UniCAIM can reduce the area-energy-delay product (AEDP) by $8.2 \sim 831 \times$ over the state-of-the-art CIM-based LLM accelerators at the circuit level, along with high accuracy comparable with dense attention at the application level, showing its great potential for efficient long-context LLM inference.

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
