# Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference

**ArXiv ID:** 2602.07397
**Published:** 2026-02-07
**Authors:** Hoang Anh Le, Sahil Joshi, Zeyu Yang, Zhaozhuo Xu, Anshumali Shrivastava
**URL:** https://arxiv.org/abs/2602.07397
**PDF:** https://arxiv.org/pdf/2602.07397
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Self-attention dominates the computational and memory cost of long-context LLM inference across both prefill and decode phases. To address this challenge, we introduce Sketch&Walk Attention, a training-free sparse attention method that determines sparsity with lightweight sketches and deterministic walk. Sketch&Walk applies Hadamard sketching to get inexpensive approximations of attention scores, then aggregates these estimates across layers via a walk mechanism that captures attention influence beyond direct interactions between tokens. The accumulated walk scores are used to select top-k attention blocks, enabling dynamic sparsity with a single training-free algorithm that applies uniformly to both the prefill and decode phases, together with custom sparse attention kernels. Across a wide range of models and tasks, Sketch&Walk maintains near-lossless accuracy at 20% attention density and can slightly outperform dense attention in some settings, while achieving up to 6x inference speedup.

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
