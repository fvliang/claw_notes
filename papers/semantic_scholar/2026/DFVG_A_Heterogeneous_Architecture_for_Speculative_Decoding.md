# DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU

**ArXiv ID:** N/A
**Published:** 2026-03-22
**Authors:** Shaoqiang Lu, Yangbo Wei, Junhong Qian, Dongge Qin, Shiji Gao, Yizhi Ding, Qifan Wang, Chen Wu, Xiao Shi, Lei He
**URL:** https://www.semanticscholar.org/paper/61739326c6c37e9138c60d3c5d87ba957855bf4c
**PDF:** N/A
**GitHub:** ShaoqiangLu/DFVG
**Fields:** Computer Science

## Abstract (English)

Speculative decoding is a promising paradigm that accelerates LLM inference by generating drafts and performing verification. However, such systems still face three major challenges: (1) The imbalance in resource requirements between draft and verification models result in low utilization and energy inefficiency when deployed together. (2) Fixed-pattern token trees produce many candidates but few valid paths, resulting in redundant drafts due to the lack of full leverage of the inherent confidence in dynamic generation. (3) Asynchronous execution with frequent alternation between the two stages suffers from idle waiting and rollback overhead. To address these issues, we propose DFVG, a heterogeneous speculative decoding architecture that offloads draft generation to FPGAs and verification to GPUs, exploiting their complementary strengths. We introduce three key contributions: (1) Heterogeneous architecture design that partitions speculative decoding into FPGA-based drafting and GPU-based verification, exploiting complementary hardware strengths with an overlap processor for high-throughput execution; (2) Hardware-aware dynamic draft generation that dynamically predicts speculative branches and token lengths based on model confidence while considering hardware parallelism limits; (3) Tightly-coupled heterogeneous pipeline with stagedecoupled scheduling that allocates execution windows between stages, combined with lightweight cross-device alignment and rollback prediction strategies. Comprehensive evaluation on mainstream models (OPT, LLaMA, Qwen) demonstrates DFVG achieves up to 3.26× speedup and 5.8× energy efficiency improvement over existing approaches. The source code at: https://github.com/ShaoqiangLu/DFVG

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

ShaoqiangLu/DFVG

---
*注: 此文件由自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
