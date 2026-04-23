# FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design

**ArXiv ID:** 2508.04405
**Published:** 2025-08-06
**Authors:** Hao Zhang, Aining Jia, Weifeng Bu, Yu Cai, Kai Sheng, Hao Chen, Xin He
**Conference/Venue:** arXiv.org
**URL:** https://arxiv.org/abs/2508.04405
**PDF:** https://arxiv.org/pdf/2508.04405
**GitHub:** FlyFoxPlayer/FlexQ
**Categories:** Computer Science

## Abstract (English)

Large Language Models (LLMs) demonstrate exceptional performance but entail significant memory and computational costs, restricting their practical deployment. While existing INT4/INT8 quantization reduces these costs, they often degrade accuracy or lack optimal efficiency. INT6 quantization offers a superior trade-off between model accuracy and inference efficiency, but lacks hardware support in modern GPUs, forcing emulation via higher-precision arithmetic units that limit acceleration. In this paper, we propose FlexQ, a novel post-training INT6 quantization framework combining algorithmic innovation with system-level optimizations. FlexQ employs uniform 6-bit weight quantization across all layers, with adaptive retention of 8-bit activations in layers identified through layer-wise sensitivity analysis. To maximize hardware efficiency, we develop a specialized high-performance GPU kernel supporting matrix multiplication for W6A6 and W6A8 representations via Binary Tensor Core (BTC) equivalents, effectively bypassing the lack of native INT6 tensor cores. Evaluations on LLaMA family models show FlexQ maintains near-FP16 accuracy, with perplexity increases of no more than 0.1 on WikiText2. The proposed kernel achieves an average 1.39$\times$ speedup over ABQ-LLM on LLaMA-2-70B linear layers. End-to-end, FlexQ delivers 1.33$\times$ inference acceleration and 1.21$\times$ memory savings over SmoothQuant. Code is released at https://github.com/FlyFoxPlayer/FlexQ.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

https://github.com/FlyFoxPlayer/FlexQ

---
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
