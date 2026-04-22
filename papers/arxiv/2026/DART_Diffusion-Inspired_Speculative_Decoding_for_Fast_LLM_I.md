# DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference

**ArXiv ID:** 2601.19278
**Published:** 2026-01-27
**Authors:** Fuliang Liu, Xue Li, Ketai Zhao, Yi Gao, Ziyan Zhou, Zhonghui Zhang, Zhibin Wang, Wanchun Dou, Sheng Zhong, Chen Tian
**URL:** https://arxiv.org/abs/2601.19278
**PDF:** https://arxiv.org/pdf/2601.19278
**GitHub:** fvliang/DART
**Fields:** Computer Science

## Abstract (English)

Speculative decoding is an effective and lossless approach for accelerating LLM inference. However, existing widely adopted model-based draft designs, such as EAGLE3, improve accuracy at the cost of multi-step autoregressive inference, resulting in high drafting latency and ultimately rendering the drafting stage itself a performance bottleneck. Inspired by diffusion-based large language models (dLLMs), we propose DART, which leverages parallel generation to reduce drafting latency. DART predicts logits for multiple future masked positions in parallel within a single forward pass based on hidden states of the target model, thereby eliminating autoregressive rollouts in the draft model while preserving a lightweight design. Based on these parallel logit predictions, we further introduce an efficient tree pruning algorithm that constructs high-quality draft token trees with N-gram-enforced semantic continuity. DART substantially reduces draft-stage overhead while preserving high draft accuracy, leading to significantly improved end-to-end decoding speed. Experimental results demonstrate that DART achieves a 2.03x--3.44x wall-clock time speedup across multiple datasets, surpassing EAGLE3 by 30% on average and offering a practical speculative decoding framework. Code is released at https://github.com/fvliang/DART.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

fvliang/DART

---
*注: 此文件由自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
