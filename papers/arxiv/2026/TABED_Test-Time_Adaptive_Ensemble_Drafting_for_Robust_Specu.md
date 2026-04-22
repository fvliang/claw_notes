# TABED: Test-Time Adaptive Ensemble Drafting for Robust Speculative Decoding in LVLMs

**ArXiv ID:** 2601.20357
**Published:** 2026-01-28
**Authors:** Minjae Lee, Wonjun Kang, Byeongkeun Ahn, Christian Classen, Kevin Galim, Seunghyuk Oh, Minghao Yan, Hyung Il Koo, Kangwook Lee
**URL:** https://arxiv.org/abs/2601.20357
**PDF:** https://arxiv.org/pdf/2601.20357
**GitHub:** furiosa-ai/TABED
**Fields:** Computer Science

## Abstract (English)

Speculative decoding (SD) has proven effective for accelerating LLM inference by quickly generating draft tokens and verifying them in parallel. However, SD remains largely unexplored for Large Vision-Language Models (LVLMs), which extend LLMs to process both image and text prompts. To address this gap, we benchmark existing inference methods with small draft models on 11 datasets across diverse input scenarios and observe scenario-specific performance fluctuations. Motivated by these findings, we propose Test-time Adaptive Batched Ensemble Drafting (TABED), which dynamically ensembles multiple drafts obtained via batch inference by leveraging deviations from past ground truths available in the SD setting. The dynamic ensemble method achieves an average robust walltime speedup of 1.74x over autoregressive decoding and a 5% improvement over single drafting methods, while remaining training-free and keeping ensembling costs negligible through parameter sharing. With its plug-and-play compatibility, we further enhance TABED by integrating advanced verification and alternative drafting methods. Code and custom-trained models are available at https://github.com/furiosa-ai/TABED.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

furiosa-ai/TABED

---
*注: 此文件由自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
