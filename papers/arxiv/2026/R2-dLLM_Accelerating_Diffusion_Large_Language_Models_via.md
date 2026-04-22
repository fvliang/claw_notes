# $R^2$-dLLM: Accelerating Diffusion Large Language Models via Spatio-Temporal Redundancy Reduction

**ArXiv ID:** 2604.18995v1
**Published:** 2026-04-21
**Authors:** Zhenbang Du, Kejing Xia, Xinrui Zhong, Yonggan Fu, Nicolai Oswald, Binfei Ji, Brucek Khailany, Pavlo Molchanov, Yingyan Lin
**URL:** https://arxiv.org/abs/2604.18995v1
**PDF:** https://arxiv.org/pdf/2604.18995v1
**GitHub:** 暂无
**Categories:** cs.CL, cs.AI, cs.LG

## Abstract (English)

Diffusion Large Language Models (dLLMs) have emerged as a promising alternative to autoregressive generation by enabling parallel token prediction. However, practical dLLM decoding still suffers from high inference latency, which limits deployment. In this work, we observe that a substantial part of this inefficiency comes from recurring redundancy in the decoding process, including spatial redundancy caused by confidence clusters and positional ambiguity, and temporal redundancy caused by repeatedly remasking predictions that have already stabilized. Motivated by these patterns, we propose $R^2$-dLLM, a unified framework for reducing decoding redundancy from both inference and training perspectives. At inference time, we introduce training-free decoding rules that aggregate local confidence and token predictions, and finalize temporally stable tokens to avoid redundant decoding steps. We further propose a redundancy-aware supervised fine-tuning pipeline that aligns the model with efficient decoding trajectories and reduces reliance on manually tuned thresholds. Experiments demonstrate that $R^2$-dLLM consistently reduces the number of decoding steps by up to 75% compared to existing decoding strategies, while maintaining competitive generation quality across different models and tasks. These results validate that decoding redundancy is a central bottleneck in dLLMs, and that explicitly reducing it yields substantial practical efficiency gains.

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
