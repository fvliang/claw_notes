# Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner

**ArXiv ID:** 2508.15044
**Published:** 2025-08-20
**Authors:** Bolian Li, Yanran Wu, Xinyu Luo, Ruqi Zhang
**Conference/Venue:** Conference on Empirical Methods in Natural Language Processing
**URL:** https://arxiv.org/abs/2508.15044
**PDF:** https://arxiv.org/pdf/2508.15044
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

Aligning large language models (LLMs) with human preferences has become a critical step in their development. Recent research has increasingly focused on test-time alignment, where additional compute is allocated during inference to enhance LLM safety and reasoning capabilities. However, these test-time alignment techniques often incur substantial inference costs, limiting their practical application. We are inspired by the speculative sampling acceleration, which leverages a small draft model to efficiently predict future tokens, to address the efficiency bottleneck of test-time alignment. We introduce the reward-shifted speculative sampling (SSS) algorithm, in which the draft model is aligned with human preferences, while the target model remains unchanged. We theoretically demonstrate that the distributional shift between the aligned draft model and the unaligned target model can be exploited to recover the RLHF optimal solution without actually obtaining it, by modifying the acceptance criterion and bonus token distribution. Our algorithm achieves superior gold reward scores at a significantly reduced inference cost in test-time weak-to-strong alignment experiments, thereby validating both its effectiveness and efficiency.

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
