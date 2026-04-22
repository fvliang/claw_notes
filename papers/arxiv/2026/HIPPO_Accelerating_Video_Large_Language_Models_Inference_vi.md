# HIPPO: Accelerating Video Large Language Models Inference via Holistic-aware Parallel Speculative Decoding

**ArXiv ID:** 2601.08273
**Published:** 2026-01-13
**Authors:** Qitan Lv, Tianyu Liu, Wen Wu, Xuenan Xu, Bowen Zhou, Feng Wu, Chao Zhang
**URL:** https://arxiv.org/abs/2601.08273
**PDF:** https://arxiv.org/pdf/2601.08273
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Speculative decoding (SD) has emerged as a promising approach to accelerate LLM inference without sacrificing output quality. Existing SD methods tailored for video-LLMs primarily focus on pruning redundant visual tokens to mitigate the computational burden of massive visual inputs. However, existing methods do not achieve inference acceleration comparable to text-only LLMs. We observe from extensive experiments that this phenomenon mainly stems from two limitations: (i) their pruning strategies inadequately preserve visual semantic tokens, degrading draft quality and acceptance rates; (ii) even with aggressive pruning (e.g., 90% visual tokens removed), the draft model's remaining inference cost limits overall speedup. To address these limitations, we propose HIPPO, a general holistic-aware parallel speculative decoding framework. Specifically, HIPPO proposes (i) a semantic-aware token preservation method, which fuses global attention scores with local visual semantics to retain semantic information at high pruning ratios; (ii) a video parallel SD algorithm that decouples and overlaps draft generation and target verification phases. Experiments on four video-LLMs across six benchmarks demonstrate HIPPO's effectiveness, yielding up to 3.51x speedup compared to vanilla auto-regressive decoding.

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
