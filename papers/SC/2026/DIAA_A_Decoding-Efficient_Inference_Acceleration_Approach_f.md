# DIAA: A Decoding-Efficient Inference Acceleration Approach for On-Device Large Language Models

**ArXiv ID:** N/A
**Published:** 2026-03-14
**Authors:** Hao Tian, Sheng Lu, Fuwen Tian, Guangming Cui, Zheng Li, Xuyun Zhang, Quan Z. Sheng, Wanchun Dou
**Conference/Venue:** AAAI Conference on Artificial Intelligence
**URL:** N/A
**PDF:** N/A
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

Large Language Models (LLMs) have revolutionized intelligent interactions, enabling mobile applications such as personal assistants on edge devices for local execution. Speculative decoding (SD) has emerged as a promising paradigm to accelerate LLM inference without compromising generation quality, employing a draft-then-verify manner. However, due to the constrained computing and memory resources on edge devices, existing SD works heavily rely on an auxiliary draft model that incurs additional memory burden and hinders the adaptability, as well as static token trees that yield suboptimal inference performance. To this end, we propose DIAA, a Decoding-efficient Inference Acceleration Approach for on-device LLMs. DIAA achieves plug-and-play and model-agnostic inference speedup with memory and computation efficiency for edge devices. Specifically, a pair of lightweight look-up tables (LUTs) is constructed by Top-K token sampling to cache historical tokens and probabilities for rapid candidate drafting. DIAA integrates a dynamic token tree with prior LUTs enabling paralleled verification, updated during decoding process, to adapt the online context. A computation overlap is then employed to pipeline the update operations of token tree, LUTs, and KV cache to improve the computational efficiency. Finally, through extensive experiments implemented on edge platform NVIDIA Jetson, DIAA outperforms existing baselines in generation speed and inference wall-clock time, while incurring minimal memory overhead.

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
