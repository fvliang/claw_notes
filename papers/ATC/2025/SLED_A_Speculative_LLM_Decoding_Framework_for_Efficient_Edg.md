# SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving

**ArXiv ID:** 2506.09397
**Published:** 2025-06-11
**Authors:** Xiangchen Li, Dimitrios Spatharakis, Saeid Ghafouri, Jiakun Fan, Hans Vandierendonck, Deepu John, Bo Ji, Dimitrios S. Nikolopoulos
**Conference/Venue:** IFIP International Information Security Conference
**URL:** https://arxiv.org/abs/2506.09397
**PDF:** https://arxiv.org/pdf/2506.09397
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

The growing gap between the increasing complexity of large language models (LLMs) and the limited computational budgets of edge devices poses a key challenge for efficient on-device inference, despite gradual improvements in hardware capabilities. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new framework that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose SLED, a framework that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server verifies the tokens utilizing a more precise target model. To further increase the efficiency of verification, the edge server batches the diverse verification requests from devices. This approach supports heterogeneous devices and reduces server-side memory footprint by sharing a single upstream target model across devices. Our initial experiments with Jetson Orin Nano, Raspberry Pi 4B/5, and an edge server equipped with 4 Nvidia A100 GPUs indicate substantial benefits: ×2.2 higher system throughput, ×2.8 higher system capacity, and better cost efficiency, all without sacrificing model accuracy.

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
