# Dynamic Micro-Batch and Token-Budget Scheduling for IoT-Scale Pipeline-Parallel LLM Inference

**ArXiv ID:** N/A
**Published:** 2026-02-01
**Authors:** Juncheol Ahn, Yubin Son, Daemin Kim, Sejin Park
**URL:** https://www.semanticscholar.org/paper/4156ff1bee3fdff0c526dadb0afefd62ce562a87
**PDF:** N/A
**GitHub:** 暂无
**Fields:** Medicine

## Abstract (English)

Large language models in IoT–edge–cloud settings face bursty, heterogeneous requests that make pipeline-parallel inference prone to micro-batch imbalance and communication stalls, causing GPU idle time and SLO violations. We propose a runtime-adaptive scheduler that jointly tunes token budgets and micro-batch counts to balance prefill/decode workloads and minimize pipeline bubbles under changing compute and network conditions. On a four-node pipeline-parallel cluster across Llama-2-13b and Qwen2.5-14b at 100/1000 Mbps, our method outperforms vLLM and SGLang, reducing GPU idle time by up to 55% and improving throughput by up to 1.61 × while improving TTFT/ITL SLO satisfaction. These results show that dynamic scheduling is essential for scalable, latency-stable LLM inference in IoT–edge–cloud environments.

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
