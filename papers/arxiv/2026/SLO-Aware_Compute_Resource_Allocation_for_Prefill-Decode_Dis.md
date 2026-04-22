# SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference

**ArXiv ID:** 2603.04716
**Published:** 2026-03-05
**Authors:** Luchang Li, Dongfang Li, Bozhao Gong, Yu Zhang
**URL:** https://arxiv.org/abs/2603.04716
**PDF:** https://arxiv.org/pdf/2603.04716
**GitHub:** 暂无
**Fields:** Computer Science, Mathematics

## Abstract (English)

Prefill-Decode (P/D) disaggregation has emerged as a widely adopted optimization strategy for Large Language Model (LLM) inference. However, there currently exists no well-established methodology for determining the optimal number of P/D hardware resources, subject to constraints on total throughput, service level objectives (SLOs), and request characteristics - specifically input and output lengths. To address this gap, we propose a hybrid approach that combines theoretical modeling with empirical benchmarking. First, we present a theoretical model for calculating P/D resource counts, which is based on total throughput requirements, request input and output lengths, as well as prefill and decode throughput. Then, to obtain the actual prefill and decode throughput under SLO constraints, we model the prefill process using M/M/1 queuing theory, deriving the achieved prefill throughput from the benchmarked maximum prefill throughput and Time-To-First-Token (TTFT). For the decode phase, we determine the decode batch sizes that meet Time-Per-Output-Token (TPOT) requirements and obtain the corresponding decode throughput through empirical measurements. Our experimental results demonstrate that the proposed method can accurately predict optimal P/D resource allocation in real-world LLM inference scenarios.

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
