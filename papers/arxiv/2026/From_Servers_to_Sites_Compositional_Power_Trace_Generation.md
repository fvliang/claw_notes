# From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning

**ArXiv ID:** 2603.18383
**Published:** 2026-03-19
**Authors:** Grant Wilkins, Fiodar Kazhamiaka, Ram Rajagopal
**URL:** https://arxiv.org/abs/2603.18383
**PDF:** https://arxiv.org/pdf/2603.18383
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Datacenter operators and electrical utilities rely on power traces at different spatiotemporal scales. Operators use fine-grained traces for provisioning, facility management, and scheduling, while utilities use site-level load profiles for capacity and interconnection planning. Existing datacenter power models do not capture LLM inference workloads, in which GPUs shift rapidly among compute-intensive prefill, lower-power decode, and idle states, and facility demand depends on how these states evolve and synchronize across many devices. We show that LLM inference power can be represented compositionally through two components: workload-driven transitions among operating states and configuration-specific power distributions within those states. Building on this observation, we develop a trace-generation framework that learns from measured traces and synthesizes power profiles for new traffic conditions and serving configurations. These traces aggregate from GPU servers to rack-, row-, and facility-scale load profiles at the temporal granularity required by the study. Across multiple LLMs, tensor-parallel settings, and GPU generations, our framework achieves median absolute energy error below 5% for most configurations while preserving temporal autocorrelation structure. The resulting traces support downstream analyses including oversubscription, power modulation, and utility-facing load characterization, enabling infrastructure evaluations that flat nameplate assumptions and static trace replay cannot support.

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
