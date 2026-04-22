# Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control

**ArXiv ID:** 2602.02987
**Published:** 2026-02-03
**Authors:** Ruihan Lin, Zezhen Ding, Zean Han, Jiheng Zhang
**URL:** https://arxiv.org/abs/2602.02987
**PDF:** https://arxiv.org/pdf/2602.02987
**GitHub:** 暂无
**Fields:** Computer Science, Mathematics

## Abstract (English)

Large Language Models (LLMs) are rapidly becoming critical infrastructure for enterprise applications, driving unprecedented demand for GPU-based inference services. A key operational challenge arises from the two-phase nature of LLM inference: a compute-intensive \emph{prefill} phase that processes user input, followed by a memory-bound \emph{decode} phase that generates output tokens. When these phases share GPU resources, prefill tasks throttle the processing speed of concurrent decodes, creating state-dependent contention. This contention is further complicated by workload heterogeneity, as different applications exhibit vastly different input and output lengths. We develop a stochastic control framework for scheduling heterogeneous LLM workloads across large GPU clusters. We formulate LLM inference as a multiclass many-server queueing network with state-dependent service rates, grounded in empirical iteration-time measurements. We analyze the fluid approximation of this system and solve steady-state linear programs that characterize optimal resource allocation. We design gate-and-route policies that regulate prefill admission and decode routing, and prove that they are asymptotically optimal in the many-GPU limit under both bundled and separate token-pricing schemes. We further extend the framework to incorporate Service Level Indicators (SLIs) such as latency and fairness, providing a general approach to constrained scheduling. Numerical experiments calibrated to empirical iteration-time data demonstrate that our policies outperform standard serving heuristics.

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
