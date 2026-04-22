# GreenScheduler: Coordinated Two-Tier Energy Optimization for Disaggregated LLM Serving

**ArXiv ID:** N/A
**Published:** 2026-03-01
**Authors:** Waled Milad Abulgasem Alashheb, Mabruka Khlifa Ali Karkeb, Sabria AbdulGader Ali Elmusrati, Sumia Abdussalam Milad Elagtel
**URL:** https://www.semanticscholar.org/paper/2dd8ee54b1240817f924fca3261bb94f28680137
**PDF:** N/A
**GitHub:** 暂无
**Fields:** N/A

## Abstract (English)

Large Language Model (LLM) inference has become a dominant consumer of en- ergy in modern AI data centers, often accounting for over 90% of total operational power [1].Recent architectural shifts toward prefill/decode disaggregation have improved perfor- mance but created complex energy optimization challenges. This paper introduces Green- Scheduler, a novel two-tier framework designed to jointly optimize GPU placement and Dynamic Voltage and Frequency Scaling (DVFS) in disaggregated environments. Tier 1 performs coarse-grained (minute-scale) phase-aware provisioning using predictive work- load modeling, while Tier 2 executes fine-grained (millisecond-scale) frequency control. For the compute-bound prefill stage, GreenScheduler employs Model Predictive Control (MPC) to manage queue dynamics; for the memory-bound decode stage, it utilizes a lightweight slack-aware adaptation mechanism. Evaluations using production Azure traces 
[2] on an H100 cluster demonstrate that GreenScheduler achieves significant energy reduc- tion in both decode and prefill pools compared to performance-optimized baselines like DistServe [3], while strictly maintaining Time to First Token (TTFT) and Time Per Output Token (TPOT) Service-Level Objectives (SLOs).

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
