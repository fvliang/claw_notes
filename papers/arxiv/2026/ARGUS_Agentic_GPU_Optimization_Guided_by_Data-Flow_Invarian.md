# ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants

**ArXiv ID:** 2604.18616
**Published:** 2026-04-16
**Authors:** Haohui Mai, Xiaoyan Guo, Xiangyun Ding, Daifeng Li, Qiuchu Yu, Chenzhun Guo, Cong Wang, Jiacheng Zhao, Christos Kozyrakis, Binhang Yuan
**URL:** https://arxiv.org/abs/2604.18616
**PDF:** https://arxiv.org/pdf/2604.18616
**GitHub:** 暂无
**Categories:** cs.DC, cs.AI, cs.PL

## Abstract (English)

LLM-based coding agents can generate functionally correct GPU kernels, yet their performance remains far below hand-optimized libraries on critical computations such as matrix multiplication, attention, and Mixture-of-Experts (MoE). Peak GPU performance requires coordinated reasoning over tightly coupled optimizations, including tiling, shared-memory staging, software pipelining, and instruction scheduling, while existing agents rely on sparse pass/fail feedback, leaving them unable to diagnose global constraint violations.   We present Argus, an agentic framework that addresses this through data-flow invariants: compile-time specifications encoding how data must be choreographed throughout kernel execution. Argus introduces a tile-based, Pythonic DSL exposing hardware instructions and compiler policies while hiding low-level representations. The DSL provides tag functions to propagate symbolic annotations through data and control flow, and tag assertions to enforce relational constraints at use sites. When violations occur, the compiler returns concrete counterexamples identifying the thread, data element, and program point, enabling dense, structured feedback for targeted fixes. Invariants are verified at compile time via abstract interpretation over a layout algebra and SMT solving, with zero runtime overhead. An in-context reinforcement learning planner learns to select optimizations and synthesize effective invariants, supported by a curated knowledge base of GPU optimization techniques.   We evaluate Argus on the AMD MI300X GPU across GEMM, flash attention, and MoE kernels accounting for over 90% of GPU time in LLM inference. Generated kernels achieve 99-104% of state-of-the-art hand-optimized assembly throughput and are 2-1543x faster than existing agentic systems. Argus further generalizes to 200 KernelBench tasks, solving 100% of Level 1 and 90% of Level 2 problems.

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
