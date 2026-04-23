# Efficient In-Memory Acceleration of Sparse Block Diagonal LLMs

**ArXiv ID:** 2510.11192
**Published:** 2025-10-08
**Authors:** J. Lima, Marc Dietrich, J. Castrillón, Asif Ali Khan
**Conference/Venue:** 2025 Cross-Disciplinary Conference on Memory-Centric Computing (CCMCC)
**URL:** https://arxiv.org/abs/2510.11192
**PDF:** https://arxiv.org/pdf/2510.11192
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

Structured sparsity enables deploying large language models (LLMs) on resource-constrained systems. Approaches like dense-to-sparse fine-tuning are particularly compelling, achieving remarkable structured sparsity by reducing the model size by over 6.7×, while still maintaining acceptable accuracy. Despite this reduction, LLM inference, especially the decode stage being inherently memory-bound, is extremely expensive on conventional Von-Neumann architectures. Compute-in-memory (CIM) architectures mitigate this by performing computations directly in memory, and when paired with sparse LLMs, enable storing and computing the entire model in memory – eliminating the data movement on the off-chip bus and improving efficiency. Nonetheless, naively mapping sparse matrices onto CIM arrays leads to poor array utilization and diminished computational efficiency. In this paper, we present an automated framework with novel mapping and scheduling strategies to accelerate sparse LLM inference on CIM accelerators. By exploiting block-diagonal sparsity, our approach improves CIM array utilization by over 50%, achieving more than 4× reduction in both memory footprint and the number of required floating-point operations.

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
