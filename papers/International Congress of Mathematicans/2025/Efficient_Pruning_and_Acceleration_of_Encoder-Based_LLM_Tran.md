# Efficient Pruning and Acceleration of Encoder-Based LLM Transformers on eFPGAs

**ArXiv ID:** N/A
**Published:** 2025-12-14
**Authors:** Omar Elayat, Vincent Gaudet, M. Elmasry
**Conference/Venue:** International Congress of Mathematicans
**URL:** N/A
**PDF:** N/A
**GitHub:** 暂无
**Categories:** N/A

## Abstract (English)

Transformer encoders such as Bidirectional Encoder Representations from Transformers (BERT) are widely adopted for Natural Language Processing (NLP) tasks, yet their computational and memory requirements hinder deployment on edge devices. While pruning reduces model size, most hardware friendly methods rely on structured, semi-structured, or pattern pruning at the expense of accuracy. Recent unstructured pruning methods have shown promising accuracy-efficiency tradeoffs but have only been demonstrated on decoder models and lack clear hardware deployment pathways. To address this problem, this work applies the state-of-the-art WANDA [17] pruning algorithm to BERT encoders and evaluates the accuracy of both unstructured and semi-structured sparsity regimes. To complement pruning and maximize inference efficiency, we propose a hardware architecture featuring a double-buffered systolic matrix-matrix multiplier with skip-zero support. This approach minimizes bitmask memory storage overhead and hides memory latency through fully overlapping compute and stream operations. Our approach is evaluated using the linear layers of the encoder as a representative workload and is implemented on the Digilent PYNQ-Z1 board, a resource-constrained Field-Programmable Gate Array (FPGA). Compared to dense baselines, our design achieves up to $1.67 \times$ latency improvement with negligible accuracy degradation on the GLUE [28] benchmark.

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
