# QIGen: A Kernel Generator for Inference on Nonuniformly Quantized Large Language Models

## Metadata
- **Authors:** Tommaso Pegolotti, Dan Alistarh, Markus Püschel
- **Conference:** arXiv 2026
- **Topic:** Quantization
- **arXiv ID:** 
- **Published:** 2026-01-31
- **GitHub:** humanrouter/ddtree-mlx

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Efficient inference on large language models (LLMs) has become a popular topic in both academia and industry. Roughly speaking, LLMs consist of a collection of weight matrices, and generative inference on these models essentially computes a sequence of matrix-vector products and thus can be heavily memory-bound. Consequently, much work has been devoted to reducing the size of the weights to lower bit-widths through various forms of quantization. In turn, these diverse precision formats complicate both the arithmetic and optimized kernel implementation. So far, the vast majority of implementation work for mixed-precision LLM computation has been done manually. Currently, one of the most powerful and complex scalar LLM compression techniques is nonuniform quantization, in which a matrix is divided unevenly into parts that are quantized with different bit-widths, minimizing the output compression error. In this paper, we present QIGen, the first kernel generator for LLM inference on CPUs to support nonuniform quantization in full generality. Given a nonuniformly quantized LLM and target CPU characteristics, QIGen first generates the diverse set of needed custom matrix-vector product kernels and combines them with a suitable storage format. We benchmark and analyze QIGen-generated code in various experiments. In particular, we show that our code achieves Pareto optimality in terms of performance and accuracy with respect to the most used open-source tool. We show a speedup of up to 1.3× for the matrix-vector and 3.4× for the matrix-matrix computations even when using uniform quantization.

## 摘要 (中文)

[中文翻译待补充] Efficient inference on large language models (LLMs) has become a popular topic in both academia and industry. Roughly speaking, LLMs consist of a collection of weight matrices, and generative inferenc...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

humanrouter/ddtree-mlx

---
*Auto-collected on 2026-04-26 evening*
