# Dynamic Offloading Optimization for Multi-Pim LLM Inference

## Metadata
- **Authors:** Jeonghoon Kang, Jae Hyung Ko, Taeho Hwang, Kyu Hyun Choi
- **Conference:** arXiv 2025
- **Topic:** Prefill/Disaggregation
- **arXiv ID:** 
- **Published:** 2025-10-27
- **GitHub:** kvcache-ai/ktransformers

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

The growing demand for energy-efficient on-device AI in consumer appliances is drawing significant attention to Processing-In-Memory (PIM) architectures. This trend is largely driven by the proliferation of Large Language Models (LLMs). While these models have enabled remarkable advances in reasoning and generative tasks, they increasingly face the memory wall problem, where data movement between processors and memory dominates both latency and energy consumption. Processing-In-Memory (PIM) architectures have emerged as a promising solution by executing memory-intensive workloads directly within memory, thereby reducing data transfer overhead. However, efficient integration of PIM into existing heterogeneous systems requires careful management of data transfers between host processors and PIM devices, as well as scheduling policies for inference graph execution. This paper introduces a Node-Aware tensor offloading method that leverages topological analysis of the model computational graph to capture tensor layout relationships across consumer nodes. An Affinity-Based device scheduling algorithm is also proposed, which uses a shared hash table to align operand tensors with their execution device for each operation kernel, eliminating redundant inter-device communication. Experimental results demonstrate that the proposed method extends a GDDR6-based PIM emulation framework integrated with ONNX Runtime to support 4D matrix multiplication. The optimizations achieve up to 15.63× and 29.3× latency reduction for QK and SV multiplications in the prefill stage, resulting in 93.6% and 94.0% energy savings, respectively. Additionally, in the decoding stage, execution time is reduced by 1.09× and 2.01×, resulting in 8.3% and 50.4% energy savings, respectively.

## 摘要 (中文)

[中文翻译待补充] The growing demand for energy-efficient on-device AI in consumer appliances is drawing significant attention to Processing-In-Memory (PIM) architectures. This trend is largely driven by the proliferat...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

kvcache-ai/ktransformers

---
*Auto-collected on 2026-04-25 evening*
