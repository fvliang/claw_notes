# Optimizing LLM inference for FPGAs

**ArXiv ID:** N/A
**Published:** 2025-10-21
**Authors:** J. R. de Freitas, J. G. Coutinho, Ce Guo, S. Demirsoy, Wayne Luk, Zhiqiang Que
**Conference/Venue:** International Conference on ASIC
**URL:** N/A
**PDF:** N/A
**GitHub:** custom-computing-ic/llm-oneapi-fpga
**Categories:** N/A

## Abstract (English)

Large Language Models (LLMs) deliver state-of-the-art performance but demand high computation and memory, making deployment in resource-limited settings challenging. Field-Programmable Gate Arrays (FPGAs) offer parallelism and efficiency, yet most prior FPGA accelerators rely on low-level, platform-specific flows that hinder portability. This work presents oneLLM, to our knowledge, the first FPGA-based LLM inference design using Intel’s oneAPI, enabling a unified high-level programming model across CPUs, GPUs, and FPGAs. Our deeply pipelined, multi-kernel hardware architecture connects specialized kernels via oneAPI pipes for on-chip streaming, reducing host–device communication. Implemented on an Intel Agilex 7 FPGA, it achieves 3 times faster than a CPU implementation, and 8.8 times faster than a non-pipelined baseline while meeting resource constraints, demonstrating the potential of portable FPGA development for LLM acceleration. Code available at https://github.com/custom-computing-ic/llm-oneapi-fpga.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

https://github.com/custom-computing-ic/llm-oneapi-fpga

---
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
