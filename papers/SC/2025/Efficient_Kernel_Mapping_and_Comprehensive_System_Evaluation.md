# Efficient Kernel Mapping and Comprehensive System Evaluation of LLM Acceleration on a CGLA

**ArXiv ID:** 2512.00335
**Published:** 2025-11-29
**Authors:** Takuto Ando, Yu Eto, Ayumu Takeuchi, Yasuhiko Nakashima
**Conference/Venue:** IEEE Access
**URL:** https://arxiv.org/abs/2512.00335
**PDF:** https://arxiv.org/pdf/2512.00335
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

Large Language Models (LLMs) demand substantial computational resources, resulting in high energy consumption on GPUs. To address this challenge, we focus on Coarse-Grained Reconfigurable Arrays (CGRAs) as an effective alternative that provides a trade-off between energy efficiency and programmability. This paper presents the first comprehensive, end-to-end evaluation of a non-AI-specialized Coarse-Grained Linear Array (CGLA) accelerator for the state-of-the-art Qwen3 LLM family. The architecture has a general-purpose, task-agnostic design, yet its flexible instruction set allows for domain-specific adaptations. This flexibility enables the architecture to achieve high efficiency for sustainable LLM inference. We assess the performance of our architecture on an FPGA prototype using the widely adopted llama.cpp framework. We then project its potential as a 28 nm ASIC and compare it against a high-performance GPU (NVIDIA RTX 4090) and an edge AI device (NVIDIA Jetson AGX Orin). While GPUs exhibit lower latency, our non-AI-specific accelerator achieves higher energy efficiency, improving the Power-Delay Product (PDP) by up to <inline-formula> <tex-math notation="LaTeX">${44.4} \times $ </tex-math></inline-formula> and <inline-formula> <tex-math notation="LaTeX">${13.6} \times $ </tex-math></inline-formula> compared with the RTX 4090 and Jetson, respectively. Similarly, it reduces the Energy-Delay Product (EDP) by up to <inline-formula> <tex-math notation="LaTeX">${11.5} \times $ </tex-math></inline-formula> compared to the high-performance GPU, demonstrating a favorable performance-energy trade-off. Critically, our system-level analysis identifies host-accelerator data transfer as the primary performance bottleneck, a factor often overlooked in kernel-level studies. These findings provide design guidance for next-generation LLM accelerators. This work validates CGRAs as a suitable platform for LLM inference in power-constrained environments, without being confined to specific algorithms.

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
