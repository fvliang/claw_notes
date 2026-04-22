# Adelia: A 4-nm LLM Processing Unit With Streamlined Dataflow and Dual-Mode Parallelism for Maximizing Hardware Efficiency

**ArXiv ID:** N/A
**Published:** 2026-04-01
**Authors:** Sukbin Lim, Jung-Hoon Kim, Seungjae Moon, Junseo Cha, Dongjin Seo, Jongho Kim, Hunjong Lee, Jinwon Lee, Joo-Young Kim
**URL:** https://www.semanticscholar.org/paper/9fcea338d318a93968a4153f17c7d1008fccb57a
**PDF:** N/A
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

The proliferation of large language models (LLMs) as cross-domain foundation models is fueled by aggressive scaling in both parameter counts and inference-time computation. The emergence of sophisticated reasoning models further accelerates this trend, demanding longer context windows and escalating the computational and memory burdens of inference. A fundamental challenge arises from the bimodal nature of LLM inference, which consists of a compute-bound prefill phase and a memory-bound decode phase. This duality creates a significant performance bottleneck for conventional architectures such as GPUs and neural processing units (NPUs), which are typically optimized for only one phase, leading to severe underutilization of resources in the other. This article introduces Adelia, a novel LLM inference accelerator designed to resolve this challenge by co-optimizing for both memory and compute efficiency. Adelia features a streamlined dataflow that aligns sustained compute throughput with available external memory bandwidth, maximizing utilization across both inference phases. Its architecture features powerful matrix and vector execution engines (VXEs) fed by a systolic parameter path (SPP) interconnect, which efficiently multicasts model parameters and key–value (KV) caches for spatial reuse, all orchestrated by an RISC control processor (RCP) that provides dynamic instruction dispatch and enables dual-mode parallelism. By holistically addressing the bimodal bottleneck, Adelia demonstrates high performance and scalable efficiency during inference. Fabricated in Samsung’s 4-nm process, Adelia achieves a high throughput of 512.6 Tokens/s on Phi-3-mini with an end-to-end runtime of 5.993 s and 65.72 Tokens/s on LLaMA-30B with a runtime of 46.741 s.

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
