# HpT: Hybrid Acceleration of Spatio-Temporal Attention Model Training on Heterogeneous Manycore Architectures

**ArXiv ID:** N/A
**Published:** 2025-03-01
**Authors:** S. Dahal, Pratyush Dhingra, Krishu K. Thapa, P. Pande, Ananth Kalyanaraman
**Conference/Venue:** IEEE Transactions on Parallel and Distributed Systems
**URL:** N/A
**PDF:** N/A
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

Transformer models have become widely popular in numerous applications, and especially for building foundation large language models (LLMs). Recently, there has been a surge in the exploration of transformer-based architectures in non-LLM applications. In particular, the self-attention mechanism within the transformer architecture offers a way to exploit any hidden relations within data, making it widely applicable for a variety of spatio-temporal tasks in scientific computing domains (e.g., weather, traffic, agriculture). Most of these efforts have primarily focused on accelerating the inference phase. However, the computational resources required to train these attention-based models for scientific applications remain a significant challenge to address. Emerging non-volatile memory (NVM)-based processing-in-memory (PIM) architectures can achieve higher performance and better energy efficiency than their GPU-based counterparts. However, the frequent weight updates during training would necessitate write operations to NVM cells, posing a significant barrier for considering stand-alone NVM-based PIM architectures. In this paper, we present <monospace>HpT</monospace>, a new hybrid approach to accelerate the training of attention-based models for scientific applications. Our approach is hybrid at two different layers: at the software layer, our approach dynamically switches from a full-parameter training mode to a lower-parameter training mode by incorporating intrinsic dimensionality; and at the hardware layer, our approach harnesses the combined power of GPUs, resistive random-access memory (ReRAM)-based PIM devices, and systolic arrays. This software-hardware co-design approach is aimed at adaptively reducing both runtime and energy costs during the training phase, without compromising on quality. Experiments on four concrete real-world scientific applications demonstrate that our hybrid approach is able to significantly reduce training time (up to <inline-formula><tex-math notation="LaTeX">$11.9\times$</tex-math><alternatives><mml:math><mml:mrow><mml:mn>11</mml:mn><mml:mo>.</mml:mo><mml:mn>9</mml:mn><mml:mo>×</mml:mo></mml:mrow></mml:math><inline-graphic xlink:href="kalyanaraman-ieq1-3522781.gif"/></alternatives></inline-formula>) and energy consumption (up to <inline-formula><tex-math notation="LaTeX">$12.05\times$</tex-math><alternatives><mml:math><mml:mrow><mml:mn>12</mml:mn><mml:mo>.</mml:mo><mml:mn>05</mml:mn><mml:mo>×</mml:mo></mml:mrow></mml:math><inline-graphic xlink:href="kalyanaraman-ieq2-3522781.gif"/></alternatives></inline-formula>), compared to the corresponding full-parameter training executing on only GPUs. Our approach serves as an example for accelerating the training of attention-based models on heterogeneous platforms including ReRAMs.

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
