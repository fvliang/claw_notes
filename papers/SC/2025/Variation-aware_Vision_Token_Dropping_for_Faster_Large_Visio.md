# Variation-aware Vision Token Dropping for Faster Large Vision-Language Models

**ArXiv ID:** 2509.01552
**Published:** 2025-09-01
**Authors:** Junjie Chen, Xuyang Liu, Zichen Wen, Yiyu Wang, Siteng Huang, Honggang Chen
**Conference/Venue:** arXiv.org
**URL:** https://arxiv.org/abs/2509.01552
**PDF:** https://arxiv.org/pdf/2509.01552
**GitHub:** 暂无
**Categories:** Computer Science

## Abstract (English)

Large vision-language models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding tasks. However, the increasing demand for high-resolution image and long-video understanding results in substantial token counts, consequently leading to reduced inference efficiency. Token compression offers a direct solution by reducing the number of tokens to be processed, thereby improving computational efficiency without architectural changes. Through extensive analysis, we identify two critical limitations in existing inner-LLM token compression methods: positional bias and incompatibility with efficient operators, which critically hinder their practical deployment for LVLM acceleration. This paper presents the first approach from a dynamic token variation perspective, revealing that visual token variations within LLMs exhibit task-agnostic properties. We propose Variation-aware Vision Token Dropping (\textit{i.e.}, \textbf{V$^2$Drop}), which progressively removes visual tokens with minimal variation during LVLM inference, thereby enhancing computational efficiency. Extensive experiments across multiple models and benchmarks consistently demonstrate that V$^2$Drop maintains \textbf{94.0\%} and \textbf{98.6\%} of the original performance for image and video understanding tasks respectively, while reducing LLM generation latency by \textbf{31.5\%} and \textbf{74.2\%}.

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
