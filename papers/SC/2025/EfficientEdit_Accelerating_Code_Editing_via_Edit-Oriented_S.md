# EfficientEdit: Accelerating Code Editing via Edit-Oriented Speculative Decoding

**ArXiv ID:** 2506.02780
**Published:** 2025-06-03
**Authors:** Peiding Wang, Li Zhang, Fang Liu, Yinghao Zhu, Wang Xu, Lin Shi, Xiaoli Lian, Minxiao Li, Bo Shen, An Fu
**Conference/Venue:** International Conference on Automated Software Engineering
**URL:** https://arxiv.org/abs/2506.02780
**PDF:** https://arxiv.org/pdf/2506.02780
**GitHub:** zhu-zhu-ding/EfficientEdit
**Categories:** Computer Science

## Abstract (English)

Large Language Models (LLMs) have demonstrated remarkable capabilities in code editing, substantially enhancing software development productivity. However, the inherent complexity of code editing tasks forces existing approaches to rely on LLMs’ autoregressive end-to-end generation, where decoding speed plays a critical role in efficiency. While inference acceleration techniques like speculative decoding are applied to improve the decoding efficiency, these methods fail to account for the unique characteristics of code editing tasks, where changes are typically localized and existing code segments are reused. To address this limitation, we propose EfficientEdit, a novel method that improves LLM-based code editing efficiency through two key mechanisms based on speculative decoding: (1) effective reuse of original code segments while identifying potential edit locations, and (2) efficient generation of edit content via high-quality drafts from edit-oriented draft models and a dynamic verification mechanism that balances quality and acceleration. Experimental results show that EfficientEdit can achieve up to 10.38× and 13.09× speedup compared to standard autoregressive decoding in CanItEdit and CodeIF-Bench, respectively, outperforming state-of-the-art inference acceleration approaches by up to 90.6%. The code and data are available at https://github.com/zhu-zhu-ding/EfficientEdit.

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

https://github.com/zhu-zhu-ding/EfficientEdit

---
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
