# A Hybrid Online and Offline Requests Inference Serving System for LLM in Private Computer Environment

**ArXiv ID:** N/A
**Published:** 2026-03-01
**Authors:** Yuchen Shen, Yuning Zhang, Dong Yuan
**URL:** https://www.semanticscholar.org/paper/ef9af770c2ab36e31e57bd896abd0e8a005fb491
**PDF:** N/A
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

While advancements in Large Language Models (LLMs) have broadened their applications, performing multitask LLM inference on a single GPU remains challenging due to insufficient GPU memory to load all model parameters. Existing methods that offload unneeded parameters to main memory and prefetch them back introduce high latency due to data transfer overhead. We propose FixGen, a single GPU LLM online-offline mixed inference serving system that supports multi-task inference. First, we develop FixPool to optimize memory management by centralizing storage in a fixed memory space and optimizing parameter storage procedures, thus reducing PCIe resource consumption and improving efficiency. Secondly, we use a router to select appropriate operators to compute requests in the prefill and decode stages within the same batch to reduce the response latency for sudden online requests. FixGen reduces the online request-response latency between 1.06 and 1.84 times compared to the conventional method while maintaining the throughput to offline requests on a single GPU from OPT-6.7b to OPT-30b.

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
