---
title: KV Cache Offloading for Context-Intensive Tasks
authors: Andrey Bocharnikov, Ivan Ermakov, Denis Kuznedelev, Vyacheslav Zhdanovskiy, Yegor Yershov
arxiv_id: 2604.08426
conference: arxiv
full_conference: arXiv
year: "2026"
topic: KV Cache
url: https://arxiv.org/abs/2604.08426
pdf_url: https://arxiv.org/pdf/2604.08426
added_date: 2026-04-11
---

# KV Cache Offloading for Context-Intensive Tasks

## 论文信息

- **arXiv**: [2604.08426](https://arxiv.org/abs/2604.08426)
- **作者**: Andrey Bocharnikov, Ivan Ermakov, Denis Kuznedelev, Vyacheslav Zhdanovskiy, Yegor Yershov

## 摘要 (Abstract)

With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offloading has emerged as a promising approach to reduce memory footprint and inference latency while preserving accuracy. Prior evaluations have largely focused on tasks that do not require extracting large amounts of information from the context. In this work, we study KV-cache offloading on context-intensive tasks: problems where the solution requires looking up a lot of information from the input prompt. We create and release the Text2JSON benchmark, a highly context-intensive task that requires extracting structured knowledge from raw text. We evaluate modern KV offloading on Text2JSON and other context-intensive tasks and find significant performance degradation on both Llama 3 and Qwen 3 models.

## 摘要中文

随着长上下文LLM在各种应用中的需求增长，键值（KV）缓存已成为延迟和内存使用的关键瓶颈。最近，KV缓存卸载已成为一种有前景的方法，可以在保持准确性的同时减少内存占用和推理延迟。之前的评估主要集中在不需要从上下文中提取大量信息的任务上。在这项工作中，我们研究了上下文密集型任务上的KV缓存卸载：解决方案需要从输入提示中查找大量信息的问题。我们创建并发布了Text2JSON基准，这是一个高度上下文密集的任务，需要从原始文本中提取结构化知识。我们评估了现代KV卸载在Text2JSON和其他上下文密集型任务上的性能，发现Llama 3和Qwen 3模型都有显著的性能下降。

## 引言 (Introduction)

KV cache offloading is becoming important for long-context LLM serving. This paper identifies issues with current offloading techniques on context-intensive tasks.

## 引言中文

KV缓存卸载对长上下文LLM服务变得越来越重要。本文发现了当前卸载技术在上下文密集型任务上的问题。

## 主要发现

1. KV cache offloading shows significant performance degradation on context-intensive tasks
2. Two key reasons: low-rank projection of keys and unreliable landmarks
3. Propose simpler alternative strategy that significantly improves accuracy
4. Findings highlight need for comprehensive evaluation of long-context compression techniques

## Text2JSON Benchmark

We create and release the Text2JSON benchmark - a highly context-intensive task that requires extracting structured knowledge from raw text.