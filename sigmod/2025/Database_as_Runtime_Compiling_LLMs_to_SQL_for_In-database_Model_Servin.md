# Database as Runtime: Compiling LLMs to SQL for In-database Model Serving

## Metadata
- **Authors:** Wenbo Sun, Ziyu Li, Rihan Hai
- **Conference:** SIGMOD 2025
- **Topic:** Edge Inference
- **arXiv ID:** 
- **Published:** 2025-06-22
- **GitHub:** 

## 原文链接
- arXiv: https://arxiv.org/abs/
- PDF: https://arxiv.org/pdf/

## 摘要 (Abstract)

Deploying large language models (LLMs) often requires specialized hardware and complex frameworks, creating barriers for CPU-based environments with resource constraints. These systems, common in air-gapped or edge scenarios, lack support for maintenance due to security, budget, or technical limits. To address this, we introduce TranSQL+, a compiler that translates LLM inference into SQL queries, enabling deployment on relational databases. By converting transformer operations into relational algebra, TranSQL+ generates vector-oriented SQL queries that leverage native database features (buffer management, indexing) to manage computations without hardware accelerators or deep learning frameworks. Demonstrated with the LLaMA3.1 8B model on DuckDB, results show relational databases can effectively serve LLMs, reducing deployment barriers and expanding access to advanced AI.

## 摘要 (中文)

[中文翻译待补充] Deploying large language models (LLMs) often requires specialized hardware and complex frameworks, creating barriers for CPU-based environments with resource constraints. These systems, common in air-...

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

[GitHub仓库待搜索补充]

---
*Auto-collected on 2026-04-28 morning*
