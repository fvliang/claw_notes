# Transformer-Based Resource and Stage-Aware Scheduling for Model-Parallel LLM Inference

**ArXiv ID:** N/A
**Published:** 2026-01-04
**Authors:** Rami Naeem, Tengis Buyantogtokh, Hamada Rizk, Tatsuya Amano, Hirozumi Yamaguchi
**URL:** https://www.semanticscholar.org/paper/5be714b739f3f60bb5ea5d20a8d6168ac8192450
**PDF:** N/A
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Current large language model (LLM) serving systems face three key limitations in distributed scheduling. First, most parallelization strategies are not stage-aware: they treat prefill and decode as uniform workloads despite their distinct compute and communication profiles. Second, many assume homogeneous hardware and ignore resource diversity in memory and bandwidth across nodes. Third, they overlook network congestion, as they are primarily designed for data-center environments with abundant interconnect bandwidth. We address these gaps with a resource- and stage-aware scheduler that models heterogeneous GPU clusters, communication costs, and per-stage characteristics. We compare three approaches: a heuristic stage-based policy, a continuous-batching (vLLM-style) baseline, and a transformer-based scheduler trained by imitation to replicate and improve the heuristic. Our evaluation spans eight representative scenarios covering large models that exceed a single GPU, prefill-dominant and mixed workloads, heterogeneous and bandwidth-limited clusters, strict SLO constraints, and multi-tenant or elastic deployments. The learned scheduler reduces latency by up to 50% under bandwidth-constrained or heterogeneous conditions while maintaining throughput within 20–30% of vLLM. It further improves latency by 3–17% over its heuristic teacher while preserving 100% feasibility. Continuous batching remains superior on high-bandwidth fabrics. These results identify bandwidth as a first-order determinant of optimal scheduling and demonstrate that learned schedulers can unify heuristic feasibility with adaptive, resource-aware optimization.

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
