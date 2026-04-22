# RAP: KV-Cache Compression via RoPE-Aligned Pruning

**ArXiv ID:** 2602.02599
**Published:** 2026-02-01
**Authors:** Jihao Xin, Tian Lyu, David E. Keyes, H. Ltaief, Marco Canini
**URL:** https://arxiv.org/abs/2602.02599
**PDF:** https://arxiv.org/pdf/2602.02599
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Long-context inference in large language models is increasingly bottlenecked by the memory and compute cost of the KV-Cache. Low-rank factorization compresses KV projections by writing $W \approx A * B$, where A produces latent KV states and B can be absorbed into downstream weights. In modern RoPE-based LLMs, this absorption fails: RoPE forces latent KV states to be reconstructed to full dimension, reintroducing substantial memory and compute overhead. We propose RoPE-Aligned Pruning (RAP), which prunes entire RoPE-aligned column pairs to preserve RoPE's 2x2 rotation structure, restore B absorption, and eliminate reconstruction. Our evaluation on LLaMA-3-8B and Mistral-7B shows that RAP enables joint reduction of KV-Cache, attention parameters, and FLOPs by 20-30%, all at once, while maintaining strong accuracy. Notably, RAP reduces attention latency to 83% (prefill) and 77% (decode) of baseline.

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
