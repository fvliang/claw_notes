# B-PASTE: Beam-Aware Pattern-Guided Speculative Execution for Resource-Constrained LLM Agents

**ArXiv ID:** 2604.16469
**Published:** 2026-04-09
**Authors:** Yanfei Song
**URL:** https://arxiv.org/abs/2604.16469
**PDF:** https://arxiv.org/pdf/2604.16469
**GitHub:** 暂无
**Categories:** cs.DC, cs.AI

## Abstract (English)

LLM agents execute in an interleaved reasoning-and-action loop, where future tool calls cannot be launched until the current reasoning step completes. This serial dependency inflates end-to-end latency and leaves the model idle while waiting for tool execution. Prior work, Pattern-Aware Speculative Tool Execution (PASTE), mitigates this bottleneck by speculating likely future tool invocations from mined control-flow and data-flow regularities. However, PASTE is tool-centric and speculates only individual invocations rather than bounded future branches.   We propose B-PASTE, a beam-aware extension that lifts speculation from single tools to local branch hypotheses under strict resource constraints. B-PASTE maintains a bounded beam of future execution subgraphs, ranks them by expected critical-path reduction rather than raw execution probability, and schedules only high-value branch prefixes on transient slack resources. It explicitly models co-run interference, downstream unlock value, and state-safety constraints, enabling the system to prioritize serial fast-path execution when early completion unlocks valuable future work, while still exploiting safe parallelism under low contention.   This design is especially important for edge-side deployments, where speculative work must not steal scarce resources from latency-critical authoritative execution. Preliminary internal testing on Thor-class edge environments shows up to 1.4X end-to-end speedup, suggesting that branch-aware speculative execution remains effective even under tight resource budgets.

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
