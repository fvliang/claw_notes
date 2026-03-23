# ASPLOS 2026 LLM Serving 相关论文汇总

> 论文搜集工作文件夹
> 更新日期：2026-03-23

---

## Session 1A: LLM Serving - 吞吐量优化

| 论文 | 作者单位 | arXiv | 状态 |
|------|-----------|-------|------|
| Towards High-Goodput LLM Serving with PD Multiplexing | 上海交大、NUS、港大 | 2504.14489 | ✅ 已完成 |
| Bullet: Boosting GPU Utilization for LLM Serving | 中山大学 | 2504.19516 | ✅ 已完成 |
| QoServe: Breaking the Silos of LLM Inference Serving | Microsoft Research India | - | 🔍 未找到 |
| Shift Parallelism: Low-Latency, High-Throughput LLM Inference | Snowflake | - | 🔍 未找到 |
| XY-Serve: End-to-End Versatile Production Serving | 华为、清华 | - | 🔍 未找到 |

---

## Session 1B: LLM Serving - 延迟与调度

| 论文 | 作者单位 | arXiv | 状态 |
|------|-----------|-------|------|
| PAT: Prefix-Aware Attention | 天津大学 | - | 🔍 未找到 |
| ZipServ: Hardware-Aware Lossless Compression | HKUST(广州) | 2603.17435 | ✅ 已完成 |
| BlendServe: Resource-Aware Batching | UC Berkeley、华盛顿大学 | - | 🔍 未找到 |
| BAT: Bipartite Attention | 浙大、阿里、NTU | - | 🔍 未找到 |
| MoE-APEX: Adaptive Precision Expert Offloading | 上海交大、港中大 | - | 🔍 未找到 |

---

## Session 2B: Speculative Decoding (投机解码)

| 论文 | 作者单位 | arXiv | 状态 |
|------|-----------|-------|------|
| DFVG: Draft-on-FPGA + Verify-on-GPU | 上海交大、东大 | - | 🔍 未找到 |
| SwiftSpec: Disaggregated Speculative Decoding | ByteDance、UChicago | - | 🔍 未找到 |
| SpeContext: Speculative Context Sparsity | 上海交大、清华 | 2512.00722 | ✅ 已完成 |
| SpecProto: Speculative Decoding for Protocol Buffers | UC Riverside、Google | - | 🔍 未找到 |
| EARTH: Entropy-Aware Speculative Prefetch | 上海交大、国防科大 | - | 🔍 未找到 |

---

## Session 3A: KV Cache 相关 (待补充)

(后续补充...)

---

## 已完成论文列表 (10篇)

1. 09_Towards_High-Goodput_LLM_Serving.md
2. 10_Bullet.md
3. 01_ZipServ.md
4. 02_JigsawServe.md
5. 03_Ouroboros.md
6. 04_LAER-MoE.md
7. 05_SpeContext.md
8. 06_RedFuser.md
9. 07_SNIP.md
10. 08_M2XFP.md

---

## 说明

- ✅ 已完成：已在 arXiv 找到原文，包含摘要和引言
- 🔍 未找到：未在 arXiv 找到，可能需要通过其他渠道获取
- 很多论文可能刚刚发表或即将发表，arXiv 还未上线