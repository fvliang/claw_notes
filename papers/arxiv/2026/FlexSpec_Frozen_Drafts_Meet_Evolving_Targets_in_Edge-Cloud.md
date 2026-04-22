# FlexSpec: Frozen Drafts Meet Evolving Targets in Edge-Cloud Collaborative LLM Speculative Decoding

**ArXiv ID:** 2601.00644
**Published:** 2026-01-02
**Authors:** Yuchen Li, Rui Kong, Zhonghao Lyu, Qiyang Li, Xinran Chen, Hengyi Cai, Lingyong Yan, Shuaiqiang Wang, Jiashu Zhao, Guangxu Zhu, Linghe Kong, Guihai Chen, Haoyi Xiong, Dawei Yin
**URL:** https://arxiv.org/abs/2601.00644
**PDF:** https://arxiv.org/pdf/2601.00644
**GitHub:** 暂无
**Fields:** Computer Science

## Abstract (English)

Deploying large language models (LLMs) in mobile and edge computing environments is constrained by limited on-device resources, scarce wireless bandwidth, and frequent model evolution. Although edge-cloud collaborative inference with speculative decoding (SD) can reduce end-to-end latency by executing a lightweight draft model at the edge and verifying it with a cloud-side target model, existing frameworks fundamentally rely on tight coupling between the two models. Consequently, repeated model synchronization introduces excessive communication overhead, increasing end-to-end latency, and ultimately limiting the scalability of SD in edge environments. To address these limitations, we propose FlexSpec, a communication-efficient collaborative inference framework tailored for evolving edge-cloud systems. The core design of FlexSpec is a shared-backbone architecture that allows a single and static edge-side draft model to remain compatible with a large family of evolving cloud-side target models. By decoupling edge deployment from cloud-side model updates, FlexSpec eliminates the need for edge-side retraining or repeated model downloads, substantially reducing communication and maintenance costs. Furthermore, to accommodate time-varying wireless conditions and heterogeneous device constraints, we develop a channel-aware adaptive speculation mechanism that dynamically adjusts the speculative draft length based on real-time channel state information and device energy budgets. Extensive experiments demonstrate that FlexSpec achieves superior performance compared to conventional SD approaches in terms of inference efficiency.

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
