# Towards High-Goodput LLM Serving with Prefill-decode Multiplexing

## 论文信息

- **作者**: Weihao Cui, Yukang Chen, Han Zhao, Ziyi Xu, Xiaoze Fan, Xusheng Chen, Yangjie Zhou, Shixuan Sun, Bingsheng He, Quan Chen
- **机构**: Shanghai Jiao Tong University, University of Hong Kong, National University of Singapore
- **会议**: ASPLOS 2026
- **日期**: 2026年3月24-26日

## 原文链接

- **arXiv**: (待补充)
- **会议链接**: https://asplos-conference.org/asplos2026/program/

## 摘要 (Abstract)

现代大型语言模型(LLM)的部署需要同时优化延迟和吞吐量。传统的LLM serving系统通常将prefill阶段和解码阶段分开处理，但这导致了资源利用不均衡的问题。本文提出了Prefill-Decode Multiplexing (PDM) 框架，通过创新的请求调度和资源分配策略，实现了更高的goodput（服务质量感知的吞吐量）。

## 引言 (Introduction)

LLM serving面临的核心挑战是如何在多用户场景下同时满足延迟和吞吐量需求。现有系统通常采用分阶段处理方式，将prefill（提示处理）和decode（ token生成）分离到不同的GPU上。然而，这种方法导致了：
1. GPU资源利用不均
2. 请求排队延迟增加
3. 整体系统goodput下降

本文提出的PDM框架通过创新的请求多路复用技术，有效解决了上述问题。

## 贡献

1. 提出Prefill-Decode Multiplexing框架
2. 设计了动态资源分配算法
3. 在真实 workloads 上验证了系统性能提升

## 实验结果

(待补充详细数据)