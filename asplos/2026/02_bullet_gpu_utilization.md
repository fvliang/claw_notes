# Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration

## 论文信息

- **作者**: Zejia Lin, Hongxin Xu, Guanyi Chen, Zhiguang Chen, Yutong Lu, Xianwei Zhang
- **机构**: Sun Yat-sen University
- **会议**: ASPLOS 2026
- **日期**: 2026年3月24-26日

## 原文链接

- **会议链接**: https://asplos-conference.org/asplos2026/program/

## 摘要 (Abstract)

本文提出Bullet系统，通过动态空间-时间协调来提升LLM serving的GPU利用率。传统的LLM serving系统存在GPU计算资源浪费的问题，Bullet通过创新的调度策略实现了更高效的GPU资源利用。

## 引言 (Introduction)

在LLM serving场景中，GPU利用率低下是一个普遍问题。原因是：
- 请求的动态到达模式
- prefill和decode阶段的计算特性差异
- 批处理调度不当

Bullet系统通过空间（多个GPU之间）和时间（不同请求之间）的动态协调，显著提升了GPU利用率。