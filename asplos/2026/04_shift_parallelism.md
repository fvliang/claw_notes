# Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads

## 论文信息

- **作者**: Mert Hidayetoglu, Aurick Qiao, Michael Wyatt, Jeff Rasley, Yuxiong He, Samyam Rajbhandari
- **机构**: Snowflake
- **会议**: ASPLOS 2026
- **日期**: 2026年3月24-26日

## 原文链接

- **会议链接**: https://asplos-conference.org/asplos2026/program/

## 摘要 (Abstract)

Shift Parallelism提出了一种新的并行策略，用于在动态工作负载下实现低延迟和高吞吐量的LLM推理。该方法通过创新的请求调度和计算分配，实现了在变化负载下的稳定性能。

## 引言 (Introduction)

动态工作负载是LLM serving的常见场景，请求速率和复杂度随时间变化。传统方法难以适应这种变化，导致：
- 高延迟峰值
- 资源浪费
- 服务质量不稳定

Shift Parallelism通过动态调整并行度来适应工作负载变化。