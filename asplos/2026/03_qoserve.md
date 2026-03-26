# QoServe: Breaking the Silos of LLM Inference Serving

## 论文信息

- **作者**: Kanishk Goel, Jayashree Mohan, Nipun Kwatra, Ravi Anupindi, Ram Ramjee
- **机构**: Microsoft Research India
- **会议**: ASPLOS 2026
- **日期**: 2026年3月24-26日

## 原文链接

- **会议链接**: https://asplos-conference.org/asplos2026/program/

## 摘要 (Abstract)

QoServe是一个统一的LLM推理服务框架，打破了传统系统中的"孤岛"设计。传统LLM serving系统针对特定场景优化，导致系统碎片化。QoServe通过创新的架构设计，实现了跨场景的高质量服务。

## 引言 (Introduction)

现有的LLM serving系统通常针对特定场景进行优化：
- 延迟优化系统
- 吞吐量优化系统
- 成本优化系统

这种碎片化导致：
1. 系统维护困难
2. 资源利用效率低
3. 难以保证服务质量

QoServe通过统一调度和资源管理解决了这些问题。