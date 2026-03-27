# SGLang

高性能LLM和多模态模型服务框架。

## 项目信息

- **GitHub**: [sgl-project/sglang](https://github.com/sgl-project/sglang)
- **Stars**: 14.3k+
- **License**: Apache 2.0

## 简介

SGLang是一个高性能的LLM服务框架，被用于生产环境中每日处理数万亿个token。它提供了以下核心功能：

- **RadixAttention**: 前缀缓存优化，实现高达5倍的推理加速
- **PD分离架构**: 支持预填充和解码的分离部署
- **大规模专家并行(EP)**: 支持DeepSeek等MoE模型的大规模部署
- **压缩有限状态机**: JSON解码加速3倍
- **多模态支持**: 支持图像和视频的LLaVA-OneVision
- **多后端支持**: NVIDIA、AMD、TPU (JAX后端)

## 最新更新 (2026)

- 2026/02: 在NVIDIA GB300 NVL72上实现25倍推理性能提升
- 2026/01: SGLang Diffusion加速视频和图像生成
- 2025/12: 为最新开源模型提供Day-0支持

## 关键特性

1. **高吞吐量**: 针对生产环境优化
2. **PD分离部署**: 支持预填充-解码分离的分布式部署
3. **Expert Parallelism**: 支持大规模MoE模型部署
4. **多硬件支持**: NVIDIA、AMD、TPU
5. **结构化输出**: 高效的JSON/结构化输出支持

## 文档

- [官网](https://sgl-project.github.io/)
- [文档](https://docs.sglang.io/)
- [Blog](https://lmsys.org/blog/)

---

*收集日期: 2026-03-27*