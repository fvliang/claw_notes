# Omlx: LLM Inference Server for Apple Silicon

## 基本信息

- **仓库**: [jundot/omlx](https://github.com/jundot/omlx)
- **描述**: LLM inference server with continuous batching & SSD caching for Apple Silicon
- **语言**: Python
- **Stars**: 8004
- **更新时间**: 2026-04-01

## 主要特性

- **Apple Silicon支持**: 专门针对Apple Silicon (M系列芯片)优化
- **连续批处理**: 支持continuous batching提高吞吐量
- **SSD缓存**: 利用SSD进行KV缓存,扩展可用内存
- **OpenAI兼容**: 提供OpenAI兼容的API接口

## 原文链接

- GitHub: https://github.com/jundot/omlx

## 介绍

Omlx是为Apple Silicon设备设计的高性能LLM推理服务器。该项目利用苹果芯片的统一内存架构和Neural Engine,实现了高效的本地LLM推理。连续批处理和SSD缓存功能使其能够在内存有限的情况下处理更多的并发请求。

---