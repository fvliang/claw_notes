# llama.cpp: LLM Inference in C/C++

## 项目信息

- **GitHub**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Stars**: 99.1k+
- **语言**: C++
- **许可证**: MIT

## 简介

llama.cpp是纯C/C++实现的LLM推理框架，无需Python依赖，支持在各种硬件平台上高效运行大语言模型。

## 主要特性

1. **纯C/C++实现**
   - 无Python依赖
   - 轻量级二进制文件
   - 易于编译和部署

2. **多硬件支持**
   - CPU推理（通过GGML库）
   - CUDA GPU加速
   - Metal加速（Apple Silicon）
   - Vulkan支持

3. **量化支持**
   - 2-bit到8-bit量化
   - K-Quant, Q-Quant, IQ-Quant
   - 大幅降低内存需求

4. **模型格式**
   - GGUF格式（新版）
   - GGML格式（兼容）
   - 转换工具齐全

## 性能

- 支持Apple Silicon的Metal加速
- 高效的CPU推理
- 批处理优化

## 相关博客

- 官方README提供了详细的构建和使用说明
- 支持多种推理前端

## 使用场景

- 本地LLM部署
- 边缘设备推理
- 资源受限环境
- 快速原型开发

---

*更新时间: 2026-03-24*