# NIXL: NVIDIA Inference Xfer Library

- **GitHub**: https://github.com/ai-dynamo/nixl
- **Stars**: ⭐
- **Conference/Source**: GitHub
- **Year**: 2025

## 摘要 (CN)

NIXL (NVIDIA Inference Xfer Library) 是用于加速AI推理框架中点对点通信的库，特别是为NVIDIA Dynamo等推理框架设计。NIXL提供了对各种类型内存（CPU和GPU）和存储（文件、块和对象存储）的抽象，通过模块化插件架构实现。

## 摘要 (EN)

NVIDIA Inference Xfer Library (NIXL) is targeted for accelerating point to point communications in AI inference frameworks such as NVIDIA Dynamo, while providing an abstraction over various types of memory (e.g., CPU and GPU) and storage (e.g., file, block and object store) through a modular plug-in architecture.

## 特性

- **高性能点对点通信**: 优化AI推理框架中的数据传输
- **内存抽象**: 支持CPU和GPU内存的统一抽象
- **存储抽象**: 支持文件、块和对象存储
- **模块化插件架构**: 易于扩展和定制
- **支持多种后端**: 包括UCX、GDRCopy、Mooncake等

## 安装

```bash
# CUDA 12
pip install nixl[cu12]

# CUDA 13
pip install nixl[cu13]
```

## 相关项目

- Mooncake: 支持Mooncake插件
- GDS: 支持GPU Direct Storage后端
- POSIX: 支持POSIX插件

## 文档

- [NIXL overview](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md) - Core concepts/architecture overview
- [Python API](https://github.com/ai-dynamo/nixl/blob/main/docs/python_api.md) - Python API usage and examples
- [Backend guide](https://github.com/ai-dynamo/nixl/blob/main/docs/BackendGuide.md) - Backend/plugin development guide

## 适用场景

NIXL特别适用于需要高效数据传输的LLM serving场景，可以与vLLM、SGLang等推理框架配合使用，提升分布式推理性能。