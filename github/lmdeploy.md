# LMDeploy: LLM部署与服务工具包

## 原文链接
- GitHub: https://github.com/InternLM/lmdeploy
- Stars: 7.7k
- 文档: https://lmdeploy.readthedocs.io

## 概述
LMDeploy是由MMRazor和MMDeploy团队开发的用于压缩、部署和服务LLM的工具包。它具有以下核心功能：

## 主要特性

### 高效推理
LMDeploy通过引入关键特性，如持久批处理（continuous batching）、分页KV缓存、动态split&fuse、张量并行、高性能CUDA内核等，提供比vLLM高1.8倍的请求吞吐量。

### 有效量化
- 支持仅权重量化（weight-only quantization）
- 支持KV缓存量化
- 4位推理性能比FP16高2.4倍
- 量化质量已通过OpenCompass评估确认

### 轻松部署分布式服务器
利用请求分发服务，LMDeploy可以轻松高效地跨多台机器和多张显卡部署多模型服务。

### 优秀的兼容性
- 支持KV Cache量化
- 支持AWQ（Activation-aware Weight Quantization）
- 支持自动前缀缓存（Automatic Prefix Caching）
- 以上特性可同时使用

## 支持的模型

### LLM
- InternLM 系列
- Qwen 系列（包括Qwen1.5, Qwen2, Qwen2.5, Qwen3）
- LLaMA 系列（包括LLaMA2, LLaMA3, LLaMA3.1）
- DeepSeek 系列（包括DeepSeek-V2, V3, R1）
- Mistral, Mixtral
- Baichuan2
- Gemma
- 以及更多...

### VLM (视觉语言模型)
- InternVL 系列
- InternLM-XComposer 系列
- CogVLM2
- Mini-InternVL
- LlaVA-Next
- 等等

## 更新日志 (2025-2026)

- [2026/02] 支持 Qwen3.5
- [2026/02] 支持 vllm-project/llm-compressor 4bit 对称/非对称量化
- [2025/09] TurboMind 在 NVIDIA V100+ GPU 上支持 MXFP4，在 H800 上实现比 vLLM 1.5倍的性能提升
- [2025/06] FP8 MoE 模型的全面推理优化
- [2025/06] 通过集成 DLSlime 和 Mooncake 支持 DeepSeek PD 分离部署
- [2025/04] 通过集成 deepseek-ai 技术提升 DeepSeek 推理性能：FlashMLA, DeepGemm, DeepEP, MicroBatch 和 eplb
- [2025/01] 支持 DeepSeek V3 和 R1
- [2024/09] LMDeploy PyTorchEngine 支持华为 Ascend 平台
- [2024/08] 集成到 modelscope/swift 作为 VLM 推理的默认加速器
- [2024/07] 支持 InternVL2 全系列模型
- [2024/04] TurboMind 升级 GQA，internlm2-20b 推理达到 16+ RPS，比 vLLM 快 1.8 倍
- [2024/04] 支持 4-bit 权重仅量化
- [2023/11] TurboMind 重大升级：Paged Attention、无序列长度限制的更快注意力、KV8 内核快2倍、Split-K 解码（Flash Decoding）、W4A16 推理

## TurboMind 引擎特性

- Paged Attention（分页注意力）
- 连续批处理（Continuous Batching）
- 动态 Split & Fuse
- 张量并行（Tensor Parallelism）
- 高性能 CUDA 内核
- Flash Attention 2 支持
- 4-bit 推理（W4A16）
- KV Cache 量化
- 自动前缀缓存
- 多模态输入支持

## 安装

```bash
pip install lmdeploy
```

## 快速开始

### 服务 LLM
```bash
lmdeploy serve api_server /path/to/llama模型
```

### 量化模型
```bash
lmdeploy lite quantize /path/to/模型 --output-dir /path/to/量化模型
```

### 推理
```bash
lmdeploy chat /path/to/模型
```

---

*本文档由自动化任务生成于 2026-03-24*