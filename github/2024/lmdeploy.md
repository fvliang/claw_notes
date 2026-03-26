# LMDeploy

## 项目信息

- **项目名称**: LMDeploy
- **GitHub**: https://github.com/InternLM/lmdeploy
- **Stars**: 7.7k+
- **语言**: Python
- **最新更新**: 8小时前
- **维护者**: InternLM (上海人工智能实验室)

## 简介

LMDeploy是一个用于压缩、部署和服务LLM的工具包，提供高效的推理性能。

## 核心特性

### 推理引擎
- **TurboMind**: 自研高性能推理引擎
- **PyTorchEngine**: 纯Python开发的推理引擎，便于快速实验

### 量化支持
- W4A16 (4-bit weight, 16-bit activation)
- INT8/INT4 KV Cache量化
- AWQ量化
- FP8 MoE模型优化
- MXFP4量化 (V100起，支持H800上1.5倍性能提升)

### 部署特性
- 多模型、多机、多卡推理服务
- VLM (视觉语言模型) 部署支持
- DeepSeek PD (Prefill-Decode) 分离部署
- Prefix Caching
- Paged Attention
- Split-K Decoding (Flash Decoding)
- CUDA Graph加速

### 支持的模型
- LLaMA系列 (LLaMA2, LLaMA3, LLaMA3.1)
- Qwen系列 (Qwen1.5, Qwen2, Qwen3)
- InternLM系列
- DeepSeek系列 (V2, V3, R1)
- Mistral, Mixtral
- 多模态模型 (InternVL, LLaVA, etc.)

### 平台支持
- NVIDIA GPU
- Huawei Ascend

## 更新日志

### 2026
- [2026/02] 支持Qwen3.5
- [2026/02] 支持vllm-project/llm-compressor 4bit对称/非对称量化

### 2025
- [2025/09] TurboMind支持MXFP4，在H800上实现1.5倍性能提升
- [2025/06] FP8 MoE模型全面优化
- [2025/06] 支持DeepSeek PD分离部署
- [2025/04] 集成DeepSeek推理优化技术 (FlashMLA, DeepGemm, DeepEP等)
- [2025/01] 支持DeepSeek V3和R1

### 2024
- [2024/11] 支持Mono-InternVL
- [2024/10] PyTorchEngine支持图模式 (Ascend平台)
- [2024/09] 支持Huawei Ascend
- [2024/07] 支持LLaMA3.1, InternVL2, InternLM-XComposer2.5
- [2024/05] VLM 4-bit weight-only量化
- [2024/04] KV Cache在线量化
- [2024/02] 支持Qwen1.5, Gemma, Mistral, Mixtral等

## 安装

```bash
pip install lmdeploy
```

## 使用示例

### API Server
```bash
lmdeploy serve api_server /path/to/llama模型
```

### 命令行推理
```bash
lmdeploy chat /path/to/llama模型
```

### 量化
```bash
lmdeploy lite 量化模型
```

## 相关资源

- [文档](https://lmdeploy.readthedocs.io/)
- [HuggingFace Hub](https://huggingface.co/lmdeploy)
- [OpenAOE](https://github.com/InternLM/OpenAOE) - Web UI