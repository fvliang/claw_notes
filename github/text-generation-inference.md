# Text Generation Inference (TGI)

## 原文链接
- GitHub: https://github.com/huggingface/text-generation-inference
- 文档: https://huggingface.github.io/text-generation-inference

## 概述
Text Generation Inference (TGI) 是一个用于部署和服务大型语言模型 (LLM) 的工具包。TGI 为最流行的开源 LLM（包括 Llama、Falcon、StarCoder、BLOOM、GPT-NeoX 等）实现高性能文本生成。

## 主要特性

### 性能优化
- **Tensor Parallelism**: 在多 GPU 上加速推理
- **Continuous Batching**: 增加总吞吐量
- **Flash Attention**: 优化的注意力机制
- **Paged Attention**: 来自 vLLM 的分页注意力
- **Quantization**: 支持 bitsandbytes、GPTQ、EETQ、AWQ、Marlin、FP8

### 生产就绪
- 分布式追踪 (Open Telemetry)
- Prometheus 指标
- 消息 API (兼容 OpenAI Chat Completion API)

### 功能特性
- Token 流式输出 (Server-Sent Events)
- 量化支持
- Watermarking
- Logits warper (temperature, top-p, top-k, repetition penalty)
- Stop sequences
- Log probabilities
- Speculation (~2x 延迟改进)
- Guidance/JSON 输出格式

### 硬件支持
- NVIDIA GPU
- AMD GPU (ROCm)
- Intel GPU
- Google TPU
- AWS Inferentia
- Intel Gaudi

## 技术架构
TGI 使用 Rust、Python 和 gRPC 构建，用于文本生成推理。在生产环境中被 Hugging Face 使用，为 Hugging Chat、Inference API 和 Inference Endpoints 提供支持。

## 使用示例

```bash
# 使用 Docker 启动
model=HuggingFaceH4/zephyr-7b-beta
volume=$PWD/data

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
 ghcr.io/huggingface/text-generation-inference:3.3.5 --model-id $model

# 生成请求
curl 127.0.0.1:8080/generate_stream \
 -X POST \
 -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
 -H 'Content-Type: application/json'
```

## 与 OpenAI API 兼容的接口

```bash
curl localhost:8080/v1/chat/completions \
 -X POST \
 -d '{
 "model": "tgi",
 "messages": [
   {"role": "system", "content": "You are a helpful assistant."},
   {"role": "user", "content": "What is deep learning?"}
 ],
 "stream": true,
 "max_tokens": 20
 }' \
 -H 'Content-Type: application/json'
```

## 状态
⚠️ TGI 现已进入维护模式。未来将接受 minor bug 修复、文档改进和轻量级维护任务的 PR。推荐使用的下游推理引擎：vLLM, SGLang, llama.cpp, MLX。

## 参考博客
- [LLM inference at scale with TGI](https://www.adyen.com/knowledge-hub/llm-inference-at-scale-with-tgi) - Adyen

---

*本文档由自动化任务生成于 2026-03-24*