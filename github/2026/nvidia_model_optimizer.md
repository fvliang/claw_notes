# NVIDIA Model Optimizer

## 项目信息

- **项目名称**: NVIDIA Model Optimizer
- **仓库**: [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- **语言**: Python
- **星标**: 2.3k
- **更新频率**: 活跃 (7小时前更新)

## 简介

A unified library of SOTA model optimization techniques like quantization, pruning, distillation, speculative decoding, etc. It compresses LLMs with minimal accuracy loss and maximizes serving throughput.

## 主要特性

- **量化 (Quantization)**
  - INT8/INT4 量化
  - AWQ, GPTQ, FP8 量化
  - 训练后量化 (PTQ)

- **剪枝 (Pruning)**
  - 结构化剪枝
  - 非结构化剪枝
  - 动态剪枝

- **蒸馏 (Distillation)**
  - 知识蒸馏
  - 任务蒸馏

- **投机解码 (Speculative Decoding)**
  - 集成多种SOTA方法
  - 减少推理延迟

## 支持的模型

- LLaMA系列
- Mistral系列
- Qwen系列
- Baichuan系列
- 其他主流LLM

## 安装

```bash
pip install nvidia-model-optimizer
```

## 使用示例

```python
import torch
from model_optimizer import optimize_model

# 加载模型
model = YourModel()

# 优化模型 (量化)
optimized_model = optimize_model(
    model,
    technique="quantization",
    quantization_method="awq",
    bits=4,
)

# 使用优化后的模型进行推理
output = optimized_model(input_ids)
```

## 性能提升

- 推理速度提升: 最高4倍
- 内存占用减少: 最高4倍
- 精度损失: <1%

## 文档

- [官方文档](https://docs.nvidia.com/model-optimizer/)
- [GitHub README](https://github.com/NVIDIA/Model-Optimizer)