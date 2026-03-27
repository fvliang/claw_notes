# vLLM Speculators: Unified Speculative Decoding Library

## 项目信息

- **项目名称**: vLLM Speculators
- **仓库**: [vllm-project/speculators](https://github.com/vllm-project/speculators)
- **语言**: Python
- **星标**: 305
- **更新频率**: 活跃 (2小时前更新)

## 简介

A unified library for building, evaluating, and storing speculative decoding algorithms for LLM inference in vLLM

## 主要特性

- **统一接口**: 为各种投机解码算法提供统一接口
- **算法实现**: 包含多种SOTA投机解码算法
- **vLLM集成**: 原生集成到vLLM推理框架
- **可扩展性**: 易于添加新的投机解码方法

## 支持的算法

- 标准投机解码 (Speculative Decoding)
- 树形投机解码 (Tree Speculative Decoding)
- 提前退出 (Early Exit)
- 自投机解码 (Self-Speculative Decoding)
- 其他SOTA方法

## 安装

```bash
pip install vllm
# speclators作为vLLM的一部分自动安装
```

## 使用示例

```python
from vllm import LLM
from vllm.speculative_decoding import speculative_model

# 使用投机解码
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",  # 使用相同的模型作为草稿
    speculative_model_quantization="awq",
    max_num_seqs=256,
    max_model_len=1024,
)
```

## 应用场景

- 加速LLM推理
- 降低延迟
- 提高吞吐量
- 生产级LLM服务

## 文档

- [vLLM官方文档](https://docs.vllm.ai/)
- [Speculative Decoding指南](https://docs.vllm.ai/en/latest/serving/serving-with-sglang.html)