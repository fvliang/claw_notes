# FlashInfer

## 项目信息

- **项目名称**: FlashInfer
- **GitHub**: https://github.com/flashinfer-ai/flashinfer
- **Stars**: 5.2k+
- **语言**: Python
- **最新更新**: 1小时前

## 简介

FlashInfer是一个为推理提供卓越性能的库和内核生成器，在各种GPU架构上提供最先进的性能。

## 核心特性

### 注意力机制 (Attention)
- 状态-of-the-art性能：针对prefill、decode和混合批处理场景优化的内核
- 多种后端：自动选择最佳后端 (FlashAttention-2/3, cuDNN, CUTLASS, TensorRT-LLM)
- 现代架构支持：支持SM75 (Turing) 到 Blackwell
- 低精度计算：FP8和FP4量化
- 生产就绪：兼容CUDAGraph和torch.compile
- Paged和Ragged KV-Cache
- Decode、Prefill和Append优化
- MLA Attention：原生支持DeepSeek的多潜伏注意力
- Cascade Attention：共享前缀的内存高效分层KV-Cache
- 稀疏注意力：块稀疏和可变块稀疏模式
- POD-Attention：混合批处理的融合prefill+decode

### GEMM (矩阵乘法)
- BF16 GEMM：用于SM10.0+ GPUs
- FP8 GEMM：per-tensor和groupwise scaling
- FP4 GEMM：Blackwell GPUs的NVFP4和MXFP4
- Grouped GEMM：LoRA和多专家路由的高效批处理矩阵运算

### MoE (混合专家)
- 融合MoE内核
- 多种路由方法：DeepSeek-V3, Llama-4和标准top-k路由
- 量化MoE：FP8和FP4专家权重与块级scaling

### 其他
- 排序自由采样：高效的Top-K、Top-P和Min-P
- **Speculative Decoding：链式投机采样支持**
- AllReduce：自定义实现
- 多节点NVLink：MNNVL支持多节点推理
- NVSHMEM集成：分布式内存操作
- RoPE：LLaMA风格旋转位置嵌入
- 归一化：RMSNorm, LayerNorm, Gemma风格融合操作
- 激活函数：SiLU, GELU与融合门控

## 支持的GPU架构

| 架构 | Compute Capability | 示例GPU |
|------|-------------------|---------|
| Turing | SM 7.5 | T4, RTX 20系列 |
| Ampere | SM 8.0, 8.6 | A100, A10, RTX 30系列 |
| Ada Lovelace | SM 8.9 | L4, L40, RTX 40系列 |
| Hopper | SM 9.0 | H100, H200 |
| Blackwell | SM 10.0, 10.3 | B200, B300 |
| Blackwell | SM 12.0, 12.1 | RTX 50系列, DGX Spark, Jetson Thor |

## 安装

```bash
# 核心包
pip install flashinfer-python

# 预编译内核
pip install flashinfer-python flashinfer-cubin

# JIT缓存 (替换cu129为你的CUDA版本)
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

## 快速开始

```python
import torch
import flashinfer

# 单次decode attention
q = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # [num_qo_heads, head_dim]
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)  # [kv_len, num_kv_heads, head_dim]
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

output = flashinfer.single_decode_with_kv_cache(q, k, v)
```

## 文档

- [完整文档](https://docs.flashinfer.ai/)
- [博客](https://flashinfer.ai)
- [Slack社区](https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA)

## 更新日志

- [2025-10-08] Blackwell支持 (v0.4.0)
- [2025-03-10] [博客文章] LLM采样的无排序GPU内核