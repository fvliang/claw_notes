# FlashInfer: Kernel Library for LLM Serving

## 原文链接
- GitHub: https://github.com/flashinfer-ai/flashinfer
- Stars: 5.2k
- 文档: https://docs.flashinfer.ai
- 官方博客: https://flashinfer.ai

## 概述
FlashInfer is a library and kernel generator for inference that delivers state-of-the-art performance across diverse GPU architectures. It provides unified APIs for attention, GEMM, and MoE operations with multiple backend implementations including FlashAttention-2/3, cuDNN, CUTLASS, and TensorRT-LLM.

## 主要特性

### 注意力机制 (Attention)
- 状态-of-the-art 性能：针对prefill、decode和混合批处理场景优化的内核
- 多个后端：自动为硬件和工作负载选择最佳后端
- 现代架构支持：从SM75 (Turing)到Blackwell的支持
- 低精度计算：FP8和FP4量化支持
- 生产就绪：CUDAGraph和torch.compile兼容
- Paged和Ragged KV-Cache：动态批处理服务的高效内存管理
- Decode、Prefill和Append：针对所有注意力阶段优化
- MLA Attention：DeepSeek的多潜注意力原生支持
- Cascade Attention：用于共享前缀的内存高效分层KV-Cache
- Sparse Attention：块稀疏和可变块稀疏模式
- POD-Attention：混合批处理的融合prefill+decode

### GEMM (矩阵乘法)
- BF16 GEMM：针对SM10.0+ GPU
- FP8 GEMM：per-tensor和groupwise缩放
- FP4 GEMM：NVFP4和MXFP4矩阵乘法（Blackwell GPU）
- Grouped GEMM：针对LoRA和多专家路由的高效批处理矩阵运算

### MoE (混合专家)
- 融合MoE内核
- 多种路由方法：DeepSeek-V3、Llama-4和标准top-k路由
- 量化MoE：FP8和FP4专家权重，带块级缩放

### 其他
- Sorting-Free Sampling：高效的Top-K、Top-P和Min-P，无需排序
- Speculative Decoding：链式投机采样支持
- AllReduce：自定义实现
- 多节点NVLink：MNNVL支持多节点推理
- NVSHMEM集成：用于分布式内存操作
- RoPE：LLaMA风格旋转位置嵌入（包括LLaMA 3.1）
- Normalization：RMSNorm、LayerNorm、Gemma风格融合操作
- Activations：SiLU、GELU与融合门控

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
pip install flashinfer-python
```

或安装完整包：
```bash
pip install flashinfer-python flashinfer-cubin
# JIT cache (替换cu129为你的CUDA版本)
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu129
```

## 应用领域
FlashInfer为以下项目提供推理支持：
- SGLang
- vLLM
- TensorRT-LLM
- TGI (Text Generation Inference)
- MLC-LLM
- LightLLM
- lorax
- ScaleLLM

## 引用
如果你在研究中使用FlashInfer，请引用我们的论文：

```
@article{ye2025flashinfer,
 title = {FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving},
 author = {
   Ye, Zihao and
   Chen, Lequn and
   Lai, Ruihang and
   Lin, Wuwei and
   Zhang, Yineng and
   Wang, Stephanie and
   Chen, Tianqi and
   Kasikci, Baris and
   Grover, Vinod and
   Krishnamurthy, Arvind and
   Ceze, Luis
 },
 journal = {arXiv preprint arXiv:2501.01005},
 year = {2025},
 url = {https://arxiv.org/abs/2501.01005}
}
```

---

*本文档由自动化任务生成于 2026-03-24*