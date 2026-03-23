# SNIP: An Adaptive Mixed Precision Framework for Subbyte Large Language Model Training

**论文链接**: [arXiv:2602.01410](https://arxiv.org/abs/2602.01410)

**作者**: Yunjie Pan, Yao Fu, Ziheng Qiao, Chengmai Mao, Yining Qi, Tianchen Du, Hongxiang Li, Lanbo Li, Chen Liang, Yong Li, Dilin Wang, Wei Liu

**会议**: ASPLOS 2026

---

## Abstract (摘要)

Training large language models (LLMs) efficiently while preserving model quality poses significant challenges, particularly with subbyte precision supported by state-of-the-art GPUs. Current mixed-precision training approaches either apply uniform precision to all GEMM operations or rely on heuristic-based methods that fail to generalize during training, leading to suboptimal convergence and instability. To address these challenges, this paper introduces SNIP, a fine-grained adaptive mixed-precision training framework for LLM pretraining that supports subbyte precision.

SNIP periodically collects statistics on activations, gradients, and optimizer states to assess the precision loss impact on model quality. We define two key metrics:

- **Loss divergence in the forward pass**: Caused by quantization-induced increases in training loss
- **Weight divergence in the backward pass**: Measures error propagation through gradients affecting model updates

These metrics guide an Integer Linear Programming (ILP) problem that systematically optimizes layerwise precision to minimize overall quality loss while meeting efficiency targets.

Experiments on 1B, 3B, 7B and 70B Llama-like models demonstrate that SNIP consistently outperforms existing baselines, reducing FLOPs by up to 80% while preserving model quality across different model sizes and training phases with minimal computational overhead.

---

在保持模型质量的同时高效训练大型语言模型（LLM）提出了重大挑战，特别是支持最新GPU的子字节精度。当前的混合精度训练方法要么对所有GEMM操作应用统一精度，要么依赖基于启发式的方法，这些方法在训练期间无法泛化，导致次优收敛和不稳定性。为了应对这些挑战，本文介绍了SNIP，这是一个用于LLM预训练的细粒度自适应混合精度训练框架，支持子字节精度。

SNIP定期收集激活、梯度和优化器状态的统计数据，以评估精度损失对模型质量的影响。我们定义了两个关键指标：

- **前向传播中的损失发散**：由量化导致的训练损失增加引起
- **后向传播中的权重发散**：通过影响模型更新的梯度测量误差传播

这些指标指导整数线性规划（ILP）问题，系统地优化逐层精度，以在满足效率目标的同时最小化整体质量损失。

在1B、3B、7B和70B Llama类模型上的实验表明，SNIP始终优于现有基线，在不同模型大小和训练阶段以最小计算开销保持模型质量的同时，最多减少80%的FLOP。

---

## 主要贡献

1. **细粒度自适应混合精度**：逐层优化精度
2. **双关键指标**：
   - 前向传播损失发散
   - 后向传播权重发散
3. **ILP优化**：系统地优化精度分配

---

## 实验结果

- 在1B, 3B, 7B, 70B模型上验证
- **最多减少80% FLOPs**
- 保持模型质量