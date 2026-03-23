# SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs

**论文链接**: [arXiv:2512.00722](https://arxiv.org/abs/2512.00722)

**作者**: Jiaming Xu, Hong Cao, Yuhan Lin, Jinyang Li, Zheng Liu, Jie Liu, Xingyu Li, Jin Wang, Jingyuan Jia, Ge Li

**会议**: ASPLOS 2026

---

## Abstract (摘要)

In this paper, we point out that the objective of the retrieval algorithms is to align with the LLM, which is similar to the objective of knowledge distillation in LLMs. We analyze the similarity in information focus between the distilled language model (DLM) and the original LLM from the perspective of information theory, and thus propose a novel paradigm that leverages a DLM as the retrieval algorithm. Based on the insight, we present SpeContext, an algorithm and system co-design for long-context reasoning.

1. **At the algorithm level**: SpeContext proposes lightweight retrieval head based on the head-level attention weights of DLM, achieving >90% parameters reduction by pruning the redundancy.

2. **At the system level**: SpeContext designs an asynchronous prefetch dataflow via the elastic loading strategy, effectively overlapping KV cache retrieval with the LLM computation.

3. **At the compilation level**: SpeContext constructs the theoretical memory model and implements an adaptive memory management system to achieve acceleration by maximizing GPU memory utilization.

We deploy and evaluate SpeContext in two resource-constrained environments, cloud and edge. Extensive experiments show that, compared with the Huggingface framework, SpeContext achieves up to 24.89x throughput improvement in cloud and 10.06x speedup in edge with negligible accuracy loss, pushing the Pareto frontier of accuracy and throughput.

---

在本文中，我们指出检索算法的目标是与LLM对齐，这类似于LLM中知识蒸馏的目标。我们从信息论的角度分析了蒸馏语言模型（DLM）和原始LLM之间信息聚焦的相似性，从而提出了一种利用DLM作为检索算法的新范式。基于这一洞察，我们提出了SpeContext，一个用于长上下文推理的算法和系统协同设计。

1. **算法层面**：SpeContext提出了基于DLM头级注意力权重的轻量级检索头，通过剪枝冗余实现了超过90%的参数减少。

2. **系统层面**：SpeContext通过弹性加载策略设计了异步预取数据流，有效地将KV缓存检索与LLM计算重叠。

3. **编译层面**：SpeContext构建了理论内存模型并实现了自适应内存管理系统，以通过最大化GPU内存利用率来实现加速。

我们在云和边缘两种资源受限环境中部署和评估SpeContext。大量实验表明，与Huggingface框架相比，SpeContext在云端实现了高达24.89倍的吞吐量提升，在边缘实现了10.06倍的加速，而精度损失可忽略不计，推动了精度和吞吐量的帕累托前沿。

---

## 主要贡献

1. **轻量级检索头**：基于蒸馏语言模型（DLM）头级注意力权重，实现>90%参数减少
2. **异步预取数据流**：通过弹性加载策略重叠KV缓存检索与LLM计算
3. **自适应内存管理**：最大化GPU内存利用率

---

## 实验结果

| 环境 | 吞吐量提升 |
|------|------------|
| 云端 | 24.89× |
| 边缘 | 10.06× |

- 精度损失可忽略不计