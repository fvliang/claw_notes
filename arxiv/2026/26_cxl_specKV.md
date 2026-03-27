# CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving

## 论文信息

- **标题**: CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving
- **作者**: Dong Liu, Yanxuan Yu
- **arXiv**: [arXiv:2512.11920](https://arxiv.org/abs/2512.11920)
- **提交日期**: 2025年12月11日
- **领域**: Artificial Intelligence (cs.AI)
- **会议**: FPGA'26 Oral

## 摘要 (Abstract)

Large Language Models (LLMs) have revolutionized natural language processing tasks, but their deployment in datacenter environments faces significant challenges due to the massive memory requirements of key-value (KV) caches. During the autoregressive decoding process, KV caches consume substantial GPU memory, limiting batch sizes and overall system throughput. To address these challenges, we propose **CXL-SpecKV**, a novel disaggregated KV-cache architecture that leverages Compute Express Link (CXL) interconnects and FPGA accelerators to enable efficient speculative execution and memory disaggregation. Our approach introduces three key innovations: (i) a CXL-based memory disaggregation framework that offloads KV-caches to remote FPGA memory with low latency, (ii) a speculative KV-cache prefetching mechanism that predicts and preloads future tokens' cache entries, and (iii) an FPGA-accelerated KV-cache compression and decompression engine that reduces memory bandwidth requirements by up to 4×. When evaluated on state-of-the-art LLM models, CXL-SpecKV achieves up to 3.2× higher throughput compared to GPU-only baselines, while reducing memory costs by 2.8× and maintaining accuracy. Our system demonstrates that intelligent memory disaggregation combined with speculative execution can effectively address the memory wall challenge in large-scale LLM serving. Our code implementation has been open-sourced at [this https URL](https://github.com/FastLM/CXL-SpecKV).

## 摘要 (中文)

大型语言模型(LLM)已经彻底改变了自然语言处理任务，但其在数据中心环境中的部署面临着重大的内存挑战，因为需要大量的键值(KV)缓存。在自回归解码过程中，KV缓存消耗大量GPU内存，限制了批处理大小和整体系统吞吐量。为了应对这些挑战，我们提出了CXL-SpecKV，这是一种新颖的解耦KV缓存架构，利用Compute Express Link (CXL)互连和FPGA加速器来实现高效的投机执行和内存解耦。我们的方法引入了三个关键创新：(i)基于CXL的内存解耦框架，以低延迟将KV缓存卸载到远程FPGA内存，(ii)投机性KV缓存预取机制，预测并预加载未来标记的缓存条目，以及(iii)FPGA加速的KV缓存压缩和解压缩引擎，将内存带宽需求减少高达4倍。在最先进的LLM模型上进行评估时，CXL-SpecKV与仅GPU基线相比实现了高达3.2倍的吞吐量提升，同时将内存成本降低2.8倍并保持准确性。我们的系统表明，智能内存解耦与投机执行相结合可以有效解决大规模LLM服务中的内存墙挑战。

## 引言 (Introduction)

LLM的部署面临以下挑战：

1. **KV缓存内存巨大**：随着序列长度增加，内存需求呈二次增长
2. **批处理大小受限**：GPU内存限制影响系统吞吐量
3. **内存墙问题**：带宽成为瓶颈

CXL-SpecKV提出以下解决方案：
- **CXL内存解耦**：利用CXL互连将KV缓存卸载到远程FPGA
- **投机预取**：预测未来标记并预加载缓存
- **FPGA压缩引擎**：减少带宽需求4倍

## 实验结果

- 吞吐量提升：最高3.2倍
- 内存成本降低：2.8倍
- 准确性：保持不变

## GitHub

- 仓库: [FastLM/CXL-SpecKV](https://github.com/FastLM/CXL-SpecKV)
- 状态: 已开源