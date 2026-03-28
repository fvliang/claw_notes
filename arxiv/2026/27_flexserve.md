# FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation

## 论文信息

- **作者**: Yinpeng Wu, Yitong Chen, Lixiang Wang, Jinyu Gu, Zhichao Hua, Yubin Xia
- **机构**: (待补充)
- **arXiv**: [2603.09046](https://arxiv.org/abs/2603.09046)
- **提交日期**: 2026年3月9日
- **领域**: Cryptography and Security (cs.CR); Machine Learning (cs.LG); Operating Systems (cs.OS)

## 摘要 (Abstract)

Device-side Large Language Models (LLMs) have witnessed explosive growth, offering higher privacy and availability compared to cloud-side LLMs. During LLM inference, both model weights and user data are valuable, and attackers may even compromise the OS kernel to steal them. ARM TrustZone is the de facto hardware-based isolation technology on mobile devices, used to protect sensitive applications from a compromised OS. However, protecting LLM inference with TrustZone incurs significant overhead due to its inflexible isolation of memory and the NPU. To address these challenges, this paper introduces FlexServe, a fast and secure LLM serving system for mobile devices. It first introduces a Flexible Resource Isolation mechanism to construct Flexible Secure Memory (Flex-Mem) and Flexible Secure NPU (Flex-NPU). Both memory pages and the NPU can be efficiently switched between unprotected and protected modes. Based on these mechanisms, FlexServe designs a fast and secure LLM inference framework within TrustZone's secure world. The LLM-Aware Memory Management and Secure Inference Pipeline are introduced to accelerate inference. A Multi-Model Scheduler is proposed to optimize multi-model workflows. We implement a prototype of FlexServe and compare it with two TrustZone-based strawman designs. The results show that FlexServe achieves an average 10.05× speedup in Time to First Token (TTFT) compared to the strawman, and an average 2.44× TTFT speedup compared to an optimized strawman with pipeline and secure NPU enabled. For multi-model agent workflows, the end-to-end speedup is up to 24.30× and 4.05× compared to the strawman and optimized strawman, respectively.

## 摘要 (中文)

设备端大语言模型(LLM)近年来呈现爆发式增长，相比云端LLM具有更高的隐私性和可用性。在LLM推理过程中，模型权重和用户数据都具有很高价值，攻击者甚至可能入侵操作系统内核来窃取这些数据。ARM TrustZone是移动设备上事实上的基于硬件的隔离技术，用于保护敏感应用免受被入侵操作系统的攻击。然而，使用TrustZone保护LLM推理会因其对内存和NPU的僵化隔离而产生显著开销。为了应对这些挑战，本文提出了FlexServe，一种用于移动设备的快速安全LLM服务系统。它首先引入了灵活资源隔离机制来构建灵活安全内存(Flex-Mem)和灵活安全NPU(Flex-NPU)。内存页面和NPU都可以在非保护模式和保护模式之间高效切换。基于这些机制，FlexServe在TrustZone的安全世界中设计了快速安全的LLM推理框架。引入了LLM感知的内存管理和安全推理管道来加速推理。提出了多模型调度器来优化多模型工作流。我们实现了FlexServe的原型，并与两种基于TrustZone的朴素设计进行了比较。结果表明，FlexServe在首 token 时间(TTFT)上相比朴素设计平均实现了10.05倍的加速，相比启用了管道和安全NPU的优化朴素设计平均实现了2.44倍的TTFT加速。对于多模型代理工作流，端到端加速分别高达24.30倍和4.05倍。

## 引言 (Introduction)

随着移动设备上LLM的快速发展，设备端推理成为趋势。然而，设备端LLM面临安全和隐私挑战。ARM TrustZone技术可用于保护敏感应用，但传统TrustZone实现存在显著性能开销。FlexServe通过创新的灵活资源隔离机制解决了这一问题。

## GitHub

暂无官方GitHub仓库。

## 博客

暂无公开博客。