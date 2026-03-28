# LLM Serving 论文索引 (2026-03-29 更新)

## 🆕 2026年3月29日新增论文

### arXiv 新论文 (新增4篇)
| 论文 | 作者 | 领域 | 备注 |
|------|------|------|------|
| FlexServe: A Fast and Secure LLM Serving System for Mobile Devices | Yinpeng Wu 等 | cs.CR/OS | 移动设备安全LLM服务，TrustZone隔离 |
| SpecEyes: Accelerating Agentic Multimodal LLMs | Haoyu Huang 等 | cs.CV/CL | 代理多模态LLM投机感知与规划 |
| RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse | Yingsheng Geng 等 | cs.LG | 多代理LLM协作，KV缓存复用 |
| CacheSolidarity: Preventing Prefix Caching Side Channels | Panagiotis Pennas 等 | cs.CR/DC | 多租户LLM服务安全 |

### GitHub 新项目 (新增1个)
| 项目 | 描述 | Stars |
|------|------|-------|
| [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) | 投机解码训练框架，可平滑移植到SGLang | - |

---

## 🆕 2026年3月28日新增论文

### arXiv 新论文 (新增4篇)
| 论文 | 作者 | 领域 | 备注 |
|------|------|------|------|
| Nightjar: Dynamic Adaptive Speculative Decoding | Rui Li 等 | cs.DC | 自适应投机框架，动态调整投机长度 |
| CXL-SpecKV: Disaggregated FPGA Speculative KV-Cache | Dong Liu 等 (FPGA'26 Oral) | cs.AI | CXL内存解耦，FPGA加速 |
| ReSpec: Optimizing Speculative Decoding in RL Systems | Qiaoling Chen 等 | cs.LG | RL训练中的投机解码优化 |

### GitHub 新项目 (新增3个)
| 项目 | 描述 | Stars |
|------|------|-------|
| [vllm-project/speculators](https://github.com/vllm-project/speculators) | vLLM统一投机解码库 | 305 |
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | SOTA模型优化技术集(量化/剪枝/蒸馏/投机解码) | 2.3k |
| [xlite-dev/Awesome-LLM-Inference](https://github.com/xlite-dev/Awesome-LLM-Inference) | 精选LLM推理论文列表 | 5.1k |

---

## 🆕 2026年3月26日新增论文

### ASPLOS 2026 (新增10篇)
| 论文 | 作者 | 领域 |
|------|------|------|
| Towards High-Goodput LLM Serving with Prefill-decode Multiplexing | Weihao Cui 等 | LLM Serving: 吞吐量优化 |
| Bullet: Boosting GPU Utilization for LLM Serving | Zejia Lin 等 | GPU利用率优化 |
| QoServe: Breaking the Silos of LLM Inference Serving | Kanishk Goel 等 (Microsoft) | 统一服务框架 |
| Shift Parallelism: Low-Latency, High-Throughput LLM Inference | Mert Hidayetoglu 等 (Snowflake) | 动态工作负载 |
| XY-Serve: End-to-End Versatile Production Serving | Mingcong Song 等 (Huawei) | 生产级服务 |
| PAT: Prefix-Aware Attention | Jinjun Yi 等 | 注意力优化 |
| ZipServ: Hardware-Aware Lossless Compression | Ruibo Fan 等 | 内存优化 |
| BlendServe: Resource-Aware Batching | Yilong Zhao 等 (UC Berkeley) | 离线推理 |
| MoE-APEX: Adaptive Precision Expert Offloading | Peng Tang 等 | MoE推理优化 |

### GitHub 新项目 (新增5个)
| 项目 | 描述 |
|------|------|
| inferinse | 高吞吐量speculative decoding网关 |
| llm-serving-at-scale | 100K+并行查询支持 |
| long-context-serving-lab | PagedAttention + 分层KV卸载演示 |
| llm-inference-optimization-lab | 基准对比平台(vLLM/TGI/TensorRT-LLM) |
| mlc-llm-rest-api | OpenAI兼容REST API + speculative decoding |

---

## 🆕 2026年3月25日新增论文

| 论文 | 作者 | 领域 | 备注 |
|------|------|------|------|
| DFlash: Block Diffusion for Flash Speculative Decoding | Jian Chen 等 | cs.CL | 扩散模型+投机解码 |
| P-EAGLE: Parallel-Drafting EAGLE | Mude Hui 等 | cs.LG | 并行drafting |
| MoE-Spec: Expert Budgeting for Efficient Speculative Decoding | Bradley McDanel 等 | cs.LG | MoE模型优化 |
| FLYING SERVING: On-the-Fly Parallelism Switching | Shouwei Gao 等 | cs.DC | **ICS 2026** |
| DualPath: Breaking Storage Bandwidth Bottleneck | Yongtong Wu 等 | cs.DC | Agentic LLM推理 |

---

## 📚 顶级会议论文 (2025年及以后)

### 2025 年

#### SOSP 2025
1. **Pie: A Programmable Serving System for Emerging LLM Applications**
   - 作者: In Gim 等
   - 论文: [arXiv:2510.24051](https://arxiv.org/abs/2510.24051)
   - GitHub: [pie-project/pie](https://github.com/pie-project/pie)

#### OSDI 2025
2. **NanoFlow: Towards Optimal Large Language Model Serving Throughput**
   - 作者: Kan Zhu, Yufei Gao 等 (University of Washington, Tsinghua)
   - 论文: [arXiv:2408.13040](https://arxiv.org/abs/2408.13040)

#### ICLR 2025
3. **SWIFT: On-the-Fly Self-Speculative Decoding**
   - GitHub: [hemingkx/SWIFT](https://github.com/hemingkx/SWIFT)

#### ICML 2025
4. **TokenSwift: Ultra Long Sequence Generation**
   - GitHub: [bigai-nlco/TokenSwift](https://github.com/bigai-nlco/TokenSwift)

#### ACL 2024
5. **LayerSkip: Early Exit and Self-Speculative Decoding**
   - GitHub: [facebookresearch/LayerSkip](https://github.com/facebookresearch/LayerSkip)

#### NAACL 2024
6. **REST: Retrieval-Based Speculative Decoding**
   - GitHub: [FasterDecoding/REST](https://github.com/FasterDecoding/REST)

#### COLM 2024
7. **TriForce: Hierarchical Speculative Decoding**
   - GitHub: [Infini-AI-Lab/TriForce](https://github.com/Infini-AI-Lab/TriForce)

---

## 📚 arXiv 论文 (2024-2025)

### 2024-2025年新增论文

| 论文 | 作者 | arXiv | GitHub |
|------|------|-------|--------|
| DistServe | PKU | [2401.09670](https://arxiv.org/abs/2401.09670) | [DistServe](https://github.com/LLMServe/DistServe) |
| Mooncake | Moonshot AI | - | [Mooncake](https://github.com/kvcache-ai/Mooncake) |
| Star-Attention | NVIDIA | [2411.17116](https://arxiv.org/abs/2411.17116) | [Star-Attention](https://github.com/NVIDIA/Star-Attention) |
| FlashAttention-3 | TriDao et al. | - | [flash-attention](https://github.com/Dao-AILab/flash-attention) |
| EAGLE (ICML/EMNLP/NeurIPS) | SafeAILab | - | [EAGLE](https://github.com/SafeAILab/EAGLE) |
| ShadowKV | - | [2410.21485](https://arxiv.org/abs/2410.21485) | - |
| ALISE | - | [2410.19690](https://arxiv.org/abs/2410.19690) | - |
| MiniKV | - | [2411.19092](https://arxiv.org/abs/2411.19092) | - |
| InstCache | - | [2411.12410](https://arxiv.org/abs/2411.12410) | - |
| Chameleon | - | [2411.18550](https://arxiv.org/abs/2411.18550) | - |
| PipeInfer | - | - | [PipeInfer](https://github.com/AutonomicPerfectionist/PipeInfer) |
| DART | - | - | [DART](https://github.com/fvliang/DART) |
| Sequoia | Infini-AI-Lab | - | [Sequoia](https://github.com/Infini-AI-Lab/Sequoia) |

### 2023 年
1. **vLLM: PagedAttention** - SOSP 2023
   - 作者: Woosuk Kwon 等
   - 论文: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
   - GitHub: [vllm-project/vllm](https://github.com/vllm-project/vllm)

---

## 🖥️ GitHub 项目

| 项目 | 描述 | Stars |
|------|------|-------|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 纯C/C++实现的LLM推理框架 | 99.1k |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | 高吞吐量LLM推理服务引擎 | 74.1k |
| [microsoft/BitNet](https://github.com/microsoft/BitNet) | 1-bit LLM推理框架 | 36.4k |
| [mlc-ai/web-llm](https://github.com/mlc-ai/web-llm) | 浏览器内LLM推理引擎 | 17.6k |
| [ModelTC/LightLLM](https://github.com/ModelTC/LightLLM) | Python LLM推理框架 | 4k |
| [skyzh/tiny-llm](https://github.com/skyzh/tiny-llm) | Apple Silicon LLM学习项目 | 4k |
| [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) | TGI - Large Language Model Text Generation Inference | - |
| [sgl-project/SpecForge](https://github.com/sgl-project/SpecForge) | Speculative Decoding训练框架 | 741 |
| [smart-lty/nano-PEARL](https://github.com/smart-lty/nano-PEARL) | Draft-Target Disaggregation | 180 |
| [samsungsds-opensource/DSDE](https://github.com/samsungsds-opensource/DSDE) | Dynamic Speculative Decoding | - |

---

## 🔄 更新日志

- **2026-03-25**: 新增18篇论文 (DistServe, Mooncake, Star-Attention, FlashAttention-3, EAGLE, LayerSkip, TriForce, REST, SWIFT, TokenSwift, ShadowKV, ALISE, MiniKV, InstCache, Chameleon, PipeInfer, DART, Sequoia)
- **2026-03-24**: 新增 SOSP 2025 和 OSDI 2025 论文
- **2026-03-24**: 删除所有没有真实出处的论文，只保留有arXiv/GitHub链接的真实论文
- **2026-03-24**: 初始化论文库

*本索引由自动化任务生成*