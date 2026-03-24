# LLM Serving 论文索引 (2026-03-25 更新)

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