# LLM Serving 论文搜集索引

本目录收集了LLM Serving、Speculative Decoding和LLM Inference相关的学术论文和开源项目。

## 目录结构

```
~/claw_notes/
├── arxiv/
│   ├── 2026/          # 2026年论文
│   ├── 2025/          # 2025年论文
│   ├── 2024/          # 2024年论文
│   └── 2023/          # 2023年论文
├── github/            # GitHub开源项目
├── osdi/              # OSDI会议论文
└── sosp/              # SOSP会议论文
```

## 2026年新增论文

### Speculative Decoding
1. S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation
2. ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding
3. Speculating Experts Accelerates Inference for Mixture-of-Experts
4. A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference
5. SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding
6. MMSpec: Benchmarking Speculative Decoding for Vision-Language Models
7. Self-Speculative Decoding for LLM-based ASR with CTC Encoder Drafts
8. Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple
9. ConFu: Contemplate the Future for Better Speculative Sampling
10. EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs
11. Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning
12. Quasar: Quantized Self-Speculative Acceleration for Rapid Inference
13. LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding
14. Make Every Draft Count: Hidden State based Speculative Decoding
15. KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection

### LLM Serving / KV Cache
1. Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention
2. CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving
3. xLLM Technical Report
4. Reasoning Language Model Inference Serving Unveiled: An Empirical Study
5. Efficiently Align Draft Models via Parameter- and Data-Efficient Adaptation
6. WANSpec: Leveraging Global Compute Capacity for LLM Inference

## 2025年论文

### Speculative Decoding
1. DSD: A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving
2. ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems
3. Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving
4. TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference
5. StarSD: One-for-Many Speculative Decoding

### LLM Serving / KV Cache
1. KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head
2. vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention
3. PagedEviction: Structured Block-wise KV Cache Pruning for Efficient LLM Inference
4. Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency
5. Rethinking Key-Value Cache Compression Techniques
6. Direct Multi-Token Decoding
7. NEZHA: A Zero-sacrifice and Hyperspeed Decoding Architecture for Generative Recommendations

## 2023年及之前的重要论文

### SOSP 2023
1. **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention** - Woosuk Kwon et al.
   - arXiv: 2309.06180
   - GitHub: vllm-project/vllm

## GitHub 开源项目

### 框架/系统
1. **SpecForge** (749 stars) - Training speculative decoding models
2. **nano-PEARL** (182 stars) - Draft-Target Disaggregation
3. **mini-infer** (124 stars) - LLM inference engine from scratch
4. **vLLM** - 生产级LLM服务系统

### 工具/实验
5. inferinse - Speculative decoding gateway
6. DSDE - Distributed Speculative Decoding
7. long-context-serving-lab - 长上下文服务实验
8. llm-inference-optimization-lab - 推理优化基准测试

## 关键词

- LLM Serving
- Speculative Decoding
- PagedAttention
- KV Cache
- LLM Inference
- Edge-Cloud Collaboration
- Batching
- Memory Optimization