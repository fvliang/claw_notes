# vLLM

## 项目信息

- **项目名称**: vLLM
- **GitHub**: https://github.com/vllm-project/vllm
- **Stars**: 74.4k+
- **语言**: Python
- **最新更新**: 持续活跃 (5分钟前更新)

## 简介

vLLM是一个高性能、易于使用的LLM推理和服务框架。

最初由UC Berkeley的Sky Computing Lab开发，现已发展为社区驱动的项目。

## 核心特性

### 高性能
- 业界领先的吞吐量和内存效率
- PagedAttention技术高效管理Attention KV缓存
- 持续批处理 incoming requests
- CUDA/HIP图加速模型执行
- 支持GPTQ, AWQ, AutoRound, INT4, INT8, FP8量化
- 优化CUDA内核 (集成FlashAttention和FlashInfer)
- **支持Speculative Decoding**
- **支持Chunked Prefill**

### 灵活性
- 无缝集成Hugging Face模型
- 支持多种解码算法 (parallel sampling, beam search等)
- 支持Tensor、Pipeline、Data和Expert并行
- 流式输出
- OpenAI兼容API
- 支持NVIDIA/AMD/Intel/PowerPC/ARM/TPU
- 支持Prefix caching
- 支持Multi-LoRA

### 支持的模型
- Transformer类LLM (Llama等)
- MoE LLMs (Mixtral, DeepSeek-V2/V3)
- Embedding模型 (E5-Mistral)
- 多模态LLM (LLaVA)

## 安装

```bash
pip install vllm
```

## 论文引用

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## 相关资源

- [文档](https://docs.vllm.ai)
- [博客](https://blog.vllm.ai/)
- [论文](https://arxiv.org/abs/2309.06180)
- [Twitter](https://x.com/vllm_project)
- [用户论坛](https://discuss.vllm.ai)
- [开发者Slack](https://slack.vllm.ai)

## SOSP 2023 论文

本文发表于SOSP 2023 (ACM Symposium on Operating Systems Principles 2023)