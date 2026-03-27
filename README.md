# LLM Serving 论文收集索引

本目录收集了LLM serving、speculative decoding、LLM inference相关的论文和项目资源。

## 目录结构

```
~/claw_notes/
├── arxiv/                    # arXiv论文
│   ├── 2023/
│   │   └── vllm_pagedattention.md
│   ├── 2024/
│   │   └── powerinfer2.md
│   └── 2025/
│       ├── specforge.md
│       ├── minedraft.md
│       ├── pcr_rag_cache.md
│       ├── mmspec.md
│       ├── specsteer.md
│       ├── parallelvlm.md
│       └── heisd.md
│   └── 2026/
│       └── minedraft.md
├── github/                   # GitHub项目
│   ├── 2024/
│   │   ├── vllm.md
│   │   ├── flashinfer.md
│   │   ├── lmdeploy.md
│   │   ├── powerinfer.md
│   │   ├── lightllm.md
│   │   ├── serverlessllm.md
│   │   ├── routellm.md
│   │   ├── tiny_llm.md
│   │   ├── llm_engineer_handbook.md
│   │   └── llm_applications.md
│   └── 2026/
│       └── sglang.md
└── (其他会议目录待添加)
```

## 收集的内容

### GitHub项目 (11个)
- SGLang: 高性能LLM服务框架 (14.3k stars)
- vLLM: 高吞吐量LLM推理引擎 (74.4k stars)
- FlashInfer: LLM Serving内核库 (5.2k stars)
- LMDeploy: 推理部署工具包 (7.7k stars)
- PowerInfer: 本地部署推理引擎 (9.1k stars)
- LightLLM: 轻量级推理框架 (4k stars)
- ServerlessLLM: 无服务器LLM服务 (664 stars)
- RouteLLM: LLM路由框架 (4.7k stars)
- Tiny LLM: Apple Silicon学习项目 (4k stars)
- LLM Engineer Handbook: 资源列表 (4.8k stars)
- LLM Applications: RAG应用指南 (1.9k stars)

### arXiv论文 (11篇)
- vLLM (SOSP 2023)
- PowerInfer-2 (2024)
- SpecForge (2025)
- MineDraft (2026)
- PCR (2025)
- MMSpec (2025)
- SpecSteer (2025)
- ParallelVLM (2025)
- HeiSD (2025)
- Pipelined Collaborative SD (2025)
- 以及更多...

## 关键词
- llm serving
- speculative decoding  
- llm inference
- paged attention
- kv cache

## 来源
- arXiv
- GitHub
- 系统会议论文 (待收集)

## 更新日期
2026-03-27

---

*收集目标：OSDI, SOSP, NSDI, SIGCOMM, SIGMOD, ATC, EuroSys, DAC, ASPLOS, SC, NeurIPS, ICLR, ICML, ACL, EMNLP*