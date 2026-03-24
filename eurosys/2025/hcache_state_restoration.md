# Fast State Restoration in LLM Serving with HCache

- **会议**: EuroSys 2025
- **arXiv**: [2410.05004](https://arxiv.org/abs/2410.05004)
- **作者**: Shiwei Gao 等
- **年份**: 2024 (2025年发表)

## 摘要

The growing complexity of LLM usage today, e.g., multi-round conversation and retrieval-augmented generation (RAG), makes contextual states (i.e., KV cache) reusable across user requests. Given the capacity constraints of GPU memory, only a limited number of contexts can be cached on GPU for reusing. Existing inference systems typically evict part of the KV cache and restore it by recomputing it from the original tokens or offloading it to host storage for later retrieval, both of which introduce substantial computational or I/O overheads.

We propose **HCache**, a novel LLM state restoration method. Its key idea is to restore LLM states from intermediate activations and thus utilize computational and I/O resources with low overhead.

## 核心贡献

1. **从中间激活恢复状态**: 新的LLM状态恢复方法
2. **无气泡恢复调度器**: 整合资源互补方法优化计算和IO任务的平衡
3. **基于块的存储管理器**: 解决布局不匹配问题

## 实验结果

- 相比KV offload，TTFT降低最高1.93X，存储空间减少1.92-2.40X
- 相比token重新计算，TTFT降低最高5.73X

---

*论文来源：EuroSys 2025，数据真实可验证*