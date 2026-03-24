# POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference

- **会议**: ASPLOS 2025
- **arXiv**: [2410.18038](https://arxiv.org/abs/2410.18038)
- **DOI**: [10.1145/3676641.3715996](https://doi.org/10.1145/3676641.3715996)
- **作者**: Aditya Kamath 等
- **年份**: 2024 (2025年发表)

## 摘要

Each request in LLM inference goes through two phases: compute-bound prefill and memory-bandwidth-bound decode. To improve GPU utilization, recent systems use hybrid batching that combines the prefill and decode phases of different requests into the same batch. This approach optimizes linear operations but remains inefficient for attention computation because existing attention kernels specialize execution independently for the prefill and decode phases.

In this paper, we present **POD-Attention** - the first GPU kernel that efficiently computes attention for hybrid batches. POD-Attention aims to maximize the utilization of both compute and memory bandwidth by carefully allocating the GPU's resources such that prefill and decode operations happen concurrently on the same multiprocessor.

## 核心贡献

1. **首个混合批次GPU内核**: 第一个高效计算混合批次注意力的GPU内核
2. **Prefill-Decode重叠**: 最大化计算和内存带宽利用率
3. **性能提升**: 注意力计算加速高达59%（平均28%）

## 实验结果

- 注意力计算加速高达59%（平均28%）
- 相比独立优化的prefill和decode注意力内核，实现更高吞吐量和更低延迟

---

*论文来源：ASPLOS 2025，数据真实可验证*