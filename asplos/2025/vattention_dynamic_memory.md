# vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention

- **会议**: ASPLOS 2025
- **arXiv**: [2405.04437](https://arxiv.org/abs/2405.04437)
- **作者**: Ashish Panwar, Rishabh Prabhu, et al.
- **年份**: 2024 (2025年发表)

## 摘要

PagedAttention is a popular approach for dynamic memory allocation in LLM serving systems. It enables on-demand allocation of GPU memory to mitigate KV cache fragmentation -- a phenomenon that crippled the batch size (and consequently throughput) in prior systems. However, in trying to allocate physical memory at runtime, PagedAttention ends up changing the virtual memory layout of the KV cache from contiguous to non-contiguous. Such a design leads to non-trivial programming and performance overheads.

We present **vAttention** -- an approach that mitigates fragmentation in physical memory while retaining the contiguity of KV cache in virtual memory. We achieve this by decoupling the allocation of virtual and physical memory using CUDA virtual memory management APIs.

## 核心贡献

1. **解耦虚拟内存和物理内存分配**: 使用CUDA虚拟内存管理API
2. **保持KV cache连续性**: 在虚拟内存中保持KV cache的连续性
3. **性能提升**: 相比PagedAttention-based kernels，提升高达1.23x

## 实验结果

- 相比FlashAttention和FlashInfer的PagedAttention内核，吞吐量提升最高1.23x

---

*论文来源：ASPLOS 2025，数据真实可验证*