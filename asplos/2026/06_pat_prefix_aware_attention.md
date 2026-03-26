# PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel

## 论文信息

- **作者**: Jinjun Yi, Zhixin Zhao, Yitao Hu, Ke Yan, Weiwei Sun, Hao Wang, Laiping Zhao, Yuhao Zhang, Wenxin Li, Keqiu Li
- **机构**: Tianjin University, Stevens Institute of Technology
- **会议**: ASPLOS 2026
- **日期**: 2026年3月24-26日

## 原文链接

- **会议链接**: https://asplos-conference.org/asplos2026/program/

## 摘要 (Abstract)

PAT（前缀感知注意力）提出了一种新的注意力机制优化，通过资源高效的多tile内核加速LLM解码。该方法显著降低了注意力计算的内存和计算开销。

## 引言 (Introduction)

注意力机制是LLM的核心组件，但也是计算瓶颈。现有优化方法忽略了prefix的特殊性。PAT利用prefix的共享特性，实现更高效的注意力计算。