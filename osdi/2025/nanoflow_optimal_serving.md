# NanoFlow: Towards Optimal Large Language Model Serving Throughput

- **会议**: OSDI 2025
- **arXiv**: [2408.13040](https://arxiv.org/abs/2408.13040) (最初提交于2024年8月，2025年5月更新)
- **作者**: Kan Zhu, Yufei Gao, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Tian Tang, Qinyu Xu, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Ziren Wang, Stephanie Wang, Arvind Krishnamurthy, Baris Kasikci
- **单位**: University of Washington, Tsinghua University, University of California Berkeley, University of Michigan
- **年份**: 2025

## 摘要

Large Language Models (LLMs) have resulted in a surging demand for planet-scale serving. Despite significant advancements in LLM inference systems, achieving optimal throughput remains challenging due to the complex interplay between GPU compute and memory resources across the entire serving pipeline.

Through a detailed analysis, we show that despite having memory-intensive components, end-to-end LLM serving is **compute bound** for most common workloads and LLMs.

## 核心贡献

1. **计算瓶颈分析**: 揭示了端到端LLM服务在大多数常见工作负载和LLM上是计算 bound
2. **优化吞吐量**: 针对GPU计算和内存资源的复杂交互进行优化

## 相关链接

- [USENIX OSDI 2025 Presentation](https://www.usenix.org/conference/osdi25/presentation/zhu-kan)

---

*论文来源：OSDI 2025，数据真实可验证*