# MoE-Spec: Expert Budgeting for Efficient Speculative Decoding

## 论文信息
- **标题**: MoE-Spec: Expert Budgeting for Efficient Speculative Decoding
- **作者**: Bradley McDanel, Steven Li, Sruthikesh Surineni, Harshit Khaitan
- **arXiv**: [2602.16052](https://arxiv.org/abs/2602.16052)
- **提交时间**: 2026年2月17日
- **领域**: Machine Learning (cs.LG)

## 摘要 (Abstract)
Speculative decoding accelerates Large Language Model (LLM) inference by verifying multiple drafted tokens in parallel. However, for Mixture-of-Experts (MoE) models, this parallelism introduces a severe bottleneck: large draft trees activate many unique experts, significantly increasing memory pressure and diminishing speedups from speculative decoding relative to autoregressive decoding. Prior methods reduce speculation depth when MoE verification becomes expensive. We propose MoE-Spec, a training-free verification-time expert budgeting method that decouples speculation depth from memory cost by enforcing a fixed expert capacity limit at each layer, loading only the experts that contribute most to verification and dropping the long tail of rarely used experts that drive bandwidth overhead. Experiments across multiple model scales and datasets show that this method yields 10--30% higher throughput than state-of-the-art speculative decoding baselines (EAGLE-3) at comparable quality, with flexibility to trade accuracy for further latency reductions through tighter budgets.

## 摘要 (中文)
投机解码通过并行验证多个draft tokens来加速大型语言模型（LLM）推理。然而，对于混合专家（MoE）模型，这种并行性引入了严重的瓶颈：大型draft树激活许多独特的专家，大大增加了内存压力，并减少了相对于自回归解码的投机解码加速。之前的方法在MoE验证变得昂贵时减少投机深度。我们提出了MoE-Spec，这是一种无需训练的验证时专家预算方法，通过在每一层强制执行固定的专家容量限制来解耦投机深度和内存成本，只加载对验证贡献最大的专家，并丢弃驱动带宽开销的很少使用的专家尾部的专家。在多个模型规模和数据集上的实验表明，与基线（EAGLE-3）相比，该方法在相同质量下提供10-30%更高的吞吐量，并且通过更严格的预算可以灵活地权衡准确性以进一步降低延迟。

## 核心贡献
1. **专家预算机制**: 首次提出针对MoE模型的验证时专家预算方法
2. **解耦深度和内存成本**: 通过固定专家容量限制来解耦投机深度和内存成本
3. **无需训练**: 完全无需训练的验证时方法

## 技术细节
- **方法**: 在每层强制执行固定的专家容量限制
- **优势**: 比EAGLE-3提供10-30%更高的吞吐量
- **灵活性**: 可以通过更严格的预算来权衡准确性以降低延迟

---

*更新时间: 2026-03-25*