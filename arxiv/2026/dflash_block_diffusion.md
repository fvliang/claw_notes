# DFlash: Block Diffusion for Flash Speculative Decoding

## 论文信息
- **标题**: DFlash: Block Diffusion for Flash Speculative Decoding
- **作者**: Jian Chen, Yesheng Liang, Zhijian Liu
- **arXiv**: [2602.06036](https://arxiv.org/abs/2602.06036)
- **提交时间**: 2026年2月5日
- **领域**: Computation and Language (cs.CL)

## 摘要 (Abstract)
Autoregressive large language models (LLMs) deliver strong performance but require inherently sequential decoding, leading to high inference latency and poor GPU utilization. Speculative decoding mitigates this bottleneck by using a fast draft model whose outputs are verified in parallel by the target LLM; however, existing methods still rely on autoregressive drafting, which remains sequential and limits practical speedups. Diffusion LLMs offer a promising alternative by enabling parallel generation, but current diffusion models typically underperform compared with autoregressive models. In this paper, we introduce DFlash, a speculative decoding framework that employs a lightweight block diffusion model for parallel drafting. By generating draft tokens in a single forward pass and conditioning the draft model on context features extracted from the target model, DFlash enables efficient drafting with high-quality outputs and higher acceptance rates. Experiments show that DFlash achieves over 6x lossless acceleration across a range of models and tasks, delivering up to 2.5x higher speedup than the state-of-the-art speculative decoding method EAGLE-3.

## 摘要 (中文)
自回归大型语言模型（LLM）虽然性能强大，但需要固有的顺序解码，导致高推理延迟和GPU利用率低。投机解码通过使用快速draft模型来缓解这一瓶颈，该模型的输出由目标LLM并行验证；然而，现有方法仍然依赖于自回归drafting，这仍然是顺序的，限制了实际的加速。扩散LLM通过实现并行生成提供了另一种有前途的方案，但当前的扩散模型通常比自回归模型性能差。在本文中，我们引入了DFlash，这是一种使用轻量级块扩散模型进行并行drafting的投机解码框架。通过在单个前向传播中生成draft tokens，并将draft模型 conditioning 在从目标模型提取的上下文特征上，DFlash能够生成高质量输出和更高的接受率。实验表明，DFlash在各种模型和任务上实现了超过6倍的无损加速，比最新的投机解码方法EAGLE-3高出2.5倍。

## 核心贡献
1. **并行生成**: 首次将块扩散模型应用于投机解码，实现单次前向传播生成多个draft tokens
2. **上下文条件化**: 通过从目标模型提取上下文特征来提高draft质量
3. **高性能**: 在各种模型和任务上实现超过6倍的无损加速

## 技术细节
- **方法**: 使用轻量级块扩散模型进行并行drafting
- **特点**: 在单个前向pass中生成draft tokens
- **优势**: 比EAGLE-3高2.5倍的加速

---

*更新时间: 2026-03-25*