# Accelerating Speculative Decoding with Block Diffusion Draft Trees (DDTree)

**Authors:** Liran Ringel, Yaniv Romano

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.12989](<https://arxiv.org/abs/2604.12989>)

**Topic:** Speculative Decoding

---

## Abstract (English)

Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a block diffusion drafter can generate an entire draft block in a single forward pass and achieve state-of-the-art speculative decoding performance, outperforming strong autoregressive drafters such as EAGLE-3. Vanilla DFlash, however, still verifies only a single drafted trajectory per round, potentially limiting its acceptance length. We introduce DDTree (Diffusion Draft Tree), a method that constructs a draft tree directly from the per-position distributions of a block diffusion drafter. Under a fixed node budget, DDTree uses a simple best-first heap algorithm to select the continuations that are most likely to match the target model according to a surrogate defined by the draft model's output. The resulting tree is verified efficiently in a single target model forward pass using an ancestor-only attention mask. Because DDTree builds on DFlash, a leading draft model for speculative decoding, these gains place DDTree among the leading approaches to speculative decoding.

## Abstract (Chinese / 中文摘要)

投机解码通过使用轻量级草案器提议多个未来token来加速自回归语言模型，目标模型并行验证这些token。DFlash表明块扩散草案器可以在单次前向传播中生成整个草案块，并实现最先进的投机解码性能，超越了EAGLE-3等强自回归草案器。然而，Vanilla DFlash每轮仍只验证单个草案轨迹，可能限制其接受长度。我们引入DDTree(扩散草案树)，一种直接从块扩散草案器的逐位置分布构建草案树的方法。在固定节点预算下，DDTree使用简单的最佳优先堆算法选择最可能匹配目标模型的延续。结果树使用仅祖先注意力掩码在单次目标模型前向传播中高效验证。由于DDTree建立在DFlash之上，这些增益使DDTree跻身投机解码的领先方法之列。

---

*Auto-collected from arXiv on 2026-04-17*
