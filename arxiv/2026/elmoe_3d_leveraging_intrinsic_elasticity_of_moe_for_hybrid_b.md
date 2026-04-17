# ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving

**Authors:** Yuseon Choi, Jingu Lee, Jungjun Oh, Sunjoo Whang, Byeongcheol Kim, Minsung Kim, Hoi-Jun Yoo, Sangjin Kim

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.14626](<https://arxiv.org/abs/2604.14626>)

**Topic:** Speculative Decoding

---

## Abstract (English)

Mixture-of-Experts (MoE) models have become the dominant architecture for large-scale language models, yet on-premises serving remains fundamentally memory-bound as batching turns sparse per-token compute into dense memory activation. Memory-centric architectures (PIM, NMP) improve bandwidth but leave compute underutilized under MoE's low arithmetic intensity at high batch sizes. Speculative decoding (SD) trades idle compute for fewer target invocations, yet verification must load experts even for rejected tokens, severely limiting its benefit in MoE especially at low batch sizes. We propose ELMoE-3D, a hybrid-bonding (HB)-based HW-SW co-designed framework that unifies cache-based acceleration and speculative decoding to offer overall speedup across batch sizes. We identify two intrinsic elasticity axes of MoE — expert and bit — and jointly scale them to construct Elastic Self-Speculative Decoding (Elastic-SD), which serves as both an expert cache and a strongly aligned self-draft model accelerated by high HB bandwidth. On our 3D-stacked hardware, ELMoE-3D achieves an average 6.6x speedup and 4.4x energy efficiency gain over naive MoE serving on xPU across batch sizes 1-16, and delivers 2.2x speedup and 1.4x energy efficiency gain over the best-performing prior accelerator baseline.

## Abstract (Chinese / 中文摘要)

混合专家(MoE)模型已成为大规模语言模型的主导架构，但在本地服务中仍然从根本上受内存限制，因为批处理将稀疏的每token计算转变为密集的内存激活。内存中心架构(PIM, NMP)改善了带宽，但在MoE的高批量低算术强度下使计算未充分利用。投机解码(SD)用空闲计算换取更少的目标调用，但验证必须为拒绝的token加载专家，严重限制了其在MoE中的收益，特别是在低批量下。我们提出ELMoE-3D，一个基于混合绑定(HB)的硬件-软件协同设计框架，统一缓存加速和投机解码以提供跨批量的整体加速。我们识别了MoE的两个内在弹性轴——专家和位——并联合缩放它们以构建弹性自投机解码(Elastic-SD)，它既作为专家缓存又作为强对齐的自草案模型。在我们的3D堆叠硬件上，ELMoE-3D在批量1-16范围内平均实现了6.6倍的加速和4.4倍的能效提升。

---

*Auto-collected from arXiv on 2026-04-17*
