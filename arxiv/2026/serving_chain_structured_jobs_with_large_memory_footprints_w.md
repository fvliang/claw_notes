# Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving

**Authors:** Tingyang Sun, Ting He, I-Hong Hou

**Conference:** arXiv 2026

**Year:** 2026

**ArXiv:** [2604.14993](<https://arxiv.org/abs/2604.14993>)

**Topic:** LLM Serving

---

## Abstract (English)

As a current trend in Artificial Intelligence (AI), large foundation models are increasingly employed as the core of AI services. However, even after training, serving such models at scale remains a challenging task due to their heavy resource footprints, particularly in terms of GPU memory. While recent works revealed unique characteristics of systems serving foundation models that distinguish them from traditional distributed computing systems, there is still a lack of fundamental understanding of the underlying system management problems. This work aims at addressing this gap by extracting a novel problem of "server chain composition" via block placement and cache allocation for serving chain-structured jobs with large memory footprints, which models a fundamental problem in serving large foundation models through pipeline parallelism. After showing the NP-hardness of the optimal solution, the focus is turned to developing scalable algorithms with guaranteed performance under state-of-the-art load balancing. Application of the proposed solution to a distributed large language model (LLM) serving system shows significant reduction of response times compared to state-of-the-art solutions.

## Abstract (Chinese / 中文摘要)

随着AI的发展趋势，大型基础模型越来越多地被用作AI服务的核心。然而，即使在训练之后，大规模服务这些模型仍然是一项具有挑战性的任务，因为它们的资源占用庞大，特别是在GPU内存方面。虽然最近的工作揭示了服务基础模型的系统的独特特征，但仍然缺乏对底层系统管理问题的基础性理解。本工作旨在填补这一空白，通过提取一个新问题——「服务器链组合」——即通过块放置和缓存分配来服务具有大内存占用的链结构作业，这建模了通过流水线并行服务大型基础模型的基本问题。在证明最优解的NP硬度后，重点转向开发在最先进负载平衡下具有保证性能的可扩展算法。将所提出的解决方案应用于分布式大语言模型(LLM)服务系统，与最先进的解决方案相比，显示了显著的响应时间减少。

---

*Auto-collected from arXiv on 2026-04-17*
