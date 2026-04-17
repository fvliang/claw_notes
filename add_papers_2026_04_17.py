#!/usr/bin/env python3
"""Add new LLM serving/inference/speculative decoding papers to database and create markdown files."""
import json
import os
import re
from datetime import datetime

DB_PATH = "database.json"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# New papers to add
new_papers = [
    {
        "title": "ScePsy: Serving Agentic Workflows Using Aggregate LLM Pipelines",
        "authors": "Marcel Wagenländer, Otto White, Britannio Jarrett, Pedro Silvestre, Yanda Tao, Guo Li, Huanzhou Zhu, Llúis Vilanova, Peter Pietzuch",
        "arxiv_id": "2604.15186",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they can be defined using arbitrary agentic frameworks and exhibit unpredictable execution times: execution may branch, fan-out, or recur in data-dependent ways. Since LLMs in workflows often outnumber available GPUs, their execution also leads to GPU oversubscription. We describe ScePsy, a new agentic serving system that efficiently schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. ScePsy exploits the insight that, while agentic workflows have unpredictable end-to-end latencies, the shares of each LLM's total execution times are comparatively stable across executions. ScePsy decides on GPU allocations based on these aggregate shares: first, it profiles the LLMs under different parallelism degrees. It then uses these statistics to construct an Aggregate LLM Pipeline, which is a lightweight latency/throughput predictor for allocations. To find a GPU allocation that minimizes latency while achieving a target throughput, ScePsy uses the Aggregate LLM Pipeline to explore a search space over fractional GPU shares, tensor parallelism degrees, and replica counts. It uses a hierarchical heuristic to place the best allocation onto the GPU cluster, minimizing fragmentation, while respecting network topology constraints. Our evaluation on realistic agentic workflows shows that ScePsy achieves up to 2.4x higher throughput and 27x lower latency compared to systems that optimize LLMs independently or rely on user-specified allocations.",
        "abstract_cn": "Agentic workflows通过编排多个大语言模型(LLM)和工具来执行复杂任务。在目标吞吐量下以低延迟服务这些工作流具有挑战性，因为它们可以使用任意的agentic框架定义，并表现出不可预测的执行时间：执行可能分支、扇出或以数据依赖的方式递归。由于工作流中的LLM通常多于可用GPU，其执行也会导致GPU超额订阅。我们描述了ScePsy，一个新的agentic服务系统，可以高效地将任意多LLM agentic工作流调度到GPU集群上。ScePsy利用了一个洞察：虽然agentic工作流具有不可预测的端到端延迟，但每个LLM的总执行时间份额在不同执行中相对稳定。ScePsy基于这些聚合份额决定GPU分配：首先，在不同并行度下对LLM进行性能分析，然后使用这些统计数据构建Aggregate LLM Pipeline——一个轻量级的延迟/吞吐量预测器。为了找到最小化延迟同时实现目标吞吐量的GPU分配，ScePsy使用Aggregate LLM Pipeline探索分数GPU份额、张量并行度和副本计数的搜索空间，并使用分层启发式方法将最佳分配放置到GPU集群上，最小化碎片化同时尊重网络拓扑约束。在真实的agentic工作流上的评估显示，ScePsy比独立优化LLM或依赖用户指定分配的系统实现了高达2.4倍的吞吐量和27倍的更低延迟。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "LLM Serving",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter",
        "authors": "Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang",
        "arxiv_id": "2604.15039",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but in practice its deployment boundary is still determined by KVCache transfer. In conventional dense-attention models, prefill generates huge KVCache traffics that keep prefill and decode tightly coupled within a single high-bandwidth network domain, limiting heterogeneous deployment and resource elasticity. Recent hybrid-attention architectures substantially reduce KVCache size, making cross-cluster KVCache transport increasingly plausible. However, smaller KVCache alone does not make heterogeneous cross-datacenter PD serving practical: real workloads remain bursty, request lengths are highly skewed, prefix caches are unevenly distributed, and inter-cluster bandwidth fluctuates. A naive design that fully externalizes prefill can therefore still suffer from congestion, unstable queueing, and poor utilization. We present Prefill-as-a-Service (PrfaaS), a cross-datacenter serving architecture that selectively offloads long-context prefill to standalone, compute-dense prefill clusters and transfers the resulting KVCache over commodity Ethernet to local PD clusters for decode. Rather than treating reduced KVCache as sufficient, PrfaaS combines model-side KV efficiency with system-side selective offloading, bandwidth-aware scheduling, and cache-aware request placement. This design removes the requirement that heterogeneous accelerators share the same low-latency RDMA fabric, enabling independent scaling of prefill and decode capacity across loosely coupled clusters. In a case study using an internal 1T-parameter hybrid model, a PrfaaS-augmented heterogeneous deployment achieves 54% and 32% higher serving throughput than homogeneous PD and naive heterogeneous baselines, respectively, while consuming only modest cross-datacenter bandwidth.",
        "abstract_cn": "Prefill-decode(PD)解耦已成为大规模LLM服务的标准架构，但在实践中其部署边界仍由KVCache传输决定。在传统的密集注意力模型中，prefill产生巨大的KVCache流量，使prefill和decode紧密耦合在单一高带宽网络域内，限制了异构部署和资源弹性。最近的混合注意力架构大幅减少了KVCache大小，使跨集群KVCache传输变得越来越可行。然而，仅靠较小的KVCache并不能使异构跨数据中心PD服务变得实用：真实工作负载仍然是突发性的，请求长度高度偏斜，前缀缓存分布不均，集群间带宽波动。我们提出Prefill-as-a-Service(PrfaaS)，一种跨数据中心服务架构，选择性地将长上下文prefill卸载到独立的、计算密集的prefill集群，并通过商品以太网将产生的KVCache传输到本地PD集群进行decode。PrfaaS将模型侧的KV效率与系统侧的选择性卸载、带宽感知调度和缓存感知请求放置相结合。这一设计消除了异构加速器共享相同低延迟RDMA fabric的要求，使prefill和decode容量能在松耦合集群间独立扩展。在内部1T参数混合模型的案例研究中，PrfaaS增强的异构部署比同构PD和朴素异构基线分别实现了54%和32%的更高服务吞吐量。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Disaggregated Serving",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving",
        "authors": "Tingyang Sun, Ting He, I-Hong Hou",
        "arxiv_id": "2604.14993",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "As a current trend in Artificial Intelligence (AI), large foundation models are increasingly employed as the core of AI services. However, even after training, serving such models at scale remains a challenging task due to their heavy resource footprints, particularly in terms of GPU memory. While recent works revealed unique characteristics of systems serving foundation models that distinguish them from traditional distributed computing systems, there is still a lack of fundamental understanding of the underlying system management problems. This work aims at addressing this gap by extracting a novel problem of \"server chain composition\" via block placement and cache allocation for serving chain-structured jobs with large memory footprints, which models a fundamental problem in serving large foundation models through pipeline parallelism. After showing the NP-hardness of the optimal solution, the focus is turned to developing scalable algorithms with guaranteed performance under state-of-the-art load balancing. Application of the proposed solution to a distributed large language model (LLM) serving system shows significant reduction of response times compared to state-of-the-art solutions.",
        "abstract_cn": "随着AI的发展趋势，大型基础模型越来越多地被用作AI服务的核心。然而，即使在训练之后，大规模服务这些模型仍然是一项具有挑战性的任务，因为它们的资源占用庞大，特别是在GPU内存方面。虽然最近的工作揭示了服务基础模型的系统的独特特征，但仍然缺乏对底层系统管理问题的基础性理解。本工作旨在填补这一空白，通过提取一个新问题——「服务器链组合」——即通过块放置和缓存分配来服务具有大内存占用的链结构作业，这建模了通过流水线并行服务大型基础模型的基本问题。在证明最优解的NP硬度后，重点转向开发在最先进负载平衡下具有保证性能的可扩展算法。将所提出的解决方案应用于分布式大语言模型(LLM)服务系统，与最先进的解决方案相比，显示了显著的响应时间减少。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "LLM Serving",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding",
        "authors": "Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao",
        "arxiv_id": "2604.14885",
        "github_repo": "https://github.com/hkr04/RACER",
        "conference": "ACL Findings 2026",
        "year": 2026,
        "abstract_en": "Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but existing training-free variants face trade-offs: retrieval-based drafts break when no exact match exists, while logits-based drafts lack structural guidance. We propose RACER (Retrieval-Augmented Contextual Rapid Speculative Decoding), a lightweight and training-free method that integrates retrieved exact patterns with logit-driven future cues. This unification supplies both reliable anchors and flexible extrapolation, yielding richer speculative drafts. Experiments on Spec-Bench, HumanEval, and MGSM-ZH demonstrate that RACER consistently accelerates inference, achieving more than 2x speedup over autoregressive decoding, and outperforms prior training-free methods, offering a scalable, plug-and-play solution for efficient LLM decoding.",
        "abstract_cn": "大语言模型(LLM)中的自回归解码每步生成一个token，导致高推理延迟。投机解码(SD)通过猜测-验证策略缓解这一问题，但现有的免训练变体面临权衡：基于检索的草案在没有精确匹配时会失效，而基于logits的草案缺乏结构指导。我们提出RACER(检索增强上下文快速投机解码)，一种轻量级免训练方法，将检索到的精确模式与logit驱动的未来线索整合。这种统一既提供了可靠的锚点又提供了灵活的推断，产生更丰富的投机草案。在Spec-Bench、HumanEval和MGSM-ZH上的实验证明，RACER持续加速推理，实现超过2倍的加速比，优于之前的免训练方法，提供可扩展的即插即用解决方案。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Speculative Decoding",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": True,
    },
    {
        "title": "Acceptance Dynamics Across Cognitive Domains in Speculative Decoding",
        "authors": "Saif Mahmoud",
        "arxiv_id": "2604.14682",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "Speculative decoding accelerates large language model (LLM) inference. It uses a small draft model to propose a tree of future tokens. A larger target model then verifies these tokens in a single batched forward pass. Despite the growing body of work on speculative methods, the degree to which the cognitive characteristics of a task affect acceptance probability remains largely unexplored. We present an empirical study of tree-based speculative decoding acceptance dynamics. Our study spans four well-established NLP benchmark domains: code generation, mathematical reasoning, logical reasoning, and open-ended chat. For this, we use TinyLlama-1.1B as the draft model against Llama-2-7B-Chat-GPTQ as the target. Over 99,768 speculative nodes collected from 200 prompts, we derive per-domain acceptance rates, expected accepted lengths, depth-acceptance profiles, and entropy-acceptance correlations. We find that task type is a stronger predictor of acceptance than tree depth. Furthermore, only the chat domain consistently yields an expected accepted length exceeding 1.0 token per step. We also show that the entropy-acceptance correlation is consistently negative but weak across all domains. Counterintuitively, chat produces the highest entropy yet the highest acceptance rate. We attribute this divergence to the lexical predictability of RLHF-aligned register. These findings have direct implications for domain-aware speculation budgets and draft-model selection strategies.",
        "abstract_cn": "投机解码加速大语言模型(LLM)推理。它使用一个小型草案模型来提议未来token树，然后更大的目标模型在单次批量前向传播中验证这些token。尽管关于投机方法的研究越来越多，但任务的认知特征对接受概率的影响程度在很大程度上仍未被探索。我们对树基投机解码的接受动态进行了实证研究，跨越四个成熟的NLP基准领域：代码生成、数学推理、逻辑推理和开放式聊天。我们使用TinyLlama-1.1B作为草案模型，Llama-2-7B-Chat-GPTQ作为目标模型。从200个提示收集的99,768个投机节点中，我们推导出每个领域的接受率、预期接受长度、深度-接受率剖面和熵-接受率相关性。我们发现任务类型比树深度是更强的接受率预测因子。此外，只有聊天领域持续产生超过1.0 token/步的预期接受长度。反直觉地，聊天产生最高的熵但最高的接受率。我们将这种分歧归因于RLHF对齐寄存器的词汇可预测性。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Speculative Decoding",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving",
        "authors": "Yuseon Choi, Jingu Lee, Jungjun Oh, Sunjoo Whang, Byeongcheol Kim, Minsung Kim, Hoi-Jun Yoo, Sangjin Kim",
        "arxiv_id": "2604.14626",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "Mixture-of-Experts (MoE) models have become the dominant architecture for large-scale language models, yet on-premises serving remains fundamentally memory-bound as batching turns sparse per-token compute into dense memory activation. Memory-centric architectures (PIM, NMP) improve bandwidth but leave compute underutilized under MoE's low arithmetic intensity at high batch sizes. Speculative decoding (SD) trades idle compute for fewer target invocations, yet verification must load experts even for rejected tokens, severely limiting its benefit in MoE especially at low batch sizes. We propose ELMoE-3D, a hybrid-bonding (HB)-based HW-SW co-designed framework that unifies cache-based acceleration and speculative decoding to offer overall speedup across batch sizes. We identify two intrinsic elasticity axes of MoE — expert and bit — and jointly scale them to construct Elastic Self-Speculative Decoding (Elastic-SD), which serves as both an expert cache and a strongly aligned self-draft model accelerated by high HB bandwidth. On our 3D-stacked hardware, ELMoE-3D achieves an average 6.6x speedup and 4.4x energy efficiency gain over naive MoE serving on xPU across batch sizes 1-16, and delivers 2.2x speedup and 1.4x energy efficiency gain over the best-performing prior accelerator baseline.",
        "abstract_cn": "混合专家(MoE)模型已成为大规模语言模型的主导架构，但在本地服务中仍然从根本上受内存限制，因为批处理将稀疏的每token计算转变为密集的内存激活。内存中心架构(PIM, NMP)改善了带宽，但在MoE的高批量低算术强度下使计算未充分利用。投机解码(SD)用空闲计算换取更少的目标调用，但验证必须为拒绝的token加载专家，严重限制了其在MoE中的收益，特别是在低批量下。我们提出ELMoE-3D，一个基于混合绑定(HB)的硬件-软件协同设计框架，统一缓存加速和投机解码以提供跨批量的整体加速。我们识别了MoE的两个内在弹性轴——专家和位——并联合缩放它们以构建弹性自投机解码(Elastic-SD)，它既作为专家缓存又作为强对齐的自草案模型。在我们的3D堆叠硬件上，ELMoE-3D在批量1-16范围内平均实现了6.6倍的加速和4.4倍的能效提升。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Speculative Decoding",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding",
        "authors": "Walaa Amer, Uday Das, Fadi Kurdahi",
        "arxiv_id": "2604.14612",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "Self-speculative decoding is an inference technique for large language models designed to speed up generation without sacrificing output quality. It combines fast, approximate decoding using a compact version of the model as a draft model with selective re-evaluation by the full target model. Some existing methods form the draft model by dynamically learning which layers to skip during inference, effectively creating a smaller subnetwork to speed up computation. However, using heuristic-based approaches to select layers to skip can often be simpler and more effective. In this paper, we propose ConfLayers, a dynamic plug-and-play approach to forming the draft model in self-speculative decoding via confidence-based intermediate layer skipping. The process iteratively computes confidence scores for all layers, selects layers to skip based on an adaptive threshold, evaluates the performance of the resulting set, and updates the best selection until no further improvement is achieved or a maximum number of iterations is reached. This framework avoids the overhead and complexity of training a layer skipping policy and can provide more consistent speed-quality trade-offs while preserving the adaptivity of the draft model to diverse tasks and datasets. The performance evaluation of ConfLayers across different models and datasets shows that our novel approach offers up to 1.4x speedup over vanilla LLM generation.",
        "abstract_cn": "自投机解码是大语言模型的一种推理技术，旨在在不牺牲输出质量的情况下加速生成。它使用模型的紧凑版本作为草案模型进行快速近似解码，并与完整目标模型的选择性重新评估相结合。一些现有方法通过动态学习推理期间跳过哪些层来形成草案模型，有效地创建更小的子网络以加速计算。然而，使用启发式方法选择跳过层通常更简单且更有效。在本文中，我们提出ConfLayers，一种动态即插即用方法，通过基于置信度的中间层跳过来形成自投机解码的草案模型。该过程迭代计算所有层的置信度分数，基于自适应阈值选择跳过的层，评估结果集的性能，并更新最佳选择。此框架避免了训练层跳过策略的开销和复杂性，可以提供更一致的速度-质量权衡。性能评估显示ConfLayers提供高达1.4倍的加速。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Speculative Decoding",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning (SpecGuard)",
        "authors": "Authors from arXiv:2604.15244",
        "arxiv_id": "2604.15244",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature allows erroneous steps to propagate. Prior approaches mitigate this using external reward models, but incur additional latency, computational overhead, and limit generalizability. We propose SpecGuard, a verification-aware speculative decoding framework that performs step-level verification using only model-internal signals. At each step, SpecGuard samples multiple draft candidates and selects the most consistent step, which is then validated using an ensemble of two lightweight model-internal signals: (i) an attention-based grounding score that measures attribution to the input and previously accepted steps, and (ii) a log-probability-based score that captures token-level confidence. These signals jointly determine whether a step is accepted or recomputed using the target, allocating compute selectively. Experiments across a range of reasoning benchmarks show that SpecGuard improves accuracy by 3.6% while reducing latency by ~11%, outperforming both SD and reward-guided SD.",
        "abstract_cn": "投机解码(SD)通过允许轻量级草案模型提议由更强的目标模型验证的输出，加速大语言模型推理。然而，其以token为中心的性质允许错误步骤传播。先前的方法使用外部奖励模型来缓解此问题，但会产生额外的延迟、计算开销并限制泛化性。我们提出SpecGuard，一个验证感知的投机解码框架，仅使用模型内部信号执行步骤级验证。在每个步骤中，SpecGuard采样多个草案候选并选择最一致的步骤，然后使用两个轻量级模型内部信号的集合进行验证：(i)基于注意力的基础分数，衡量对输入和先前接受步骤的归因；(ii)基于log概率的分数，捕获token级置信度。实验表明SpecGuard将准确性提高3.6%同时减少约11%的延迟。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Speculative Decoding",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "Accelerating Speculative Decoding with Block Diffusion Draft Trees (DDTree)",
        "authors": "Liran Ringel, Yaniv Romano",
        "arxiv_id": "2604.12989",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a block diffusion drafter can generate an entire draft block in a single forward pass and achieve state-of-the-art speculative decoding performance, outperforming strong autoregressive drafters such as EAGLE-3. Vanilla DFlash, however, still verifies only a single drafted trajectory per round, potentially limiting its acceptance length. We introduce DDTree (Diffusion Draft Tree), a method that constructs a draft tree directly from the per-position distributions of a block diffusion drafter. Under a fixed node budget, DDTree uses a simple best-first heap algorithm to select the continuations that are most likely to match the target model according to a surrogate defined by the draft model's output. The resulting tree is verified efficiently in a single target model forward pass using an ancestor-only attention mask. Because DDTree builds on DFlash, a leading draft model for speculative decoding, these gains place DDTree among the leading approaches to speculative decoding.",
        "abstract_cn": "投机解码通过使用轻量级草案器提议多个未来token来加速自回归语言模型，目标模型并行验证这些token。DFlash表明块扩散草案器可以在单次前向传播中生成整个草案块，并实现最先进的投机解码性能，超越了EAGLE-3等强自回归草案器。然而，Vanilla DFlash每轮仍只验证单个草案轨迹，可能限制其接受长度。我们引入DDTree(扩散草案树)，一种直接从块扩散草案器的逐位置分布构建草案树的方法。在固定节点预算下，DDTree使用简单的最佳优先堆算法选择最可能匹配目标模型的延续。结果树使用仅祖先注意力掩码在单次目标模型前向传播中高效验证。由于DDTree建立在DFlash之上，这些增益使DDTree跻身投机解码的领先方法之列。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Speculative Decoding",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration",
        "authors": "Xinyu Liu, Xin Liu, Bo Jin, Runsong Zhao, Pengcheng Huang, Junhao Ruan, Bei Li, Chunyang Xiao, Tong Xiao, Jingbo Zhu",
        "arxiv_id": "2604.14889",
        "github_repo": "",
        "conference": "arXiv 2026",
        "year": 2026,
        "abstract_en": "While Chain-of-thought (CoT) reasoning enables LLMs to solve challenging reasoning problems, as KV cache grows linearly with the number of generated tokens, CoT reasoning faces scaling issues in terms of speed and memory usage. In this work, we propose MemoSight (Memory-Foresight-based reasoning), a unified framework that integrates both context compression and multi-token prediction to mitigate the efficiency issues while maintaining CoT reasoning performance. Our framework adopts the same minimalist design for both context compression and multi-token prediction via special tokens and their corresponding position layout tailored to each token type. Comprehensive experiments on four reasoning benchmarks demonstrate that MemoSight reduces the KV cache footprint by up to 66% and accelerates inference by 1.56x, while outperforming existing CoT compression methods.",
        "abstract_cn": "虽然思维链(CoT)推理使LLM能够解决具有挑战性的推理问题，但随着KV缓存随生成的token数量线性增长，CoT推理在速度和内存使用方面面临扩展性问题。在这项工作中，我们提出MemoSight(基于记忆-预见推理)，一个统一框架，集成了上下文压缩和多token预测以缓解效率问题同时保持CoT推理性能。我们的框架对上下文压缩和多token预测采用了相同的最小化设计，通过特殊token及其相应的位置布局为每种token类型量身定制。在四个推理基准上的综合实验表明，MemoSight将KV缓存占用减少高达66%并将推理加速1.56倍。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "KV Cache",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
    {
        "title": "P/D-Serve: Serving Disaggregated Large Language Model at Scale",
        "authors": "Yibo Jin, Tao Wang, Huimin Lin, Mingyang Song, Peiyang Li, Yipeng Ma, Yicheng Shan, Zhengfan Yuan, Cailong Li, Yajing Sun, Tiandeng Wu, Xing Chu, Ruizhi Huan, Li Ma, Xiao You, Wenting Zhou, Yunpeng Ye, Wen Liu, Xiangkun Xu, Yongsheng Zhang, Tiantian Dong, Jiawei Zhu, Zhe Wang, Xijian Ju, Jianxun Song 等",
        "arxiv_id": "2408.08147",
        "github_repo": "",
        "conference": "arXiv 2024",
        "year": 2024,
        "abstract_en": "Serving disaggregated large language models (LLMs) over tens of thousands of xPU devices (GPUs or NPUs) with reliable performance faces multiple challenges. 1) Ignoring the diversity (various prefixes and tidal requests), treating all the prompts in a mixed pool is inadequate. To facilitate the similarity per scenario and minimize the inner mismatch on P/D (prefill and decoding) processing, fine-grained organization is required, dynamically adjusting P/D ratios for better performance. 2) Due to inaccurate estimation on workload (queue status or maintained connections), the global scheduler easily incurs unnecessary timeouts in prefill. 3) Block-fixed device-to-device (D2D) KVCache transfer over cluster-level RDMA (remote direct memory access) fails to achieve desired D2D utilization as expected. To overcome previous problems, this paper proposes an end-to-end system P/D-Serve, which models end-to-end (E2E) P/D performance and enables: 1) fine-grained P/D organization, mapping the service with RoCE as needed; 2) on-demand forwarding upon rejections for idle prefill; and 3) efficient KVCache transfer via optimized D2D access. P/D-Serve is implemented upon Ascend and MindSpore, has been deployed over tens of thousands of NPUs for more than eight months in commercial use, and further achieves 60%, 42% and 46% improvements on E2E throughput, TTFT SLO and D2D transfer time.",
        "abstract_cn": "在数万个xPU设备(GPU或NPU)上服务解耦的大语言模型(LLM)并保持可靠性能面临多重挑战。1)忽略多样性（各种前缀和潮汐请求），将所有提示放在混合池中是不充分的。为了促进每个场景的相似性并最小化P/D(prefill和decoding)处理的内部不匹配，需要细粒度组织，动态调整P/D比率以获得更好的性能。2)由于对工作负载的不准确估计，全局调度器容易在prefill中产生不必要的超时。3)块固定的设备到设备(D2D) KVCache传输通过集群级RDMA无法实现预期的D2D利用率。为了克服这些问题，本文提出端到端系统P/D-Serve，它建模端到端(E2E)P/D性能并启用：1)细粒度P/D组织；2)空闲prefill的按需转发；3)通过优化的D2D访问的高效KVCache传输。P/D-Serve在Ascend和MindSpore上实现，已在数万个NPU上商业部署超过八个月。",
        "intro_en": "",
        "intro_cn": "",
        "topic": "Disaggregated Serving",
        "has_content": True,
        "is_placeholder_arxiv": False,
        "is_github_project": False,
    },
]

def slugify(title):
    """Create a slug from a paper title for filenames."""
    # Remove special chars, keep alphanumeric and spaces
    s = re.sub(r'[^\w\s-]', '', title.lower())
    s = re.sub(r'[\s-]+', '_', s.strip())
    # Truncate
    return s[:60]

def get_dir_for_conference(conference, year):
    """Get directory path based on conference and year."""
    conf_lower = conference.lower()
    if 'arxiv' in conf_lower:
        return f"./arxiv/{year}"
    elif 'asplos' in conf_lower:
        return f"./asplos/{year}"
    elif 'osdi' in conf_lower:
        return f"./osdi/{year}"
    elif 'sosp' in conf_lower:
        return f"./sosp/{year}"
    elif 'nsdi' in conf_lower:
        return f"./nsdi/{year}"
    elif 'sigcomm' in conf_lower:
        return f"./sigcomm/{year}"
    elif 'eurosys' in conf_lower:
        return f"./eurosys/{year}"
    elif 'atc' in conf_lower:
        return f"./atc/{year}"
    elif 'sc' in conf_lower:
        return f"./sc/{year}"
    elif 'mlsys' in conf_lower:
        return f"./mlsys/{year}"
    elif 'acl' in conf_lower:
        return f"./acl/{year}"
    elif 'neurips' in conf_lower or 'nips' in conf_lower:
        return f"./neurips/{year}"
    elif 'iclr' in conf_lower:
        return f"./iclr/{year}"
    elif 'icml' in conf_lower:
        return f"./icml/{year}"
    elif 'emnlp' in conf_lower:
        return f"./emnlp/{year}"
    elif 'dac' in conf_lower:
        return f"./dac/{year}"
    elif 'sigmod' in conf_lower:
        return f"./sigmod/{year}"
    else:
        return f"./other/{year}"

def generate_md_content(paper, arxiv_url):
    """Generate markdown file content for a paper."""
    md = f"# {paper['title']}\n\n"
    md += f"**Authors:** {paper['authors']}\n\n"
    md += f"**Conference:** {paper['conference']}\n\n"
    md += f"**Year:** {paper['year']}\n\n"
    md += f"**ArXiv:** [{paper['arxiv_id']}](<{arxiv_url}>)\n\n"
    if paper.get('github_repo'):
        md += f"**GitHub:** [{paper['github_repo']}]({paper['github_repo']})\n\n"
    md += f"**Topic:** {paper['topic']}\n\n"
    md += "---\n\n"
    md += "## Abstract (English)\n\n"
    md += paper['abstract_en'] + "\n\n"
    md += "## Abstract (Chinese / 中文摘要)\n\n"
    md += paper['abstract_cn'] + "\n\n"
    md += "---\n\n"
    md += "*Auto-collected from arXiv on 2026-04-17*\n"
    return md

def main():
    # Load database
    with open(DB_PATH, 'r') as f:
        db = json.load(f)
    
    # Find max numeric id
    max_id = 0
    for p in db['papers']:
        try:
            max_id = max(max_id, int(p['id']))
        except (ValueError, TypeError):
            pass
    
    existing_titles_lower = set(p['title'].lower() for p in db['papers'])
    
    added_count = 0
    for paper in new_papers:
        if paper['title'].lower() in existing_titles_lower:
            print(f"SKIP (already exists): {paper['title']}")
            continue
        
        max_id += 1
        paper['id'] = max_id
        paper['added_date'] = '2026-04-17'
        
        dir_path = get_dir_for_conference(paper['conference'], paper['year'])
        slug = slugify(paper['title'])
        file_path = f"{dir_path}/{slug}.md"
        paper['file'] = file_path
        
        # Create directory and write markdown file
        full_dir = os.path.join(BASE_DIR, dir_path.lstrip('./'))
        os.makedirs(full_dir, exist_ok=True)
        
        arxiv_url = f"https://arxiv.org/abs/{paper['arxiv_id']}"
        md_content = generate_md_content(paper, arxiv_url)
        
        full_file_path = os.path.join(BASE_DIR, file_path.lstrip('./'))
        with open(full_file_path, 'w') as f:
            f.write(md_content)
        
        db['papers'].append(paper)
        existing_titles_lower.add(paper['title'].lower())
        added_count += 1
        print(f"ADD: {paper['title']} -> {file_path}")
    
    # Save database
    with open(DB_PATH, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    
    print(f"\nTotal added: {added_count} papers")
    print(f"Total in database: {len(db['papers'])} papers")

if __name__ == '__main__':
    main()