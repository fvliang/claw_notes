#!/usr/bin/env python3
"""Add new LLM serving papers found on 2026-04-16"""

import json, os, re, time

DB_PATH = "/home/admin/claw_notes/database.json"

# Load existing database
db = json.load(open(DB_PATH))
existing_titles = set()
for p in db['papers']:
    existing_titles.add(p.get('title', '').strip().lower())

base_ts = int(time.time())
# Handle mixed id formats (some are int, some are str like "paper_xxx_yyy")
id_nums = []
for p in db['papers']:
    pid = p.get('id', 0)
    if isinstance(pid, int):
        id_nums.append(pid)
    elif isinstance(pid, str) and '_' in pid:
        id_nums.append(int(pid.split('_')[-1]))
    else:
        id_nums.append(0)
base_id_num = max(id_nums) if id_nums else 0

# New papers to add (only LLM serving/inference/speculative decoding related)
new_papers_raw = [
    {
        "title": "Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference",
        "arxiv_id": "2604.13634",
        "date": "2026-04-15",
        "venue": "ACL 2026",
        "authors": "Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan",
        "abstract_en": "Speculative decoding accelerates autoregressive generation by letting draft tokens bypass full verification, but conventional frameworks suffer from frequent false rejections, particularly when draft models produce semantically correct but lexically divergent outputs. In this paper, we present Calibrated Speculative Decoding (CSD), a training-free framework that recovers valid tokens discarded by standard verification. Guided by the principle of \"Frequency-Guided Candidate Selection and Probability-Guarded Acceptance,\" CSD incorporates two lightweight modules: Online Correction Memory, which aggregates historical rejections to propose recurring divergence patterns as rescue candidates, and Semantic Consistency Gating, which verifies candidate admissibility using probability ratios instead of exact token matching. Our evaluation across diverse large language models demonstrates that CSD outperforms existing methods, achieving a peak throughput speedup of 2.33x. CSD preserves model accuracy across all tasks while further boosting performance on complex reasoning datasets. These results establish CSD as a highly effective, lightweight solution for practical LLM deployments.",
        "abstract_cn": "投机解码通过让草稿token绕过完整验证来加速自回归生成，但传统框架经常遭受假拒绝，尤其是当草稿模型产生语义正确但词汇不同的输出时。本文提出校准投机解码(CSD)，一个无需训练的框架，用于恢复被标准验证丢弃的有效token。遵循\"频率引导候选选择与概率保护接受\"原则，CSD包含两个轻量模块：在线校正记忆，聚合历史拒绝以提出重复分歧模式作为救援候选；语义一致性门控，使用概率比而非精确token匹配验证候选可接受性。跨多种大语言模型的评估表明CSD优于现有方法，峰值吞吐量加速2.33倍。",
        "url": "https://arxiv.org/abs/2604.13634",
        "tags": ["speculative decoding", "LLM inference", "inference acceleration"],
        "github": "",
    },
    {
        "title": "YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference",
        "arxiv_id": "2604.13556",
        "date": "2026-04-15",
        "venue": "arXiv",
        "authors": "You Wu, Ziheng Chen, Yizhen Zhang, Haoyi Wu, Chengting Yu, Yuchi Xu, Wenbo Su, Bo Zheng, Kewei Tu",
        "abstract_en": "Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models (LLMs). Although they reduce the memory consumption of the KV cache, such methods usually introduce non-negligible performance degradation. In this work, we aim to enhance the performance of YOCO, a cross-layer KV compression method that shares the KVs of the middle layer with the top-half layers. We propose YOCO++, an enhanced YOCO that incorporates a weighted residual connection between the KVs of each bottom-half layer and the bottom layer. Compared to YOCO, YOCO++ increases model capacity while maintaining the same training and inference efficiency. Our experiments show that YOCO++ achieves state-of-the-art performance among the cross-layer KV compression methods at a 50% KV cache compression rate, outperforming the standard Transformer.",
        "abstract_cn": "跨层KV压缩已被证明在大语言模型高效推理中有效。虽然这些方法减少了KV缓存的内存消耗，但通常引入不可忽视的性能退化。本文旨在增强YOCO的性能——一种将中间层KV与上半层共享的跨层KV压缩方法。我们提出YOCO++，在每个下半层和底层KV之间引入加权残差连接的增强版YOCO。相比YOCO，YOCO++增加了模型容量同时保持相同的训练和推理效率。实验表明YOCO++在50%KV缓存压缩率下达到跨层KV压缩方法中最优性能，甚至超越标准Transformer。",
        "url": "https://arxiv.org/abs/2604.13556",
        "tags": ["KV cache", "LLM inference", "inference efficiency"],
        "github": "",
    },
    {
        "title": "ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding",
        "arxiv_id": "2604.13519",
        "date": "2026-04-15",
        "venue": "arXiv",
        "authors": "Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li",
        "abstract_en": "Tool calling has greatly expanded the practical utility of large language models (LLMs) by enabling them to interact with external applications. As LLM capabilities advance, effective tool use increasingly involves multi-step, multi-turn interactions to solve complex tasks. However, the resulting growth in tool interactions incurs substantial latency, posing a key challenge for real-time LLM serving. Through empirical analysis, we find that tool-calling traces are highly structured, conform to constrained schemas, and often exhibit recurring invocation patterns. Motivated by this, we propose ToolSpec, a schema-aware, retrieval-augmented speculative decoding method for accelerating tool calling. ToolSpec exploits predefined tool schemas to generate accurate drafts, using a finite-state machine to alternate between deterministic schema token filling and speculative generation for variable fields. In addition, ToolSpec retrieves similar historical tool invocations and reuses them as drafts to further improve efficiency. ToolSpec presents a plug-and-play solution that can be seamlessly integrated into existing LLM workflows. Experiments across multiple benchmarks demonstrate that ToolSpec achieves up to a 4.2x speedup, substantially outperforming existing training-free speculative decoding methods.",
        "abstract_cn": "工具调用极大扩展了大语言模型的实用价值，使其能与外部应用交互。随着LLM能力提升，有效的工具使用越来越涉及多步骤多轮交互。但工具交互增长带来显著延迟，成为实时LLM服务的关键挑战。通过实证分析，我们发现工具调用轨迹高度结构化，遵循受约束的schema，并常展现重复调用模式。据此提出ToolSpec，一种schema感知、检索增强的投机解码方法，用于加速工具调用。ToolSpec利用预定义工具schema生成准确草稿，使用有限状态机交替确定性schema token填充和变量字段投机生成，还检索相似历史工具调用作为草稿。实验表明ToolSpec实现最高4.2倍加速。",
        "url": "https://arxiv.org/abs/2604.13519",
        "tags": ["speculative decoding", "LLM serving", "tool calling", "inference acceleration"],
        "github": "",
    },
    {
        "title": "Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel",
        "arxiv_id": "2604.13327",
        "date": "2026-04-14",
        "venue": "MLSys 2026",
        "authors": "Hongyi Jin, Bohan Hou, Guanjie Wang, Ruihang Lai, Jinqi Chen, Zihao Ye, Yaxing Cai, Yixin Dong, Xinhao Cheng, Zhihao Zhang, Yilong Zhao, Yingyi Huang, Lijie Yang, Jinchen Jiang, Gabriele Oliaro, Jianan Ji, Xupeng Miao, Vinod Grover, Todd C. Mowry, Zhihao Jia, Tianqi Chen",
        "abstract_en": "Modern GPU workloads, especially large language model (LLM) inference, suffer from kernel launch overheads and coarse synchronization that limit inter-kernel parallelism. Recent megakernel techniques fuse multiple operators into a single persistent kernel to eliminate launch gaps and expose inter-kernel parallelism, but struggle to handle dynamic shapes and data-dependent computation in real workloads. We present Event Tensor, a unified compiler abstraction for dynamic megakernels. Event Tensor encodes dependencies between tiled tasks, and enables first-class support for both shape and data-dependent dynamism. Built atop this abstraction, our Event Tensor Compiler (ETC) applies static and dynamic scheduling transformations to generate high-performance persistent kernels. Evaluations show that ETC achieves state-of-the-art LLM serving latency while significantly reducing system warmup overhead.",
        "abstract_cn": "现代GPU工作负载（尤其是LLM推理）受内核启动开销和粗粒度同步限制，制约了内核间并行性。最近的megakernel技术将多个算子融合到单个持久内核中以消除启动间隙，但难以处理真实工作负载中的动态形状和数据依赖计算。本文提出Event Tensor，一种用于动态megakernel的统一编译器抽象。Event Tensor编码分块任务间的依赖关系，实现对形状和数据依赖动态性的原生支持。基于此抽象，Event Tensor Compiler (ETC)应用静态和动态调度变换生成高性能持久内核。评估表明ETC实现了最先进的LLM服务延迟，同时显著降低系统预热开销。",
        "url": "https://arxiv.org/abs/2604.13327",
        "tags": ["LLM inference", "megakernel", "GPU kernel", "LLM serving", "compiler"],
        "github": "",
    },
    {
        "title": "KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs",
        "arxiv_id": "2604.13226",
        "date": "2026-04-14",
        "venue": "arXiv",
        "authors": "Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Bing Li, Ulf Schlichtmann",
        "abstract_en": "Large Language Models (LLMs) rely heavily on Key-Value (KV) caching to minimize inference latency. However, standard KV caches are context-dependent: reusing a cached document in a new context requires recomputing KV states to account for shifts in attention distribution. Existing solutions such as CacheBlend, EPIC, and SAM-KV mitigate this issue by selectively recomputing a subset of tokens; however, they still incur non-negligible computational overhead (FLOPs) and increased Time-to-First-Token (TTFT) latency. In this paper, we propose KV Packet, a recomputation-free cache reuse framework that treats cached documents as immutable \"packets\" wrapped in light-weight trainable soft-token adapters, which are trained via self-supervised distillation to bridge context discontinuities. Experiments on Llama-3.1 and Qwen2.5 demonstrate that the proposed KV Packet method achieves near-zero FLOPs and lower TTFT than recomputation-based baselines, while retaining F1 scores comparable to those of the full recomputation baseline.",
        "abstract_cn": "大语言模型严重依赖KV缓存来最小化推理延迟。但标准KV缓存是上下文依赖的：在新上下文中复用缓存文档需要重新计算KV状态以适应注意力分布变化。现有解决方案如CacheBlend、EPIC和SAM-KV通过选择性重计算部分token来缓解，但仍引入不可忽视的计算开销和增加的TTFT延迟。本文提出KV Packet，一种免重计算的缓存复用框架，将缓存文档视为不可变\"包\"，包裹在轻量可训练软token适配器中，通过自监督蒸馏训练来桥接上下文不连续性。Llama-3.1和Qwen2.5上的实验表明KV Packet实现近零FLOPs和更低的TTFT。",
        "url": "https://arxiv.org/abs/2604.13226",
        "tags": ["KV cache", "LLM inference", "context-independent caching"],
        "github": "",
    },
    {
        "title": "Latent-Condensed Transformer for Efficient Long Context Modeling",
        "arxiv_id": "2604.12452",
        "date": "2026-04-14",
        "venue": "ACL 2026",
        "authors": "Zeng You, Yaofo Chen, Qiuwu Chen, Ying Sun, Shuhai Zhang, Yingjian Li, Yaowei Wang, Mingkui Tan",
        "abstract_en": "Large language models (LLMs) face significant challenges in processing long contexts due to the linear growth of the key-value (KV) cache and quadratic complexity of self-attention. Existing approaches address these bottlenecks separately: Multi-head Latent Attention (MLA) reduces the KV cache by projecting tokens into a low-dimensional latent space, while sparse attention reduces computation. However, sparse methods cannot operate natively on MLA's compressed latent structure, missing opportunities for joint optimization. In this paper, we propose Latent-Condensed Attention (LCA), which directly condenses context within MLA's latent space, where the representation is disentangled into semantic latent vectors and positional keys. LCA separately aggregates semantic vectors via query-aware pooling and preserves positional keys via anchor selection. This approach jointly reduces both computational cost and KV cache without adding parameters. Beyond MLA, LCA's design is architecture-agnostic and readily extends to other attention mechanisms such as GQA. Theoretically, we prove a length-independent error bound. Experiments show LCA achieves up to 2.5× prefilling speedup and 90% KV cache reduction at 128K context while maintaining competitive performance.",
        "abstract_cn": "大语言模型在处理长上下文时面临显著挑战，原因是KV缓存的线性增长和自注意力的二次复杂度。现有方法分别解决这些瓶颈：多头潜在注意力(MLA)通过将token投影到低维潜在空间减少KV缓存，稀疏注意力减少计算量。但稀疏方法无法原生操作MLA压缩的潜在结构，错过了联合优化机会。本文提出潜在压缩注意力(LCA)，直接在MLA潜在空间中压缩上下文。LCA通过查询感知池化聚合语义向量，通过锚点选择保留位置键，联合降低计算成本和KV缓存而不增加参数。实验显示LCA在128K上下文下实现2.5倍预填充加速和90%KV缓存减少。",
        "url": "https://arxiv.org/abs/2604.12452",
        "tags": ["KV cache", "long context", "LLM inference", "attention optimization"],
        "github": "",
    },
    {
        "title": "Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning",
        "arxiv_id": "2604.12374",
        "date": "2026-04-14",
        "venue": "arXiv",
        "authors": "NVIDIA Team",
        "abstract_en": "We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the first model in the Nemotron 3 family to 1) be pre-trained in NVFP4, 2) leverage LatentMoE, a new Mixture-of-Experts architecture that optimizes for both accuracy per FLOP and accuracy per parameter, and 3) include MTP layers for inference acceleration through native speculative decoding. We pre-trained Nemotron 3 Super on 25 trillion tokens followed by post-training using supervised fine tuning (SFT) and reinforcement learning (RL). The final model supports up to 1M context length and achieves comparable accuracy on common benchmarks, while also achieving up to 2.2x and 7.5x higher inference throughput compared to GPT-OSS-120B and Qwen3.5-122B, respectively. Nemotron 3 Super datasets, along with the base, post-trained, and quantized checkpoints, are open-sourced on HuggingFace.",
        "abstract_cn": "本文描述了Nemotron 3 Super的预训练、后训练和量化——一个120亿(活跃12亿)参数的混合Mamba-Attention MoE模型。Nemotron 3 Super是Nemotron 3系列中首个1)以NVFP4预训练、2)利用LatentMoE新MoE架构优化每FLOP和每参数精度、3)包含MTP层通过原生投机解码加速推理的模型。在25万亿token上预训练后进行SFT和RL后训练。最终模型支持1M上下文长度，推理吞吐量相比GPT-OSS-120B和Qwen3.5-122B分别高达2.2倍和7.5倍提升。模型数据集和检查点已在HuggingFace开源。",
        "url": "https://arxiv.org/abs/2604.12374",
        "tags": ["MoE", "speculative decoding", "LLM inference", "hybrid model", "inference throughput"],
        "github": "https://huggingface.co/nvidia/Nemotron-3-Super",
    },
    {
        "title": "A Full-Stack Performance Evaluation Infrastructure for 3D-DRAM-based LLM Accelerators",
        "arxiv_id": "2604.08044",
        "date": "2026-04-09",
        "venue": "arXiv",
        "authors": "Cong Li, Chenhao Xue, Yi Ren, Xiping Dong, Yu Cheng, Yinbo Hu, Fujun Bai, Yixin Guo, Xiping Jiang, Qiang Wu, Zhi Yang, Zhe Cheng, Yuan Xie, Guangyu Sun",
        "abstract_en": "Large language models (LLMs) exhibit memory-intensive behavior during decoding, making it a key bottleneck in LLM inference. To accelerate decoding execution, hybrid-bonding-based 3D-DRAM has been adopted in LLM accelerators. While this emerging technology provides strong performance gains over existing hardware, current 3D-DRAM accelerators (3D-Accelerators) rely on closed-source evaluation tools, limiting access to publicly available performance analysis methods. Moreover, existing designs are highly customized for specific scenarios, lacking a general and reusable full-stack modeling for 3D-Accelerators across diverse use cases. To bridge this fundamental gap, we present ATLAS, the first silicon-proven Architectural Three-dimensional-DRAM-based LLM Accelerator Simulation framework. Built on commercially deployed multi-layer 3D-DRAM technology, ATLAS introduces unified abstractions for both 3D-Accelerator system architecture and programming primitives to support arbitrary LLM inference scenarios. Validation against real silicon shows that ATLAS achieves ≤8.57% simulation error and 97.26-99.96% correlation with measured performance. ATLAS will be open-sourced upon publication.",
        "abstract_cn": "大语言模型在解码阶段表现出内存密集行为，成为LLM推理的关键瓶颈。为加速解码执行，混合键合3D-DRAM已被用于LLM加速器。虽然此新兴技术提供了强大的性能提升，但当前3D-DRAM加速器依赖闭源评估工具，缺乏公开的性能分析方法。现有设计高度定制特定场景，缺乏通用可复用的全栈建模。为填补这一差距，我们提出ATLAS——首个经过硅验证的3D-DRAM LLM加速器仿真框架。基于商用部署的多层3D-DRAM技术，ATLAS引入统一抽象支持任意LLM推理场景。真实硅片验证显示ATLAS仿真误差≤8.57%，相关性97.26-99.96%。",
        "url": "https://arxiv.org/abs/2604.08044",
        "tags": ["LLM accelerator", "3D-DRAM", "LLM inference", "hardware acceleration"],
        "github": "",
    },
    {
        "title": "Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions",
        "arxiv_id": "2604.07931",
        "date": "2026-04-09",
        "venue": "arXiv",
        "authors": "Jing Wang, Yu-Yang Qian, Ke Xue, Chao Qian, Peng Zhao, Zhi-Hua Zhou",
        "abstract_en": "Output-length prediction is important for efficient LLM serving, as it directly affects batching, memory reservation, and scheduling. For prompt-only length prediction, most existing methods use a one-shot sampled length as the label, implicitly treating each prompt as if it had one true target length. We show that this is unreliable: even under a fixed model and decoding setup, the same prompt induces a prompt-conditioned output length distribution, not a deterministic scalar, and this distribution is consistent with heavy-tailed behavior. Motivated by this, we cast length prediction as robust estimation from heavy-tailed prompt-conditioned length distributions. We propose prompt-conditioned length distribution (ProD) methods, which construct training targets from multiple independent generations of the same prompt. Two variants are developed to reuse the served LLM's hidden states: ProD-M, which uses a median-based target for robust point prediction, and ProD-D, which uses a distributional target that preserves prompt-conditioned uncertainty. We provide theoretical justifications by analyzing the estimation error under a surrogate model. Experiments across diverse scenarios show consistent gains in prediction quality.",
        "abstract_cn": "输出长度预测对高效LLM服务至关重要，直接影响批处理、内存预留和调度。对于仅基于提示的长度预测，大多数现有方法使用单次采样长度作为标签，隐式地将每个提示视为有一个真实目标长度。我们表明这是不可靠的：即使在固定模型和解码设置下，同一提示也产生提示条件输出长度分布而非确定性标量，且该分布呈现重尾行为。据此，我们将长度预测建模为重尾提示条件分布的鲁棒估计。提出ProD方法，从同一提示的多次独立生成构建训练目标：ProD-M使用中位数目标进行鲁棒点预测，ProD-D使用分布目标保留提示条件不确定性。",
        "url": "https://arxiv.org/abs/2604.07931",
        "tags": ["LLM serving", "length prediction", "scheduling", "batching"],
        "github": "",
    },
    {
        "title": "Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate",
        "arxiv_id": "2604.07874",
        "date": "2026-04-09",
        "venue": "arXiv",
        "authors": "Fangyue Liu, Hua Liu, Xinyuan Lyu, Shuo Ai, Hao Liang, Lingpeng Chen, Ziqian Hu, Chong Zha, Xin Jin, Hanmei Luo, Peng Chen",
        "abstract_en": "LLM inference powers latency-critical production services nowadays. The bursty nature of inference traffic results in over-provisioning, which in turn leads to resource underutilization. While online-offline colocation promises to utilize idle capacity, broad production deployment must overcome two major challenges: (i) large online interference due to slow or frequent preemptions, and (ii) extensive frameworks and drivers modifications, to colocate different models and support preemptions. We present Valve, a production-friendly colocation system that jointly bounds preemption latency and preemption rate. Specifically, Valve enables sub-millisecond compute preemption at most once per online request, and rate-limited sub-layer memory reclamation. Deployed on 8,054 GPUs in production, Valve improves cluster utilization by 34.6%, which translates to a 2,170 GPU save. This efficiency gains is achieved with minimal online interference, incurring <5% TTFT increase and <2% TPOT increase across workloads.",
        "abstract_cn": "LLM推理目前驱动延迟敏感的生产服务。推理流量的突发性导致过度配置，进而造成资源利用率低下。虽然在线-离线混合部署有望利用闲置容量，但广泛生产部署需克服两大挑战：(i)缓慢或频繁抢占导致的大量在线干扰，(ii)大量框架和驱动修改。我们提出Valve，一种生产友好的混合部署系统，联合约束抢占延迟和抢占率。Valve实现亚毫秒级计算抢占（每个在线请求最多一次）和速率限制的子层内存回收。在8,054 GPU的生产部署中，Valve将集群利用率提高34.6%（相当于节省2,170 GPU），在线干扰极小（TTFT增加<5%，TPOT增加<2%）。",
        "url": "https://arxiv.org/abs/2604.07874",
        "tags": ["LLM serving", "GPU colocation", "inference serving", "production deployment"],
        "github": "",
    },
    {
        "title": "AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention",
        "arxiv_id": "2604.07815",
        "date": "2026-04-09",
        "venue": "arXiv",
        "authors": "Yuxuan Hu, Jianchao Tan, Jiaqi Zhang, Wen Zan, Pingwei Sun, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai, Jing Zhang",
        "abstract_en": "Long-context inference in LLMs faces the dual challenges of quadratic attention complexity and prohibitive KV cache memory. While token-level sparse attention offers superior accuracy, its indexing overhead is costly; block-level methods improve efficiency but sacrifice precision. We propose AsyncTLS, a hierarchical sparse attention system that combines coarse-grained block filtering with fine-grained token selection to balance accuracy and efficiency, coupled with an asynchronous offloading engine that overlaps KV cache transfers with computation via temporal locality exploitation. Evaluated on Qwen3 and GLM-4.7-Flash across GQA and MLA architectures, AsyncTLS achieves accuracy comparable to full attention while delivering 1.2x - 10.0x operator speedups and 1.3x - 4.7x end-to-end throughput improvements on 48k - 96k contexts.",
        "abstract_cn": "LLM长上下文推理面临注意力二次复杂度和KV缓存内存的双重挑战。token级稀疏注意力精度更高但索引开销大；block级方法效率更高但牺牲精度。我们提出AsyncTLS，一种分层稀疏注意力系统，结合粗粒度block过滤与细粒度token选择以平衡精度和效率，并配合异步卸载引擎利用时间局部性重叠KV缓存传输与计算。在Qwen3和GLM-4.7-Flash的GQA和MLA架构上评估，AsyncTLS实现与全注意力相当的精度，同时提供1.2-10.0倍算子加速和1.3-4.7倍端到端吞吐提升。",
        "url": "https://arxiv.org/abs/2604.07815",
        "tags": ["KV cache", "sparse attention", "LLM inference", "long context"],
        "github": "",
    },
    {
        "title": "ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge--Cloud Speculative LLM Serving",
        "arxiv_id": "2604.09722",
        "date": "2026-04-08",
        "venue": "TDIS 2026",
        "authors": "Xiangchen Li, Saeid Ghafouri, Jiakun Fan, Babar Ali, Hans Vandierendonck, Dimitrios S. Nikolopoulos",
        "abstract_en": "Speculative decoding enables collaborative Large Language Model (LLM) inference across cloud and edge by separating lightweight token drafting from heavyweight verification. While prior systems show performance and cost benefits, practical deployment requires navigating a large configuration space spanning draft model variants, quantisation levels, speculative lengths, and heterogeneous edge devices. This paper presents ConfigSpec, a configuration selection framework for distributed speculative LLM serving. ConfigSpec profiles edge devices and draft-target alignment, and models drafting throughput, acceptance rate, and power to evaluate goodput, verification cost efficiency, and energy efficiency across the joint configuration space. Our analysis reveals structurally conflicting optima, underscoring the need for profiling-based configuration selection in disaggregated edge-cloud LLM inference.",
        "abstract_cn": "投机解码通过分离轻量token起草与重量级验证，实现了云端和边缘的协作LLM推理。虽然已有系统显示性能和成本优势，但实际部署需导航大型配置空间（草稿模型变体、量化级别、投机长度、异构边缘设备）。本文提出ConfigSpec，一种用于分布式投机LLM服务的配置选择框架。ConfigSpec对边缘设备和草稿-目标对齐进行性能分析，建模起草吞吐量、接受率和功耗以评估联合配置空间中的有效吞吐、验证成本效率和能效。分析揭示了结构性冲突最优解，强调在解耦边缘-云LLM推理中需要基于分析的配置选择。",
        "url": "https://arxiv.org/abs/2604.09722",
        "tags": ["speculative decoding", "LLM serving", "edge-cloud", "configuration optimization"],
        "github": "",
    },
    {
        "title": "DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification",
        "arxiv_id": "2604.07622",
        "date": "2026-04-08",
        "venue": "AISTATS 2026",
        "authors": "Ziyi Wang, Siva Rajesh Kasa, Ankith M S, Santhosh Kumar Kasa, Jiaru Zou, Sumit Negi, Ruqi Zhang, Nan Jiang, Qifan Song",
        "abstract_en": "Speculative decoding is an effective technique for accelerating large language model inference by drafting multiple tokens in parallel. In practice, its speedup is often bottlenecked by a rigid verification step that strictly enforces the accepted token distribution to exactly match the target model. This constraint leads to the rejection of many plausible tokens, lowering the acceptance rate and limiting overall time speedup. To overcome this limitation, we propose Dynamic Verification Relaxed Speculative Decoding (DIVERSED), a relaxed verification framework that improves time efficiency while preserving generation quality. DIVERSED learns an ensemble-based verifier that blends the draft and target model distributions with a task-dependent and context-dependent weight. We provide theoretical justification and demonstrate empirically that DIVERSED achieves substantially higher inference efficiency compared to standard speculative decoding methods.",
        "abstract_cn": "投机解码是一种通过并行起草多个token加速大语言模型推理的有效技术。实际上，其加速常受刚性验证步骤瓶颈限制——严格要求接受token分布精确匹配目标模型。此约束导致许多合理token被拒绝，降低接受率并限制整体加速。为克服此限制，我们提出DIVERSED（动态验证放宽投机解码），一种放宽验证框架，提高时间效率同时保持生成质量。DIVERSED学习集成验证器，以任务依赖和上下文依赖的权重混合草稿和目标模型分布。理论分析和实证表明DIVERSED相比标准投机解码方法实现显著更高的推理效率。",
        "url": "https://arxiv.org/abs/2604.07622",
        "tags": ["speculative decoding", "LLM inference", "verification relaxation"],
        "github": "https://github.com/comeusr/diversed",
    },
    {
        "title": "Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC",
        "arxiv_id": "2604.07609",
        "date": "2026-04-08",
        "venue": "arXiv",
        "authors": "Mohammad Siavashi, Mariano Scazzariello, Gerald Q. Maguire, Dejan Kostić, Marco Chiesa",
        "abstract_en": "Large Language Model (LLM) inference is rapidly becoming a core datacenter service, yet current serving stacks keep the host CPU on the critical path for orchestration and token-level control. This makes LLM performance sensitive to CPU interference, undermining application colocation and forcing operators to reserve CPU headroom, leaving substantial capacity unutilized. We introduce Blink, an end-to-end serving architecture that removes the host CPU from the steady-state inference path by redistributing responsibilities across a SmartNIC and a GPU. Blink offloads request handling to the SmartNIC, which delivers inputs directly into GPU memory via RDMA, and replaces host-driven scheduling with a persistent GPU kernel that performs batching, scheduling, and KV-cache management without CPU involvement. Evaluated against TensorRT-LLM, vLLM, and SGLang, Blink outperforms all baselines even in isolation, reducing pre-saturation P99 TTFT by up to 8.47× and P99 TPOT by up to 3.40×, improving decode throughput by up to 2.1×, and reducing energy per token by up to 48.6%. Under CPU interference, Blink maintains stable performance, while existing systems degrade by up to two orders of magnitude.",
        "abstract_cn": "LLM推理正成为数据中心核心服务，但当前服务栈将主机CPU保留在编排和token级控制的关键路径上。这使LLM性能对CPU干扰敏感，削弱应用混合部署并迫使运营商预留CPU余量，留下大量闲置容量。我们提出Blink，一种端到端服务架构，通过将职责重新分配到SmartNIC和GPU来移除稳态推理路径中的主机CPU。Blink将请求处理卸载到SmartNIC（通过RDMA直接将输入送入GPU内存），并用持久GPU内核替代主机驱动调度，无CPU参与地执行批处理、调度和KV缓存管理。相比TensorRT-LLM、vLLM和SGLang，Blink降低P99 TTFT最高8.47倍、P99 TPOT最高3.40倍，解码吞吐提升2.1倍，每token能耗降低48.6%。在CPU干扰下Blink保持稳定，现有系统退化高达两个数量级。",
        "url": "https://arxiv.org/abs/2604.07609",
        "tags": ["LLM serving", "SmartNIC", "GPU", "CPU-free", "inference architecture"],
        "github": "",
    },
    {
        "title": "Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference",
        "arxiv_id": "2604.07472",
        "date": "2026-04-08",
        "venue": "arXiv",
        "authors": "Jiaming Cheng, Duong Tung Nguyen",
        "abstract_en": "Deploying large language model (LLM) inference at scale requires jointly selecting base models, provisioning heterogeneous GPUs, configuring parallelism, and distributing workloads under tight latency, accuracy, and budget constraints. Exact mixed-integer linear programming (MILP) approaches guarantee optimality but scale poorly. We propose two constraint-aware heuristics: a Greedy Heuristic (GH) for single-pass allocation, and an Adaptive Greedy Heuristic (AGH) that enhances GH via multi-start construction, relocate-based local search, and GPU consolidation. On workloads calibrated with the Azure LLM Inference Trace (2025), both heuristics produce feasible solutions in under one second, with AGH closely approaching optimal cost while achieving over 260x speedup on large-scale instances.",
        "abstract_cn": "大规模部署LLM推理需要联合选择基础模型、配置异构GPU、配置并行性并在严格的延迟、精度和预算约束下分配工作负载。精确MILP方法保证最优性但扩展性差。我们提出两种约束感知启发式方法：贪心启发式(GH)用于单次分配，自适应贪心启发式(AGH)通过多起始构造、基于迁移的局部搜索和GPU整合增强GH。在基于Azure LLM推理Trace(2025)校准的工作负载上，两种启发式在一秒内生成可行解，AGH接近最优成本并在大规模实例上实现260倍以上加速。",
        "url": "https://arxiv.org/abs/2604.07472",
        "tags": ["LLM serving", "heterogeneous GPU", "SLO", "scheduling", "resource allocation"],
        "github": "",
    },
    {
        "title": "Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics",
        "arxiv_id": "2604.07144",
        "date": "2026-04-08",
        "venue": "arXiv",
        "authors": "Youhe Jiang, Ran Yan, You Peng, Wenshuang Li, Taiyi Wang, Fangcheng Fu, Binhang Yuan",
        "abstract_en": "Modern Large Language Model (LLM) serving operates in highly volatile environments characterized by severe runtime dynamics, such as workload fluctuations and elastic cluster autoscaling. Traditional serving systems rely on static, human-engineered serving policies to manage these dynamics. However, these policies must navigate deeply intertwined runtime trade-offs whose optimal balance is workload-specific and shifts continuously, rendering any fixed policy fundamentally unable to adapt. We propose Autopoiesis, a novel online self-evolving system that shifts LLM serving from static policy deployment to continuous online policy evolution. Autopoiesis introduces an LLM-driven program synthesis workflow to evolve serving policies with respect to real-time observed dynamics. We evaluate Autopoiesis across diverse runtime dynamics and show up to 53% and on average 34% improvements over state-of-the-art LLM serving systems.",
        "abstract_cn": "现代LLM服务在高度动态的环境中运行，面临工作负载波动和弹性集群自动扩展等严重运行时动态。传统服务系统依赖静态人工工程策略来管理这些动态，但这些策略必须导航深度交织的运行时权衡，其最优平衡是工作负载特定的且持续变化，使任何固定策略根本无法适应。我们提出Autopoiesis，一种新型在线自进化系统，将LLM服务从静态策略部署转向持续在线策略演化。Autopoiesis引入LLM驱动程序合成工作流，根据实时观察的动态演化服务策略。跨多种运行时动态评估显示，相比最先进LLM服务系统，Autopoiesis实现最高53%和平均34%的改进。",
        "url": "https://arxiv.org/abs/2604.07144",
        "tags": ["LLM serving", "self-evolving", "dynamic scheduling", "runtime adaptation"],
        "github": "",
    },
    {
        "title": "MARS: Enabling Autoregressive Models Multi-Token Generation",
        "arxiv_id": "2604.07023",
        "date": "2026-04-08",
        "venue": "arXiv",
        "authors": "(Authors from paper)",
        "abstract_en": "Autoregressive (AR) language models generate text one token at a time, even when consecutive tokens are highly predictable given earlier context. We introduce MARS (Mask AutoRegreSsion), a lightweight fine-tuning method that teaches an instruction-tuned AR model to predict multiple tokens per forward pass. MARS adds no architectural modifications, no extra parameters, and produces a single model that can still be called exactly like the original AR model with no performance degradation. Unlike speculative decoding, which maintains a separate draft model alongside the target, or multi-head approaches such as Medusa, MARS requires only continued training on existing instruction data. When generating one token per forward pass, MARS matches or exceeds the AR baseline on six standard benchmarks. When allowed to accept multiple tokens per step, it maintains baseline-level accuracy while achieving 1.5-1.7x throughput. We further develop a block-level KV caching strategy for batch inference, achieving up to 1.71x wall-clock speedup over AR with KV cache on Qwen2.5-7B.",
        "abstract_cn": "自回归语言模型逐token生成文本，即使连续token在给定先前上下文时高度可预测。我们提出MARS（掩码自回归），一种轻量微调方法，教指令调优的AR模型每次前向传播预测多个token。MARS不添加架构修改或额外参数，产生的单个模型仍可像原始AR模型一样调用且无性能退化。不同于投机解码需要单独草稿模型，或Medusa等多头方法需要额外预测头，MARS仅需在现有指令数据上继续训练。当允许每步接受多个token时，维持基线精度同时实现1.5-1.7倍吞吐。进一步开发了block级KV缓存策略，在Qwen2.5-7B上实现最高1.71倍墙钟加速。",
        "url": "https://arxiv.org/abs/2604.07023",
        "tags": ["multi-token generation", "LLM inference", "speculative decoding", "KV cache"],
        "github": "",
    },
    {
        "title": "When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration",
        "arxiv_id": "2604.13349",
        "date": "2026-04-14",
        "venue": "arXiv",
        "authors": "Yiping Li, Zhiyu An, Wan Du",
        "abstract_en": "Communication in Large Language Model (LLM)-based multi-agent systems is moving beyond discrete tokens to preserve richer context. Recent work such as LatentMAS enables agents to exchange latent messages through full key-value (KV) caches. However, full KV relay incurs high memory and communication cost. We adapt eviction-style KV compression to this setting and introduce Orthogonal Backfill (OBF) to mitigate information loss from hard eviction. OBF injects a low-rank orthogonal residual from discarded KV states into the retained KV states. It achieves performance comparable to full KV relay while reducing communication cost by 79.8%--89.4%.",
        "abstract_cn": "基于LLM的多智能体系统通信正超越离散token以保留更丰富的上下文。最近的工作如LatentMAS使智能体能通过完整KV缓存交换潜在消息。但完整KV中继产生高内存和通信成本。我们在此场景中适配驱逐式KV压缩并引入正交回填(OBF)以缓解硬驱逐的信息损失。OBF将丢弃KV状态的低秩正交残差注入保留的KV状态。在9个标准基准上实现与完整KV中继相当的性能，同时降低通信成本79.8%-89.4%。",
        "url": "https://arxiv.org/abs/2604.13349",
        "tags": ["KV cache", "multi-agent", "LLM communication", "compression"],
        "github": "https://github.com/markli404/When-Less-Latent-Leads-to-Better-Relay",
    },
]

# Filter out papers that already exist
new_papers_to_add = []
for p in new_papers_raw:
    title_lower = p["title"].strip().lower()
    if title_lower not in existing_titles:
        new_papers_to_add.append(p)
    else:
        print(f"SKIPPED (duplicate): {p['title']}")

print(f"\nNew papers to add: {len(new_papers_to_add)}")

# Format and add papers to database
for i, p in enumerate(new_papers_to_add):
    paper_id = f"paper_{base_ts}_{base_id_num + i + 1}"
    
    # Determine year and venue for directory structure
    year = p["date"][:4]  # e.g. "2026"
    venue = p["venue"]
    
    paper_entry = {
        "id": paper_id,
        "title": p["title"],
        "authors": p["authors"],
        "date": p["date"],
        "venue": venue,
        "year": year,
        "abstract_en": p["abstract_en"],
        "abstract_cn": p["abstract_cn"],
        "url": p["url"],
        "arxiv_id": p["arxiv_id"],
        "tags": p["tags"],
        "github": p["github"],
        "introduction_en": "",  # Would need full paper fetch for this
        "introduction_cn": "",
        "blog_content": "",
    }
    
    db["papers"].append(paper_entry)
    
    # Create directory structure
    dir_path = f"/home/admin/claw_notes/{venue}/{year}"
    os.makedirs(dir_path, exist_ok=True)
    
    # Create markdown file for the paper
    md_content = f"""# {p['title']}

- **Authors:** {p['authors']}
- **Date:** {p['date']}
- **Venue:** {venue}
- **arXiv ID:** {p['arxiv_id']}
- **URL:** [{p['url']}]({p['url']})
- **Tags:** {', '.join(p['tags'])}
{f"- **GitHub:** [{p['github']}]({p['github']})" if p['github'] else ""}

## Abstract (English)

{p['abstract_en']}

## Abstract (Chinese / 中文摘要)

{p['abstract_cn']}

## Introduction (English)

*(To be added)*

## Introduction (Chinese / 中文引言)

*(To be added)*

## Blog Content

*(To be added)*

## GitHub Description

*(To be added)*

---
*Auto-collected on 2026-04-16*
"""
    
    md_filename = f"{p['arxiv_id']}_{p['title'].replace(' ', '_').replace(':', '').replace('/', '_').replace('--', '-').replace(',', '')[:80]}.md"
    md_path = os.path.join(dir_path, md_filename)
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"Added: [{paper_id}] {p['title']} -> {md_path}")

# Save updated database
with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2, ensure_ascii=False)

print(f"\nTotal papers in database: {len(db['papers'])}")
print(f"New papers added this run: {len(new_papers_to_add)}")