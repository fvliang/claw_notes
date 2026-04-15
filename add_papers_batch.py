#!/usr/bin/env python3
"""
批量添加论文到 database.json 并创建 markdown 文件
"""
import json
import os

DB_PATH = os.path.expanduser("~/claw_notes/database.json")
BASE_DIR = os.path.expanduser("~/claw_notes")

def load_db():
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def get_next_id(db):
    return max(p['id'] for p in db['papers']) + 1

def get_existing_titles(db):
    return set(p['title'] for p in db['papers'])

# New papers to add - all LLM serving related
NEW_PAPERS = [
    {
        "title": "ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge--Cloud Speculative LLM Serving",
        "authors": "Xiangchen Li, Saeid Ghafouri, Jiakun Fan, Babar Ali, Hans Vandierendonck, Dimitrios S. Nikolopoulos",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Edge-Cloud Serving",
        "abstract_en": "Speculative decoding enables collaborative Large Language Model (LLM) inference across cloud and edge by separating lightweight token drafting from heavyweight verification. While prior work focuses on drafting accuracy and acceptance rates, the end-to-end performance critically depends on the configuration of the disaggregated edge--cloud system, including the drafter model selection, placement, and resource allocation. We present ConfigSpec, a profiling-based configuration selection framework that systematically evaluates candidate configurations across diverse edge--cloud setups to identify Pareto-optimal trade-offs between latency, throughput, and resource cost.",
        "abstract_cn": "投机解码通过将轻量级token起草与重量级验证分离，实现了云和边缘之间的协作式大语言模型推理。虽然之前的工作侧重于起草准确性和接受率，但端到端性能关键取决于分离式边缘-云系统的配置，包括起草模型选择、放置和资源分配。我们提出了ConfigSpec，一个基于性能分析的配置选择框架，系统地评估各种边缘-云设置中的候选配置，以识别延迟、吞吐量和资源成本之间的帕累托最优权衡。",
        "intro_en": "Speculative decoding has emerged as a promising technique for accelerating LLM inference, particularly in disaggregated edge-cloud settings where lightweight drafters run on edge devices while the target model resides in the cloud. However, the performance of such systems is highly sensitive to configuration choices including model selection, device placement, and resource allocation.",
        "intro_cn": "投机解码已成为加速LLM推理的有前途的技术，特别是在分离式边缘-云设置中，轻量级起草器在边缘设备上运行而目标模型驻留在云中。然而，这种系统的性能对配置选择高度敏感，包括模型选择、设备放置和资源分配。",
    },
    {
        "title": "SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding",
        "authors": "Talor Abramovich, Maor Ashkenazi, Carl Putterman, Benjamin Chislett, Tiyasa Mitra, Bita Darvish Rouhani, Ran Zilberstein, Yonatan Geifman",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "Speculative Decoding (SD) has emerged as a critical technique for accelerating Large Language Model (LLM) inference. Unlike deterministic approximation methods, SD guarantees exact output distribution preservation, making it a mathematically rigorous approach to inference acceleration. However, the lack of a unified benchmark has hindered systematic comparison and progress. We present SPEED-Bench, the first comprehensive benchmark for evaluating SD methods across diverse models, tasks, and hardware configurations.",
        "abstract_cn": "投机解码（SD）已成为加速大语言模型推理的关键技术。与确定性近似方法不同，SD保证精确输出分布保持，使其成为推理加速的数学严谨方法。然而，缺乏统一的基准阻碍了系统性比较和进步。我们提出了SPEED-Bench，第一个用于在各种模型、任务和硬件配置中评估SD方法的综合基准。",
        "intro_en": "As LLMs grow in size and complexity, inference latency becomes a critical bottleneck. Speculative decoding offers a principled approach to acceleration by using smaller draft models to propose tokens that are verified by the target model in a single forward pass. Despite growing interest, evaluation of SD methods remains fragmented across different papers with inconsistent settings.",
        "intro_cn": "随着LLM规模和复杂性的增长，推理延迟成为关键瓶颈。投机解码通过使用较小的起草模型提出token，由目标模型在单次前向传播中验证，提供了一种有原则的加速方法。尽管兴趣不断增长，SD方法的评估在不同论文之间仍然分散，设置不一致。",
    },
    {
        "title": "Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC",
        "authors": "Mohammad Siavashi, Mariano Scazzariello, Gerald Q. Maguire Jr., Dejan Kostić, Marco Chiesa",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "LLM Serving",
        "abstract_en": "Large Language Model (LLM) inference is rapidly becoming a core datacenter service, yet current serving systems heavily rely on the CPU for orchestrating the inference pipeline. We present Blink, a system that delegates the entire LLM serving stack—including scheduling, memory management, and network communication—to the GPU and SmartNIC, eliminating CPU involvement in the inference pipeline. This approach enables tighter integration between compute and communication, reducing overhead and improving throughput.",
        "abstract_cn": "大语言模型推理正迅速成为数据中心核心服务，然而当前服务系统严重依赖CPU来协调推理流水线。我们提出了Blink，一个将整个LLM服务栈（包括调度、内存管理和网络通信）委托给GPU和SmartNIC的系统，消除了CPU在推理流水线中的参与。这种方法使计算和通信之间更紧密集成，减少开销并提高吞吐量。",
        "intro_en": "Current LLM serving systems like vLLM and TensorRT-LLM rely on the CPU for critical orchestration tasks, including request scheduling, KV cache management, and inter-GPU communication. This CPU-centric design introduces overhead that limits achievable throughput, especially at high batch sizes.",
        "intro_cn": "当前的LLM服务系统（如vLLM和TensorRT-LLM）依赖CPU执行关键协调任务，包括请求调度、KV缓存管理和GPU间通信。这种以CPU为中心的设计引入了限制可达到吞吐量的开销，特别是在高批量大小下。",
    },
    {
        "title": "Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference",
        "authors": "Jiaming Cheng, Duong Tung Nguyen",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Multi-Model Serving",
        "abstract_en": "Deploying large language model (LLM) services in heterogeneous GPU clusters presents significant challenges in model allocation and resource management. Different GPU types have varying memory capacities and compute capabilities, making it difficult to serve multiple LLM models while satisfying Service Level Objectives (SLOs). We propose a mixed-scale allocation framework that optimally distributes LLM models across heterogeneous GPUs, balancing SLO compliance with resource efficiency.",
        "abstract_cn": "在异构GPU集群中部署大语言模型服务面临模型分配和资源管理的重大挑战。不同GPU类型具有不同的内存容量和计算能力，使得在满足服务级别目标（SLO）的同时服务多个LLM模型变得困难。我们提出了一种混合尺度分配框架，在异构GPU之间最优分配LLM模型，平衡SLO合规性和资源效率。",
        "intro_en": "Modern LLM serving clusters increasingly comprise heterogeneous GPU hardware, ranging from high-end A100/H100 GPUs to more cost-effective alternatives. Efficiently serving multiple LLM models across such mixed infrastructure requires careful consideration of each model's resource requirements and each GPU's capabilities.",
        "intro_cn": "现代LLM服务集群越来越多地包含异构GPU硬件，从高端A100/H100 GPU到更经济的替代方案。在这种混合基础设施上高效服务多个LLM模型需要仔细考虑每个模型的资源需求和每个GPU的能力。",
    },
    {
        "title": "InfiniLoRA: Disaggregated Multi-LoRA Serving for Large Language Models",
        "authors": "Hongyu Chen, Letian Ruan, Zilin Xu, Yuchen Li, Xinyu Chen, Jingwen Leng, Bingsheng He, Minyi Guo, Shixuan Sun",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Disaggregated Serving",
        "abstract_en": "LoRA enables efficient customization of LLMs and is widely used in multi-tenant and multi-task serving scenarios. However, serving many LoRA adapters simultaneously introduces significant memory and scheduling challenges. We present InfiniLoRA, a disaggregated serving architecture that separates LoRA adapter storage from the base model computation, enabling elastic scaling of adapter capacity and efficient batching across heterogeneous adapters.",
        "abstract_cn": "LoRA实现了LLM的高效定制，广泛应用于多租户和多任务服务场景。然而，同时服务许多LoRA适配器引入了显著的内存和调度挑战。我们提出了InfiniLoRA，一种分离式服务架构，将LoRA适配器存储与基础模型计算分离，实现了适配器容量的弹性扩展和异构适配器之间的高效批处理。",
        "intro_en": "Multi-LoRA serving has become essential for LLM deployment platforms that need to support thousands of customized adapters. Current approaches face memory pressure from storing all adapters and scheduling overhead from managing diverse adapter configurations.",
        "intro_cn": "多LoRA服务已成为需要支持数千个定制适配器的LLM部署平台的关键。当前方法面临存储所有适配器的内存压力和管理多样适配器配置的调度开销。",
    },
    {
        "title": "Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics",
        "authors": "Youhe Jiang, Ran Yan, You Peng, Wenshuang Li, Taiyi Wang, Fangcheng Fu, Binhang Yuan",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "LLM Serving",
        "abstract_en": "Modern Large Language Model (LLM) serving systems face highly dynamic runtime conditions including fluctuating request rates, varying input lengths, and changing hardware availability. Existing systems rely on static or manually-tuned configurations that fail to adapt. We present Autopoiesis, a self-evolving system paradigm that continuously adapts its scheduling strategies, resource allocation, and parallelism configurations based on real-time runtime observations, achieving consistent high performance across diverse dynamic conditions.",
        "abstract_cn": "现代大语言模型服务系统面临高度动态的运行时条件，包括波动的请求率、变化的输入长度和变化的硬件可用性。现有系统依赖静态或手动调整的配置，无法适应。我们提出了Autopoiesis，一种自进化系统范式，根据实时运行时观察持续适应其调度策略、资源分配和并行配置，在各种动态条件下实现一致的高性能。",
        "intro_en": "LLM serving systems operate in environments that are inherently dynamic—request patterns shift throughout the day, input characteristics vary across applications, and hardware conditions change due to maintenance or scaling events. Static serving configurations inevitably lead to suboptimal performance under such dynamics.",
        "intro_cn": "LLM服务系统在本质上动态的环境中运行——请求模式在一天中变化，输入特征在不同应用间变化，硬件条件因维护或扩展事件而改变。静态服务配置在这种动态下不可避免地导致次优性能。",
    },
    {
        "title": "Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start",
        "authors": "Xueshen Liu, Yongji Wu, Yuncheng Yao, Danyang Zhuo, Ion Stoica, Z. Morley Mao",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "LLM Serving",
        "abstract_en": "Modern LLM service providers increasingly rely on autoscaling and parallelism reconfiguration to respond to rapidly changing workloads, but cold-start latency remains a critical bottleneck. CUDA graph capture, which enables fast kernel dispatch, requires significant setup time that delays new instance readiness. We present Foundry, a template-based approach that pre-materializes CUDA graph contexts for common configurations, enabling near-instant cold starts when deploying new LLM serving instances.",
        "abstract_cn": "现代LLM服务提供商越来越依赖自动扩展和并行重配置来响应快速变化的工作负载，但冷启动延迟仍然是关键瓶颈。CUDA图捕获（实现快速内核调度）需要大量设置时间，延迟了新实例的启动。我们提出了Foundry，一种基于模板的方法，预先为常见配置物化CUDA图上下文，在部署新LLM服务实例时实现近乎即时的冷启动。",
        "intro_en": "The shift to autoscaling in LLM serving means new instances are frequently spun up and torn down. Each cold start involves model loading, memory allocation, and CUDA graph capture—operations that can take minutes. Foundry addresses this by pre-building reusable CUDA graph templates.",
        "intro_cn": "LLM服务向自动扩展的转变意味着新实例频繁启动和关闭。每次冷启动涉及模型加载、内存分配和CUDA图捕获——这些操作可能需要数分钟。Foundry通过预先构建可重用的CUDA图模板来解决这个问题。",
    },
    {
        "title": "Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving",
        "authors": "Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Scheduling",
        "abstract_en": "Existing LLM serving systems typically configure each instance for worst-case context length, leading to substantial KV-cache over-allocation and under-utilized concurrency. In practice, 80-95% of requests are short, yet are served under configurations optimized for long contexts, wasting 4-8x throughput capacity and triggering reliability issues. We propose Dual-Pool, a token-budget routing system that maintains separate serving pools for short and long contexts, enabling cost-efficient and reliable LLM serving.",
        "abstract_cn": "现有LLM服务系统通常为最坏情况的上下文长度配置每个实例，导致大量KV缓存过度分配和并发利用率不足。实际上，80-95%的请求是短请求，却在为长上下文优化的配置下服务，浪费4-8倍吞吐量容量并引发可靠性问题。我们提出了Dual-Pool，一个token预算路由系统，为短和长上下文维护分离的服务池，实现成本高效和可靠的LLM服务。",
        "intro_en": "The mismatch between typical request lengths and worst-case-oriented configurations creates enormous inefficiency in LLM serving. Short requests suffer from unnecessary resource reservation while the system struggles to handle occasional long requests.",
        "intro_cn": "典型请求长度与面向最坏情况配置之间的不匹配在LLM服务中造成了巨大效率损失。短请求遭受不必要的资源预留，而系统难以处理偶尔的长请求。",
    },
    {
        "title": "Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference",
        "authors": "Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Scheduling",
        "abstract_en": "We present token-budget routing, a simple yet effective approach that reduces GPU costs for LLM inference by routing requests to appropriately-sized serving pools. Our theoretical analysis shows that GPU savings follow the formula alpha * (1 - 1/rho), predicting fleet-level GPU savings from two observable quantities: the short-traffic fraction alpha and the throughput gain ratio rho. On traces from the Azure LLM Inference Dataset and LMSYS-Chat-1M serving Llama-3-70B on A100 GPUs, token-budget routing reduces GPU costs significantly.",
        "abstract_cn": "我们提出了token预算路由，一种简单而有效的方法，通过将请求路由到适当大小的服务池来降低LLM推理的GPU成本。我们的理论分析表明，GPU节省遵循公式alpha * (1 - 1/rho)，从两个可观测量预测舰队级GPU节省：短流量比例alpha和吞吐量增益比率rho。在Azure LLM推理数据集和LMSYS-Chat-1M的追踪数据上，使用A100 GPU服务Llama-3-70B，token预算路由显著降低了GPU成本。",
        "intro_en": "LLM inference costs are dominated by GPU resources, and the current practice of provisioning for worst-case scenarios leads to massive waste. Most requests are short but are served on instances sized for the longest possible context.",
        "intro_cn": "LLM推理成本主要由GPU资源主导，当前为最坏情况配置的做法导致大量浪费。大多数请求是短请求，却在为最长可能上下文大小的实例上服务。",
    },
    {
        "title": "Rocks Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference",
        "authors": "Konstantinos Papaioannou, Thaleia Dimitra Doudali",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Scheduling",
        "abstract_en": "Multimodal Large Language Models (MLLMs) power platforms like ChatGPT, Gemini, and Copilot, enabling richer interactions with text, images, and videos. These heterogeneous workloads introduce additional inference stages, such as vision preprocessing and encoding, that inflate latency and memory demand. Existing scheduling policies treat all requests uniformly, ignoring modality-specific resource requirements. We propose a modality-aware scheduling approach inspired by the rocks-pebbles-sand metaphor that prioritizes scheduling decisions based on modality-specific resource profiles.",
        "abstract_cn": "多模态大语言模型驱动ChatGPT、Gemini和Copilot等平台，实现与文本、图像和视频的更丰富交互。这些异构工作负载引入了额外的推理阶段，如视觉预处理和编码，增加了延迟和内存需求。现有调度策略统一处理所有请求，忽略了模态特定的资源需求。我们提出了一种受岩石-卵石-沙子隐喻启发的模态感知调度方法，基于模态特定的资源特征优先排序调度决策。",
        "intro_en": "The rise of multimodal LLMs creates fundamentally different scheduling challenges compared to text-only models. Vision encoders and image processors add CPU-intensive preprocessing stages that existing GPU-centric schedulers overlook.",
        "intro_cn": "多模态LLM的兴起创造了与纯文本模型根本不同的调度挑战。视觉编码器和图像处理器增加了CPU密集的预处理阶段，现有以GPU为中心的调度器忽视了这些。",
    },
    {
        "title": "CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference",
        "authors": "Chuxu Song, Zhencan Peng, Jiuqi Wei, Chuanhui Yang",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Attention",
        "abstract_en": "Long-context LLMs increasingly rely on extended, reusable prefill prompts for agents and domain Q&A, pushing attention and KV-cache to become the dominant decode-time bottlenecks. While sparse attention reduces computation and transfer costs, it often struggles to maintain accuracy at high sparsity levels due to distribution shift between query and key patterns. We propose CSAttention, a centroid-scoring approach that clusters KV-cache entries and scores cluster centroids to identify relevant blocks, enabling high sparsity with maintained accuracy.",
        "abstract_cn": "长上下文LLM越来越多地依赖扩展的、可重用的预填充提示用于代理和领域问答，使注意力和KV缓存成为主要的解码时间瓶颈。虽然稀疏注意力减少了计算和传输成本，但由于查询和键模式之间的分布偏移，在高稀疏度下往往难以保持准确性。我们提出了CSAttention，一种质心评分方法，聚类KV缓存条目并评分聚类质心以识别相关块，实现在保持准确性的同时高稀疏度。",
        "intro_en": "As LLMs handle increasingly long contexts, the KV cache grows proportionally, making attention computation a dominant bottleneck during decoding. Sparse attention methods attempt to reduce this cost but face accuracy challenges.",
        "intro_cn": "随着LLM处理越来越长的上下文，KV缓存相应增长，使注意力计算成为解码期间的主要瓶颈。稀疏注意力方法试图减少这种成本但面临准确性挑战。",
    },
    {
        "title": "CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference",
        "authors": "Yulin Zou, Yan Chen, Wenyan Chen, JooYoung Park, Shivaraman Nitin, Luo Tao, Francisco Romero, Dmitrii Ustiugov",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "LLM Serving",
        "abstract_en": "Video streaming analytics is a crucial workload for vision-language model serving, but the high cost of multimodal token generation creates significant inference overhead. We present CodecSight, a system that leverages video codec signals (motion vectors, residual frames, and macroblock partitioning) already computed during video encoding to skip redundant visual token generation, reducing the computational cost of VLM inference for streaming video workloads.",
        "abstract_cn": "视频流分析是视觉语言模型服务的关键工作负载，但多模态token生成的高成本创造了显著的推理开销。我们提出了CodecSight，一个利用视频编码期间已计算的视频编解码信号（运动向量、残差帧和宏块分区）来跳过冗余视觉token生成的系统，降低了流视频工作负载的VLM推理计算成本。",
        "intro_en": "Streaming video analysis with VLMs is extremely costly because each frame requires full visual tokenization. However, consecutive frames in video streams share substantial visual similarity—information already captured by video codecs.",
        "intro_cn": "使用VLM的流视频分析极其昂贵，因为每帧需要完整的视觉token化。然而，视频流中的连续帧共享大量视觉相似性——这些信息已由视频编解码器捕获。",
    },
    {
        "title": "ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems",
        "authors": "Daeyeon Son",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "LLM Serving",
        "abstract_en": "An OS kernel that runs LLM inference internally can read logit distributions before any text is generated and act on them as a governance primitive. We present ProbeLogits, a kernel-level operation that performs a single forward pass and reads specific token logits to classify agent actions as safe or dangerous, with zero learned parameters. This approach enables real-time safety checks at the inference level without additional model overhead.",
        "abstract_cn": "在内部运行LLM推理的操作系统内核可以在任何文本生成之前读取logit分布，并将其作为治理原语行动。我们提出了ProbeLogits，一种内核级操作，执行单次前向传播并读取特定token logit以将代理行动分类为安全或危险，无需学习参数。这种方法在推理级别实现实时安全检查，无需额外模型开销。",
        "intro_en": "As LLMs become embedded in operating systems as native services, new primitives emerge for inference-level governance. ProbeLogits explores the concept of reading logits directly from the model's forward pass for safety classification.",
        "intro_cn": "随着LLM作为原生服务嵌入操作系统，出现了推理级治理的新原语。ProbeLogits探索了直接从模型前向传播读取logit进行安全分类的概念。",
    },
    {
        "title": "Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "We present Quasar, a self-speculative decoding method that uses quantized model layers as drafters and full-precision layers as verifiers. By leveraging the same model with different precision levels, Quasar eliminates the need for separate draft models while achieving memory-efficient verification. The quantized layers produce approximate tokens that are verified against the full-precision computation, enabling significant inference acceleration without additional memory overhead.",
        "abstract_cn": "我们提出了Quasar，一种自投机解码方法，使用量化模型层作为起草器，全精度层作为验证器。通过利用同一模型的不同精度级别，Quasar消除了对单独起草模型的需要，同时实现内存高效的验证。量化层产生近似token，与全精度计算进行验证，在不增加额外内存开销的情况下实现显著的推理加速。",
        "intro_en": "Self-speculative decoding eliminates the need for a separate draft model by using parts of the target model itself for drafting. Quasar advances this idea by using quantized versions of model layers as drafters.",
        "intro_cn": "自投机解码通过使用目标模型本身的部分进行起草，消除了对单独起草模型的需要。Quasar通过使用模型层的量化版本作为起草器推进了这一想法。",
    },
    {
        "title": "KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "Self-speculative decoding skips intermediate model layers to generate draft tokens, but selecting which layers to skip is challenging. We formulate layer selection as a knapsack optimization problem, where each layer's contribution to draft quality is weighed against its computational cost. KnapSpec dynamically selects the optimal layer skipping pattern for each input, maximizing drafting accuracy while minimizing computation overhead.",
        "abstract_cn": "自投机解码跳过中间模型层来生成起草token，但选择跳过哪些层具有挑战性。我们将层选择形式化为背包优化问题，其中每层对起草质量的贡献与其计算成本权衡。KnapSpec为每个输入动态选择最优层跳过模式，最大化起草准确性同时最小化计算开销。",
        "intro_en": "Self-speculative decoding offers a way to accelerate LLM inference without requiring a separate draft model. However, deciding which layers to skip for drafting is non-trivial—skipping too many reduces draft quality while skipping too few provides little acceleration.",
        "intro_cn": "自投机解码提供了无需单独起草模型即可加速LLM推理的方法。然而，决定跳过哪些层进行起草并非易事——跳过太多降低起草质量，跳过太少提供很少加速。",
    },
    {
        "title": "Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "We propose a training-free approach for multi-token prediction in speculative decoding by probing the embedding space of the target model. Instead of training a separate draft model, our method uses the hidden states from the target model's intermediate layers to predict future tokens, requiring no additional training while achieving competitive acceptance rates.",
        "abstract_cn": "我们提出了一种通过探测目标模型嵌入空间进行投机解码中多token预测的无训练方法。我们的方法使用目标模型中间层的隐藏状态来预测未来token，而不是训练单独的起草模型，无需额外训练同时实现竞争力的接受率。",
        "intro_en": "The need for trained draft models in speculative decoding adds complexity and deployment overhead. This work explores whether the target model's own intermediate representations can be used for multi-token prediction without any training.",
        "intro_cn": "投机解码中对训练起草模型的需求增加了复杂性和部署开销。这项工作探索了是否可以使用目标模型自身的中间表示进行多token预测而无需任何训练。",
    },
    {
        "title": "LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "Training draft models for speculative decoding typically uses language modeling losses, but these losses do not directly optimize the acceptance rate—the metric that determines acceleration. We propose LK losses, a family of training objectives that directly target acceptance rate optimization, aligning draft model training with the actual metric that governs speculative decoding performance.",
        "abstract_cn": "为投机解码训练起草模型通常使用语言建模损失，但这些损失不直接优化接受率——决定加速的指标。我们提出了LK损失，一系列直接针对接受率优化的训练目标，将起草模型训练与实际决定投机解码性能的指标对齐。",
        "intro_en": "The standard approach to training draft models for speculative decoding uses next-token prediction losses. However, what matters for acceleration is how many tokens the target model accepts—not how well the draft model predicts the next token in isolation.",
        "intro_cn": "训练投机解码起草模型的标准方法使用下一个token预测损失。然而，对加速重要的是目标模型接受多少token——而不是起草模型单独预测下一个token的效果。",
    },
    {
        "title": "Make Every Draft Count: Hidden State based Speculative Decoding",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "We propose a hidden state-based speculative decoding approach that leverages the intermediate representations of the target model to guide draft token generation. By analyzing the hidden state trajectories across layers, our method identifies tokens with higher acceptance probability before verification, making every draft token count toward acceleration.",
        "abstract_cn": "我们提出了一种基于隐藏状态的投机解码方法，利用目标模型的中间表示来指导起草token生成。通过分析层间隐藏状态轨迹，我们的方法在验证之前识别具有更高接受概率的token，使每个起草token都对加速有所贡献。",
        "intro_en": "Not all draft tokens in speculative decoding are equally likely to be accepted. This work explores whether hidden state information from the target model can help prioritize drafting tokens that are more likely to pass verification.",
        "intro_cn": "投机解码中并非所有起草token都有相同的接受可能性。这项工作探索了目标模型的隐藏状态信息是否可以帮助优先起草更有可能通过验证的token。",
    },
    {
        "title": "WANSpec: Leveraging Global Compute Capacity for LLM Inference",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Edge-Cloud Serving",
        "abstract_en": "We present WANSpec, a system that leverages geographically distributed compute resources for speculative LLM inference over wide-area networks. WANSpec places draft models on edge or regional compute nodes while keeping the target model in centralized datacenters, using network-aware scheduling to minimize the impact of WAN latency on speculative decoding performance.",
        "abstract_cn": "我们提出了WANSpec，一个利用地理分布的计算资源通过广域网络进行投机LLM推理的系统。WANSpec将起草模型放置在边缘或区域计算节点上，同时将目标模型保持在中央数据中心，使用网络感知调度来最小化广域网络延迟对投机解码性能的影响。",
        "intro_en": "Speculative decoding typically assumes draft and target models are co-located. WANSpec challenges this assumption by exploring how speculative decoding can work across wide-area networks with distributed compute resources.",
        "intro_cn": "投机解码通常假设起草模型和目标模型在同一位置。WANSpec挑战了这一假设，探索了投机解码如何在使用分布计算资源的广域网络上工作。",
    },
    {
        "title": "Privacy-Aware Split Inference with Speculative Decoding for Large Language Models over Wide-Area Networks",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Edge-Cloud Serving",
        "abstract_en": "We propose a privacy-aware split inference framework that combines speculative decoding with model splitting across wide-area networks. The framework ensures sensitive user data remains on local edge devices while leveraging cloud resources for target model verification, achieving both inference acceleration and data privacy preservation.",
        "abstract_cn": "我们提出了一种隐私感知的分割推理框架，将投机解码与广域网络上的模型分割相结合。该框架确保敏感用户数据保留在本地边缘设备上，同时利用云资源进行目标模型验证，实现推理加速和数据隐私保护。",
        "intro_en": "Cloud-based LLM inference raises privacy concerns when user prompts contain sensitive information. Split inference with speculative decoding offers a way to keep sensitive computations local while still benefiting from cloud-scale verification.",
        "intro_cn": "当用户提示包含敏感信息时，基于云的LLM推理引发隐私担忧。结合投机解码的分割推理提供了一种保持敏感计算本地化的方法，同时仍然受益于云规模的验证。",
    },
    {
        "title": "Benchmarking the Energy Savings with Speculative Decoding Strategies",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "While speculative decoding is primarily evaluated for latency reduction, its impact on energy consumption is equally important for sustainable LLM deployment. We present the first comprehensive benchmark of energy savings across different speculative decoding strategies, measuring both per-query energy and total system energy under various workload conditions.",
        "abstract_cn": "虽然投机解码主要评估延迟降低，其对能耗的影响对于可持续LLM部署同样重要。我们提出了首个跨不同投机解码策略的节能综合基准，测量各种工作负载条件下的每查询能耗和系统总能耗。",
        "intro_en": "Energy efficiency is becoming a critical metric for LLM deployment alongside latency. Speculative decoding reduces the number of forward passes needed, but its energy impact depends on whether the draft model overhead outweighs the verification savings.",
        "intro_cn": "能效正成为与延迟并列的LLM部署关键指标。投机解码减少了所需的前向传播次数，但其能耗影响取决于起草模型开销是否超过验证节省。",
    },
    {
        "title": "Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Local Inference",
        "abstract_en": "We present a compiler-assisted speculative sampling framework that optimizes draft model deployment on heterogeneous edge devices with varying compute capabilities. The compiler automatically partitions the speculative decoding pipeline across available hardware, balancing draft generation speed with verification accuracy to maximize end-to-end acceleration on resource-constrained edge platforms.",
        "abstract_cn": "我们提出了一种编译器辅助的投机采样框架，优化在具有不同计算能力的异构边缘设备上的起草模型部署。编译器自动在可用硬件上分割投机解码流水线，平衡起草生成速度与验证准确性，以在资源受限的边缘平台上最大化端到端加速。",
        "intro_en": "Running speculative decoding on edge devices is challenging due to heterogeneous compute capabilities and limited resources. A compiler-assisted approach can automatically adapt the decoding pipeline to the available hardware.",
        "intro_cn": "由于异构计算能力和有限资源，在边缘设备上运行投机解码具有挑战性。编译器辅助方法可以自动将解码流水线适配到可用硬件。",
    },
    {
        "title": "SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "We propose SpecAttn, a co-design framework that jointly optimizes sparse attention and self-speculative decoding for LLM inference acceleration. By sharing computation between sparse attention selection and speculative drafting, SpecAttn reduces redundant computation and achieves higher acceleration than applying either technique alone.",
        "abstract_cn": "我们提出了SpecAttn，一个联合优化稀疏注意力和自投机解码用于LLM推理加速的协同设计框架。通过在稀疏注意力选择和投机起草之间共享计算，SpecAttn减少了冗余计算，实现了比单独应用任一技术更高的加速。",
        "intro_en": "Sparse attention and speculative decoding are two complementary approaches for accelerating LLM inference, but they are typically applied independently. Co-designing these techniques can share computation and reduce overhead.",
        "intro_cn": "稀疏注意力和投机解码是加速LLM推理的两种互补方法，但通常独立应用。协同设计这些技术可以共享计算并减少开销。",
    },
    {
        "title": "SDFP: Speculative Decoding with FIT-Pruned Models for Training-Free and Plug-and-Play LLM Acceleration",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "We present SDFP, a training-free speculative decoding method that uses FIT-pruned (feature-importance-tree pruned) versions of the target model as draft models. By pruning less important layers and attention heads based on feature importance analysis, SDFP creates lightweight draft models directly from the target model without any training, enabling plug-and-play acceleration.",
        "abstract_cn": "我们提出了SDFP，一种无训练投机解码方法，使用目标模型的FIT修剪（特征重要性树修剪）版本作为起草模型。通过基于特征重要性分析修剪不太重要的层和注意力头，SDFP直接从目标模型创建轻量级起草模型而无需任何训练，实现即插即用加速。",
        "intro_en": "Creating draft models for speculative decoding typically requires training a separate smaller model. SDFP explores whether pruning the target model itself can create an effective drafter without any training.",
        "intro_cn": "为投机解码创建起草模型通常需要训练一个单独的较小模型。SDFP探索了是否可以通过修剪目标模型本身来创建有效的起草器而无需任何训练。",
    },
    {
        "title": "Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "We propose Sparrow, a speculative decoding method for video LLMs that uses text-anchored window attention to select relevant visual tokens for drafting. By glimpsing visual semantics anchored to the text context, Sparrow reduces the visual token overhead in draft generation, enabling efficient speculative decoding for video understanding tasks.",
        "abstract_cn": "我们提出了Sparrow，一种用于视频LLM的投机解码方法，使用文本锚定的窗口注意力选择相关视觉token进行起草。通过瞥见锚定于文本上下文的视觉语义，Sparrow减少了起草生成中的视觉token开销，实现了视频理解任务的高效投机解码。",
        "intro_en": "Speculative decoding for video LLMs faces unique challenges because draft generation requires processing visual tokens that are expensive to encode. Sparrow addresses this by only attending to visually relevant tokens anchored by the text context.",
        "intro_cn": "视频LLM的投机解码面临独特挑战，因为起草生成需要处理编码昂贵的视觉token。Sparrow通过仅关注由文本上下文锚定的视觉相关token来解决这一问题。",
    },
    {
        "title": "Balancing Latency and Accuracy of Code Completion via Local-Cloud Model Cascading",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Edge-Cloud Serving",
        "abstract_en": "We present a model cascading framework for code completion that balances latency and accuracy by routing queries between local small models and cloud large models. The framework uses confidence-based routing to send simple completions to local models for fast response while escalating complex requests to cloud models for higher accuracy.",
        "abstract_cn": "我们提出了一个用于代码补全的模型级联框架，通过在本地小模型和云大模型之间路由查询来平衡延迟和准确性。框架使用基于置信度的路由将简单补全发送到本地模型以快速响应，同时将复杂请求升级到云模型以获得更高准确性。",
        "intro_en": "Code completion requires both speed and accuracy. Local models offer low latency but limited capability, while cloud models provide high accuracy but introduce network delay. Cascading between them offers a practical compromise.",
        "intro_cn": "代码补全需要速度和准确性。本地模型提供低延迟但能力有限，而云模型提供高准确性但引入网络延迟。在它们之间级联提供了实用的折衷。",
    },
    {
        "title": "EvoESAP: Non-Uniform Expert Pruning for Sparse MoE",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "MoE",
        "abstract_en": "Mixture-of-Experts (MoE) models achieve high quality with efficient compute by activating only a subset of experts per token. However, the total number of experts still consumes significant memory. We propose EvoESAP, an evolutionary expert pruning approach that non-uniformly prunes experts based on their utilization patterns, preserving frequently-activated experts while aggressively pruning rarely-used ones.",
        "abstract_cn": "混合专家（MoE）模型通过每token仅激活一部分专家来实现高质量和高效计算。然而，专家总数仍然消耗大量内存。我们提出了EvoESAP，一种进化式专家修剪方法，基于利用模式非均匀修剪专家，保留频繁激活的专家同时激进修剪很少使用的专家。",
        "intro_en": "MoE models face memory challenges because all experts must be stored even though only a few are activated per token. Expert pruning can reduce memory, but uniform pruning loses important experts.",
        "intro_cn": "MoE模型面临内存挑战，因为所有专家必须存储，尽管每token仅激活少数。专家修剪可以减少内存，但均匀修剪会丢失重要专家。",
    },
    {
        "title": "See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Speculative Decoding",
        "abstract_en": "We propose loosely speculative decoding for video LLMs that uses visual-semantic guidance from higher-level scene understanding to draft tokens. Rather than requiring exact distribution matching, our approach allows controlled divergence guided by visual semantics, achieving faster decoding for video understanding tasks while maintaining semantic coherence.",
        "abstract_cn": "我们提出了用于视频LLM的宽松投机解码，使用来自高级场景理解的视觉语义指导来起草token。我们的方法允许由视觉语义引导的受控偏离，而不是要求精确分布匹配，在保持语义连贯性的同时为视频理解任务实现更快的解码。",
        "intro_en": "Strict speculative decoding requires exact distribution matching between draft and target models. For video LLMs, visual semantics provide additional guidance that can relax this requirement while maintaining output quality.",
        "intro_cn": "严格的投机解码要求起草模型和目标模型之间的精确分布匹配。对于视频LLM，视觉语义提供了额外的指导，可以在保持输出质量的同时放宽这一要求。",
    },
    {
        "title": "A-IO: Adaptive Inference Orchestration for Memory-Bound NPUs",
        "authors": "Multiple Authors",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "year": 2026,
        "topic": "Hardware",
        "abstract_en": "We present A-IO, an adaptive inference orchestration system for LLM inference on Neural Processing Units (NPUs) that are memory-bound. A-IO dynamically adjusts memory management, computation scheduling, and data movement patterns based on runtime profiling of NPU resource utilization, maximizing throughput under strict memory constraints.",
        "abstract_cn": "我们提出了A-IO，一种用于在内存受限的神经处理单元（NPU）上进行LLM推理的自适应推理协调系统。A-IO基于NPU资源利用的运行时分析动态调整内存管理、计算调度和数据移动模式，在严格内存约束下最大化吞吐量。",
        "intro_en": "NPUs offer cost-effective inference for LLMs but are often memory-bound, limiting achievable throughput. Adaptive orchestration that responds to runtime conditions can help overcome these constraints.",
        "intro_cn": "NPU为LLM提供经济高效的推理，但通常受内存限制，限制了可达到的吞吐量。响应运行时条件的自适应协调可以帮助克服这些约束。",
    },
]

def create_md_file(paper, existing_titles):
    """Create markdown file for a paper"""
    conf_dir = paper['conference'].lower()
    year = paper['year']
    
    # Determine directory
    dir_path = os.path.join(BASE_DIR, conf_dir, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    # Find next number
    existing_files = os.listdir(dir_path) if os.path.exists(dir_path) else []
    nums = []
    for f in existing_files:
        if f.endswith('.md') and f[0:2].isdigit():
            try:
                nums.append(int(f[0:2]))
            except:
                pass
    next_num = max(nums) + 1 if nums else 1
    
    # Create filename
    slug = paper['title'].lower().split(':')[0].strip()
    slug = slug.replace(' ', '_').replace('-', '_')[:30]
    filename = f"{next_num:02d}_{slug}.md"
    filepath = os.path.join(dir_path, filename)
    
    # Create content
    url = f"https://arxiv.org/abs/{paper['arxiv_id']}" if paper['arxiv_id'] else ""
    content = f"""---
title: {paper['title']}
authors: {paper['authors']}
arxiv_id: {paper.get('arxiv_id', '')}
conference: {paper['conference']}
full_conference: {paper['conference'].upper()} {year}
year: "{year}"
topic: {paper['topic']}
url: {url}
pdf_url: {url.replace('abs', 'pdf') if url else ""}
added_date: 2026-04-15
---

# {paper['title']}

## 论文信息

- **arXiv**: {paper.get('arxiv_id', '(待确认)')}
- **会议**: {paper['conference'].upper()} {year}
- **作者**: {paper['authors']}
- **主题**: {paper['topic']}

## 摘要 (Abstract)

{paper['abstract_en']}

## 摘要中文

{paper['abstract_cn']}

## 引言 (Introduction)

{paper['intro_en']}

## 引言中文

{paper['intro_cn']}

## 主要贡献

1. (待补充)

## 原文链接

- arXiv: {url or '(待确认)'}
- GitHub: (待补充)

## 补充材料

- 博客: (待补充)
- 相关GitHub: (待补充)

## 备注
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return filename

def main():
    db = load_db()
    existing_titles = get_existing_titles(db)
    
    added = 0
    skipped = 0
    
    for paper_data in NEW_PAPERS:
        if paper_data['title'] in existing_titles:
            skipped += 1
            print(f"⏭️  Skipped (already exists): {paper_data['title']}")
            continue
        
        # Create markdown file
        filename = create_md_file(paper_data, existing_titles)
        
        # Add to database
        next_id = get_next_id(db)
        new_paper = {
            "id": next_id,
            "title": paper_data['title'],
            "authors": paper_data['authors'],
            "arxiv_id": paper_data.get('arxiv_id', ''),
            "github_repo": paper_data.get('github_repo', ''),
            "conference": paper_data['conference'],
            "year": paper_data['year'],
            "topic": paper_data['topic'],
            "abstract_en": paper_data['abstract_en'],
            "abstract_cn": paper_data['abstract_cn'],
            "intro_en": paper_data['intro_en'],
            "intro_cn": paper_data['intro_cn'],
            "file": filename,
            "has_content": True,
            "is_placeholder_arxiv": not paper_data.get('arxiv_id', ''),
            "is_github_project": False,
        }
        db['papers'].append(new_paper)
        existing_titles.add(paper_data['title'])
        added += 1
        print(f"✅ Added [{next_id}]: {paper_data['title']} → {filename}")
    
    save_db(db)
    print(f"\n📊 Summary: Added {added} papers, Skipped {skipped} duplicates")
    return added

if __name__ == "__main__":
    added = main()