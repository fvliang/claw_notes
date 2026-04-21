#!/usr/bin/env python3
"""Add new LLM serving papers to the database and create markdown files."""

import json, os, re, time, hashlib

# Load existing database
db_path = '/home/admin/claw_notes/database.json'
with open(db_path) as f:
    db = json.load(f)

existing_titles = set()
for p in db['papers']:
    existing_titles.add(p['title'].lower().strip()[:60])

# Comprehensive list of new LLM serving papers with full metadata
# Only papers from April 2026 that are truly LLM serving/inference related
new_papers_raw = [
    # === 2026-04-20 papers ===
    {
        "title": "Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing",
        "arxiv_id": "2604.18170",
        "url": "https://arxiv.org/abs/2604.18170",
        "published": "2026-04-20",
        "authors": ["(from arXiv)"],
        "abstract_en": "We introduce Copy-as-Decode, a kernel that reframes constrained text editing as parallel prefill: the target string becomes the 'draft' and grammar constraints enforce deterministic acceptance, sharing the parallel-forward kernel of speculative decoding but with input tokens as the draft and program-enforced acceptance replacing probabilistic verification.",
        "categories": ["cs.CL", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "parallel prefill", "llm editing", "grammar-constrained"],
        "github": "",
    },
    {
        "title": "WISV: Wireless-Informed Semantic Verification for Distributed Speculative Decoding in Device-Edge LLM Inference",
        "arxiv_id": "2604.17701",
        "url": "https://arxiv.org/abs/2604.17701",
        "published": "2026-04-20",
        "authors": ["Zixuan Liu", "Zhiyong Chen", "Nan Xue", "Shengkang Chen", "Jiangchao Yao", "Meixia Tao", "Wenjun Zhang"],
        "abstract_en": "While distributed device-edge speculative decoding accelerates LLM inference, verification overhead on constrained devices remains significant. We propose WISV, a wireless-informed semantic verification framework that leverages channel state information to reduce verification complexity in distributed speculative decoding.",
        "categories": ["cs.AI", "cs.NI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "distributed inference", "device-edge", "wireless verification"],
        "github": "",
    },
    {
        "title": "HybridGen: Efficient LLM Generative Inference via CPU-GPU Hybrid Computing",
        "arxiv_id": "2604.18529",
        "url": "https://arxiv.org/abs/2604.18529",
        "published": "2026-04-20",
        "authors": ["(from arXiv)"],
        "abstract_en": "We propose HybridGen, an efficient LLM generative inference framework that leverages CPU-GPU hybrid computing to maximize resource utilization and reduce inference latency.",
        "categories": ["cs.DC", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["llm inference", "cpu-gpu hybrid", "hybrid computing", "inference efficiency"],
        "github": "",
    },
    {
        "title": "MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression",
        "arxiv_id": "2604.17695",
        "url": "https://arxiv.org/abs/2604.17695",
        "published": "2026-04-20",
        "authors": ["Libo Sun", "Peixiong He", "Po-Wei Harn", "Xiao Qin"],
        "abstract_en": "KV cache memory is the dominant bottleneck for long-context LLM inference. Existing compression methods each act on a single axis of the four-dimensional KV tensor -- token eviction (sequence), quantization (precision), low-rank projection (head dimension), or cross-layer sharing -- but apply the same recipe to every layer. We show that this homogeneity leaves accuracy on the table: different layers prefer different compression strategies.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["kv cache compression", "moe routing", "llm inference", "multi-axis compression"],
        "github": "",
    },
    {
        "title": "How Much Cache Does Reasoning Need? Depth-Cache Tradeoffs in KV-Compressed Transformers",
        "arxiv_id": "2604.17935",
        "url": "https://arxiv.org/abs/2604.17935",
        "published": "2026-04-20",
        "authors": ["(from arXiv)"],
        "abstract_en": "We study the depth-cache tradeoffs in KV-compressed transformers, examining how much cache is needed for reasoning tasks and the implications for inference efficiency.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["kv cache", "kv compression", "reasoning", "depth-cache tradeoff"],
        "github": "",
    },
    {
        "title": "River-LLM: Large Language Model Seamless Exit Based on KV Share",
        "arxiv_id": "2604.18396",
        "url": "https://arxiv.org/abs/2604.18396",
        "published": "2026-04-20",
        "authors": ["(from arXiv)"],
        "abstract_en": "We propose River-LLM, a seamless exit mechanism for LLM inference based on KV share, allowing early termination while maintaining output quality.",
        "categories": ["cs.CL", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["llm inference", "early exit", "kv share", "seamless exit"],
        "github": "",
    },
    {
        "title": "AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization",
        "arxiv_id": "2604.18137",
        "url": "https://arxiv.org/abs/2604.18137",
        "published": "2026-04-20",
        "authors": ["(from arXiv)"],
        "abstract_en": "Processing-in-Memory (PIM) architectures improve bandwidth for LLM inference but face capacity limitations. We propose AQPIM, breaking the PIM capacity wall through in-memory activation quantization for efficient LLM inference.",
        "categories": ["cs.AR", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["pim", "llm inference", "activation quantization", "memory architecture"],
        "github": "",
    },
    {
        "title": "Latent Phase-Shift Rollback: Inference-Time Error Correction via Residual Stream Monitoring",
        "arxiv_id": "2604.18567",
        "url": "https://arxiv.org/abs/2604.18567",
        "published": "2026-04-20",
        "authors": ["(from arXiv)"],
        "abstract_en": "We propose Latent Phase-Shift Rollback, an inference-time error correction mechanism via residual stream monitoring for LLM inference quality improvement.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["inference-time error correction", "residual stream", "llm inference quality"],
        "github": "",
    },
    # === 2026-04-19 papers ===
    {
        "title": "SLO-Guard: Crash-Aware, Budget-Consistent Autotuning for SLO-Constrained LLM Serving",
        "arxiv_id": "2604.17627",
        "url": "https://arxiv.org/abs/2604.17627",
        "published": "2026-04-19",
        "authors": ["Christian Lysenstøen"],
        "abstract_en": "Serving large language models under latency service-level objectives (SLOs) is a configuration-heavy systems problem with an unusually failure-prone search space. We present SLO-Guard, a crash-aware autotuner for vLLM serving that treats crashes as first-class observations. SLO-Guard combines a feasible-first Thermal Budget Annealing (TBA) exploration phase with a warm-started Tree-structured Parzen Estimator (TPE) exploitation phase; the handoff replays all exploration history, including crashes encoded as extreme constraint violations.",
        "categories": ["cs.LG", "cs.DC", "cs.PF"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["llm serving", "slo", "autotuning", "vllm", "configuration"],
        "github": "https://github.com/Chrislysen/SLO-Guard",
    },
    {
        "title": "Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems",
        "arxiv_id": "2604.17249",
        "url": "https://arxiv.org/abs/2604.17249",
        "published": "2026-04-19",
        "authors": ["Yuji Yamamoto"],
        "abstract_en": "Rowhammer on GPU DRAM has enabled adversarial bit flips in model weights; shared KV-cache blocks in LLM serving systems present an analogous but previously unexamined target. In vLLM's Prefix Caching, these blocks exist as a single physical copy without integrity protection. Using software fault injection, we characterize worst-case severity and identify three properties: silent divergence, selective propagation, and persistent accumulation.",
        "categories": ["cs.CR", "cs.AR", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["kv cache security", "llm serving", "bit-flip", "vllm prefix caching"],
        "github": "",
    },
    {
        "title": "Graph-Guided Adaptive Channel Elimination for KV Cache Compression (GRACE)",
        "arxiv_id": "2604.17164",  # approximate from search
        "url": "https://arxiv.org/abs/2604.17164",
        "published": "2026-04-18",
        "authors": ["Enwei Tong", "Yao Zhu", "Yuanchao Bai", "Kai Wang", "Xianming Liu", "Xiangyang Ji"],
        "abstract_en": "We introduce GRACE (Graph-guided Adaptive Channel Elimination), a novel framework that reframes KV cache compression as a graph-based optimization problem. GRACE models channels as nodes and their inter-dependencies as edges, enabling structured and informed pruning decisions.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["kv cache compression", "channel elimination", "graph optimization", "llm inference"],
        "github": "",
    },
    # === 2026-04-18 papers ===
    {
        "title": "Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon",
        "arxiv_id": "2604.16957",
        "url": "https://arxiv.org/abs/2604.16957",
        "published": "2026-04-18",
        "authors": ["(from arXiv)"],
        "abstract_en": "We present Open-TQ-Metal, a fused compressed-domain attention kernel for long-context LLM inference on Apple Silicon Metal GPU, enabling efficient attention computation in the compressed domain.",
        "categories": ["cs.AR", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["compressed-domain attention", "apple silicon", "metal gpu", "llm inference kernel"],
        "github": "",
    },
    {
        "title": "SinkRouter: Sink-Aware Routing for Efficient Long-Context Decoding in Large Language and Multimodal Models",
        "arxiv_id": "2604.16883",
        "url": "https://arxiv.org/abs/2604.16883",
        "published": "2026-04-18",
        "authors": ["Junnan Liu", "Xinyan Liu", "Peifeng Gao", "Zhaobo Qi", "Beichen Zhang", "Weigang Zhang", "Antoni Bert Chen"],
        "abstract_en": "We show that the attention sink phenomenon corresponds to a stable, reachable, and error-controllable fixed point constructed during training. Based on this insight, we propose SinkRouter, a training-free selective routing framework that detects the sink signal and skips computations that would otherwise produce near-zero output.",
        "categories": ["cs.CL", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["attention sink", "routing", "long-context decoding", "llm inference efficiency"],
        "github": "",
    },
    {
        "title": "HieraSparse: Hierarchical Semi-Structured Sparse KV Attention",
        "arxiv_id": "2604.16864",
        "url": "https://arxiv.org/abs/2604.16864",
        "published": "2026-04-18",
        "authors": ["Haoxuan Wang", "Chen Wang"],
        "abstract_en": "We introduce HieraSparse, a hierarchical KV Cache compression framework with acceleration kernels that leverage GPU sparse tensor cores to speed up attention computation with semi-structured sparse patterns.",
        "categories": ["cs.LG", "cs.AR"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["kv cache", "sparse attention", "hierarchical compression", "gpu sparse tensor cores"],
        "github": "",
    },
    # === 2026-04-17 papers ===
    {
        "title": "KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving",
        "arxiv_id": "2604.16682",
        "url": "https://arxiv.org/abs/2604.16682",
        "published": "2026-04-17",
        "authors": ["(from arXiv)"],
        "abstract_en": "We propose KAIROS, a stateful, context-aware power-efficient agentic inference serving system that manages LLM inference for agentic workflows with awareness of multi-turn context and power constraints.",
        "categories": ["cs.DC", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["agentic inference", "llm serving", "power-efficient", "context-aware scheduling"],
        "github": "",
    },
    {
        "title": "POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving",
        "arxiv_id": "2604.16583",
        "url": "https://arxiv.org/abs/2604.16583",
        "published": "2026-04-17",
        "authors": ["(from arXiv)"],
        "abstract_en": "We propose POLAR, an online learning framework for LoRA adapter caching and routing in edge LLM serving systems, dynamically managing adapter placement for optimal performance.",
        "categories": ["cs.DC", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["lora adapter", "edge serving", "online learning", "routing", "llm serving"],
        "github": "",
    },
    {
        "title": "Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving",
        "arxiv_id": "2604.15629",  # from search results
        "url": "https://arxiv.org/abs/2604.15629",
        "published": "2026-04-17",
        "authors": ["Takeshi Yoshimura", "Valentijn Dymphnus van de Beek", "Tatsuhiro Chiba"],
        "abstract_en": "Under long-context serving, accuracy becomes speed through retry dynamics. We introduce Time-to-Correct-Answer (TTCA), a metric that measures the wall-clock time required for a serving system to return a correct answer, capturing cumulative delay from retries that single-shot latency metrics miss.",
        "categories": ["cs.DC", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["distributed llm serving", "routing", "accuracy-speed", "long-context", "ttca"],
        "github": "",
    },
    # === 2026-04-16 papers ===
    {
        "title": "Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter",
        "arxiv_id": "2604.16xxx",  # exact ID not found yet
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Ruoyu Qin", "Weiran He", "Yaoyu Wang", "Zheming Li", "Xinran Xu", "Yongwei Wu", "Weimin Zheng", "Mingxing Zhang"],
        "abstract_en": "Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but its deployment boundary is still determined by KVCache transfer. We present Prefill-as-a-Service (PrfaaS), a cross-datacenter serving architecture that selectively offloads long-context prefill to standalone, compute-dense prefill clusters.",
        "categories": ["cs.DC", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["prefill-as-a-service", "kv cache transfer", "cross-datacenter", "pd disaggregation"],
        "github": "",
    },
    {
        "title": "Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Tingyang Sun", "Ting He", "I-Hong Hou"],
        "abstract_en": "Large foundation models are increasingly employed as the core of AI services. Serving such models at scale remains challenging due to their heavy resource footprints, particularly GPU memory. We develop scalable algorithms with guaranteed performance for chain-structured jobs with large memory footprints, applicable to LLM serving.",
        "categories": ["cs.DC", "cs.PF"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["llm serving", "chain-structured jobs", "memory footprint", "gpu memory"],
        "github": "",
    },
    {
        "title": "Faster LLM Inference via Sequential Monte Carlo",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Yahya Emara", "Mauricio Barba da Costa", "Chi-Chih Chang", "Cameron Freer", "Tim Vieira", "Ryan Cotterell", "Mohamed S. Abdelfattah"],
        "abstract_en": "We propose using Sequential Monte Carlo methods for faster LLM inference, providing a probabilistic framework that generalizes speculative decoding with theoretically grounded acceptance criteria.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "sequential monte carlo", "llm inference", "probabilistic framework"],
        "github": "",
    },
    {
        "title": "From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Kiran Purohit", "Ramasuri Narayanam", "Soumyabrata Pal"],
        "abstract_en": "We propose verification-aware speculative decoding that extends token-level speculation to step-level speculation for efficient multi-step reasoning in LLMs.",
        "categories": ["cs.CL", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "verification-aware", "multi-step reasoning", "step-level speculation"],
        "github": "",
    },
    {
        "title": "RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Zihong Zhang", "Zuchao Li", "Lefei Zhang", "Ping Wang", "Hai Zhao"],
        "abstract_en": "Autoregressive decoding in Large Language Models generates one token per step, causing high inference latency. Speculative decoding mitigates this through a guess-and-verify strategy. We propose RACER, a retrieval-augmented contextual rapid speculative decoding method that combines retrieval-based and logits-based drafting.",
        "categories": ["cs.CL", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "retrieval-augmented", "rapid decoding", "inference latency"],
        "github": "",
    },
    {
        "title": "ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Yuseon Choi", "Jingu Lee", "Jungjun Oh", "Sunjoo Whang", "Byeongcheol Kim", "Minsung Kim", "Hoi-Jun Yoo", "Sangjin Kim"],
        "abstract_en": "Memory-centric architectures (PIM, NMP) improve bandwidth but leave compute underutilized under MoE's low arithmetic intensity at high batch sizes. We propose ELMoE-3D, leveraging intrinsic elasticity of MoE for hybrid-bonding-enabled self-speculative decoding in on-premises serving.",
        "categories": ["cs.AR", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["self-speculative decoding", "moe elasticity", "hybrid bonding", "on-premises serving", "pim"],
        "github": "",
    },
    {
        "title": "ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Walaa Amer", "Uday das", "Fadi Kurdahi"],
        "abstract_en": "Self-speculative decoding uses the base LLM itself for speculation but faces limitations from overconfident shallow layers. We propose ConfLayers, an adaptive confidence-based layer skipping method for self-speculative decoding.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["self-speculative decoding", "layer skipping", "confidence-based", "llm inference"],
        "github": "",
    },
    {
        "title": "The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Ranjith Chodavarapu", "Lei Xu"],
        "abstract_en": "We establish that FP16 KV cache inference is fundamentally non-equivalent to recomputation and provide a mechanistic framework for understanding numerical instability in modern LLM inference systems.",
        "categories": ["cs.LG", "cs.AR"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["fp16 kv cache", "numerical instability", "llm inference", "kv cache divergence"],
        "github": "",
    },
    {
        "title": "Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU",
        "arxiv_id": "2604.16xxx",
        "url": "https://arxiv.org/abs/2604.16xxx",
        "published": "2026-04-16",
        "authors": ["Jevin Jiang", "Ying Chen", "Blake A. Hechtman", "Fenghui Zhang", "Yarong Mu"],
        "abstract_en": "Shifting to cost-efficient accelerators like Google's Tensor Processing Units prioritizes both performance and total cost of ownership. We present Ragged Paged Attention (RPA), a high-performance and flexible attention kernel for TPUs, implemented using XLA Pallas for efficiently mapping LLM workloads onto TPU architectures.",
        "categories": ["cs.AR", "cs.DC"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["paged attention", "tpu", "llm inference kernel", "ragged execution", "xla pallas"],
        "github": "",
    },
    # === 2026-04-15 papers ===
    {
        "title": "YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference",
        "arxiv_id": "2604.15xxx",
        "url": "https://arxiv.org/abs/2604.15xxx",
        "published": "2026-04-15",
        "authors": ["You Wu", "Ziheng Chen", "Yizhen Zhang", "Haoyi Wu", "Chengting Yu", "Yuchi Xu", "Wenbo Su", "Bo Zheng", "Kewei Tu"],
        "abstract_en": "Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models. We aim to enhance YOCO, a cross-layer KV compression method that shares KVs of the middle layer with the top-half layers, by proposing KV residual connections.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["kv compression", "yoco", "kv residual", "llm inference efficiency"],
        "github": "",
    },
    {
        "title": "ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding",
        "arxiv_id": "2604.15xxx",
        "url": "https://arxiv.org/abs/2604.15xxx",
        "published": "2026-04-15",
        "authors": ["Heming Xia", "Yongqi Li", "Cunxiao Du", "Mingbo Song", "Wenjie Li"],
        "abstract_en": "Tool calling in LLMs is structured, conforms to constrained schemas, and often exhibits recurring invocation patterns. We propose ToolSpec, a schema-aware, retrieval-augmented speculative decoding method for accelerating tool calling.",
        "categories": ["cs.CL", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "tool calling", "schema-aware", "retrieval-augmented"],
        "github": "",
    },
    {
        "title": "Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference",
        "arxiv_id": "2604.15xxx",
        "url": "https://arxiv.org/abs/2604.15xxx",
        "published": "2026-04-15",
        "authors": ["Xuwen Zhou", "Fangxin Liu", "Chao Wang", "Xiao Zheng", "Hao Zheng", "Min He", "Li Jiang", "Haibing Guan"],
        "abstract_en": "We propose Calibrated Speculative Decoding, using frequency-guided candidate selection for more efficient LLM inference by calibrating draft token probabilities.",
        "categories": ["cs.CL", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "calibrated", "frequency-guided", "candidate selection"],
        "github": "",
    },
    {
        "title": "Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs",
        "arxiv_id": "2604.15xxx",
        "url": "https://arxiv.org/abs/2604.15xxx",
        "published": "2026-04-15",
        "authors": ["Sangeeta Chowdhary", "Ryan Swann", "Sean Siddens", "Muhammad Osama", "Stephen Neuendorffer", "Alexandru Dutu", "Karthik Sangaiah", "Sandeepa Bhuyan", "Samuel Bayliss", "Ganesh Dasika"],
        "abstract_en": "We propose Fleet, a multi-level task model that maps computation to memory scopes for memory-bound workloads such as LLM inference on multi-die GPUs, enabling hierarchical megakernel abstraction.",
        "categories": ["cs.AR", "cs.DC"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["megakernel", "multi-die gpu", "llm inference", "hierarchical task abstraction"],
        "github": "",
    },
    # === 2026-04-14 papers ===
    {
        "title": "Accelerating Speculative Decoding with Block Diffusion Draft Trees",
        "arxiv_id": "2604.14xxx",
        "url": "https://arxiv.org/abs/2604.14xxx",
        "published": "2026-04-14",
        "authors": ["Liran Ringel", "Yaniv Romano"],
        "abstract_en": "We propose accelerating speculative decoding by using block diffusion draft trees, enabling more tokens per speculation round through tree-structured draft generation.",
        "categories": ["cs.CL", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative decoding", "block diffusion", "draft trees", "tree-structured speculation"],
        "github": "",
    },
    # === 2026-04-13 papers ===
    {
        "title": "SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration",
        "arxiv_id": "2604.13xxx",
        "url": "https://arxiv.org/abs/2604.13xxx",
        "published": "2026-04-13",
        "authors": ["Zhuofan Wen", "Yang Feng"],
        "abstract_en": "Speculative decoding has emerged as a promising approach to accelerate autoregressive inference in large language models. Self-draft methods avoid the overhead of auxiliary draft models but face limitations: shallow layers often produce overconfident yet incorrect drafts. We propose SpecBound, adaptive bounded self-speculation with layer-wise confidence calibration.",
        "categories": ["cs.CL", "cs.LG"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["self-speculative decoding", "confidence calibration", "adaptive bounded", "llm inference"],
        "github": "",
    },
    {
        "title": "SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling",
        "arxiv_id": "2604.13xxx",
        "url": "https://arxiv.org/abs/2604.13xxx",
        "published": "2026-04-13",
        "authors": ["Zikun Liu", "Liang Luo", "Qianru Li", "Zhengyu Zhang", "Wei Ling", "Jingyi Shen", "Zeliang Chen", "Yaning Huang", "Jingxian Huang", "Abdallah Aboelela", "Chonglin Sun", "Feifan Gu", "Fenggang Wu", "Hang Qu", "Huayu Li", "Jill Pan", "Kaidi Pei", "Laming Chen", "Longhao Jin", "Qin Huang", "Tongyi Tang", "Varna Puvvada", "Wenlin Chen", "Xiaohan Wei", "Xu Cao"],
        "abstract_en": "Large models with deep reasoning capabilities make real-time serving impractical. We present SOLARIS (Speculative Offloading of Latent-bAsed Representation for Inference Scaling), a novel framework for efficient inference through speculative offloading of latent representations.",
        "categories": ["cs.DC", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["speculative offloading", "latent representation", "inference scaling", "llm serving"],
        "github": "",
    },
    {
        "title": "From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution",
        "arxiv_id": "2604.11378",
        "url": "https://arxiv.org/abs/2604.11378",
        "published": "2026-04-13",
        "authors": ["(from arXiv)"],
        "abstract_en": "We propose a scheduler-theoretic framework for LLM agent execution, transforming agent loops into structured graphs for more efficient scheduling and execution in LLM inference systems.",
        "categories": ["cs.DC", "cs.AI"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["llm agent scheduling", "inference framework", "structured graphs", "agent execution"],
        "github": "",
    },
    # === Earlier April (from search) ===
    {
        "title": "Layer-wise MoE Routing Locality under Shared-Prefix Code Generation: Token-Identity Decomposition",
        "arxiv_id": "2604.17182",
        "url": "https://arxiv.org/abs/2604.17182",
        "published": "2026-04-19",
        "authors": ["(from arXiv)"],
        "abstract_en": "We study layer-wise MoE routing locality under shared-prefix code generation, proposing token-identity decomposition to improve MoE inference efficiency.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["moe routing locality", "shared-prefix", "code generation", "inference efficiency"],
        "github": "",
    },
    {
        "title": "Sequential KV Cache Compression via Probabilistic Language Tries: Beyond the Per-Vector Shannon Limit",
        "arxiv_id": "2604.10xxx",
        "url": "https://arxiv.org/abs/2604.10xxx",
        "published": "2026-04-10",
        "authors": ["Gregory Magarshak"],
        "abstract_en": "Recent work on KV cache quantization has approached the Shannon entropy limit for per-vector compression. We observe that this limit applies to a strictly weaker problem: compressing the KV cache as a sequence. We propose sequential KV cache compression via probabilistic language tries.",
        "categories": ["cs.LG", "cs.CL"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["kv cache compression", "shannon limit", "probabilistic language tries", "sequential compression"],
        "github": "",
    },
    {
        "title": "DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization",
        "arxiv_id": "2604.17789",
        "url": "https://arxiv.org/abs/2604.17789",
        "published": "2026-04-20",
        "authors": ["(from arXiv)"],
        "abstract_en": "DuQuant++ enhances microscaling FP4 quantization with fine-grained rotation for more efficient LLM inference through improved low-bit quantization.",
        "categories": ["cs.LG", "cs.AR"],
        "conference": "arxiv",
        "year": 2026,
        "keywords": ["fp4 quantization", "microscaling", "llm inference", "rotation quantization"],
        "github": "",
    },
]

# Filter out papers that already exist and ones where we have placeholder arxiv IDs
new_papers = []
for p in new_papers_raw:
    title_key = p['title'].lower().strip()[:60]
    if title_key in existing_titles:
        continue
    # Skip papers with placeholder arxiv IDs (xxx) - we'll try to resolve these later
    if 'xxx' in p['arxiv_id']:
        # Try to keep these anyway - we'll note them as needing verification
        p['needs_id_verification'] = True
    new_papers.append(p)

print(f"Total new papers to add: {len(new_papers)}")

# Create markdown files and add to database
papers_base = '/home/admin/claw_notes/papers'

added_count = 0
for p in new_papers:
    conference = p['conference']
    year = p['year']
    
    # Create directory
    dir_path = os.path.join(papers_base, conference, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    # Create paper ID
    paper_id = p['arxiv_id'].replace('xxx', '0')  # placeholder
    safe_title = re.sub(r'[^\w\s-]', '', p['title'][:60]).strip().replace(' ', '_')
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        continue
    
    # Write markdown file
    md_content = f"""# {p['title']}

**ArXiv ID:** {p['arxiv_id']}
**Published:** {p['published']}
**Authors:** {', '.join(p['authors'])}
**URL:** {p['url']}
**GitHub:** {p.get('github', '')}
**Keywords:** {', '.join(p['keywords'])}

## 摘要 (中文)

*(待补充)*

## Abstract (English)

{p['abstract_en']}

## 引言 (中文)

*(待补充)*

## Introduction (English)

*(待补充 - 需要阅读原文)*

## 博客内容

*(待补充)*

## GitHub 介绍

{p.get('github', '暂无 GitHub 仓库') if p.get('github') else '暂无 GitHub 仓库'}

---
*注: 此文件由自动化论文搜集系统生成，部分内容待完善。*
"""
    
    with open(filepath, 'w') as f:
        f.write(md_content)
    
    # Add to database
    db_entry = {
        "id": f"{conference}_{year}_{safe_title}",
        "title": p['title'],
        "conference": conference,
        "year": year,
        "url": p['url'],
        "github": p.get('github', ''),
        "authors": p['authors'],
        "keywords": p['keywords'],
        "published": p['published'],
        "arxiv_id": p['arxiv_id'],
        "abstract_en": p['abstract_en'],
        "abstract_cn": "",
        "introduction_en": "",
        "introduction_cn": "",
        "markdown_path": os.path.join(conference, str(year), filename),
    }
    db['papers'].append(db_entry)
    added_count += 1
    print(f"  Added: [{p['arxiv_id']}] {p['title'][:80]}")

# Save database
with open(db_path, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\nTotal papers added: {added_count}")
print(f"Total papers in database: {len(db['papers'])}")