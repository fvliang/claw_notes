# LLM Serving Papers Index

_Generated: 2026-07-29 06:18_


## 2026


### Distributed Inference

- **BloomBee: Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization** — Jiu Chen, Shuangyan Yang, Xu Xiong, Hexiao Duan, Xinran Zhang, Jie Ren, Dong Li
  [arXiv](https://arxiv.org/abs/2604.21072)
  > Decentralized LLM inference distributes computation among heterogeneous nodes across the internet, offering a performant and cost-efficient solution, alternative to traditional centralized inference. ...

- **HybridGen: Efficient LLM Generative Inference via CPU-GPU Hybrid Computing** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.18529)
  > We propose HybridGen, an efficient LLM generative inference framework that leverages CPU-GPU hybrid computing to maximize resource utilization and reduce inference latency....

- **[GitHub] parallax: Parallax is a distributed model serving framework that lets you build your own AI cluster anywhere** — GradientHQ
  [GitHub](https://github.com/GradientHQ/parallax)
  > Parallax is a distributed model serving framework that lets you build your own AI cluster anywhere...


### Early Exit

- **River-LLM: Large Language Model Seamless Exit Based on KV Share** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.18396)
  > We propose River-LLM, a seamless exit mechanism for LLM inference based on KV share, allowing early termination while maintaining output quality....


### Edge Inference

- **Black-Box Skill Stealing Attack from Proprietary LLM Agents: An Empirical Study** — Zihan Wang, Rui Zhang, Yu Liu, Chi Liu, Qingchuan Zhao
  [arXiv](https://arxiv.org/abs/2604.21829v1)
  > LLM agents increasingly rely on skills to encapsulate reusable capabilities via progressively disclosed instructions. High-quality skills inject expert knowledge into general-purpose models, improving...

- **EdgeFlow: Fast Cold Starts for LLMs on Mobile Devices** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2604.09083)
  > EdgeFlow targets fast cold starts for LLMs on mobile devices, addressing the challenge of deploying large language models on resource-constrained edge devices by optimizing initialization and inferenc...

- **HaS: Accelerating RAG through Homology-Aware Speculative Retrieval** — Peng Peng, Weiwei Lin, Wentai Wu, Xinyang Wang, Yongheng Liu
  [arXiv](https://arxiv.org/abs/2604.20452v1)
  > Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) at inference by retrieving external documents as context. However, retrieval becomes increasingly ti...


### Edge LLM Serving

- **Unlocking the Edge Deployment and On-Device Acceleration of Multi-LoRA Enabled One-for-All Foundational LLM** — Sravanth Kodavanti, Sowmya Vajrala, Srinivas Miriyala
  [arXiv](https://arxiv.org/abs/2604.18655)
  > Deploying large language models (LLMs) on smartphones poses significant engineering challenges due to stringent constraints on memory, latency, and runtime. This paper explores edge deployment and on-...


### Hardware Acceleration

- **A Full-Stack Performance Evaluation Infrastructure for 3D-DRAM-based LLM Accelerators** — Cong Li, Chenhao Xue, Yi Ren, Xiping Dong, Yu Cheng, Yinbo Hu, Fujun Bai, Yixin Guo, Xiping Jiang, Qiang Wu, Zhi Yang, Zhe Cheng, Yuan Xie, Guangyu Sun
  [arXiv](https://arxiv.org/abs/2604.08044)
  > Large language models (LLMs) exhibit memory-intensive behavior during decoding, making it a key bottleneck in LLM inference. To accelerate decoding execution, hybrid-bonding-based 3D-DRAM has been ado...

- **Ouroboros: Wafer-Scale SRAM CIM with Token-Grained Pipelining for Large Language Model Inference** — Yiqi Liu, Cheng Liu, Zhen Gu, Tianchen Ding, Zongyue Zhao, Ziyu Yang, Yufei Ding, Yibo Lin, Mingjie Lin, Xiaowei Li, Zidong Du, Chen Liu, Yunji Chen
  [arXiv](https://arxiv.org/abs/2603.02737)
  > Conventional LLM inference architectures suffer from high energy and latency due to frequent data movement across memory hierarchies. We propose Ouroboros, a wafer-scale SRAM-based Computing-in-Memory...


### Inference Kernel

- **FastTree: Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference** — Zaifeng Pan, Yitong Ding, Yue Guan, Zheng Wang, Zhongkai Yu
  [GitHub](https://github.com/aerlabsAI/ai-inference-resources)
  > Tree-structured prefix sharing is prevalent in recent large language model (LLM) applications. Existing LLM serving systems use a radix tree to organize the global key-value (KV) cache, facilitating c...

- **[GitHub] lite_llama: A light llama-like llm inference framework based on the triton kernel.** — harleyszhang
  [GitHub](https://github.com/harleyszhang/lite_llama)
  > A light llama-like llm inference framework based on the triton kernel....


### Inference Routing

- **Layer-wise MoE Routing Locality under Shared-Prefix Code Generation: Token-Identity Decomposition** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.17182)
  > We study layer-wise MoE routing locality under shared-prefix code generation, proposing token-identity decomposition to improve MoE inference efficiency....

- **SinkRouter: Sink-Aware Routing for Efficient Long-Context Decoding in Large Language and Multimodal Models** — ['Junnan Liu', 'Xinyan Liu', 'Peifeng Gao', 'Zhaobo Qi', 'Beichen Zhang', 'Weigang Zhang', 'Antoni Bert Chen']
  [arXiv](https://arxiv.org/abs/2604.16883)
  > We show that the attention sink phenomenon corresponds to a stable, reachable, and error-controllable fixed point constructed during training. Based on this insight, we propose SinkRouter, a training-...


### Inference Scheduling

- **98$\times$ Faster LLM Routing Without a Dedicated GPU: Flash Attention, Prompt Compression, and Near-Streaming for the vLLM Semantic Router** — Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang
  [arXiv](https://arxiv.org/abs/2603.12646)
  > System-level routers that intercept LLM requests for safety classification, domain routing, and PII detection must be both fast and operationally lightweight: they should add minimal latency to every ...

- **A Task Decomposition and Planning Framework for Efficient LLM Inference in AI-Enabled WiFi-Offload Networks** — Mingqi Han, Xing Sun
  [arXiv](https://arxiv.org/abs/2604.21399)
  > AI WiFi offload is emerging as a promising approach for providing large language model (LLM) services to resource-constrained wireless devices. However, unlike conventional edge computing, LLM inferen...

- **DeepServe: Hierarchical Model Placement and Dynamic Batching for Cost-Efficient Multi-Tenant LLM Inference at Scale** — Tejas Pravinbhai Patel, P. Agarwal
  [GitHub](https://github.com/sgl-project/SpecForge)
  > Large Language Model (LLM) inference serving at scale presents critical challenges in multi-tenant cloud environments, where organizations must balance conflicting objectives of cost efficiency, respo...

- **From Research Question to Scientific Workflow: Leveraging Agentic AI for Science Automation** — Bartosz Balis, Michal Orzechowski, Piotr Kica, Michal Dygas, Michal Kuszewski
  [arXiv](https://arxiv.org/abs/2604.21910v1) | [GitHub](https://github.com/containers/ramalama)
  > Scientific workflow systems automate execution -- scheduling, fault tolerance, resource management -- but not the semantic translation that precedes it. Scientists still manually convert research ques...

- **Hive: A Multi-Agent Infrastructure for Algorithm- and Task-Level Scaling** — Zizhang Luo, Yuhao Luo, Youwei Xiao, Yansong Xu, Runlin Guo, Yun Liang
  [arXiv](https://arxiv.org/abs/2604.17353)
  > Large language models are increasingly deployed as complex agentic systems that scale with task complexity. While prior work has extensively explored model- and system-level scaling, algorithm- and ta...

- **RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment** — Yingfeng Luo, Hongyu Liu, Dingyang Lin, Kaiyan Chang, Chenglong Wang
  [arXiv](https://arxiv.org/abs/2604.22520v1)
  > Large Language Models (LLMs) have achieved remarkable performance in Machine Translation (MT), but deploying them at scale remains prohibitively expensive. A widely adopted remedy is the hybrid system...

- **Thinking with Reasoning Skills: Fewer Tokens, More Accuracy** — Guangxiang Zhao, Qilong Shi, Xusen Xiao, Xiangzheng Zhang, Tong Yang
  [arXiv](https://arxiv.org/abs/2604.21764v1) | [GitHub](https://github.com/dilab-zju/self-speculative-decoding)
  > Reasoning LLMs often spend substantial tokens on long intermediate reasoning traces (e.g., chain-of-thought) when solving new problems. We propose to summarize and store reusable reasoning skills dist...

- **TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale** — Jun Wang, Ziyin Zhang, Rui Wang, Hang Yu, Peng Di
  [arXiv](https://arxiv.org/abs/2604.21889v1)
  > Real-time detection and mitigation of technical anomalies are critical for large-scale cloud-native services, where even minutes of downtime can result in massive financial losses and diminished user ...

- **[GitHub] llm-scheduling-artifact: Artifact of OSDI '24 paper, ”Llumnix: Dynamic Scheduling for Large Language Model Serving“** — alibaba
  [GitHub](https://github.com/alibaba/llm-scheduling-artifact)
  > Artifact of OSDI '24 paper, ”Llumnix: Dynamic Scheduling for Large Language Model Serving“...

- **[GitHub] ome: Open Model Engine (OME) — Kubernetes operator for LLM serving, GPU scheduling, and model lifecycle m** — ome-projects
  [GitHub](https://github.com/ome-projects/ome)
  > Open Model Engine (OME) — Kubernetes operator for LLM serving, GPU scheduling, and model lifecycle management. Works with SGLang, vLLM, TensorRT-LLM, and Triton...


### KV Cache

- **Adaptive Multi-Objective Tiered Storage Configuration for KV Cache in LLM Service** — Xianzhe Zheng, Zhengheng Wang, Ruiyang Ma, Rui Wang, Xiyu Wang
  [arXiv](https://arxiv.org/abs/2603.08739)
  > The memory-for-computation paradigm of KV caching is essential for accelerating large language model (LLM) inference service, but limited GPU high-bandwidth memory (HBM) capacity motivates offloading ...

- **Bit-Flip Vulnerability of Shared KV-Cache Blocks in LLM Serving Systems** — ['Yuji Yamamoto']
  [arXiv](https://arxiv.org/abs/2604.17249)
  > Rowhammer on GPU DRAM has enabled adversarial bit flips in model weights; shared KV-cache blocks in LLM serving systems present an analogous but previously unexamined target. In vLLM's Prefix Caching,...

- **CASK: Core-Aware Selective KV Compression for Reasoning Traces** — Buseong Kim, Heejun Gwon
  [arXiv](https://arxiv.org/abs/2604.10900)
  > In large language models performing long-form reasoning, the KV cache grows rapidly with decode length, creating bottlenecks in memory and inference speed. CASK proposes core-aware selective KV compre...

- **CodeComp: Structural KV Cache Compression for Agentic Coding** — ['Qiujiang Chen', 'Jing Xiong', 'Chenyang Zhao', 'Sidi Yang', 'Ngai Wong']
  [arXiv](https://arxiv.org/abs/2604.10235)
  > Agentic code tasks such as fault localization and patch generation require processing long codebases under tight memory constraints, where the Key-Value (KV) cache becomes the primary inference bottle...

- **Comparative Characterization of KV Cache Management Strategies for LLM Inference** — ['Oteo Mamo', 'Olga Kogiou', 'Hyunjin Yi', 'Weikuan Yu']
  [arXiv](https://arxiv.org/abs/2604.05012)
  > Efficient inference with Large Language Models (LLMs) increasingly relies on Key-Value (KV) caches to store previously computed key and value vectors at each layer. These caches are essential to minim...

- **DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing** — Jinyu Guo, Zhihan Zhang, Yutong Li, Jiehui Xie, Md. Tamim Iqbal, Dongshen Han, Lik-Hang Lee, Sung-Ho Bae, Jie Zou, Yang Yang, Chaoning Zhang
  [arXiv](https://arxiv.org/abs/2604.19351)
  > The quadratic computational complexity of the standard attention mechanism constitutes a fundamental bottleneck for large language models in long-context scenarios. While KV cache compression methods ...

- **Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs** — ['Sayed Pedram Haeri Boroujeni', 'Niloufar Mehrabi', 'Patrick Woods', 'Gabriel Hillesheim', 'Abolfazl Razi']
  [arXiv](https://arxiv.org/abs/2604.04722)
  > Large Language Models (LLMs) have achieved remarkable progress across reasoning, generation, and decision-making tasks, yet deploying them on mobile, embedded, and edge devices remains particularly ch...

- **Graph-Guided Adaptive Channel Elimination for KV Cache Compression (GRACE)** — ['Enwei Tong', 'Yao Zhu', 'Yuanchao Bai', 'Kai Wang', 'Xianming Liu', 'Xiangyang Ji']
  [arXiv](https://arxiv.org/abs/2604.17164)
  > We introduce GRACE (Graph-guided Adaptive Channel Elimination), a novel framework that reframes KV cache compression as a graph-based optimization problem. GRACE models channels as nodes and their int...

- **KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs** — Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Bing Li, Ulf Schlichtmann
  [arXiv](https://arxiv.org/abs/2604.13226)
  > Large Language Models (LLMs) rely heavily on Key-Value (KV) caching to minimize inference latency. However, standard KV caches are context-dependent: reusing a cached document in a new context require...

- **KVSculpt: KV Cache Compression as Distillation** — ['Bo Jiang', 'Sian Jin']
  [arXiv](https://arxiv.org/abs/2603.27819)
  > KV cache compression is critical for efficient long-context LLM inference. Approaches that reduce the per-pair footprint -- quantization and low-rank decomposition -- are orthogonal to those that redu...

- **Latent-Condensed Transformer for Efficient Long Context Modeling** — Zeng You, Yaofo Chen, Qiuwu Chen, Ying Sun, Shuhai Zhang, Yingjian Li, Yaowei Wang, Mingkui Tan
  [arXiv](https://arxiv.org/abs/2604.12452)
  > Large language models (LLMs) face significant challenges in processing long contexts due to the linear growth of the key-value (KV) cache and quadratic complexity of self-attention. Existing approache...

- **Long-Context Aware Upcycling: A New Frontier for Hybrid LLM Scaling** — Parsa Ashrafi Fashi, Utkarsh Saxena, Mehdi Rezagholizadeh, Aref Jafari, Akash Haridas
  [arXiv](https://arxiv.org/abs/2604.24715v1)
  > Hybrid sequence models that combine efficient Transformer components with linear sequence modeling blocks are a promising alternative to pure Transformers, but most are still pretrained from scratch a...

- **M2XFP: A Metadata-Augmented Microscaling Data Format for Efficient Low-bit Quantization** — Weiming Hu, Zihan Zhang, Haoyan Zhang, Chen Zhang, Cong Guo, Yu Feng, Tianchi Hu, Guanglin Li, Guipeng Hu, Junsong Wang, Jingwen Leng
  [arXiv](https://arxiv.org/abs/2601.19213)
  > Existing low-bit Microscaling (MX) formats, such as MXFP4, often suffer from substantial accuracy degradation due to the use of a shared scaling factor with the Power-of-Two format. In this work, we e...

- **MEMENTO: Teaching LLMs to Manage Their Own Context** — Vasilis Kontonis, Yuchen Zeng, Shivam Garg
  [arXiv](https://arxiv.org/abs/2604.09852)
  > Reasoning models think in long, unstructured streams with no mechanism for compressing or organizing their own intermediate state. MEMENTO introduces a framework for teaching LLMs to manage their own ...

- **MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration** — Xinyu Liu, Xin Liu, Bo Jin, Runsong Zhao, Pengcheng Huang, Junhao Ruan, Bei Li, Chunyang Xiao, Tong Xiao, Jingbo Zhu
  [arXiv](https://arxiv.org/abs/2604.14889)
  > While Chain-of-thought (CoT) reasoning enables LLMs to solve challenging reasoning problems, as KV cache grows linearly with the number of generated tokens, CoT reasoning faces scaling issues in terms...

- **MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression** — ['Libo Sun', 'Peixiong He', 'Po-Wei Harn', 'Xiao Qin']
  [arXiv](https://arxiv.org/abs/2604.17695)
  > KV cache memory is the dominant bottleneck for long-context LLM inference. Existing compression methods each act on a single axis of the four-dimensional KV tensor -- token eviction (sequence), quanti...

- **Neural Garbage Collection: Learning to Forget while Learning to Reason** — Michael Y. Li, Jubayer Ibn Hamid, Emily B. Fox
  [arXiv](https://arxiv.org/abs/2604.18002)
  > Chain-of-thought reasoning has driven striking advances in language model capability, yet every reasoning step grows the KV cache, creating a bottleneck. Neural Garbage Collection learns to forget int...

- **On-Device-First Hybrid LLM Inference on AI-PCs: Closing the Enterprise GenAI Divide** — ['S. Begum', 'Kris Fleming', 'T. Lewellen']
  [GitHub](https://github.com/kvcache-ai/Mooncake)
  > AI-PCs and modern PCs equipped with capable CPUs, GPUs, and NPUs now make on-device inference for small language models (SLMs) practical across many enterprise workloads. This enables assistants that ...

- **PRISM: Breaking the O(n) Memory Wall in Long-Context LLM Inference via O(1) Photonic Block Selection** — ['Hyoseok Park', 'Yeonsang Park']
  [arXiv](https://arxiv.org/abs/2603.21576)
  > Long-context LLM inference is bottlenecked not by compute but by the O(n) memory bandwidth cost of scanning the KV cache at every decode step -- a wall that no amount of arithmetic scaling can break. ...

- **Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines** — Junwan Kim, Hyunkyung Bae
  [arXiv](https://arxiv.org/abs/2604.16734)
  > Multimodal large language models (MLLMs) have recently demonstrated strong capabilities in understanding and generating responses from diverse visual inputs, including high-resolution images and long ...

- **SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving** — Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, Xiaoxia Wu
  [arXiv](https://arxiv.org/abs/2604.19157)
  > KV-cache memory is a major bottleneck in real-world LLM serving, where systems must simultaneously support latency-sensitive small-batch requests and high-throughput concurrent workloads. Although man...

- **ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference** — ['Qiuyang Zhang', 'Kai Zhou', 'Ding Tang', 'Kai Lu', 'Cheng Li', 'Zhenyu Yang']
  [arXiv](https://arxiv.org/abs/2603.27138)
  > Large language models encounter critical GPU memory capacity constraints during long-context inference, where KV cache memory consumption severely limits decode batch sizes. While existing research ha...

- **Sequential KV Cache Compression via Probabilistic Language Tries: Beyond the Per-Vector Shannon Limit** — ['Gregory Magarshak']
  [arXiv](https://arxiv.org/abs/2604.10xxx)
  > Recent work on KV cache quantization has approached the Shannon entropy limit for per-vector compression. We observe that this limit applies to a strictly weaker problem: compressing the KV cache as a...

- **SnapStream: Efficient Long Sequence Decoding on Dataflow Accelerators** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2511.03092)
  > The proliferation of 100B+ parameter Large Language Models (LLMs) with 100k+ context length support have resulted in increasing demands for on-chip memory to support large KV caches. Techniques such a...

- **SparKV: Overhead-Aware KV Cache Loading for Efficient On-Device LLM Inference** — Hongyao Liu, Liuqun Zhai, Junyi Wang, Zhengru Fang
  [arXiv](https://arxiv.org/abs/2604.21231)
  > Efficient inference for on-device Large Language Models (LLMs) remains challenging due to limited hardware resources and the high cost of the prefill stage, which processes the full input context to c...

- **TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference** — Gradwell Dzikanyanga, Weihao Yang, Hao Huang, Donglei Wu, Shihao Wang, Wen Xia, Sanjeeb K C
  [arXiv](https://arxiv.org/abs/2604.19769)
  > Key-value (KV) caching is critical for efficient inference in large language models (LLMs), yet its memory footprint scales linearly with context length, resulting in a severe scalability bottleneck. ...

- **Transactional Attention: Semantic Sponsorship for KV-Cache Retention** — Abhinaba Basu
  [arXiv](https://arxiv.org/abs/2604.11288)
  > At K=16 tokens (0.4% of a 4K context), every existing KV-cache compression method achieves 0% on credential retrieval. The failure mode is dormant token eviction. Transactional Attention proposes sema...

- **When Less Latent Leads to Better Relay: Information-Preserving Compression for Latent Multi-Agent LLM Collaboration** — Yiping Li, Zhiyu An, Wan Du
  [arXiv](https://arxiv.org/abs/2604.13349) | [GitHub](https://github.com/https://github.com/markli404/When-Less-Latent-Leads-to-Better-Relay)
  > Communication in Large Language Model (LLM)-based multi-agent systems is moving beyond discrete tokens to preserve richer context. Recent work such as LatentMAS enables agents to exchange latent messa...

- **YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference** — You Wu, Ziheng Chen, Yizhen Zhang, Haoyi Wu, Chengting Yu, Yuchi Xu, Wenbo Su, Bo Zheng, Kewei Tu
  [arXiv](https://arxiv.org/abs/2604.13556)
  > Cross-layer key-value (KV) compression has been found to be effective in efficient inference of large language models (LLMs). Although they reduce the memory consumption of the KV cache, such methods ...

- **Zipage: Maintain High Request Concurrency for LLM Reasoning through Compressed PagedAttention** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2603.08743)
  > Zipage maintains high request concurrency for LLM reasoning through Compressed PagedAttention, enabling efficient serving of long-reasoning chains while preserving KV cache efficiency and reducing mem...

- **[GitHub] Awesome-KV-Cache-Optimization: [ACL 2026] Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimiz** — jjiantong
  [GitHub](https://github.com/jjiantong/Awesome-KV-Cache-Optimization)
  > [ACL 2026] Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization...

- **[GitHub] Awesome-LLM-KV-Cache: Awesome-LLM-KV-Cache: A curated list of 📙Awesome LLM KV Cache Papers with Codes. ** — Zefan-Cai
  [GitHub](https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache)
  > Awesome-LLM-KV-Cache: A curated list of 📙Awesome LLM KV Cache Papers with Codes. ...

- **[GitHub] Fast-dLLM: Official implementation of "Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Ca** — NVlabs
  [GitHub](https://github.com/NVlabs/Fast-dLLM)
  > Official implementation of "Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding"...

- **[GitHub] InfiniStore: KV cache store for distributed LLM inference** — bytedance
  [arXiv](https://arxiv.org/abs/2604.24971v1) | [GitHub](https://github.com/bytedance/InfiniStore)
  > We present PolyKV, a system in which multiple concurrent inference agents share a single, asymmetrically compressed KV cache pool. Rather than allocating a separate KV cache per agent -- the standard ...

- **[GitHub] KVCache-Factory: Unified KV Cache Compression Methods for Auto-Regressive Models** — Zefan-Cai
  [GitHub](https://github.com/Zefan-Cai/KVCache-Factory)
  > Unified KV Cache Compression Methods for Auto-Regressive Models...

- **[GitHub] R-KV: [Neurips 2025] R-KV: Redundancy-aware KV Cache Compression for Reasoning Models** — Zefan-Cai
  [GitHub](https://github.com/Zefan-Cai/R-KV)
  > [Neurips 2025] R-KV: Redundancy-aware KV Cache Compression for Reasoning Models...

- **[GitHub] RWKV-LM: RWKV (pronounced RwaKuv) is an RNN with great LLM performance, which can also be directly trained li** — BlinkDL
  [GitHub](https://github.com/BlinkDL/RWKV-LM)
  > RWKV (pronounced RwaKuv) is an RNN with great LLM performance, which can also be directly trained like a GPT transformer (parallelizable). We are at RWKV-7 "Goose". So it's combining the best of RNN a...

- **[GitHub] ShadowKV: [ICML 2025 Spotlight] ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference** — ByteDance-Seed
  [GitHub](https://github.com/ByteDance-Seed/ShadowKV)
  > [ICML 2025 Spotlight] ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference...

- **[GitHub] SwiftLM: ⚡ Native MLX Swift LLM inference server for Apple Silicon. OpenAI-compatible API, SSD streaming for ** — SharpAI
  [GitHub](https://github.com/SharpAI/SwiftLM)
  > ⚡ Native MLX Swift LLM inference server for Apple Silicon. OpenAI-compatible API, SSD streaming for 100B+ MoE models, TurboQuant KV cache compression, MACOS + iOS iPhone app....

- **[GitHub] fox: High-performance LLM inference engine — drop-in replacement for Ollama with faster multi-turn infere** — ferrumox
  [GitHub](https://github.com/ferrumox/fox)
  > High-performance LLM inference engine — drop-in replacement for Ollama with faster multi-turn inference, lower TTFT, and higher throughput through prefix caching and continuous batching....

- **[GitHub] kvcached: Virtualized Elastic KV Cache for Dynamic GPU Sharing and Beyond** — ovg-project
  [GitHub](https://github.com/ovg-project/kvcached)
  > Virtualized Elastic KV Cache for Dynamic GPU Sharing and Beyond...

- **[GitHub] kvpress: LLM KV cache compression made easy** — NVIDIA
  [arXiv](https://arxiv.org/abs/2604.24971v1) | [GitHub](https://github.com/NVIDIA/kvpress)
  > We present PolyKV, a system in which multiple concurrent inference agents share a single, asymmetrically compressed KV cache pool. Rather than allocating a separate KV cache per agent -- the standard ...

- **[GitHub] turboquant-pytorch: From-scratch PyTorch implementation of Google's TurboQuant (ICLR 2026) for LLM KV cache compression.** — tonbistudio
  [GitHub](https://github.com/tonbistudio/turboquant-pytorch)
  > From-scratch PyTorch implementation of Google's TurboQuant (ICLR 2026) for LLM KV cache compression. 5x compression at 3-bit with 99.5% attention fidelity....

- **[GitHub] turboquant: TurboQuant: Near-optimal KV cache quantization for LLM inference (3-bit keys, 2-bit values) with Tri** — 0xSero
  [GitHub](https://github.com/0xSero/turboquant)
  > TurboQuant: Near-optimal KV cache quantization for LLM inference (3-bit keys, 2-bit values) with Triton kernels + vLLM integration...

- **[GitHub] uccl: UCCL is an efficient communication library for GPUs, covering collectives, P2P (e.g., KV cache trans** — uccl-project
  [GitHub](https://github.com/uccl-project/uccl)
  > UCCL is an efficient communication library for GPUs, covering collectives, P2P (e.g., KV cache transfer, RL weight transfer), and EP (e.g., GPU-driven)...

- **[GitHub] vllm-kvcompress: KV cache compression for high-throughput LLM inference** — IsaacRe
  [GitHub](https://github.com/IsaacRe/vllm-kvcompress)
  > KV cache compression for high-throughput LLM inference...


### LLM Pruning/Serving

- **GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models** — Ziyang Wang, Jiangfeng Xiao, Chuan Xiao, Ruoxiang Li, Rui Mao, Jianbin Qin
  [arXiv](https://arxiv.org/abs/2604.19398)
  > Large language models (LLMs) are expensive to serve because model parameters, attention computation, and KV caches impose substantial memory and latency costs. GRASPrune presents a structured pruning ...

- **SimDiff: Depth Pruning via Similarity and Difference** — Yuli Chen, Shuhao Zhang, Fanshen Meng, Bo Cheng, Jiale Han
  [arXiv](https://arxiv.org/abs/2604.19520)
  > Depth pruning removes entire layers of LLMs to reduce inference cost. SimDiff proposes a novel depth pruning approach via similarity and difference analysis between adjacent layers, enabling more effe...


### LLM Serving

- **31.1 A 14.08-to-135.69Token/s ReRAM-on-Logic Stacked Outlier-Free Large-Language-Model Accelerator with Block-Clustered Weight-Compression and Adaptive Parallel-Speculative-Decoding** — Pingcheng Dong, Yonghao Tan, Xuejiao Liu, Peng Luo, Yu Liu, Di Pang, Songchen Ma, Xijie Huang, Shih-Yang Liu, Dong Zhang, Zhichao Lu, Luhong Liang, Chi-Ying Tsui, Fengbin Tu, Liang Zhao, Kwang-Ting Cheng
  > This work presents a 55nm speculative decoding-based LLM accelerator with bumpingbased face-to-face ReRAM-on-logic stacking technology. It features a local rotation unit for outlier-free low-bit quant...

- **A 28nm Speculative-Decoding LLM Processor Achieving 105-to-685µs/Token Latency for Billion-Parameter Models** — Yang Wang, Huanyu Wang, Jiaxin Yang, Yutong Su, Ruiqi Guo, Zhiheng Yue, Jiangyuan Gu, Shaojun Wei, Yang Hu, Shouyi Yin
  > LLMs face decoding bottlenecks. Speculative Decoding (SD) reduces latency via a small draft model for serial decoding and a large target model to verify in parallel. Despite this advantage, it still s...

- **A Frozen 12B Beats Frontier Models on Verified Work: 100% Accuracy, 0 Tokens, Bit-Exact, Forever** — Sietse Schelpe
  [arXiv](https://arxiv.org/abs/2607.23806v1)
  > Improving a language model today means retraining it: enormous compute, a new opaque model each cycle, non-deterministic output. We take the opposite path: the model stays frozen, and a persistent mem...

- **A Generative Partially Specified Finite State Machine Approach to Complex Behaviour Planning** — Kalana Ratnayake, Michael Pritchard, David Hinwood, Maleen Jayasuriya, Damith Herath
  [arXiv](https://arxiv.org/abs/2607.15674v1)
  > Autonomous robots operating in dynamic environments require behaviour planning systems that combine reactivity, interpretability, and adaptability. While Large Language Models have been successfully i...

- **A Hybrid Online and Offline Requests Inference Serving System for LLM in Private Computer Environment** — Yuchen Shen, Yuning Zhang, Dong Yuan
  > While advancements in Large Language Models (LLMs) have broadened their applications, performing multitask LLM inference on a single GPU remains challenging due to insufficient GPU memory to load all ...

- **A Metamorphic Testing Approach to Diagnosing Memorization in LLM-Based Program Repair** — Milan De Koning, Ali Asgari, Pouria Derakhshanfar, Annibale Panichella
  [arXiv](https://arxiv.org/abs/2604.21579v1)
  > LLM-based automated program repair (APR) techniques have shown promising results in reducing debugging costs. However, prior results can be affected by data leakage: large language models (LLMs) may m...

- **A Roadmap to Impactful Pluralistic Alignment Research** — Elinor Poole-Dayan, Jillian Fisher, Atoosa Kasirzadeh, Jacob Andreas, Mitchell Gordon, Michiel A. Bakker
  [arXiv](https://arxiv.org/abs/2607.22305v1)
  > Pluralistic value alignment---the goal of building AI systems that represent and serve diverse human values and perspectives---has emerged as an active research agenda. Yet, there's no public evidence...

- **ACRL: Adaptive Control of Training-Inference Discrepancy for Stable Reinforcement Learning** — Wenwu Fan, Qihong Lin, Zhijie Xia, Zhuo Zheng, Sihao Wang, Qiang Chen, Liangsheng Zhu
  [arXiv](https://arxiv.org/abs/2607.24062v1)
  > Reinforcement Learning (RL) training for Large Language Models (LLMs) often suffers from instability due to the discrepancy between training and inference. This training-inference discrepancy stems fr...

- **AGG: Jacobian-Aggregated Group Gradient for Efficient GRPO Training of Diffusion Models** — Ruiyi Ding, Jie Li, He Kang, Ziyan Liu, Chengru Song, Yuan chen
  [arXiv](https://arxiv.org/abs/2607.17572v1)
  > Group Relative Policy Optimization (GRPO) is a powerful reinforcement learning algorithm for aligning generative models with human preferences. While successful in large language models~\cite{shao2024...

- **AIGB-R1: Self-Evolving Generative Auto-Bidding via Hierarchical Planner-Executor Optimization** — Yuejia Dou, Hesong Wang, Xinyu Zhang, Tianyu Wang, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng, Qi Qi
  [arXiv](https://arxiv.org/abs/2607.17281v1)
  > Auto-bidding plays an essential role in online advertising, automatically adjusting bids for advertisers to optimize their commercial goals. The emerging AI-Generated Bidding (AIGB) paradigm widely ad...

- **ARBITER: Guarded Agentic Control for SLO-Oriented Kubernetes Remediation** — Pooyan Habibi, Alberto Leon-Garcia
  [arXiv](https://arxiv.org/abs/2607.19182v1)
  > Maintaining service-level objectives (SLOs) on Kubernetes microservices remains difficult because autoscalers observe coarse resource metrics, recent SLO controllers often depend on custom telemetry, ...

- **ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants** — Haohui Mai, Xiaoyan Guo, Xiangyun Ding, Daifeng Li, Qiuchu Yu, Chenzhun Guo, Cong Wang, Jiacheng Zhao, Christos Kozyrakis, Binhang Yuan
  [arXiv](https://arxiv.org/abs/2604.18616)
  > LLM-based coding agents can generate functionally correct GPU kernels, yet their performance remains far below hand-optimized libraries on critical computations such as matrix multiplication, attentio...

- **Accelerating Speculative Decoding with Block Diffusion Draft Trees** — Liran Ringel
  [arXiv](https://arxiv.org/abs/2604.12989)
  > Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a ...

- **Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving** — Takeshi Yoshimura, Valentijn Dymphnus van de Beek, Tatsuhiro Chiba
  [arXiv](https://arxiv.org/abs/2604.15732)
  > Distributed LLM serving systems optimize per-request latency and throughput. However, under long-context workloads, inference accuracy becomes more variable. When incorrect responses trigger retries, ...

- **AdaFlash: Adaptive Speculative Decoding via On-Policy Distilled Diffusion Drafters** — Yu-Yang Qian, Hao-Cong Wu, Chen Chen, Jiacheng Sun, Zhenhua Dong, Peng Zhao, Zhi-Hua Zhou
  [arXiv](https://arxiv.org/abs/2607.19223v1)
  > Speculative decoding, in which a lightweight draft model first generates a draft sequence that is then verified in parallel by the target model, has become a prevalent paradigm for accelerating large ...

- **AdaHome: An Adaptive Smart Home Assistant using Local Small Language Models** — Eu Jin Lim, Zhaoxing Li, Sebastian Stein
  [arXiv](https://arxiv.org/abs/2607.18034v1)
  > Smart home assistants interpret a wide range of user commands, from explicit device control to underspecified and preference dependent requests. While recent systems based on Large Language Models (LL...

- **AdaSpec: Adaptive Multilingual Speculative Decoding with Self-Synthesized Language-Aware Training and Vocabulary Simplification** — Dinh-Truong Do, Nguyen-Khang Le, Le-Minh Nguyen
  > Speculative decoding accelerates large language model (LLM) inference by using a lightweight drafter to propose multiple tokens, which are then verified in parallel by the base model. While effective ...

- **Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration** — Zhuofan Wen
  [arXiv](https://arxiv.org/abs/2604.12247)
  > Speculative decoding has emerged as a promising approach to accelerate autoregressive inference in LLMs. Self-draft methods leverage the base LLM itself for speculation, but shallow layers often produ...

- **Adaptive Depth Sparse Framework: Similarity-Driven Resource Allocation for Pre-Trained LLMs** — Yidu Wu, Xiang Wang, Kejie Zhao, Zhangchi Wang, Qinghai Guo, Xiaoying Tang
  [arXiv](https://arxiv.org/abs/2607.21291v1)
  > Large language models (LLMs) achieve strong generation and reasoning performance, but the Transformer architecture incurs high inference cost. Existing acceleration methods often rely on task-specific...

- **Adelia: A 4-nm LLM Processing Unit With Streamlined Dataflow and Dual-Mode Parallelism for Maximizing Hardware Efficiency** — Sukbin Lim, Jung-Hoon Kim, Seungjae Moon, Junseo Cha, Dongjin Seo, Jongho Kim, Hunjong Lee, Jinwon Lee, Joo-Young Kim
  > The proliferation of large language models (LLMs) as cross-domain foundation models is fueled by aggressive scaling in both parameter counts and inference-time computation. The emergence of sophistica...

- **AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU** — Yuning Zhang et al.
  [arXiv](https://arxiv.org/abs/2603.10342)
  > AgentServe presents a single-GPU serving system that ensures stable multi-agent execution by isolating prefills from decodes, applying dynamic budgeting to resume prefills, and allocating GPU resource...

- **Agentic CPU-GPU Scheduling for Heterogeneous AI Workloads** — Tianxi Lu, Sherief Reda
  [arXiv](https://arxiv.org/abs/2607.22242v1)
  > Agentic AI systems compose heterogeneous tool workloads on shared GPU/CPU infrastructure, yet existing frameworks assign all GPU-capable tools to the GPU by default. We profile 19 AI tools across GPU ...

- **Agentic Root Cause Analysis through Evidence-Grounded Reasoning** — Amaury Wei, Olga Fink
  [arXiv](https://arxiv.org/abs/2607.22385v1)
  > Diagnosing the root cause of anomalies is essential for safe industrial operation. Despite extensive sensor instrumentation, formulating hypotheses and gathering evidence remains a manual process, cre...

- **Agents in the Wild: Where Research Meets Deployment** — Grace Hui Yang, Pranav N. Venkit, Hooman Sedghamiz, Enrico Santus, Victor Dibia, Ioana Baldini
  [arXiv](https://arxiv.org/abs/2607.19336v1)
  > Agentic systems large language model (LLM) based architectures capable of reasoning, planning, acting, and coordinating with tools and other agents are rapidly transitioning from research prototypes t...

- **An MLIR-Based Compilation Method for Large Language Models** — Pengchao Hu, Zhibin Xin, Yifan Chen, Yangyang Zhou, Liang Wang
  [arXiv](https://arxiv.org/abs/2607.15865v1)
  > Large Language Models (LLMs) have become the dominant workload on modern AI accelerators, yet deploying them on specialized hardware still faces two core challenges: how to import a trained model into...

- **AnovaX: A Local, Multi-Agent Voice Assistant with LLM Planning, Typed Executors, and Adaptive Recovery** — Raunak B Sinha
  [arXiv](https://arxiv.org/abs/2607.15367v1)
  > Desktop voice assistants are still dominated by cloud pipelines that ship raw audio off the machine and expose a fixed set of skills. We describe AnovaX, a small local-first assistant that runs entire...

- **AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention** — Yuxuan Hu, Jianchao Tan, Jiaqi Zhang, Wen Zan, Pingwei Sun, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai, Jing Zhang
  [arXiv](https://arxiv.org/abs/2604.07815)
  > Long-context inference in LLMs faces quadratic attention complexity and prohibitive KV cache memory. AsyncTLS proposes a hierarchical sparse attention system combining coarse-grained block filtering w...

- **B-PASTE: Beam-Aware Pattern-Guided Speculative Execution for Resource-Constrained LLM Agents** — Yanfei Song
  [arXiv](https://arxiv.org/abs/2604.16469)
  > LLM agents execute in an interleaved reasoning-and-action loop, where future tool calls cannot be launched until the current reasoning step completes. This serial dependency inflates end-to-end latenc...

- **Benchmarking Compound AI Applications for Hardware-Software Co-Design** — Paramuth Samuthrsindh, Angel Cervantes, Varun Gohil, Gohar Irfan Chaudhry, Christina Delimitrou, Adam Belay
  [arXiv](https://arxiv.org/abs/2604.09593)
  > Compound AI applications, composed from interactions between Large Language Models (LLMs), Machine Learning (ML) models, external tools and data sources are quickly becoming an integral workload in da...

- **Break the Optimization Barrier of LLM-Enhanced Recommenders: A Theoretical Analysis and Practical Framework** — Zhangchi Zhu, Wei Zhang
  [arXiv](https://arxiv.org/abs/2604.20490v1) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > Large language model (LLM)-enhanced recommendation models inject LLM representations into backbone recommenders to exploit rich item text without inference-time LLM cost. However, we find that existin...

- **C$^2$KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference** — Chuheng Du, Junyi Chen, Hanlin Tang, Kan Liu, Tao Lan, Lin Qu, Chaoyue Niu, Shengzhong Liu, Guihai Chen, Fan Wu
  [arXiv](https://arxiv.org/abs/2607.17715v1)
  > Long-context inference is central to modern large language model (LLM) applications such as retrieval-augmented generation and multi-document reasoning. To mitigate the growing inference cost, recent ...

- **C-PTQ: Fisher-weighted Channel-wise Sensitivity for Post-training Quantization of MLLMs** — Jiameng Li, Han Zhou, Matthew B. Blaschko
  [arXiv](https://arxiv.org/abs/2607.21076v1)
  > Multimodal large language models (MLLMs) require huge memory and computational costs, which limits their practical deployment. Post-training quantization (PTQ) techniques offer an efficient solution f...

- **CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands** — Weiye Wang, Chen Chen, Junxue Zhang, Zhusheng Wang, Hui Yuan, Zixuan Guan, Xiaolong Zheng, Qizhen Weng, Yin Chen, Minyi Guo
  [arXiv](https://arxiv.org/abs/2603.21257)
  > Distributed prefix caching has become a core technique for efficient LLM serving. However, for long-context requests with high cache hit ratios, retrieving reusable KVCache blocks from remote servers ...

- **CCCL: In-GPU Compression-Coupled Collective Communication** — Chon Lam Lao, Zhiying Xu, Zhuang Wang, Ziming Mao, Delong Meng, Jia Zhen, Jun Wu, Ion Stoica, Yida Wang, Yang Zhou
  [arXiv](https://arxiv.org/abs/2604.17172)
  > Collective communication incurs significant overhead in LLM workloads. Although overlapping communication with computation in application-level is a common strategy, it often requires substantial code...

- **CONSISTRE: A Unified Consistency-Aware Framework for Document-Level Relation Extraction with Large Language Models** — Mingxuan Sun
  [arXiv](https://arxiv.org/abs/2607.24312v1)
  > Document-level relation extraction (DocRE) aims to extract relations among multiple entities across extended contexts while maintaining consistency across predicted triples. Although large language mo...

- **Cache-Aware Prompt Compression:A Two-Tier Cost Model for LLM API Caching** — Yan Song
  [arXiv](https://arxiv.org/abs/2607.15516v1)
  > Production LLM deployments combine two cost-reduction primitives: prompt caching (a discounted rate for re-used token prefixes) and prompt compression (fewer tokens sent). The compression literature h...

- **Can We Break LLMs Out of Self-Loops? Fine-Grained Reasoning Control with Activation Steering** — Sheldon Yu, Tong Yu, Xunyi Jiang, Rohan Surana, Gagan Mundada, Sungchul Kim, Lina Yao, Julian McAuley, Junda Wu
  [arXiv](https://arxiv.org/abs/2607.18100v1)
  > Extended reasoning has become standard for frontier Large Language Models (LLMs), yet the trajectories these models produce remain largely uncontrollable. Existing methods for shaping how a model reas...

- **CausalForge: A Formally Grounded, Self-Improving Agentic Framework for Automated Research in Causal Inference** — Jiyuan Tan, Vasilis Syrgkanis
  [arXiv](https://arxiv.org/abs/2607.22511v1)
  > Automating theoretical research is constrained not only by the generation of candidate results, but also by their reliable evaluation. A common approach is to close the research loop with a large lang...

- **Characterizing Performance-Energy Trade-offs of Large Language Models in Multi-Request Workflows** — Md. Monzurul Amin Ifath, Israat Haque
  [arXiv](https://arxiv.org/abs/2604.09611)
  > First systematic characterization of performance-energy trade-offs in multi-request LLM inference. We develop four representative workloads (sequential, interactive, agentic, composite). Using NVIDIA ...

- **Chimera: Latency- and Performance-Aware Multi-agent Serving for Heterogeneous LLMs** — Kangqi Ni, Wenyue Hua, Xiaoxiang Shi, Jiang Guo, Shiyu Chang, Tianlong Chen
  [arXiv](https://arxiv.org/abs/2603.22206)
  > Multi-agent applications execute complex tasks as multi-stage workflows where each stage is an LLM call. Existing LLM serving systems largely assume homogeneous clusters with identical model replicas,...

- **Cloud-native and Distributed Systems for Efficient and Scalable Large Language Models -- A Research Agenda** — Minxian Xu, Jingfeng Wu, Shengye Song, Satish Narayana Srirama, Bahman Javad, Rajiv Ranjan, Devki Nandan Jha, Sa Wang, Wenhong Tian, Huanle Xu, Li Li, Zizhao Mo, Shuo Ren, Thomas Kunz, Petar Kochovski, Vlado Stankovski, Kejiang Ye, Chengzhong Xu, Rajkumar Buyya
  [arXiv](https://arxiv.org/abs/2604.17227)
  > The rapid rise of Large Language Models (LLMs) has revolutionized various artificial intelligence (AI) applications, from natural language processing to code generation. However, the computational dem...

- **CoFEE: Reasoning Control for LLM-Based Feature Discovery** — Maximilian Westermann, Ben Griffin, Aaron Ontoyin Yin, Zakari Salifu, Yagiz Ihlamur
  [arXiv](https://arxiv.org/abs/2604.21584v1) | [GitHub](https://github.com/Zefan-Cai/R-KV)
  > Feature discovery from complex unstructured data is fundamentally a reasoning problem: it requires identifying abstractions that are predictive of a target outcome while avoiding leakage, proxies, and...

- **CoLLM: A Unified Framework for Co-execution of LLMs Federated Fine-tuning and Inference** — Shaoyuan Huang, Xiaokai Wang, Na Yan, Xiaofei Wang, Wenyu Wang, Yansha Deng
  [arXiv](https://arxiv.org/abs/2604.16400)
  > As Large Language Models (LLMs) are increasingly adopted in edge intelligence to power domain-specific applications and personalized services, the quality and efficiency of the LLM post-training phase...

- **Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks** — ['Songge Zhang', 'Wen Wu', 'Liang Li', 'Ye Wang', 'Xuemin', 'Shen']
  [arXiv](https://arxiv.org/abs/2604.04654)
  > Low Earth orbit (LEO) satellites play an essential role in intelligent Earth observation by leveraging artificial intelligence models. However, limited onboard memory and excessive inference delay pre...

- **ContinuityBench: A Benchmark and Systems Study of Stateful Failover in Multi-Provider LLM Routing** — Vishal Pandey, Gopal Singh
  [arXiv](https://arxiv.org/abs/2607.15899v1)
  > In production large language model (LLM) deployments, high API availability guarantees do not equate to conversational continuity. When a primary provider experiences an outage or strict rate-limiting...

- **Continuous Semantic Caching for Low-Cost LLM Serving** — Baran Atalar, Xutong Liu, Jinhang Zuo, Siwei Wang, Wei Chen, Carlee Joe-Wong
  [arXiv](https://arxiv.org/abs/2604.15873)
  > Large language models (LLMs) are increasingly studied as repositories of linguistic knowledge. In this line of work, models are commonly evaluated both as generators of language and as judges of lingu...

- **Cornserve: A Distributed Serving System for Any-to-Any Multimodal Models** — Jae-Won Chung, Jeff J. Ma, Jisang Ahn, Yizhuo Liang, Akshay Jajoo, Myungjin Lee, Mosharaf Chowdhury
  [arXiv](https://arxiv.org/abs/2603.12118) | [GitHub](https://github.com/cornserve-ai/cornserve)
  > Cornserve provides a flexible task abstraction for expressing Any-to-Any model computation graphs, enabling component disaggregation and independent scaling. The distributed runtime dispatches compute...

- **Cost-Efficient Multimodal LLM Inference via Cross-Tier GPU Heterogeneity** — Donglin Yu
  [arXiv](https://arxiv.org/abs/2603.12707)
  > Multimodal large language model (MLLM) inference splits into two phases with opposing hardware demands: vision encoding is compute-bound, while language generation is memory-bandwidth-bound. We show t...

- **D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound 3D NAND-Optimized Similarity Search with Vector Adaptation** — Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S. Rosing, Mingu Kang
  [arXiv](https://arxiv.org/abs/2607.17538v1)
  > Retrieval-Augmented Generation (RAG) enhances the factual grounding of large language model (LLM) inference by retrieving relevant information from external knowledge bases. However, its dense vector ...

- **D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding** — Tianyu Liu, Yuhao Shen, Rui Cen, Junhan Shi, Jiebin Zhang, Guangshuo Qin, Hong Liu, Song Liu, Guanghua Yu, Jianchen Zhu
  [arXiv](https://arxiv.org/abs/2607.14647v1)
  > Speculative decoding accelerates large language model (LLM) inference without compromising output quality. Recent parallel drafting methods further improve single-request performance by decoupling dra...

- **DAT: Dual-Aware Adaptive Transmission for Efficient Multimodal LLM Inference in Edge-Cloud Systems** — ['Qi Guo', 'Zheming Yang', 'Yunqing Hu', 'Chang Zhao', 'Wen Ji']
  [arXiv](https://arxiv.org/abs/2604.05375)
  > Multimodal large language models (MLLMs) have shown strong capability in semantic understanding and visual reasoning, yet their use on continuous video streams in bandwidth-constrained edge-cloud syst...

- **DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU** — Shaoqiang Lu, Yangbo Wei, Junhong Qian, Dongge Qin, Shiji Gao, Yizhi Ding, Qifan Wang, Chen Wu, Xiao Shi, Lei He
  [GitHub](https://github.com/ShaoqiangLu/DFVG)
  > Speculative decoding is a promising paradigm that accelerates LLM inference by generating drafts and performing verification. However, such systems still face three major challenges: (1) The imbalance...

- **DUET: Disaggregated Hybrid Mamba-Transformer LLMs with Prefill and Decode-Specific Packages** — Alish Kanani, Sangwan Lee, Han Lyu, Jiahao Lin, Jaehyun Park, Umit Y. Ogras
  [arXiv](https://arxiv.org/abs/2603.15530)
  > DUET introduces a disaggregated accelerator that assigns prefill and decode phases to specialized packages. The Prefill package utilizes systolic array chiplets with off-package memory. The Decode pac...

- **Data Quality over Capacity: Internalizing Documents into LoRA Adapters for Closed-Book QA** — Joan Figuerola Hurtado
  [arXiv](https://arxiv.org/abs/2607.21861v1)
  > We study baking documents directly into the weights of a 4-bit Gemma-4-e4b model via LoRA, so a system can answer questions about a corpus closed-book: no retrieval and no context-window budget. Acros...

- **DeInfer: Efficient Parallel Inferencing for Decomposed Large Language Models** — You-Liang Huang, Xinhao Huang, Chengxi Liao, Zeyi Wen
  [arXiv](https://arxiv.org/abs/2604.17709)
  > Existing works on large language model (LLM) decomposition mainly focus on improving performance on downstream tasks, but they ignore the poor parallel inference performance when trying to scale up th...

- **DepCap: Adaptive Block-Wise Parallel Decoding for Efficient Diffusion LM Inference** — Xiang Xia, Wuyang Zhang, Jiazheng Liu, Cheng Yan, Yanyong Zhang
  [arXiv](https://arxiv.org/abs/2604.15750)
  > DepCap is a training-free framework for efficient block-wise DLM inference. DepCap uses cross-step signals for determining block boundaries and token-level conflict signals for parallel decoding. DepC...

- **Do LLM Decoders Listen Fairly? Benchmarking How Language Model Priors Shape Bias in Speech Recognition** — Srishti Ginjala, Eric Fosler-Lussier, Christopher W. Myers, Srinivasan Parthasarathy
  [arXiv](https://arxiv.org/abs/2604.21276v1)
  > As pretrained large language models replace task-specific decoders in speech recognition, a critical question arises: do their text-derived priors make recognition fairer or more biased across demogra...

- **DualDiffusion: A Speculative Decoding Strategy for Masked Diffusion Models** — N/A
  [arXiv](https://arxiv.org/abs/2604.05250)
  > Masked Diffusion Models (MDMs) offer a promising alternative to autoregressive language models by enabling parallel token generation and bidirectional context modeling. However, their inference speed ...

- **DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving** — Ying Yuan, Pengfei Zuo, Bo Wang, Zhangyu Chen, Zhipeng Tan, Zhou Yu
  [arXiv](https://arxiv.org/abs/2602.06502)
  > In LLM serving, reusing the KV cache of prompts across requests is critical for reducing TTFT and serving costs. Cache-affinity scheduling, which co-locates requests with the same prompt prefix to max...

- **DualScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS** — Omar Basit, Yunzhao Liu, Z. Jonny Kong, Y. Charlie Hu
  [arXiv](https://arxiv.org/abs/2602.18755)
  > Prefill/decode disaggregation is increasingly adopted in LLM serving to improve the latency-throughput tradeoff and meet strict TTFT and TPOT SLOs. However, LLM inference remains energy-hungry: autosc...

- **DynaCalKV: Key-Value Cache Compression via Head Grouping and Adaptive Rank Allocation** — Tan T. Nguyen, Quan V. Dang
  [arXiv](https://arxiv.org/abs/2607.24331v1)
  > As the inference phase of Large Language Models (LLMs) requires handling long context windows, the Key-Value (KV) cache initially appears to address this challenge but eventually becomes a significant...

- **Dynamic Micro-Batch and Token-Budget Scheduling for IoT-Scale Pipeline-Parallel LLM Inference** — Juncheol Ahn, Yubin Son, Daemin Kim, Sejin Park
  > Large language models in IoT–edge–cloud settings face bursty, heterogeneous requests that make pipeline-parallel inference prone to micro-batch imbalance and communication stalls, causing GPU idle tim...

- **ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs** — Multiple authors
  [arXiv](https://arxiv.org/abs/2604.03298)
  > ENEC proposes a lossless AI model compression method enabling fast inference on Ascend NPUs, addressing inference acceleration on heterogeneous hardware....

- **EchoKV: Efficient KV Cache Compression via Similarity-Based Reconstruction** — Yixuan Wang, Shiyu Ji, Yijun Liu, Qingfu Zhu, Wanxiang Che
  [arXiv](https://arxiv.org/abs/2603.22910)
  > KV cache memory demand poses a significant bottleneck for LLMs in long-context applications. Existing low-rank compression methods rely on irreversible parameter transformations, sacrificing flexibili...

- **EdgeCoInfer: Hierarchical Collaborative Inference for On-Device Multimodal Large Models** — Lin Tan, David K. Y. Yau, Songtao Guo
  [arXiv](https://arxiv.org/abs/2607.17143v1)
  > Modern mobile applications predominantly execute concurrent Multimodal Large Language Models (MLLMs) to provide ubiquitous intelligence. However, satisfying this demand within edge environments faces ...

- **EduGuard: A Safe RAG-Based LLM Tutor for Programming Education** — S M Asif Hossain, Ruksat Khan Shayoni, M. F. Mridha, Jungpil Shin
  [arXiv](https://arxiv.org/abs/2607.15738v1)
  > Generative AI (GenAI) is increasingly used by students for programming explanation, debugging, and assignment support. Yet unrestricted large language model (LLM) tutors can hallucinate, contradict co...

- **Efficient Clustering with Provable Guardrails for LLM Inference at Scale** — Longshaokan Wang, Wai Tsang Keung, Punit Ghodasara, Roman Wang, Ali Dashti, Francesc Moreno-Noguer
  [arXiv](https://arxiv.org/abs/2607.19704v1)
  > Scaling LLM-based applications to millions of users is bottlenecked by the inference cost and latency of modern foundation models. A natural fix is to cluster the inputs and call the LLM only on clust...

- **Efficient Multi-round LLM Inference over Disaggregated Serving (AMPD)** — Wenhao He, Youhe Jiang, Penghao Zhao, Quanqing Xu, Eiko Yoneki, Bin Cui, Fangcheng Fu
  [arXiv](https://arxiv.org/abs/2602.14516)
  > Multi-round workflows raise hurdles for PD disaggregation — existing systems overlook interleaved prefill-decode workload patterns. AMPD adaptively coordinates prefill workloads based on real-time con...

- **Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation** — Luxi Lin, Zhihang Lin, Zhanpeng Zeng, Yuhao Chen, Qingyu Zhang, Jixiang Luo, Xuelong Li, Rongrong Ji
  [arXiv](https://arxiv.org/abs/2603.09527) | [GitHub](https://github.com/https://github.com/Lyn-Lucy/Efficient-Draft-Adaptation)
  > Speculative decoding accelerates LLM inference but suffers from performance degradation when target models are fine-tuned for specific domains. We introduce EDA (Efficient Draft Adaptation), a paramet...

- **Enhancing Rubric-based RL via Self-Distillation** — Mingxuan Xia, Yuhang Yang, Chao Ye, Shuai Zhu, Shenzhi Yang, Guangcheng Zhu, Yuhang Zhang, Cheng Peng, Haobo Wang, Siqing Wang
  [arXiv](https://arxiv.org/abs/2607.18082v1)
  > Rubric-based RL has recently shown promise in improving LLMs on open-ended tasks. A widely recognized limitation of rubric-based RL is limited exploration: criteria that no rollout manages to satisfy ...

- **Euclid-MCP: A Model Context Protocol Server for Deterministic Logical Reasoning via Prolog** — Bartolomeo Bogliolo
  [arXiv](https://arxiv.org/abs/2607.21412v1)
  > Large Language Models (LLMs) excel at natural language understanding and generation but remain unreliable for multi-step logical reasoning, especially in safety-critical or compliance-sensitive domain...

- **Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel** — Hongyi Jin, Bohan Hou, Guanjie Wang, Ruihang Lai, Jinqi Chen, Zihao Ye, Yaxing Cai, Yixin Dong, Xinhao Cheng, Zhihao Zhang, Yilong Zhao, Yingyi Huang, Lijie Yang, Jinchen Jiang, Gabriele Oliaro, Jianan Ji, Xupeng Miao, Vinod Grover, Todd C. Mowry, Zhihao Jia, Tianqi Chen
  [arXiv](https://arxiv.org/abs/2604.13327)
  > Modern GPU workloads, especially large language model (LLM) inference, suffer from kernel launch overheads and coarse synchronization that limit inter-kernel parallelism. Recent megakernel techniques ...

- **Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives** — Siyuan Shen, Anton Korzh, John Bachan, Tiancheng Chen, Arnav Goel, Ludwig Schneider, Pouya Kousha, Zhenhao He, Sylvain Jeaugey, Kamil Iskra, Nishank Chandawala, Jeff R. Hammond, Torsten Hoefler
  [arXiv](https://arxiv.org/abs/2607.16100v1)
  > GPU collective communication is typically optimized for bandwidth, yet many emerging workloads are increasingly limited by latency. Long-context decode-heavy large language model (LLM) inference is a ...

- **ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels** — Bingyang Wu, Chao Jin, Zili Zhang, Xinming Wei, Yinmin Zhong, Ruidong Zhu, Chengxu Yang, Xin Jin, Yuliang Liu
  [arXiv](https://arxiv.org/abs/2607.18002v1)
  > LLMs scale Mixture-of-Experts (MoE) parameters for superior intelligence, but massive weights and dynamic computation impede efficient serving. Existing instance-level prefill-decode disaggregation is...

- **Failures Reveal What Metrics Miss: An Evidence-Driven Agent for Recursive Refinement of ECG Classifiers** — Jinliang Deng, Yiming Niu, Yibo Pan, Zhiqi Shao, Qin Luo, Yongxin Tong
  [arXiv](https://arxiv.org/abs/2607.24419v1)
  > Deep models have substantially advanced 12-lead ECG classification, yet their refinement still relies heavily on human experts to inspect failures and iteratively revise classifier designs. Recent LLM...

- **Fast Forward: Accelerating LLM Prefill with Predictive FFN Sparsity** — Aayush Gautam, Mukul Gagrani, Junyoung Park, Mingu Lee, Chiris Lott, Narasimha Reddy
  [arXiv](https://arxiv.org/abs/2602.00397)
  > The prefill stage of large language model (LLM) inference is a key computational bottleneck for long-context workloads. At short-to-moderate context lengths (1K--16K tokens), Feed-Forward Networks (FF...

- **Faster LLM Inference via Sequential Monte Carlo** — Yahya Emara et al.
  [arXiv](https://arxiv.org/abs/2604.15672)
  > Speculative decoding (SD) accelerates language model inference by drafting tokens from a cheap proposal model and verifying them against an expensive target model via rejection sampling. Because rejec...

- **Fewer Paths, Better Performance: Understanding the ZCube Topology through Braess's Paradox** — Li Chen
  [arXiv](https://arxiv.org/abs/2607.21893v1)
  > Datacenter networks follow a multipath doctrine: provision many paths between endpoints, hash flows across them, and let redundancy absorb both failures and load imbalance. The ZCube topology violates...

- **Find Before You Fine-Tune: A Diagnostic Study of Small LLMs for Cybersecurity QA** — Shaswata Mitra, Subash Neupane, Trisha Chakraborty, Himanshu Tripathi, Sudip Mittal, Aritran Piplai, Shahram Rahimi
  [arXiv](https://arxiv.org/abs/2607.18725v1)
  > Large Language Models (LLMs) are increasingly fine-tuned for critical-domain Question-Answering (QA), yet choosing which small model to adapt, before paying the cost of adaptation, remains difficult. ...

- **Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for Mixture-of-Experts** — Minyu Cui, Anna Wingkvist, Morgan Ericsson
  [arXiv](https://arxiv.org/abs/2607.19539v1)
  > Mixture-of-Experts (MoE) architectures increase model capacity without proportionally increasing computation cost and have become a key building block for scaling large language models (LLMs) to trill...

- **FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications** — Krish Agarwal, Zhuoming Chen, Yanyuan Qin, Zhenyu Gu, Atri Rudra, Beidi Chen
  [arXiv](https://arxiv.org/abs/2607.18171v1)
  > Real-time multimodal applications, including voice agents and interactive video generation, compose heterogeneous models into pipelines whose efficient deployment requires application-specific decisio...

- **Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs** — N/A
  [arXiv](https://arxiv.org/abs/2604.15379)
  > Modern GPUs adopt chiplet-based designs with multiple private cache hierarchies, but current programming models (CUDA/HIP) expose a flat execution hierarchy that cannot express chiplet-level locality ...

- **FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design** — Jiahao Zhang, Zifan He, Nicholas Fraser, M. Blott, Yizhou Sun, Jason Cong
  [arXiv](https://arxiv.org/abs/2601.15710)
  > We present FlexLLM, a composable High-Level Synthesis (HLS) library for rapid development of domain-specific LLM accelerators. FlexLLM exposes key architectural degrees of freedom for stage-customized...

- **FlexSpec: Frozen Drafts Meet Evolving Targets in Edge-Cloud Collaborative LLM Speculative Decoding** — Yuchen Li, Rui Kong, Zhonghao Lyu, Qiyang Li, Xinran Chen, Hengyi Cai, Lingyong Yan, Shuaiqiang Wang, Jiashu Zhao, Guangxu Zhu, Linghe Kong, Guihai Chen, Haoyi Xiong, Dawei Yin
  [arXiv](https://arxiv.org/abs/2601.00644)
  > Deploying large language models (LLMs) in mobile and edge computing environments is constrained by limited on-device resources, scarce wireless bandwidth, and frequent model evolution. Although edge-c...

- **Flow-Controlled Scheduling for LLM Inference with Provable Stability Guarantees** — Zhuolun Dong, Junyu Cao
  [arXiv](https://arxiv.org/abs/2604.11001)
  > LLM inference faces a key challenge: decode lengths are unknown, so memory usage per request grows with generated tokens, potentially causing overflow and instability. We propose a simple flow-control...

- **FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models** — Bing Tian, Haikun Liu, Xiaocheng Zhong, Zhuohui Duan, Zhaokai Luo, Huayi Jin, Zhiyong Wang, Xiaofei Liao
  [arXiv](https://arxiv.org/abs/2607.17652v1)
  > Block-wise diffusion large language models (dLLMs) decode sequentially at the block level, enabling effective KV-cache reuse across blocks but making inter-block decoding strictly serial. Prior work h...

- **FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving** — Chia-chi Hsieh, Zan Zong, Xinyang Chen, Jianjiang Li, Jidong Zhai, Lijie Wen
  [arXiv](https://arxiv.org/abs/2602.16603)
  > The growing demand for large language models (LLMs) requires serving systems to handle many concurrent requests with diverse service level objectives (SLOs). This exacerbates head-of-line (HoL) blocki...

- **Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference** — ['Quantong Qiu', 'Zhiyi Hong', 'Yi Yang', 'Haitian Wang', 'Kebin Liu', 'Qingqing Dang']
  [arXiv](https://arxiv.org/abs/2604.07394)
  > The quadratic computational complexity of standard attention mechanisms presents a severe scalability bottleneck for LLMs in long-context scenarios. While hybrid attention mechanisms combining Full At...

- **ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache** — Shao Wang, Rui Ren, Lin Gui
  [arXiv](https://arxiv.org/abs/2604.06370)
  > The serving paradigm of LLMs is rapidly shifting towards complex multi-agent workflows. While LoRA enables efficient co-hosting of specialized agents, it introduces a critical memory bottleneck — uniq...

- **From Agent Loops to Structured Graphs: A Scheduler-Theoretic Framework for LLM Agent Execution** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.11378)
  > We propose a scheduler-theoretic framework for LLM agent execution, transforming agent loops into structured graphs for more efficient scheduling and execution in LLM inference systems....

- **From Feasibility to Desirability: Plan, Learn, Adapt (PLA) Framework for Personalized On-Device Itinerary Generation** — Himel Dev, Tanmoy Sen, Madhusudan Basak, Bashima Islam
  [arXiv](https://arxiv.org/abs/2607.15552v1)
  > Generating personalized trip itineraries is a complex planning task and involves a tension between hard combinatorial feasibility and soft latent desirability. Classical optimization enforces constrai...

- **From Inference Routing to Agent Orchestration: Declarative Policy Compilation with Cross-Layer Verification** — Huamin Chen, Xunzhuo Liu, Bowei He, Xue Liu
  [arXiv](https://arxiv.org/abs/2603.27299)
  > Extends the Semantic Router DSL from stateless per-request routing to multi-step agent workflows. The compiler emits verified decision nodes for orchestration frameworks (LangGraph, OpenClaw), Kuberne...

- **From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning** — Grant Wilkins, Fiodar Kazhamiaka, Ram Rajagopal
  [arXiv](https://arxiv.org/abs/2603.18383)
  > Datacenter operators and electrical utilities rely on power traces at different spatiotemporal scales. Operators use fine-grained traces for provisioning, facility management, and scheduling, while ut...

- **Fully Homomorphic Encryption on Llama 3 model for privacy preserving LLM inference** — N/A
  [arXiv](https://arxiv.org/abs/2604.12168)
  > The applications of Generative Artificial Intelligence (GenAI) and their intersections with data-driven fields, such as healthcare, finance, transportation, and information security, have led to signi...

- **GPU Acceleration of TFHE-Based High-Precision Nonlinear Layers for Encrypted LLM Inference** — ['Guoci Chen', 'Xiurui Pan', 'Qiao Li', 'Bo Mao', 'Congming Gao', 'Chengying Huan']
  [arXiv](https://arxiv.org/abs/2604.04783)
  > Deploying large language models (LLMs) as cloud services raises privacy concerns as inference may leak sensitive data. Fully Homomorphic Encryption (FHE) allows computation on encrypted data, but curr...

- **Generalizing Test Cases for Comprehensive Test Scenario Coverage** — Binhang Qi, Yun Lin, Xinyi Weng, Chenyan Liu, Hailong Sun
  [arXiv](https://arxiv.org/abs/2604.21771v1) | [GitHub](https://github.com/LMCache/LMCache)
  > Test cases are essential for software development and maintenance. In practice, developers derive multiple test cases from an implicit pattern based on their understanding of requirements and inferenc...

- **GreenScheduler: Coordinated Two-Tier Energy Optimization for Disaggregated LLM Serving** — Waled Milad Abulgasem Alashheb, Mabruka Khlifa Ali Karkeb, Sabria AbdulGader Ali Elmusrati, Sumia Abdussalam Milad Elagtel
  > Large Language Model (LLM) inference has become a dominant consumer of en- ergy in modern AI data centers, often accounting for over 90% of total operational power [1].Recent architectural shifts towa...

- **HACO: Hedged Agent Computing for Reliable LLM Systems** — Enhan Li, Hongyang Du
  [arXiv](https://arxiv.org/abs/2607.19215v1)
  > As large language model (LLM) agents move from isolated prompting to longhorizon workflows, failures increasingly arise at the role-to-instance binding boundary, where task-specific role requests must...

- **HIPPO: Accelerating Video Large Language Models Inference via Holistic-aware Parallel Speculative Decoding** — Qitan Lv, Tianyu Liu, Wen Wu, Xuenan Xu, Bowen Zhou, Feng Wu, Chao Zhang
  [arXiv](https://arxiv.org/abs/2601.08273)
  > Speculative decoding (SD) has emerged as a promising approach to accelerate LLM inference without sacrificing output quality. Existing SD methods tailored for video-LLMs primarily focus on pruning red...

- **HadAgent: Harness-Aware Decentralized Agentic AI Serving with Proof-of-Inference Blockchain Consensus** — Landy Jimenez, Mariah Weatherspoon, Bingyu Shen, Yi Sheng, Jianming Liu, Boyang Li
  [arXiv](https://arxiv.org/abs/2604.15276)
  > Proof-of-Work blockchain consensus consumes vast computational resources without producing useful output, while the rapid growth of large language model (LLM) agents has created unprecedented demand f...

- **Hardware-Software Co-design for 3D-DRAM-based LLM Serving Accelerator** — Cong Li, Yihan Yin, Chenhao Xue, Zhao Wang, Fujun Bai, Yixin Guo, Xiping Jiang, Qiang Wu, Yuan Xie, Guangyu Sun
  [arXiv](https://arxiv.org/abs/2603.04797)
  > Large language models (LLMs) have been widely deployed for online generative services, where numerous LLM instances jointly handle workloads with fluctuating request arrival rates and variable request...

- **Harness Engineering for LLM-Driven GPU Kernel Generation** — Yue Shui, Chenyu Ma, Hangfei Xu, Shengzhao Wen, Yanpeng Wang
  [arXiv](https://arxiv.org/abs/2607.17979v1)
  > Large language models (LLMs) can assist GPU kernel generation, but their practical effectiveness depends on whether generated code can be reliably constrained, validated, profiled, and selected. This ...

- **HarnessLLM: Rust Verification Harness Generation with Large Language Models** — Minghua Wang, Yuwei Liu, Lin Huang
  [arXiv](https://arxiv.org/abs/2607.22161v1)
  > Rust's ownership model and type system offer strong memory safety guarantees, but unsafe code and runtime panics still present significant risks. Formal verification is essential to ensure memory safe...

- **HiKV: Hierarchical Importance-Aware KV Cache with Hardware Acceleration for LLM Decoding** — Chao Fang, Jun Yin, Man Shi, Marian Verhelst
  [arXiv](https://arxiv.org/abs/2607.22389v1)
  > With the rapid adoption of long-context large language models (LLMs), the continuously growing KV cache during decoding has become the critical memory bottleneck. To tackle this challenge, we propose ...

- **HiTMS: A High-Throughput Multi-Stream Linguistic Steganography Framework** — Ruiyi Yan, Yugo Murawaki, Zhongliang Yang
  [arXiv](https://arxiv.org/abs/2607.23597v1)
  > Generative linguistic steganography conceals secret bits within the sampling randomness of large language models. Existing schemes are single-stream, conveying an entire secret through a single respon...

- **HijackKV: New Threat in Position-Independent KV Cache Reuse** — Yichi Zhang, Zhiqi Wang, Huan Zhang, Yuchen Yang
  [arXiv](https://arxiv.org/abs/2607.19957v1)
  > Key-Value (KV) cache reduces inference latency in large language models (LLMs). Traditional prefix-based reuse has low cache hit rates across inference requests because it requires exact token and pos...

- **HiveMind: OS-Inspired Scheduling for Concurrent LLM Agent Workloads** — Justice Owusu Agyemang, Jerry John Kponyo, Obed Kwasi Somuah, Elliot Amponsah, Godfred Manu Addo Boakye, Kwame Opuni-Boachie Obour Agyekum
  [arXiv](https://arxiv.org/abs/2604.16790)
  > When multiple LLM coding agents share a rate-limited API endpoint, they exhibit resource contention patterns analogous to unscheduled OS processes competing for CPU, memory, and I/O. HiveMind proposes...

- **How Does Alignment Tuning Shape Representations of Sycophancy and Related Cue-Induced Biases in LLMs?** — Prakhar Gupta, Terry Jingchen Zhang, Florent Draye, Bernhard Schölkopf, Zhijing Jin
  [arXiv](https://arxiv.org/abs/2607.18114v1)
  > Modern LLMs are alarmingly susceptible to surprisingly simple immaterial changes of input prompts: a casual hint, an incorrectly labeled few-shot example, or a fake prior assistant turn often flips an...

- **HyMCache: A KV Cache Framework for Multi-Turn LLM Serving with CXL-Hybrid Memory** — Hakbeom Jang, Inho Song, Sam H. Noh, Jongryool Kim
  [arXiv](https://arxiv.org/abs/2607.18141v1)
  > Long-context, multi-turn, and agentic LLM workloads increasingly reuse previously processed context, making KV-cache reuse essential for reducing redundant computation. However, this reuse shifts the ...

- **IEMAS: An Incentive-Efficiency Routing Framework for Open Agentic Web Ecosystems** — ['Hongze Liu', 'Chang Guo', 'Yingzeng Li', 'Mengru Wang', 'Jiong Lou', 'Shijing Yuan']
  [arXiv](https://arxiv.org/abs/2603.17302)
  > The transition to open, distributed Multi-Agent Systems (MAS) promises scalable intelligence but introduces a non-trivial tension: maximizing global efficiency requires cooperative, resource-aware sch...

- **ITQ3_S: High-Fidelity 3-bit LLM Inference via Interleaved Ternary Quantization with Rotation-Domain Smoothing** — ['Edward J. Yoon']
  [arXiv](https://arxiv.org/abs/2603.27914)
  > We present ITQ3_S (Interleaved Ternary Quantization -- Specialized), a novel 3-bit weight quantization format for LLMs integrating TurboQuant (TQ), a rotation-domain strategy based on the Fast Walsh-H...

- **IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs** — Yuzhen Mao et al.
  [arXiv](https://arxiv.org/abs/2604.10539)
  > Key-Value (KV) cache plays a crucial role in accelerating inference in large language models (LLMs) by storing intermediate attention states and avoiding redundant computation during autoregressive ge...

- **InnerQ: Hardware-aware Tuning-free Quantization of KV Cache for Large Language Models** — Sayed Mohammadreza Tayaranian Hosseini, Amir Ardakani, W. Gross
  [arXiv](https://arxiv.org/abs/2602.23200)
  > Reducing the hardware footprint of large language models (LLMs) during decoding is critical for efficient long-sequence generation. A key bottleneck is the key-value (KV) cache, whose size scales with...

- **InstantInfer: Enabling Fast LLM Cold Start with Communicating Finite Automata** — Yitao Yuan, Yongchao He, Shaoke Fang, Wenfei Wu
  [arXiv](https://arxiv.org/abs/2607.18957v1)
  > Cold starts in large language model (LLM) inference services significantly affect user experience, yet they remain inefficient due to sequential initialization and a massive number of fine-grained I/O...

- **Intelligent Multi-UAV Navigation in ITNTNs: A Hierarchical LLM Approach** — Zijiang Yan, Hao Zhou, Wael Jaafar, Jianhua Pei, Ping Wang, Halim Yanikomeroglu, Hina Tabassum
  [arXiv](https://arxiv.org/abs/2607.18604v1)
  > The deployment of high-speed Uncrewed Aerial Vehicles (UAVs) in 3D aerial highways necessitates robust coordination of physical flight kinematics and multi-tier network handovers. While Deep Reinforce...

- **IoUPD: IoU-Aware Privileged Distillation for Visual Grounding with Multimodal Large Language Models** — Xiuyuan Zhu, Ke Lu, Hao Wu, Zijin Du, Dongming Zhang, Jian Xue
  [arXiv](https://arxiv.org/abs/2607.15732v1)
  > Visual grounding with multimodal large language models is commonly formulated as autoregressive coordinate generation, where a model outputs bounding-box coordinates as text given an image and a refer...

- **KAP: Bridging the Knowledge Selection-Runtime Consumption Gap in LLM Systems** — Shuo Wang, Fang Xi, Wenyuan Huang, Qing Wang, Junming Su
  [arXiv](https://arxiv.org/abs/2607.24260v1)
  > Modern LLM systems increasingly rely on knowledge-selection processes that produce high-value structured priors, such as ranked evidence, graph topology, multimodal alignment, and confidence signals. ...

- **Kalypso: Relational LLM Serving** — Hojae Son, Md Ashraful Islam, Huy Gia Cao, Hui Guan, Marco Serafini
  [arXiv](https://arxiv.org/abs/2607.23815v1)
  > Large language models are increasingly used as semantic operators for filtering, extracting, ranking, joining, and transforming unstructured data. Existing semantic query processing systems invoke req...

- **Keeping the Cache Warm Pays: Keepalive Economics for Agentic Workloads** — Maxim Khailo
  [arXiv](https://arxiv.org/abs/2607.19214v1)
  > Frontier LLM providers cache a prompt's processed prefix so that a follow-up request sharing it pays ~10% of the input price and skips most of the prefill latency. Agentic workloads systematically des...

- **Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones** — Ayoub Ghriss, Sourav Chakraborty
  [arXiv](https://arxiv.org/abs/2607.17419v1)
  > Linear attention promises constant-time recurrent inference but degrades sharply on associative recall. We formulate attention recall as a spherical-packing problem and introduce Kernelized Linear Att...

- **LAPS: A Length-Aware-Prefill LLM Serving System** — Jianshu She, Zonghang Li, Hongchao Du, Shangyuan Wu, Wenhao Zheng, Eric P. Xing, Zhengzhong Liu, Huaxiu Yao, Jason Xue, Qirong Ho
  [arXiv](https://arxiv.org/abs/2601.11589)
  > LAPS identifies and disaggregates requests with different prompt lengths in LLM serving to reduce TTFT latency. While recent systems have decoupled the prefill and decode stages to improve throughput,...

- **LLM Inference at the Edge: Mobile, NPU, and GPU Performance Efficiency Trade-offs Under Sustained Load** — Pranay Tummalapalli et al.
  [arXiv](https://arxiv.org/abs/2603.23640)
  > Deploying LLMs on-device for always-on personal agents demands sustained inference from hardware tightly constrained in power, thermal envelope, and memory. We benchmark Qwen 2.5 1.5B (4-bit quantised...

- **LLM-42: Enabling Determinism in LLM Inference with Verified Speculation** — Raja Gond, Aditya K Kamath, Arkaprava Basu, R. Ramjee, Ashish Panwar
  [arXiv](https://arxiv.org/abs/2601.17768)
  > In LLM inference, the same prompt may yield different outputs across different runs. At the system level, this non-determinism arises from floating-point non-associativity combined with dynamic batchi...

- **LLM-CoOpt: A Co-Design and Optimization Framework for Efficient LLM Inference on Heterogeneous Platforms** — Jie Kong, Wei Wang, Jiehan Zhou, Chen Yu
  [arXiv](https://arxiv.org/abs/2602.09323)
  > Major challenges in LLMs inference remain frequent memory bandwidth bottlenecks, computational redundancy, and inefficiencies in long-sequence processing. To address these issues, we propose LLM-CoOpt...

- **LLM-Driven AutoML for Cross-Lingual Handwritten OCR: Closed-Loop Neural Architecture Search with GPT-5, GPT-4o, and Claude Sonnet 4** — Mobina Kashaniyan, Amirhossein Ghassemi, Nasser Mozayani
  [arXiv](https://arxiv.org/abs/2607.15509v1)
  > We present a fully automated closed-loop AutoML framework that uses GPT-5, GPT-4o, and Claude Sonnet 4 as autonomous neural architecture designers for cross-lingual handwritten optical character recog...

- **LLM-based Source Code Compression via Thresholded Symbol Ranking** — Angelo Nardone, Paolo Ferragina
  [arXiv](https://arxiv.org/abs/2607.24192v1)
  > We study the problem of lossless compression of source code, motivated by the storage demands of large-scale software archives, such as Software Heritage (https://www.softwareheritage.org/). General-p...

- **LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure** — Jaehong Cho, Hyunmin Choi, Guseul Heo, Jongse Park
  [arXiv](https://arxiv.org/abs/2602.23036)
  > Large language model (LLM) serving infrastructures are undergoing a shift toward heterogeneity and disaggregation. Modern deployments increasingly integrate diverse accelerators and near-memory proces...

- **LLMs and Agentic AI Systems for Smart Grids: A Tutorial on Architectures and Applications** — Daniela Rojas, Abdulwahab Albassam, Aidan G. Leung, Jett Ngo, Ryan Luo, Peter R. Quawas, Junpyung Kim, Kangkai Liang, Mansi Nanavati, Jonathan Mai, Meng-Chi Tsai, Yun-Tong Tsai, Yize Chen, Yuanyuan Shi
  [arXiv](https://arxiv.org/abs/2607.18147v1)
  > Large language models (LLMs) and agentic AI systems have evolved from natural language tasks to using external tools to plan, retrieve, and act in technical domains. In smart grids, recent work applie...

- **LOCKS: Page-Local Compact Key Summaries for Efficient Long-Context Decoding** — Junsung Hwang
  [arXiv](https://arxiv.org/abs/2607.24555v1)
  > Serving large language models at long context is bottlenecked by the key-value (KV) cache, which is read in full at every decode step. Attention keys are locally low-rank though globally high-rank: sh...

- **Large-Scale LLM Inference with Heterogeneous Workloads: Prefill-Decode Contention and Asymptotically Optimal Control** — Ruihan Lin, Zezhen Ding, Zean Han, Jiheng Zhang
  [arXiv](https://arxiv.org/abs/2602.02987)
  > Large Language Models (LLMs) are rapidly becoming critical infrastructure for enterprise applications, driving unprecedented demand for GPU-based inference services. A key operational challenge arises...

- **Latent Denoising Improves Visual Alignment in Large Multimodal Models** — Dhruv Parikh, Jacob Fein-Ashley, Rajgopal Kannan, Viktor Prasanna
  [arXiv](https://arxiv.org/abs/2604.21343v1) | [GitHub](https://github.com/dipampaul17/KVSplit)
  > Large Multimodal Models (LMMs) such as LLaVA are typically trained with an autoregressive language modeling objective, providing only indirect supervision to visual tokens. This often yields weak inte...

- **Learning to Communicate: Toward End-to-End Optimization of Multi-Agent Language Systems** — Ye Yu, Heming Liu, Haibo Jin, Xiaopeng Yuan, Peng Kuang
  [arXiv](https://arxiv.org/abs/2604.21794v1) | [GitHub](https://github.com/skyzh/tiny-llm)
  > Multi-agent systems built on large language models have shown strong performance on complex reasoning tasks, yet most work focuses on agent roles and orchestration while treating inter-agent communica...

- **Lil: Less is Less When Applying Post-Training Sparse-Attention Algorithms in Long-Decode Stage** — Junhao Hu, Fang Li, Mingtao Xu, Fei Meng, Shiju Zhao, Tiancheng Hu, Ting Peng, Anmin Liu, Wenrui Huang, Chenxu Liu, Ziyue Hua, Tao Xie
  [arXiv](https://arxiv.org/abs/2601.03043)
  > Large language models (LLMs) demonstrate strong capabilities across a wide range of complex tasks and are increasingly deployed at scale, placing significant demands on inference efficiency. Prior wor...

- **Local-Splitter: A Measurement Study of Seven Tactics for Reducing Cloud LLM Token Usage on Coding-Agent Workloads** — Justice Owusu Agyemang, Jerry John Kponyo, Elliot Amponsah, Godfred Manu Addo Boakye, Kwame Opuni-Boachie Obour Agyekum
  [arXiv](https://arxiv.org/abs/2604.12301)
  > We present a systematic measurement study of seven tactics for reducing cloud LLM token usage when a small local model can act as a triage layer in front of a frontier cloud model. Local routing combi...

- **Look Less, Think Faster: Joint Token-Compute Adaptation for Multimodal LLMs** — Pengcheng Wang, Zhiquan Wang, Jayoung Lee, Zhuoyan Xu, Ran Xu, Saurabh Bagchi, Yin Li, Somali Chaterji
  [arXiv](https://arxiv.org/abs/2607.20357v1)
  > Multimodal Large Language Models (MLLMs) have recently demonstrated strong performance across vision-language tasks. However, their high inference cost, arising from both the large number of input vis...

- **Lossless but Not Free: An Empirical Anatomy of Speculative Decoding on Consumer Hardware** — Param Chordiya
  [arXiv](https://arxiv.org/abs/2607.17283v1)
  > Single-stream autoregressive decoding of large language models is bound by memory bandwidth: each generated token requires one full forward pass through the target model, and successive passes cannot ...

- **Low-Latency Edge LLM Handover via Joint KV Cache Transfer and Token Prefill** — Seunghun Lee, Jihong Park, Ce Zheng, Hyuncheol Park
  [arXiv](https://arxiv.org/abs/2603.28018)
  > Edge deployment of large language models (LLMs) can reduce latency for interactive services, but mobility introduces service interruptions when an user equipment (UE) hands over between base stations ...

- **MAC-Attention: a Match-Amend-Complete Scheme for Fast and Accurate Attention Computation** — Jinghan Yao, Sam Adé Jacobs, Walid Krichene, Masahiro Tanaka, Dhabaleswar K Panda
  [arXiv](https://arxiv.org/abs/2604.00235) | [GitHub](https://github.com/YJHMITWEB/MAC-Attention)
  > Long-context decoding in LLMs is IO-bound: each token re-reads an ever-growing KV cache. Prior accelerations cut bytes via compression (lower fidelity) or selection/eviction (restricting accessibility...

- **MADA-RL: Multi-Agent Debate-Aware Reinforcement Learning for Parameter-Efficient Reasoning in Compact Models** — Martino M. L. Pulici, Cuong Xuan Chu, Evgeny Kharlamov, Zifeng Ding, Volker Tresp, Yunpu Ma
  [arXiv](https://arxiv.org/abs/2607.18006v1)
  > Large language models achieve strong reasoning performance, but often at prohibitive training cost - a challenge that is especially acute for compact models ($\leq 4 \, \mathrm{B}$ parameters) trained...

- **MARS: Unleashing the Power of Speculative Decoding via Margin-Aware Verification** — Jingwei Song, Xinyu Wang, Hanbin Wang, Xiaoxuan Lei, Bill Shi, Shixin Han, Eric Yang, Xiao-Wen Chang, Lynn Ai
  [arXiv](https://arxiv.org/abs/2601.15498) | [GitHub](https://github.com/5SSjw/MARS)
  > Speculative Decoding (SD) accelerates autoregressive large language model (LLM) inference by decoupling generation and verification. While recent methods improve draft quality by tightly coupling the ...

- **MAViE: A Multi-scale Adaptive Vision Encoder for Fine-grained Visual Perception and Efficient Multimodal Reasoning** — Shaofei Lei
  [arXiv](https://arxiv.org/abs/2607.24424v1)
  > Vision-language models commonly project all tokens produced by a pretrained vision encoder into a large language model. However, final-layer features can discard text, local attributes, and spatial re...

- **MSAO: Adaptive Modality Sparsity-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference** — Zheming Yang et al.
  [arXiv](https://arxiv.org/abs/2604.02945)
  > Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities but impose substantial computational and latency burdens, posing critical challenges for deployment on resou...

- **MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference** — Joris Köster, Zixuan Liu, Siavash Khajavi, Zizhan Zheng
  [arXiv](https://arxiv.org/abs/2603.26557)
  > Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessio...

- **MemExplorer: Navigating the Heterogeneous Memory Design Space for Agentic Inference NPUs** — Haoran Wu, Zeyu Cao, Yao Lai, Binglei Lou, Jiayi Nie, Can Xiao, T. Adeniran, Przemyslaw Forys, Kauser Johar, Catriona R Wright, Junyi Liu, Kai Shi, Nicholas D. Lane, R. Antonova, Jianyi Cheng, Timothy Jones, Aaron Zhao, Robert Mullins
  [arXiv](https://arxiv.org/abs/2604.16007)
  > Emerging agentic LLM workloads are driving rapidly growing demand on both memory capacity and bandwidth, with different phases of inference (e.g., prefill and decode) imposing distinct requirements. I...

- **MemVLN: Episodic and Procedural Memory for Vision-and-Language Navigation** — Yuqi Liu, Shengju Qian, Tianyuan Qu, Mingxian Lin, Zixuan Wang, Xin Wang, Bei Yu, Jiaya Jia
  [arXiv](https://arxiv.org/abs/2607.23504v1)
  > Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires agents to maintain long-horizon visual history for trajectory consistency while executing actions with low latency. Existing...

- **MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios** — Shuhuai Li, Jianghao Lin, Dongdong Ge, Yinyu Ye
  [arXiv](https://arxiv.org/abs/2603.09983)
  > Mixture-of-Experts (MoE) models enable scalable performance but face severe memory constraints on edge devices. Existing offloading strategies struggle with I/O bottlenecks due to the dynamic, low-inf...

- **Modularized Dynamic-Granularity Video LLM for Multi-Event Long Video Understanding** — Wei Feng, Xin Wang, Yu-Wei Zhan, Yuwei Zhou, Wenwu Zhu
  [arXiv](https://arxiv.org/abs/2607.15778v1)
  > Video Large Language Models (Video LLMs) have made significant advancements in various video understanding tasks. However, long-video scenarios remain challenging due to the tension between limited vi...

- **Multi-Layer Scheduling for MoE-Based LLM Reasoning** — Yifan Sun, Gholamreza Haffari, Minxian Xu, Rajkumar Buyya, Adel N. Toosi
  [arXiv](https://arxiv.org/abs/2602.21626)
  > Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks, but serving them efficiently at scale remains a critical challenge due to their substantial computational an...

- **Multi-stage Flow Scheduling for LLM Serving** — Yijun Sun, Xudong Liao, Songrun Xie, Hao Chen, Han Tian, Wenxue Li 等
  [arXiv](https://arxiv.org/abs/2603.17456)
  > Meeting stringent Time-To-First-Token (TTFT) requirements is crucial for LLM applications. To improve efficiency, modern LLM serving systems adopt disaggregated architectures with diverse parallelisms...

- **NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL** — Amos Goldman et al. (NVIDIA Corporation)
  [arXiv](https://arxiv.org/abs/2603.13606)
  > NCCL EP provides unified ncclEpDispatch and ncclEpCombine primitives, supporting Low-Latency (LL) mode for inference decoding and High-Throughput (HT) mode for training and inference prefill. LL targe...

- **Native LLM and MLLM Inference at Scale on Apple Silicon** — Wayner Barrios
  [arXiv](https://arxiv.org/abs/2601.19139) | [GitHub](https://github.com/https://github.com/wbarrios/vllm-mlx)
  > The growing adoption of Apple Silicon for machine learning development has created demand for efficient inference solutions that leverage its unique unified memory architecture. However, existing tool...

- **OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration** — Youhe Jiang, Fangcheng Fu, Taiyi Wang, Guoliang He, Eiko Yoneki
  [arXiv](https://arxiv.org/abs/2602.12151)
  > Serving Large Language Models (LLMs) can benefit immensely from parallelizing both the model and input requests across multiple devices, but incoming workloads exhibit substantial spatial and temporal...

- **Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.16957)
  > We present Open-TQ-Metal, a fused compressed-domain attention kernel for long-context LLM inference on Apple Silicon Metal GPU, enabling efficient attention computation in the compressed domain....

- **Oracle Gap and Signal Fidelity: A Fixed-Pool Diagnostic for Test-Time Collaboration** — Jie Hu
  [arXiv](https://arxiv.org/abs/2607.17531v1)
  > Test-time collaboration, including self-consistency, best-of-N selection, critic models, and verifier pipelines, is often credited with broadly improving LLM reasoning, yet its gains are uneven and so...

- **Orla: A Library for Serving LLM-Based Multi-Agent Systems** — Rana Shahout, Hayder Tirmazi, Minlan Yu, Michael Mitzenmacher
  [arXiv](https://arxiv.org/abs/2603.13605)
  > We introduce Orla, a library for constructing and running LLM-based agentic systems. Modern agentic applications consist of workflows that combine multiple LLM inference steps, tool calls, and heterog...

- **PAM: Processing Across Memory Hierarchy for Efficient KV-centric LLM Serving System** — Lian Liu, Shixin Zhao, Yutian Zhou, Yutian Zhou, Yintao He, Mengdi Wang, Yinhe Han, Ying Wang
  [arXiv](https://arxiv.org/abs/2602.11521)
  > The widespread adoption of Large Language Models (LLMs) has exponentially increased the demand for efficient serving systems. With growing requests and context lengths, key-value (KV)-related operatio...

- **PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models** — Eunyeong Cho, Jehyeon Bang, Ranggi Hwang, Minsoo Rhu
  [arXiv](https://arxiv.org/abs/2602.11530)
  > The emergence of reasoning-based LLMs leveraging Chain-of-Thought (CoT) inference introduces new serving challenges, as their extended reasoning phases delay user-visible output and inflate Time-To-Fi...

- **PATS: Policy-Aware Training Scaffolding for Agentic Reinforcement Learning** — Yipeng Shi, Zhipeng Ma, Yue Wang, Qitai Tan, Yang Li, Peng Chen, Zhengzhou Zhu
  [arXiv](https://arxiv.org/abs/2607.21419v1)
  > In long-horizon LLM agent reinforcement learning, weak policies often repeat similar failures, producing uninformative rollout trajectories and limiting effective policy optimization. Existing skill-c...

- **PEARL: Auditable Repair for Scientific Reasoning Graph Extraction** — Bohan Su, Pengze Li, Yuchen Lu, Xi Chen
  [arXiv](https://arxiv.org/abs/2607.17917v1)
  > Scientific Reasoning Graph Extraction (SRGE) aims to recover explicit links among observations, evidence, intermediate claims, and paper-level conclusions. LLMs can produce graph-like scientific expla...

- **PIM-SHERPA: Software Method for On-device LLM Inference by Resolving PIM Memory Attribute and Layout Inconsistencies** — Sunjung Lee, Sanghoon Cha, Hyeonsu Kim, Seung-Yeon Seo, Yuhwan Ro, Sukhan Lee, Byeongho Kim, Yongjun Park, Kyomin Sohn, Seungwon Lee, Jaehoon Yu
  [arXiv](https://arxiv.org/abs/2603.09216)
  > On-device deployments of large language models (LLMs) are rapidly proliferating across mobile and edge platforms. LLM inference comprises a compute-intensive prefill phase and a memory bandwidth-inten...

- **POLAR: Online Learning for LoRA Adapter Caching and Routing in Edge LLM Serving** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.16583)
  > We propose POLAR, an online learning framework for LoRA adapter caching and routing in edge LLM serving systems, dynamically managing adapter placement for optimal performance....

- **POP: Prefill-Only Pruning for Efficient Large Model Inference** — Junhui He, Zhihui Fu, Jun Wang, Qing'an Li
  [arXiv](https://arxiv.org/abs/2602.03295)
  > Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated remarkable capabilities. However, their deployment is hindered by significant computational costs. Existing structured ...

- **PRISM: Parametrically Refactoring Inference for Speculative Sampling Draft Models** — Xuliang Wang, Yuetao Chen, Maochan Zhen, Fang Liu, Xin Zheng, Xing Liu, Hong Xu, Ming Li
  [arXiv](https://arxiv.org/abs/2602.01762)
  > Large Language Models (LLMs), constrained by their auto-regressive nature, suffer from slow decoding. Speculative decoding methods have emerged as a promising solution to accelerate LLM decoding, attr...

- **PackInfer: Compute- and I/O-Efficient Attention for Batched LLM Inference** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2602.06072)
  > Attention efficiency is critical to large language model (LLM) inference. While prior advances optimize attention execution for individual requests (e.g., FlashAttention), production LLM serving relie...

- **PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization** — Yuchen Yang, Yifan Zhao, Anisha Dasgupta, Sasa Misailovic
  [arXiv](https://arxiv.org/abs/2607.16184v1)
  > Mixture-of-Experts (MoE) is a popular class of large language models (LLMs), offering high efficiency and accuracy. However, in KV-cache-intensive serving scenarios, MoEs often exhibit a tension betwe...

- **Pancake: Hierarchical Memory System for Multi-Agent LLM Serving** — Zhengding Hu, Zaifeng Pan, Prabhleen Kaur, Vibha Murthy, Zhongkai Yu, Yue Guan 等
  [arXiv](https://arxiv.org/abs/2602.21477)
  > In this work, we identify and address the core challenges of agentic memory management in LLM serving, where large-scale storage, frequent updates, and multiple coexisting agents jointly introduce com...

- **PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving** — Xu Bai
  [arXiv](https://arxiv.org/abs/2604.12171)
  > Pipeline parallelism (PP) is widely used to partition LLM layers across GPUs. However, existing systems rely on static PP configurations that fail to adapt to dynamic settings. PipeLive enables live i...

- **PoTRE: Test-Time Reasoning inspired by Cognitive Heterogeneity** — Anmol Kankariya, Sercan Ö. Arık
  [arXiv](https://arxiv.org/abs/2607.20268v1)
  > While Large Language Models (LLMs) excel at many tasks, they frequently struggle with complex reasoning that requires long-horizon planning and iterative error correction. Furthermore, standard single...

- **PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference** — Hyunwoo Oh, Suyeon Jang, Hanning Chen, KyungIn Nam, Sanggeon Yun, Ryozo Masukawa, Mohsen Imani
  [arXiv](https://arxiv.org/abs/2607.14618v1)
  > CPUs are the most universal target for on-device LLM inference, but existing low-bit quantization methods offer either coarse operating points or fine-grained mixed precision that is difficult to exec...

- **Power Aware Dynamic Reallocation For Inference** — Yiwei Jiang, Sangeeta Chowdhary, Nathaniel Morris, Rutwik Jain, Srilatha Manne, Samuel Bayliss
  [arXiv](https://arxiv.org/abs/2601.12241)
  > Disaggregation has emerged as a powerful strategy for optimizing large language model (LLM) inference by separating compute-intensive prefill and memory-bound decode phases across specialized GPUs. Th...

- **PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving** — Sunghyeon Woo, Hoseung Kim, Sunghwan Shim, Minjung Jo, Hyunjoon Jeong, Jeongtae Lee 等
  [arXiv](https://arxiv.org/abs/2602.12029)
  > Multi-agent systems increasingly orchestrate multiple specialized language models to solve complex real-world problems, often invoking them over a shared context. This execution pattern repeatedly pro...

- **Probabilistic Language Tries: A Unified Framework for Compression, Decision Policies, and Execution Reuse** — ['Gregory Magarshak']
  [arXiv](https://arxiv.org/abs/2604.06228)
  > We introduce probabilistic language tries (PLTs), a unified representation that makes explicit the prefix structure implicitly defined by any generative model over sequences. By assigning to each outg...

- **PyroDash: Cost-Efficient Token-Level Small-Large Language Model Collaborative Inference** — Niqi Lyu, Pengtao Shi, Wei Qiu, Jianlin Zhong, Sicong Xia, Jianyao Ma, Yicheng Ding
  [arXiv](https://arxiv.org/abs/2607.20327v1)
  > Large language models (LLMs) provide strong reasoning capabilities but are expensive to serve at scale, whereas small language models (SLMs) are cheaper but less reliable on difficult problems. We int...

- **QCFuse: Query-Centric Cache Fusion for Efficient RAG Inference** — ['Jianxin Yan', 'Zeheng Qian', 'Wangze Ni', 'Zhitao Shen', 'Zhiping Wang', 'Haoyang Li']
  [arXiv](https://arxiv.org/abs/2604.08585)
  > Cache fusion accelerates generation process of LLMs equipped with RAG through KV caching and selective token recomputation, thereby reducing computational costs and improving efficiency. However, exis...

- **QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE via QUantization-error Alignment across Dual Sides** — Zhengyang Zhuge, Hao Yu, Xin Wang, Zheng Li, Yizhong Cao, Dayiheng Liu, Jianwei Zhang
  [arXiv](https://arxiv.org/abs/2607.15810v1)
  > Rollout generation is a major bottleneck in Reinforcement Learning (RL) for Mixture-of-Experts (MoE) Large Language Models, motivating low-precision rollout acceleration such as FP8. As an emerging lo...

- **RAP: KV-Cache Compression via RoPE-Aligned Pruning** — Jihao Xin, Tian Lyu, David E. Keyes, H. Ltaief, Marco Canini
  [arXiv](https://arxiv.org/abs/2602.02599)
  > Long-context inference in large language models is increasingly bottlenecked by the memory and compute cost of the KV-Cache. Low-rank factorization compresses KV projections by writing $W \approx A * ...

- **RAPID-Serve: Resource-efficient and Accelerated P/D Intra-GPU Disaggregation** — Amna Masood, Pratishtha Gaur, N. Jayasena
  [arXiv](https://arxiv.org/abs/2601.11822)
  > Two widely adopted techniques for LLM inference serving systems today are hybrid batching and disaggregated serving. A hybrid batch combines prefill and decode tokens of different requests in the same...

- **RED-PIM: Reducing Data Movement for Transformers using Processing-in-Memory** — Zahra Yousefijamarani, Alaa Alameldeen
  [arXiv](https://arxiv.org/abs/2607.21731v1)
  > Transformers are widely used across many domains, including natural language processing, computer vision, web search, and DNA sequence analysis. Given their broad applicability, improving the performa...

- **RIS-Kernel: A Model-Agnostic Architecture for Long-Context LLM Inference via Sparse Attention** — Anderson R. Santos
  [arXiv](https://arxiv.org/abs/2607.21927v1)
  > Full self-attention in large language models scales as O(N^2), which limits long-context document analysis to 65,536 tokens and requires costly GPU clusters. The Reduced Interaction Sampling (RIS) inf...

- **Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU** — N/A
  [arXiv](https://arxiv.org/abs/2604.15464)
  > Large Language Model (LLM) deployment is increasingly shifting to cost-efficient accelerators like Google's Tensor Processing Units (TPUs), prioritizing both performance and total cost of ownership (T...

- **Rarity-Aware Discrete Diffusion with Spatially Consistent Decoding for Photo-Realistic Image Super-Resolution** — Ao Li, Yapeng Du, Yi Xin, Lei Zhu, Le Zhang, Guangtao Zhai, Ce Zhu, Xiaohong Liu
  [arXiv](https://arxiv.org/abs/2607.17612v1)
  > Continuous diffusion models have become the dominant paradigm for photo-realistic image Super-Resolution (SR), but they typically formulate reconstruction as continuous signal-level denoising and inco...

- **Rationale-Guided Knowledge Distillation for Cross-Lingual Stance Detection** — Qiuli Zhou, Jingyuan Yao, Shengeng Tang, Hongzhi Chen, Jun Tang, Richang Hong
  [arXiv](https://arxiv.org/abs/2607.18693v1)
  > Stance detection aims to identify whether a text expresses a favorable or opposing attitude toward a given target, and serves as an important task for various downstream applications. Although existin...

- **RecGPT-V3 Technical Report** — Bowen Zheng, Chao Yi, Dian Chen, Gaoyang Guo, Han Zhu, Jiakai Tang, Jian Wu, Mao Zhang, Wen Chen, Yifan Lu, Yujie Luo, Yuning Jiang, Zhujin Gao, Bo Zheng, Dixuan Wang, Hao Fang, Jiancai Liu, Jing Yu, Ke Chen, Kewei Zhu, Mingke Xu, Wenjun Yang, Xunke Xi, Zile Zhou
  [arXiv](https://arxiv.org/abs/2607.15591v1)
  > Large language models (LLMs) are transforming recommender systems from matching co-occurrence patterns in historical behavior toward reasoning about the intent that drives it. RecGPT-V1 pioneered this...

- **RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators** — Xinsheng Tang, Yuhui Zhao, Jintao Li, Jiaming Xu, Shuo Li, Jiansong Chen, Chen Zhang, Yong Li, Xiaoyong Liu, Ji Liu, Jin Wang, Wei Lin
  [arXiv](https://arxiv.org/abs/2603.10026)
  > Operator fusion, as a key performance optimization technique in the deployment of AI models, significantly improves execution efficiency and has been widely adopted in modern AI compilers. However, fo...

- **Refusal-Gated Decoding: Preserving Refusal Behavior Under High-Temperature Sampling** — Phillip Howard, Xin Su, Allen Roush, Manikandan Ravikiran, Amir Abdullah
  [arXiv](https://arxiv.org/abs/2607.20791v1)
  > High-temperature sampling is one of the primary mechanisms for increasing diversity in LLMs. Recent advances in truncation-based sampling techniques have helped mitigate drawbacks of high-temperature ...

- **Reinforcement Learning for Large Language Model Selective Evidence Adoption from Contaminated Retrieval Results** — Yanyu Chen, Yue Li, Yongyi Cui, Dongsheng Shi, Lichang Dai
  [arXiv](https://arxiv.org/abs/2607.20090v1)
  > Retrieval-augmented large language models frequently face contexts that interleave useful evidence with misleading statements or instruction-like content. Blanket refusal discards valid evidence, wher...

- **Resource Multiplexing in Tuning and Serving Large Language Models** — Yongjun He, Hao Yang, Yao Lu, Ana Klimovic, Gustavo Alonso
  [GitHub](https://github.com/aerlabsAI/ai-inference-resources)

- **Rethinking Latency Denial-of-Service: Attacking the LLM Serving Framework, Not the Model** — Tianyi Wang, Huawei Fan, Yuanchao Shu, Peng Cheng, Cong Wang
  [arXiv](https://arxiv.org/abs/2602.07878)
  > Large Language Models face an emerging and critical threat known as latency attacks. Because LLM inference is inherently expensive, even modest slowdowns can translate into substantial operating costs...

- **ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments** — Haley Li, Xinglu Wang, Cong Feng, Chunxu Zuo, Yanan Wang, Hei Lo, Yufei Cui, Bingji Wang, Duo Cui, Shuming Jing, Yizhou Shan, Ying Xiong, Jiannan Wang, Yong Zhang, Zhenan Fan
  [arXiv](https://arxiv.org/abs/2602.21140)
  > As LLM deployments scale over more hardware, the probability of a single failure increases significantly. A common recovery approach is to restart the LLM serving instance; however, this is costly in ...

- **Robust Interpretation of Historical Documents in Knowledge Graphs Through Query Inference and Execution** — Sebastià Nicolau, Adrià Molina, Oriol Ramos Terrades, Josep Lladós
  [arXiv](https://arxiv.org/abs/2607.24475v1)
  > The emergence of Large Language Models (LLMs) has redefined how users interact with information in digital environments. However, their widespread and often indiscriminate integration has raised signi...

- **Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions** — Jing Wang, Yu-Yang Qian, Ke Xue, Chao Qian, Peng Zhao, Zhi-Hua Zhou
  [arXiv](https://arxiv.org/abs/2604.07931)
  > Output-length prediction is important for efficient LLM serving, as it directly affects batching, memory reservation, and scheduling. Most existing methods use a one-shot sampled length as the label, ...

- **Rocks, Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference** — Konstantinos Papaioannou, Thaleia Dimitra Doudali, et al.
  [arXiv](https://arxiv.org/abs/2603.26498)
  > Multimodal Large Language Models (MLLMs) power platforms like ChatGPT, Gemini, and Copilot, enabling richer interactions with text, images, and videos. These heterogeneous workloads introduce addition...

- **RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving** — Hossein Hosseini Kasnavieh, Christopher Leckie, Adel N. Toosi
  [arXiv](https://arxiv.org/abs/2604.10907)
  > Multi-model LLM routing has emerged as an effective approach for reducing serving cost and latency while maintaining output quality. However, prior routing methods typically assume each model has fixe...

- **S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance** — Di Liu, Yifei Liu, Chen Chen, Zhibin Yu, Xiaoyi Fan, Quan Chen 等
  [arXiv](https://arxiv.org/abs/2603.10353)
  > With the increasing volumes of Large Language Models (LLMs) and the expanding context lengths, attention computation has become a key performance bottleneck in LLM serving. For fast attention computat...

- **SALT: Salience-Aware Lexical Trie for Long-Context Compression** — Oteo Mamo, Hyunjin Yi, Joydhriti Choudhury, Shangqian Gao, Weikuan Yu
  [arXiv](https://arxiv.org/abs/2607.17486v1)
  > As large language models (LLMs) process increasingly longer prompts, computation and KV-cache memory costs have emerged as major bottlenecks in inference systems. Existing input-level prompt compressi...

- **SHIELD: A Segmented Hierarchical Memory Architecture for Energy-Efficient LLM Inference on Edge NPUs** — ['Jintao Zhang', 'Xuanyao Fong']
  [arXiv](https://arxiv.org/abs/2604.07396)
  > Large Language Model (LLM) inference on edge Neural Processing Units (NPUs) is fundamentally constrained by limited on-chip memory capacity. Although high-density embedded DRAM (eDRAM) is attractive f...

- **SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference** — Luchang Li, Dongfang Li, Bozhao Gong, Yu Zhang
  [arXiv](https://arxiv.org/abs/2603.04716)
  > Prefill-Decode (P/D) disaggregation has emerged as a widely adopted optimization strategy for Large Language Model (LLM) inference. However, there currently exists no well-established methodology for ...

- **SLO-Guard: Crash-Aware, Budget-Consistent Autotuning for SLO-Constrained LLM Serving** — ['Christian Lysenstøen']
  [arXiv](https://arxiv.org/abs/2604.17627) | [GitHub](https://github.com/Chrislysen/SLO-Guard)
  > Serving large language models under latency service-level objectives (SLOs) is a configuration-heavy systems problem with an unusually failure-prone search space. We present SLO-Guard, a crash-aware a...

- **SMART: When is it Actually Worth Expanding a Speculative Tree?** — Lifu Wang, Pan Zhou
  [arXiv](https://arxiv.org/abs/2604.09731)
  > Tree-based speculative decoding accelerates autoregressive generation by verifying a branching tree of draft tokens in a single target-model forward pass. However, existing methods prioritize maximizi...

- **SMEFT-Pheno-Agent: a natural-language-driven AI agent for machine-learning-assisted Standard Model Effective Field Theory phenomenology** — Yu-Chen Guo, Jie Wang, Ji-Chong Yang
  [arXiv](https://arxiv.org/abs/2607.22331v1)
  > We present SMEFT-Pheno-Agent, a Python workflow guided by a natural-language AI agent to perform machine-learning-assisted Standard Model Effective Field Theory (SMEFT) phenomenology at high-energy co...

- **SMoLPU: 122.1µJ/Token Sparse MoE-Based Speculative Decoding Language Processing Unit with Adaptive-Offload NPU-CIM Core** — Sangwoo Ha, Jingu Lee, Young-Hun Moon, Sunjoo Whang, Wooyoung Jo, Gwangtae Park, Sangjin Kim, Soyeon Um, Junha Ryu, Y. Jo, H.-J. Yoo
  > SMoLPU is an energy-efficient MoE-based speculative decoding LLM processor with an NPU-CIM core. It has 3 features: 1) Token-adaptive expert refinement removes redundant expert activations and schedul...

- **SNIP: An Adaptive Mixed Precision Framework for Subbyte Large Language Model Training** — Yunjie Pan, Yao Fu, Ziheng Qiao, Chengmai Mao, Yining Qi, Tianchen Du, Hongxiang Li, Lanbo Li, Chen Liang, Yong Li, Dilin Wang, Wei Liu
  [arXiv](https://arxiv.org/abs/2602.01410)
  > Training large language models (LLMs) efficiently while preserving model quality poses significant challenges, particularly with subbyte precision supported by state-of-the-art GPUs. Current mixed-pre...

- **SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling** — Zikun Liu, Liang Luo, Qianru Li, Zhengyu Zhang, Wei Ling, Jingyi Shen 等
  [arXiv](https://arxiv.org/abs/2604.12110)
  > Recent advances in recommendation scaling laws have led to foundation models of unprecedented complexity. While these models offer superior performance, their computational demands make real-time serv...

- **SUN: Shared Use of Next-token Prediction for Efficient Multi-LLM Disaggregated Serving** — Sunghyeon Woo, Ahreum Seo, Jaegwang Lee, Jaeeun Kil, Hanbae Seo, Joonghoon Kim 等
  [arXiv](https://arxiv.org/abs/2603.02599)
  > In multi-model LLM serving, decode execution remains inefficient due to model-specific resource partitioning: since cross-model batching is not possible, memory-bound decoding often suffers from sever...

- **SWE-Pruner Pro: The Coder LLM Already Knows What to Prune** — Yuhang Wang, Yuling Shi, Shaoqiu Zhang, Jialiang Liang, Shilin He, Siyu Ye, Yuting Chen, Kai Cai, Xiaodong Gu
  [arXiv](https://arxiv.org/abs/2607.18213v1)
  > Pruning long context for coding agents has been a vital technology for efficient context management. While existing context pruning methods such as SWE-Pruner realize this by attaching a separate code...

- **Scalable LLM Agent Tool Access in the Cloud** — Mingxin Li, Enge Song, Yueshang Zuo, Xiaodong Liu, Rong Wen, Qiang Fu, Gianni Antichi, Jian He, Jing Tie, Zhou Shao, Xiaobo Xue, Xiong Xiao, Luyao Zhong, Shaokai Zhang, Jiangu Zhao, Jianyuan Lu, Shize Zhang, Xiaoqing Sun, Changgang Zheng, Zihao Fan, Haonan Li, Tian Pan, Xiaomin Wu, Yang Song, Xing Li, Biao Lyu, Meng Li, Haipeng Dai, Guihai Chen, Shunmin Zhu
  [arXiv](https://arxiv.org/abs/2607.15593v1)
  > LLM agents increasingly rely on tool calling to act on external systems, and the Model Context Protocol (MCP) has quickly become its de facto interface. Operating MCP at cloud scale, however, becomes ...

- **ScePsy: Serving Agentic Workflows Using Aggregate LLM Pipelines** — Marcel Wagenländer, Otto White, Britannio Jarrett, Pedro Silvestre, Yanda Tao, Guo Li, Huanzhou Zhu, Llúis Vilanova, Peter Pietzuch
  [arXiv](https://arxiv.org/abs/2604.15186)
  > Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving such workflows at a target throughput with low latency is challenging because they c...

- **Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions** — Haoyu Zheng, Yongqiang Zhang, Fangcheng Fu, Xiaokai Zhou, Hao Luo, Hongchao Zhu, Yuanyuan Zhu, Hao Wang, Xiao Yan, Jiawei Jiang
  [arXiv](https://arxiv.org/abs/2604.00499)
  > To schedule LLM inference, the \textit{shortest job first} (SJF) principle is favorable by prioritizing requests with short output lengths to avoid head-of-line (HOL) blocking. Existing methods usuall...

- **Scheduling the Unschedulable: Taming Black-Box LLM Inference at Scale** — Renzhong Yuan, Yijun Zeng, Xiaosong Gao, Linxi Yu, Haochun Liao, Han Wang
  [arXiv](https://arxiv.org/abs/2604.05847)
  > When output token counts can be predicted at submission time, client-side scheduling against a black-box LLM API becomes semi-clairvoyant: decisions condition on coarse token priors even though the pr...

- **Scout Before You Attend: Sketch-and-Walk Sparse Attention for Efficient LLM Inference** — Hoang Anh Le, Sahil Joshi, Zeyu Yang, Zhaozhuo Xu, Anshumali Shrivastava
  [arXiv](https://arxiv.org/abs/2602.07397)
  > Self-attention dominates the computational and memory cost of long-context LLM inference across both prefill and decode phases. To address this challenge, we introduce Sketch&Walk Attention, a trainin...

- **SelectInfer: Selective Neuron Loading and Computation for On-Device LLMs** — Huzaifa Shaaban Kabakibo, Eric Schniedermeyer, Artem Burchanow, Lin Wang
  [arXiv](https://arxiv.org/abs/2607.18081v1)
  > Large Language Models (LLMs) have demonstrated remarkable capabilities across a range of Natural Language Processing (NLP) tasks, but their high computational and memory demands pose significant chall...

- **Self-Distillation for Multi-Token Prediction** — Guoliang Zhao, Ruobing Xie, An Wang, Shuaipeng Li, Huaibing Xie, Xingwu Sun
  [arXiv](https://arxiv.org/abs/2603.23911)
  > As LLMs scale up, inference efficiency becomes a critical bottleneck. Multi-Token Prediction (MTP) could accelerate LLM inference by predicting multiple future tokens in parallel. MTP-D proposes a sim...

- **Semantic-Aware Data-Aided Channel Estimation with Large Language Models for MIMO Systems** — Sojeong Park, Jaehyun Choi, Hyun Jong Yang
  [arXiv](https://arxiv.org/abs/2607.18640v1)
  > Data-aided channel estimation enhances spectral efficiency by reusing detected symbols as virtual pilots. In this process, selecting only reliable symbols is crucial to prevent misdetected symbols fro...

- **Semiotic logical hexagon theory for LLM logical reasoning** — Yunyao Zhang, Xinglang Zhang, Zeliang Chen, Junqing Yu, Zikai Song
  [arXiv](https://arxiv.org/abs/2607.21933v1)
  > Large language models (LLMs) have become powerful tools for language understanding and logical reasoning. However, they still make mistakes when a problem requires both understanding meaning and follo...

- **Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving** — Tingyang Sun, Ting He, I-Hong Hou
  [arXiv](https://arxiv.org/abs/2604.14993)
  > As a current trend in Artificial Intelligence (AI), large foundation models are increasingly employed as the core of AI services. However, even after training, serving such models at scale remains a c...

- **Serving Compound Inference Systems on Datacenter GPUs** — Sriram Devata, Rahul Sukthankar, Saurabh Adya
  [arXiv](https://arxiv.org/abs/2603.08797)
  > Applications in emerging domains such as XR are being built as compound inference systems, where multiple ML models are composed in the form of a task graph to service each request. Serving these comp...

- **Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking** — Zizhao Mo, Junlin Chen, Huanle Xu, Chengzhong Xu
  [arXiv](https://arxiv.org/abs/2603.12831)
  > Nowadays, service providers often deploy multiple types of LLM services within shared clusters. While the service colocation improves resource utilization, it introduces significant interference risks...

- **Skill Self-Play: Pushing the Frontier of LLM Capability with Co-Evolving Skills** — Siyuan Huang, Pengyu Cheng, Haotian Liu, Tao Chen, Yihao Liu, Jingwei Ni, Shijie Zhou, Ziyi Yang, Gangwei Jiang, Mengyu Zhou, Yu Cheng, Xiaoxi Jiang, Guanjun Jiang
  [arXiv](https://arxiv.org/abs/2607.22529v1)
  > LLM training is shifting from manual design and annotation to interaction-driven self-evolution. However, existing self-evolutionary methods face a fundamental dilemma between task diversity and verif...

- **Slot Machines: How LLMs Keep Track of Multiple Entities** — Paul C. Bogdan, Jack Lindsey
  [arXiv](https://arxiv.org/abs/2604.21139v1) | [GitHub](https://github.com/turboderp-org/exllamav3)
  > Language models must bind entities to the attributes they possess and maintain several such binding relationships within a context. We study how multiple entities are represented across token position...

- **SpecBox: Speculative Sandbox Scheduling for Efficient LLM Agent Serving** — Yihui Zhang, Tianyu Wo, Jinghao Wang, Xiaoyang Sun, Menghao Zhang, Cangzhou Yuan, Li Li, Chunming Hu, Albert Y. Zomaya, Renyu Yang
  [arXiv](https://arxiv.org/abs/2607.23933v1)
  > As LLM agents increasingly rely on the Model Context Protocol (MCP) to invoke isolated external sandboxes, disaggregated sandbox deployment introduces a fundamental tension between resource utilizatio...

- **SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding** — Jehyeon Bang, Eunyeong Cho, Ranggi Hwang, Jinha Chung, Minsoo Rhu
  [arXiv](https://arxiv.org/abs/2604.10152)
  > The MoE architecture selectively activates parameters to mitigate computational costs, but high memory requirements and sub-optimal parameter efficiency pose challenges. Although CPU-offloaded MoE inf...

- **SpecSyn: LLM-based Synthesis and Refinement of Formal Specifications for Real-world Program Verification** — Lezhi Ma, Shangqing Liu, Yi Li, Qiong Wu, Han Wang
  [arXiv](https://arxiv.org/abs/2604.21570v1)
  > Program verification is a formal technique to rigorously ensure the correctness and fault-freeness of software systems. However, constructing comprehensive interprocedural specifications for full veri...

- **Speculative Speculative Decoding (SSD)** — Tanishq Kumar, Tri Dao, Avner May
  [arXiv](https://arxiv.org/abs/2603.03251) | [GitHub](https://github.com/tanishq Kumar/ssd-saguaro)
  > Speculative decoding accelerates autoregressive inference by using a fast draft model to predict upcoming tokens, then verifying them in parallel. However, speculative decoding itself relies on a sequ...

- **Stealthy Backdoor Attacks against LLMs Based on Natural Style Triggers** — Jiali Wei, Ming Fan, Guoheng Sun, Xicheng Zhang, Haijun Wang
  [arXiv](https://arxiv.org/abs/2604.21700v1)
  > The growing application of large language models (LLMs) in safety-critical domains has raised urgent concerns about their security. Many recent studies have demonstrated the feasibility of backdoor at...

- **StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving** — Azam Nouri
  [arXiv](https://arxiv.org/abs/2603.28795)
  > StepCache addresses LLM serving workloads where repeated requests share common solution structure but differ in localized constraints. Prior caching approaches reuse either full responses (semantic ca...

- **StreamServe: Adaptive Speculative Flows for Low-Latency Disaggregated LLM Serving** — Satyam Kumar, Arpit Singh Gautam, Kailash Talreja, Saurabh Jha
  [arXiv](https://arxiv.org/abs/2604.09562)
  > Efficient LLM serving must balance throughput and latency across diverse, bursty workloads. StreamServe is a disaggregated prefill decode serving architecture that combines metric-aware routing across...

- **StructKV: Preserving the Structural Skeleton for Scalable Long-Context Inference** — Zhirui Chen
  [arXiv](https://arxiv.org/abs/2604.06746)
  > As LLMs scale to support context windows exceeding one million tokens, KV cache linear growth imposes severe memory and bandwidth bottlenecks. Existing compression approaches prioritize tokens based o...

- **SwiftSpec: Disaggregated Speculative Decoding and Fused Kernels for Low-Latency LLM Inference** — Ziyi Zhang, Ziheng Jiang, Chengquan Jiang, Menghan Yu, Size Zheng, Haibin Lin, Xin Liu, Henry Hoffmann
  [GitHub](https://github.com/ByteDance-Seed/SwiftSpec)
  > Low-latency, single-request decoding of large language models is critical for interactive systems with tight SLA demands. Prior work reduces latency through speculative decoding (combining a small dra...

- **TABED: Test-Time Adaptive Ensemble Drafting for Robust Speculative Decoding in LVLMs** — Minjae Lee, Wonjun Kang, Byeongkeun Ahn, Christian Classen, Kevin Galim, Seunghyuk Oh, Minghao Yan, Hyung Il Koo, Kangwook Lee
  [arXiv](https://arxiv.org/abs/2601.20357) | [GitHub](https://github.com/furiosa-ai/TABED)
  > Speculative decoding (SD) has proven effective for accelerating LLM inference by quickly generating draft tokens and verifying them in parallel. However, SD remains largely unexplored for Large Vision...

- **TALON: Confidence-Aware Speculative Decoding with Adaptive Token Trees** — Tianyu Liu, Qitan Lv, Yuhao Shen, Xiao Sun, Xiaoyan Sun
  [arXiv](https://arxiv.org/abs/2601.07353)
  > Recent speculative decoding shifted from sequential chain-based drafting to tree-structured generation, but existing tree-based methods build fixed-width, fixed-depth draft trees that fail to adapt to...

- **TAPS: Task Aware Proposal Distributions for Speculative Sampling** — Mohamad Zbib, Mohamad Bazzi, Ammar Mohanna, Hasan Abed Al Kader Hammoud, Bernard Ghanem
  [arXiv](https://arxiv.org/abs/2603.27027)
  > Speculative decoding accelerates autoregressive generation by letting a lightweight draft model propose future tokens that a larger target model then verifies in parallel. In practice, however, draft ...

- **TRACE-ROUTER: Task-Consistent and Adaptive Online Routing for Agentic AI** — Ritik Raj, Souvik Kundu, Sarbartha Banerjee, Dheemanth Joshi, Ishita Vohra, Tushar Krishna
  [arXiv](https://arxiv.org/abs/2607.22465v1)
  > Routing to select large language models (LLMs) with different cost-quality trade-offs has become a fundamental deployment feature of enterprise AI. Existing routers, primarily make independent routing...

- **Talaria: Session-Aware Serverless Serving of Hundred-Billion-Parameter LLMs** — Utopia Meng, Unicornt Zhao, Derek Li, Goalen Gao, Frank Du
  [arXiv](https://arxiv.org/abs/2607.17181v1)
  > Serverless multi-model LLM systems multiplex popularity-skewed model catalogs over shared GPU pools, yet typically schedule each request independently. Tool-using agents break this abstraction: a sess...

- **TaxBreak: Unmasking the Hidden Costs of LLM Inference Through Overhead Decomposition** — Prabhu Vellaisamy, Shreesh Tripathi, Vignesh Natarajan, Surya Santhan Thenarasu, Shawn Blanton, John P. Shen
  [arXiv](https://arxiv.org/abs/2603.12465)
  > TaxBreak presents a trace-driven methodology for decomposing host-visible orchestration overhead into three components: framework translation time, CUDA library translation time, and kernel launch-pat...

- **TeLLMe: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs** — Ye Qiao, Zhiheng Chen, Yifan Zhang, Yian Wang, Sitao Huang
  > With the emergence of wearable devices and other embedded systems, deploying large language models (LLMs) on edge platforms becomes an urgent need. However, it is challenging because of their high com...

- **The Diminishing Returns of Early-Exit Decoding in Modern LLMs** — Rui Wei, Rui Du, Hanfei Yu, Devesh Tiwari, Jian Li, Zhaozhuo Xu, Hao Wang
  [arXiv](https://arxiv.org/abs/2603.23701)
  > In LLM inference, early-exit stops computation at an intermediate layer once prediction is sufficiently confident. Recent LLMs adopt improved pretraining recipes and architectures that reduce layer re...

- **The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference** — N/A
  [arXiv](https://arxiv.org/abs/2604.15409)
  > KV caching is a ubiquitous optimization in autoregressive transformer inference, long presumed to be numerically equivalent to cache-free computation. This assumption fails under standard FP16 precisi...

- **The Illusion of Secure LLM Code: Closing the Security Gap via Iterative Reprompting** — Ishpuneet Singh, Shreyas Mahajan, Gurjot Singh, Maninder Singh
  [arXiv](https://arxiv.org/abs/2607.23710v1)
  > Large Language Models (LLMs) are increasingly integrated into software development workflows, yet their ability to autonomously generate secure authentication code remains uncertain. This paper evalua...

- **The xPU-athalon: Quantifying the Competition of AI Acceleration** — ['Alicia Golden', 'Carole-Jean Wu', 'Gu-Yeon Wei', 'David Brooks']
  [arXiv](https://arxiv.org/abs/2604.10852)
  > The push for greater efficiency in AI computation has given rise to an array of accelerator architectures that increasingly challenge the GPU's long-standing dominance. In this work, we provide a quan...

- **TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters** — Zhiwen Mo, Yu Cheng, Lei Wang, Zhengju Tang, Lei Xu, Guoyu Li, Yuqi Dong, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, Luo Mai, Zhi Yang, Wayne Luk, Hongxiang Fan
  [arXiv](https://arxiv.org/abs/2607.22432v1)
  > Recent GPU programming frameworks such as Triton, TileLang, and CUDA Tile adopt tiles as first-class primitives, making tile-centric programming the prevailing approach for high-performance GPU kernel...

- **Token Coherence: Adapting MESI Cache Protocols to Minimize Synchronization Overhead in Multi-Agent LLM Systems** — Vladyslav Parakhin
  [arXiv](https://arxiv.org/abs/2603.15183) | [GitHub](https://github.com/hipvlady/agent-coherence)
  > Multi-agent LLM orchestration incurs synchronization costs scaling as O(n x S x |D|). This work maps synchronization cost explosion onto the cache coherence problem and adapts MESI-protocol invalidati...

- **Total Variation Distance Estimation in Autoregressive Models** — Eric Price, Kevin Tian, Zhiyang Xun, Yusong Zhu
  [arXiv](https://arxiv.org/abs/2607.19510v1)
  > Modern LLM deployments use a number of implementation choices and inference optimizations (e.g., batching, custom kernels, and quantization) on top of fixed weights, so two engines serving "the same m...

- **Transformer-Based Resource and Stage-Aware Scheduling for Model-Parallel LLM Inference** — Rami Naeem, Tengis Buyantogtokh, Hamada Rizk, Tatsuya Amano, Hirozumi Yamaguchi
  > Current large language model (LLM) serving systems face three key limitations in distributed scheduling. First, most parallelization strategies are not stage-aware: they treat prefill and decode as un...

- **Transition-Aware Backend Dispatch for Edge LLM Inference** — Alaaddin Goktug Ayar, Martin Margala
  [arXiv](https://arxiv.org/abs/2607.17415v1)
  > Efficient large language model (LLM) inference on edge platforms is limited not only by model size, but also by shape-dependent performance differences across execution backends. Static backend assign...

- **Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference** — ['Zifan He', 'Rui Ma', 'Yizhou Sun', 'Jason Cong']
  [arXiv](https://arxiv.org/abs/2603.29002)
  > Modern large language models (LLMs) increasingly depends on efficient long-context processing and generation mechanisms, including sparse attention, retrieval-augmented generation (RAG), and compresse...

- **Understanding the Impact of Linguistic Realization Choices on LLM Stance with Causal Tracing** — Langchen Huang, Sebastian Padó, Franziska Weeber
  [arXiv](https://arxiv.org/abs/2607.20115v1)
  > Large language models (LLMs) are known to be sensitive to prompt and input formulations. However, existing studies have focused on lexical realization and largely ignored constructional choice. This p...

- **UniGen-AR: Unifying Visual Generation with Auto-Regressive Modeling** — Zhipeng Bao, Zhen Zhu, Nupur Kumari, Anurag Bagchi, Yu-Xiong Wang, Pavel Tokmakov, Martial Hebert
  [arXiv](https://arxiv.org/abs/2607.24157v1)
  > Modern computer vision pipelines remain fragmented, with tasks such as text-to-image generation, editing, restoration, and classical perception handled by separate models. We study Unified Visual Gene...

- **Unified Static-Dynamic Pruning for Efficient LLM Inference** — Jinhyeok Kim, Yejoon Lee, Jaeyoung Do
  [arXiv](https://arxiv.org/abs/2607.21985v1)
  > The increasing deployment of large language models (LLMs) has magnified the computational and memory bottlenecks of autoregressive decoding, where low compute intensity and bandwidth-bound kernels dom...

- **Using Fine-Tuned LLMs to Identify Indicators of Vulnerability in UK Police Incident Logs** — Sam Relins, Daniel Birks
  [arXiv](https://arxiv.org/abs/2607.18446v1)
  > Purpose: Understanding how much of routine policing involves vulnerable people could inform resourcing, training, and multi-agency response, yet administrative data provide limited insight. We explore...

- **VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation** — Qijun Han, Haoqin Tu, Zijun Wang, Haoyue Dai, Yiyang Zhou
  [arXiv](https://arxiv.org/abs/2604.21375v1)
  > Autonomous GUI agents face two fundamental challenges: early stopping, where agents prematurely declare success without verifiable evidence, and repetitive loops, where agents cycle through the same f...

- **Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate** — Fangyue Liu, Hua Liu, Xinyuan Lyu, Shuo Ai, Hao Liang, Lingpeng Chen, Ziqian Hu, Chong Zha, Xin Jin, Hanmei Luo, Peng Chen
  [arXiv](https://arxiv.org/abs/2604.07874)
  > LLM inference powers latency-critical production services. Valve is a production-friendly colocation system that jointly bounds preemption latency and preemption rate. It enables sub-millisecond compu...

- **VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs** — Shahrzad Esmat, Dhawal Shah, Ali Jannesari
  [arXiv](https://arxiv.org/abs/2607.15498v1)
  > The key-value (KV) cache is the main memory bottleneck in long-context large language model (LLM) inference. Two leading training-free families are both structurally limited: token-selection methods (...

- **Visual Saliency Steering Distillation for Multimodal Chain-of-Thought Reasoning** — Hao Yang, Jin Wang, Xuejie Zhang
  [arXiv](https://arxiv.org/abs/2607.22013v1)
  > Multimodal chain-of-thought (CoT) reasoning integrates visual and textual cues through step-by-step inference. In small models with limited token budgets, modality-interaction fusion often suppresses ...

- **WWW.Serve: Interconnecting Global LLM Services through Decentralization** — Huanyu Wang, Ziyu Xia, Zhuoming Chen, Beidi Chen
  [arXiv](https://arxiv.org/abs/2603.20661)
  > Large language model (LLM) services are mostly centralized, leading to scalability bottlenecks and underutilization of substantial scattered GPU resources. While decentralization offers a promising al...

- **Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures** — ['Mauricio Fadel Argerich', 'Jonathan Fürst', 'Marta Patiño-Martínez']
  [arXiv](https://arxiv.org/abs/2604.09048)
  > While the large energy consumption of Large Language Models (LLMs) is recognized by the community, system operators lack guidance for energy-efficient LLM inference deployments that leverage energy tr...

- **WaveTune: Wave-aware Bilinear Modeling for Efficient GPU Kernel Auto-tuning** — ['Kaixuan Zhang', 'Chutong Ding', 'Shiyou Qian', 'Luping Wang', 'Jian Cao', 'Guangtao Xue']
  [arXiv](https://arxiv.org/abs/2604.10187)
  > The rapid adoption of Large Language Models (LLMs) has made GPU inference efficiency an increasingly critical system concern. The runtime of LLM workloads is largely dominated by tile-based kernels, p...

- **Wavefront Parallelization for Efficient Learned Image Compression** — Shimon Murai, Fangzheng Lin, Kasidis Arunruangsirilert, Jiro Katto
  [arXiv](https://arxiv.org/abs/2607.19082v1)
  > Autoregressive context models are foundational for learned image compression,but they suffer from slow serial inference. Existing acceleration methods such as checkerboard context require architectura...

- **WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning** — Juyong Jiang, Chenglin Cai, Chansung Park, Jiasi Shen, Sunghun Kim
  [arXiv](https://arxiv.org/abs/2604.20398v1) | [GitHub](https://github.com/sgl-project/sglang)
  > While Large Language Models (LLMs) excel at function-level code generation, project-level tasks such as generating functional and visually aesthetic multi-page websites remain highly challenging. Exis...

- **When LLM Defenses Backfire: Characterizing Safety, Performance, and Cost Trade-offs** — Tong Zhang, Zexin Li, Simin Chen, Yun Peng
  [arXiv](https://arxiv.org/abs/2607.24392v1)
  > Jailbreak defenses are essential for protecting large language models (LLMs), but they can also introduce secondary costs that weaken model utility. We present a systematic study of these defense trad...

- **Where FactsGo Missing: A LayerwiseTaxonomy and Per-Layer Attribution of Information Omissionin Air-Gapped LLM Agent Pipelines** — Santhiya Rajan
  [arXiv](https://arxiv.org/abs/2607.22448v1)
  > Air-gapped and on-premises deployments in regulated settings (clinical FHIR services, legal review, sovereign infrastructure) cannot call frontier APIs; they run quantized 4-8B models via llama.cpp or...

- **Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context** — Alagappan Valliappan
  [arXiv](https://arxiv.org/abs/2607.21535v1)
  > Speculative decoding accelerates autoregressive generation by having a cheap draft propose tokens that a target verifies in parallel. Frontier models increasingly ship a built-in Multi-Token-Predictio...

- **Zing: Social Mind for LLMs** —  Zing Team, Ao Xiang, Bi Jingping, Chen Jiahui, Chen Lehan, Chen Yilin, Cheng Xueqi, Fan Yixing, Gan Kairong, Gao Haowen, Gao Jinhua, Gao Shuxuan, Gong Chang, Guo Jiafeng, Guo Ruijie, Han Zhouyu, He Guangfu, He Yichun, Jiang Shuo, Jing Shaoling, Jing Ya, Lei Chenhao, Lei Yan, Li Anqi, Li Chengao, Li Haoyu, Li Shitian, Liang Xinjian, Liu Zhaoge, Lyu Xingyu, Nie Zhuwei, Pang Liang, Quan Zeping, Shan Shiguang, Shen Huawei, Tang Xinran, Tian Feng, Wang Qian, Wang Ruiping, Wang Xiaohong, Xia Zaiyu, Xiao Yi, Xu Jiayuan, Xu Kehan, Xu Qianqian, Xu Tianyu, Xu Yongjun, Yang Haoming, Yang Jun, Yao Di, Yu Xiaoming, Zhang Futong, Zhang Jie, Zhang Shixuan, Zhang Yuxuan, Zhao Xinyu, Zhao Zhuoran, Zhong Yunfei, Zhu Shengyu
  [arXiv](https://arxiv.org/abs/2607.23740v1)
  > As large language models move from isolated task solving toward long-term service in human environments, they require social intelligence: the ability to infer mental states, track social relations, r...

- **ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval** — David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen
  [arXiv](https://arxiv.org/abs/2604.10898)
  > Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rel...

- **[GitHub] BitNet: Official inference framework for 1-bit LLMs** — microsoft
  [arXiv](https://arxiv.org/abs/2411.04965v1) | [GitHub](https://github.com/microsoft/BitNet)
  > Recent research on the 1-bit Large Language Models (LLMs), such as BitNet b1.58, presents a promising direction for reducing the inference cost of LLMs while maintaining their performance. In this wor...

- **[GitHub] FastDeploy: High-performance Inference and Deployment Toolkit for LLMs and VLMs based on PaddlePaddle** — PaddlePaddle
  [GitHub](https://github.com/PaddlePaddle/FastDeploy)
  > High-performance Inference and Deployment Toolkit for LLMs and VLMs based on PaddlePaddle...

- **[GitHub] InferLLM: a lightweight LLM model inference framework** — MegEngine
  [arXiv](https://arxiv.org/abs/1411.4413v2) | [GitHub](https://github.com/MegEngine/InferLLM)
  > A joint measurement is presented of the branching fractions $B^0_s\toμ^+μ^-$ and $B^0\toμ^+μ^-$ in proton-proton collisions at the LHC by the CMS and LHCb experiments. The data samples were collected ...

- **[GitHub] JetStream: JetStream is a throughput and memory optimized engine for LLM inference on XLA devices, starting wit** — AI-Hypercomputer
  [GitHub](https://github.com/AI-Hypercomputer/JetStream)
  > JetStream is a throughput and memory optimized engine for LLM inference on XLA devices, starting with TPUs (and GPUs in future -- PRs welcome)....

- **[GitHub] LightLLM: LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for i** — ModelTC
  [GitHub](https://github.com/ModelTC/LightLLM)
  > LightLLM is a Python-based LLM (Large Language Model) inference and serving framework, notable for its lightweight design, easy scalability, and high-speed performance....

- **[GitHub] MARTI: A Framework for LLM-based Multi-Agent Reinforced Training and Inference** — TsinghuaC3I
  [GitHub](https://github.com/TsinghuaC3I/MARTI)
  > A Framework for LLM-based Multi-Agent Reinforced Training and Inference...

- **[GitHub] Nanoflow: A throughput-oriented high-performance serving framework for LLMs** — efeslab
  [GitHub](https://github.com/efeslab/Nanoflow)
  > A throughput-oriented high-performance serving framework for LLMs...

- **[GitHub] RouteLLM: A framework for serving and evaluating LLM routers - save LLM costs without compromising quality** — lm-sys
  [GitHub](https://github.com/lm-sys/RouteLLM)
  > A framework for serving and evaluating LLM routers - save LLM costs without compromising quality...

- **[GitHub] ScaleLLM: A high-performance inference system for large language models, designed for production environments.** — vectorch-ai
  [GitHub](https://github.com/vectorch-ai/ScaleLLM)
  > A high-performance inference system for large language models, designed for production environments....

- **[GitHub] TensorRT-LLM: TensorRT LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) an** — NVIDIA
  [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
  > TensorRT LLM provides users with an easy-to-use Python API to define Large Language Models (LLMs) and supports state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs. TensorRT ...

- **[GitHub] ZhiLight: A highly optimized LLM inference acceleration engine for Llama and its variants.** — zhihu
  [GitHub](https://github.com/zhihu/ZhiLight)
  > A highly optimized LLM inference acceleration engine for Llama and its variants....

- **[GitHub] asystem-awex: A high-performance RL training-inference weight synchronization framework, designed to enable second** — inclusionAI
  [GitHub](https://github.com/inclusionAI/asystem-awex)
  > A high-performance RL training-inference weight synchronization framework, designed to enable second-level parameter updates from training to inference in RL workflows...

- **[GitHub] chitu: High-performance inference framework for large language models, focusing on efficiency, flexibility,** — thu-pacman
  [GitHub](https://github.com/thu-pacman/chitu)
  > High-performance inference framework for large language models, focusing on efficiency, flexibility, and availability....

- **[GitHub] cuckoo: Cuckoo is a Decentralized AI Model-Serving Platform, starting with GPU-sharing for text-to-image gen** — cuckoo-network
  [GitHub](https://github.com/cuckoo-network/cuckoo)
  > Cuckoo is a Decentralized AI Model-Serving Platform, starting with GPU-sharing for text-to-image generation and LLM inference....

- **[GitHub] e2e-llm-workflows: Fine-tune an LLM to perform batch inference and online serving.** — anyscale
  [GitHub](https://github.com/anyscale/e2e-llm-workflows)
  > Fine-tune an LLM to perform batch inference and online serving....

- **[GitHub] llm-optimizer: Benchmark and optimize LLM inference across frameworks with ease** — bentoml
  [GitHub](https://github.com/bentoml/llm-optimizer)
  > Benchmark and optimize LLM inference across frameworks with ease...

- **[GitHub] llm_note: LLM notes, including model inference, transformer model structure, and llm framework code analysis n** — harleyszhang
  [GitHub](https://github.com/harleyszhang/llm_note)
  > LLM notes, including model inference, transformer model structure, and llm framework code analysis notes....

- **[GitHub] nncf: Neural Network Compression Framework for enhanced OpenVINO™ inference** — openvinotoolkit
  [GitHub](https://github.com/openvinotoolkit/nncf)
  > Neural Network Compression Framework for enhanced OpenVINO™ inference...

- **[GitHub] ramalama: RamaLama is an open-source developer tool that simplifies the local serving of AI models from any so** — containers
  [GitHub](https://github.com/containers/ramalama)
  > RamaLama is an open-source developer tool that simplifies the local serving of AI models from any source and facilitates their use for inference in production, all through the familiar language of con...

- **[GitHub] tiny-llm: A course of learning LLM inference serving on Apple Silicon for systems engineers: build a tiny vLLM** — skyzh
  [GitHub](https://github.com/skyzh/tiny-llm)
  > A course of learning LLM inference serving on Apple Silicon for systems engineers: build a tiny vLLM + Qwen....

- **[GitHub] transformers: 🤗 Transformers: the model-definition framework for state-of-the-art machine learning models in text,** — huggingface
  [GitHub](https://github.com/huggingface/transformers)
  > 🤗 Transformers: the model-definition framework for state-of-the-art machine learning models in text, vision, audio, and multimodal models, for both inference and training. ...


### LoRA/Adapter Serving

- **Language as a Latent Variable for Reasoning Optimization** — Linjuan Wu, Haoran Wei, Jialong Tang, Shuang Luo, Baosong Yang
  [arXiv](https://arxiv.org/abs/2604.21593v1) | [GitHub](https://github.com/containers/ramalama)
  > As LLMs reduce English-centric bias, a surprising trend emerges: non-English responses sometimes outperform English on reasoning tasks. We hypothesize that language functions as a latent variable that...

- **Separable Expert Architecture: Toward Privacy-Preserving LLM Personalization via Composable Adapters and Deletable User Proxies** — Chris Schneider, Philipp Schoenegger, Ben Bariach
  [arXiv](https://arxiv.org/abs/2604.21571v1) | [GitHub](https://github.com/SqueezeAILab/KVQuant)
  > Current model training approaches incorporate user information directly into shared weights, making individual data removal computationally infeasible without retraining. This paper presents a three-l...


### MoE Inference

- **DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for LLM Inference QoS Assurance** — Yuning Zhang, Grant Pinkert, Nan Yang, Yanli Li, Dong Yuan
  [arXiv](https://arxiv.org/abs/2509.07379)
  > Large Language Models (LLMs) are increasingly deployed as Internet/Web services (LLM-as-a-Service) with strict latency Service-Level Objectives (SLOs) under tight GPU memory budgets. Mixture-of-Expert...

- **Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs** — Afsara Benazir, Felix Xiaozhu Lin
  [arXiv](https://arxiv.org/abs/2604.18788)
  > Apple Neural Engine (ANE) is a dedicated neural processing unit (NPU) present in every Apple Silicon chip. Mixture-of-Experts (MoE) LLMs improve inference efficiency by activating only a sparse subset...

- **FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving** — Qingxiu Liu, Cyril Y. He, Hanser Jiang
  [arXiv](https://arxiv.org/abs/2604.02715)
  > Mixture-of-Experts (MoE) models have become a dominant paradigm for scaling large language models, but their rapidly growing parameter sizes introduce severe challenges for efficient serving. FluxMoE ...

- **From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill** — Gunjun Lee, Jiwon Kim, Jaiyoung Park, Younjoo Lee, Jung Ho Ahn
  [arXiv](https://arxiv.org/abs/2510.08055)
  > Large Language Model (LLM) inference in production must meet stringent service-level objectives for both time-to-first-token (TTFT) and time-between-token (TBT) while maximizing throughput under fixed...

- **LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training** — Xinyi Liu, Zijian Zhang, YongLi Zhu, Jiale Zhang, Peng Sun, XuanWang, Qi Qi, Jingren Zhou, Tong Yang, Bin Cui
  [arXiv](https://arxiv.org/abs/2602.11686)
  > Expert parallelism is vital for effectively training Mixture-of-Experts (MoE) models, enabling different devices to host distinct experts, with each device processing different input data. However, du...

- **MoEless: Efficient MoE LLM Serving via Serverless Computing** — Hanfei Yu, Bei Ouyang, Shwai He, Ang Li, Hao Wang
  [arXiv](https://arxiv.org/abs/2603.06350)
  > Large Language Models (LLMs) have become a cornerstone of AI, driving progress across diverse domains such as content creation, search and recommendation systems, and AI-assisted workflows. To allevia...

- **ReaLB: Real-Time Load Balancing for Multimodal MoE Inference** — Yingping Wang, Yi Wu, Xiangyu Wu
  [arXiv](https://arxiv.org/abs/2604.19503)
  > Mixture-of-Experts (MoE) architectures are widely used in modern large language models and multimodal models. However, inference efficiency is often limited by load imbalance across experts. ReaLB pro...

- **Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving** — Yue Pan, Zihan Xia, Po-Kai Hsu, Lanxiang Hu, Hyungyo Kim, Janak Sharda, Minxuan Zhou, Nam Sung Kim, Shimeng Yu, Tajana Rosing, Mingu Kang
  [arXiv](https://arxiv.org/abs/2510.05245)
  > As Large Language Models (LLMs) continue to evolve, Mixture of Experts (MoE) architecture has emerged as a prevailing design for achieving state-of-the-art performance across a wide range of tasks. Mo...

- **[GitHub] llm-server: Smart launcher for llama.cpp / ik_llama.cpp — auto-detects GPUs, optimizes MoE placement, crash reco** — raketenkater
  [GitHub](https://github.com/raketenkater/llm-server)
  > Smart launcher for llama.cpp / ik_llama.cpp — auto-detects GPUs, optimizes MoE placement, crash recovery...


### Offloading/Heterogeneous

- **MCAP: Deployment-Time Layer Profiling for Memory-Constrained LLM Inference** — Anurita Das
  [arXiv](https://arxiv.org/abs/2604.21026v1) | [GitHub](https://github.com/facebookresearch/LayerSkip)
  > Deploying large language models to heterogeneous hardware is often constrained by memory, not compute. We introduce MCAP (Monte Carlo Activation Profiling), a load-time per-layer importance estimator ...

- **NeuroClaw Technical Report** — Cheng Wang, Zhibin He, Zhihao Peng, Shengyuan Liu, Yufan Hu
  [arXiv](https://arxiv.org/abs/2604.24696v1)
  > Agentic artificial intelligence systems promise to accelerate scientific workflows, but neuroimaging poses unique challenges: heterogeneous modalities (sMRI, fMRI, dMRI, EEG), long multi-stage pipelin...

- **Strategic Heterogeneous Multi-Agent Architecture for Cost-Effective Code Vulnerability Detection** — Zhaohui Geoffrey Wang
  [arXiv](https://arxiv.org/abs/2604.21282v1)
  > Automated code vulnerability detection is critical for software security, yet existing approaches face a fundamental trade-off between detection accuracy and computational cost. We propose a heterogen...

- **[GitHub] ktransformers: A Flexible Framework for Experiencing Heterogeneous LLM Inference/Fine-tune Optimizations** — kvcache-ai
  [GitHub](https://github.com/kvcache-ai/ktransformers)
  > A Flexible Framework for Experiencing Heterogeneous LLM Inference/Fine-tune Optimizations...

- **[GitHub] mosec: A high-performance ML model serving framework, offers dynamic batching and CPU/GPU pipelines to full** — mosecorg
  [GitHub](https://github.com/mosecorg/mosec)
  > A high-performance ML model serving framework, offers dynamic batching and CPU/GPU pipelines to fully exploit your compute machine...


### Other

- **HieraSparse: Hierarchical Semi-Structured Sparse KV Attention** — ['Haoxuan Wang', 'Chen Wang']
  [arXiv](https://arxiv.org/abs/2604.16864)
  > We introduce HieraSparse, a hierarchical KV Cache compression framework with acceleration kernels that leverage GPU sparse tensor cores to speed up attention computation with semi-structured sparse pa...

- **How Much Cache Does Reasoning Need? Depth-Cache Tradeoffs in KV-Compressed Transformers** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.17935)
  > We study the depth-cache tradeoffs in KV-compressed transformers, examining how much cache is needed for reasoning tasks and the implications for inference efficiency....

- **KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.16682)
  > We propose KAIROS, a stateful, context-aware power-efficient agentic inference serving system that manages LLM inference for agentic workflows with awareness of multi-turn context and power constraint...

- **Latent Phase-Shift Rollback: Inference-Time Error Correction via Residual Stream Monitoring** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.18567)
  > We propose Latent Phase-Shift Rollback, an inference-time error correction mechanism via residual stream monitoring for LLM inference quality improvement....


### Prefill/Disaggregation

- **Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.18170)
  > We introduce Copy-as-Decode, a kernel that reframes constrained text editing as parallel prefill: the target string becomes the 'draft' and grammar constraints enforce deterministic acceptance, sharin...

- **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter** — Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang
  [arXiv](https://arxiv.org/abs/2604.15039)
  > Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but in practice its deployment boundary is still determined by KVCache transfer. In conventional de...

- **Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token (TTFT)** — Rajveer Bachkaniwala, Chengqi Luo, Richard So, Divya Mahajan, Kexin Rong
  [arXiv](https://arxiv.org/abs/2603.19458)
  > In this work, we investigate observable signatures of a magnetically charged Anti-de Sitter black hole in string-inspired Euler-Heisenberg theory. We analyze photon trajectories, the photon sphere, an...


### Quantization

- **AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.18137)
  > Processing-in-Memory (PIM) architectures improve bandwidth for LLM inference but face capacity limitations. We propose AQPIM, breaking the PIM capacity wall through in-memory activation quantization f...

- **Adaptive Compute Efficient Learning via Conceptual-Criticality (Student Abstract)** — Iñigo Parra, M. ManoBharathi, Mayank Kumar, Pushpa Kumar Balan, Priyadarsi Mishra
  [GitHub](https://github.com/skyzh/tiny-llm)
  > The computational cost of large language models (LLMs) is a primary obstacle to sustainable deployment. Static resource allocation is inefficient, as not all inputs require the same depth of processin...

- **An LLM-Guided Query-Aware Inference System for GNN Models on Large Knowledge Graphs** — Waleed Afandi, Hussein Abdallah, Ashraf Aboulnaga, Essam Mansour
  [arXiv](https://arxiv.org/abs/2603.04545) | [GitHub](https://github.com/skyzh/tiny-llm)
  > Efficient inference for graph neural networks (GNNs) on large knowledge graphs (KGs) is essential for many real-world applications. GNN inference queries are computationally expensive and vary in comp...

- **DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization** — ['(from arXiv)']
  [arXiv](https://arxiv.org/abs/2604.17789)
  > DuQuant++ enhances microscaling FP4 quantization with fine-grained rotation for more efficient LLM inference through improved low-bit quantization....

- **Efficient LLM Inference via Activation-Aware Weight Quantization: System Integration and Performance Analysis** — ['Tejas Pravinbhai Patel', 'Gajendra Babu Thokala', 'Sandeep Shivam', 'Chandrashekhar Medicherla', 'Vinay R Soni', 'Arun Kumar Elengovan', 'Isan Sahoo', 'Chaitanya Kulkarni']
  [GitHub](https://github.com/kvcache-ai/Mooncake)
  > The rapid scaling of large language models (LLMs) has driven extraordinary gains in natural language understanding and generation, but at the cost of substantial compute and memory demands. Efficient ...

- **Private LLM Inference on Consumer Blackwell GPUs: A Practical Guide for Cost-Effective Local Deployment in SMEs** — ['Jonathan Knoop', 'Hendrik Holtmann']
  [arXiv](https://arxiv.org/abs/2601.09527) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > SMEs increasingly seek alternatives to cloud LLM APIs, which raise data privacy concerns. Dedicated cloud GPU instances offer improved privacy but with limited guarantees and ongoing costs, while prof...

- **QIGen: A Kernel Generator for Inference on Nonuniformly Quantized Large Language Models** — Tommaso Pegolotti, Dan Alistarh, Markus Püschel
  [GitHub](https://github.com/humanrouter/ddtree-mlx)
  > Efficient inference on large language models (LLMs) has become a popular topic in both academia and industry. Roughly speaking, LLMs consist of a collection of weight matrices, and generative inferenc...

- **Tool Attention Is All You Need: Dynamic Tool Gating and Lazy Schema Loading for Eliminating the MCP/Tools Tax in Scalable Agentic Workflows** — Anuj Sadani, Deepak Kumar
  [arXiv](https://arxiv.org/abs/2604.21816v1) | [GitHub](https://github.com/containers/ramalama)
  > The Model Context Protocol (MCP) has become a common interface for connecting large language model (LLM) agents to external tools, but its reliance on stateless, eager schema injection imposes a hidde...

- **Understanding Efficiency: Quantization, Batching, and Serving Strategies in LLM Energy Use** — ['Julien Delavande', 'Régis Pierrard', 'Sasha Luccioni']
  [arXiv](https://arxiv.org/abs/2601.22362) | [GitHub](https://github.com/aerlabsAI/ai-inference-resources)
  > Large Language Models (LLMs) are increasingly deployed in production, contributing towards shifting the burden in terms of computational resources and energy demands from training to inference. While ...

- **[GitHub] exllamav3: An optimized quantization and inference library for running LLMs locally on modern consumer-class GP** — turboderp-org
  [GitHub](https://github.com/turboderp-org/exllamav3)
  > An optimized quantization and inference library for running LLMs locally on modern consumer-class GPUs ...


### Speculative Decoding

- **Accelerating OpenPangu Inference on NPU via Speculative Decoding** — Yuntao Dai, Jing Wu, Hang Gu, Teng Wang
  [arXiv](https://arxiv.org/abs/2603.0)
  > To mitigate the Memory Wall bottleneck encountered by Large Language Models (LLMs) during inference on NPU hardware, and addressing the scarcity of native support for mainstream speculative decoding a...

- **Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models** — Ally Qin, Jian Wan, Sarat Mudunuri, Srinivasan Manoharan
  [arXiv](https://arxiv.org/abs/2604.19767)
  > We evaluate speculative decoding with EAGLE3 as an inference-time optimization for PayPal&#39;s Commerce Agent, powered by a fine-tuned llama3.1-nemotron-nano-8B-v1 model. Building on prior work (NEMO...

- **Acceptance Dynamics Across Cognitive Domains in Speculative Decoding** — Saif Mahmoud
  [arXiv](https://arxiv.org/abs/2604.14682)
  > Speculative decoding accelerates large language model (LLM) inference. It uses a small draft model to propose a tree of future tokens. A larger target model then verifies these tokens in a single batc...

- **AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding** — ['Zikun Li', 'Zhuofu Chen', 'Rémi Delacourt', 'Gabriele Oliaro', 'Zeyu Wang', 'Qinghan Chen', 'Shuhuai Lin', 'April Yang', 'Zhihao Zhang', 'Zhuoming Chen', 'Sean Lai', 'Xupeng Miao', 'Zhihao Jia']
  [arXiv](https://arxiv.org/abs/2501.12162v2) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > Modern large language model (LLM) applications exhibit diverse service-level objectives (SLOs), from low-latency requirements in interactive coding assistants to more relaxed constraints in data wrang...

- **Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference** — Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan
  [arXiv](https://arxiv.org/abs/2604.13634)
  > Speculative decoding accelerates autoregressive generation by letting draft tokens bypass full verification, but conventional frameworks suffer from frequent false rejections, particularly when draft ...

- **ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding** — Walaa Amer, Uday Das, Fadi Kurdahi
  [arXiv](https://arxiv.org/abs/2604.14612)
  > Self-speculative decoding is an inference technique for large language models designed to speed up generation without sacrificing output quality. It combines fast, approximate decoding using a compact...

- **DiP-SD: Distributed Pipelined Speculative Decoding for Efficient LLM Inference at the Edge** — N/A
  [arXiv](https://arxiv.org/abs/2604.20919)
  > Speculative decoding has emerged as a promising technique for large language model (LLM) inference by accelerating autoregressive decoding via draft-then-verify. This paper studies a new edge scenario...

- **Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization** — Jiu Chen, Shuangyan Yang, Xu Xiong, Hexiao Duan, Xinran Zhang
  [arXiv](https://arxiv.org/abs/2604.21072v1) | [GitHub](https://github.com/SharpAI/SwiftLM)
  > Decentralized LLM inference distributes computation among heterogeneous nodes across the internet, offering a performant and cost-efficient solution, alternative to traditional centralized inference. ...

- **ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving** — Yuseon Choi, Jingu Lee, Jungjun Oh, Sunjoo Whang, Byeongcheol Kim, Minsung Kim, Hoi-Jun Yoo, Sangjin Kim
  [arXiv](https://arxiv.org/abs/2604.14626)
  > Mixture-of-Experts (MoE) models have become the dominant architecture for large-scale language models, yet on-premises serving remains fundamentally memory-bound as batching turns sparse per-token com...

- **FASER: Fine-Grained Phase Management for Speculative Decoding in Dynamic LLM Serving** — Wenyan Chen, Chengzhi Lu, Yanying Lin, Dmitrii Ustiugov
  [arXiv](https://arxiv.org/abs/2604.20503v1)
  > Speculative decoding (SD) is a widely used approach for accelerating decode-heavy LLM inference workloads. While online inference workloads are highly dynamic, existing SD systems are rigid and take a...

- **From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning (SpecGuard)** — Authors from arXiv:2604.15244
  [arXiv](https://arxiv.org/abs/2604.15244)
  > Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature a...

- **Multi-Drafter Speculative Decoding with Alignment Feedback** — ['Taehyeon Kim', 'Hojung Jung', 'Se-Young Yun']
  [arXiv](https://arxiv.org/abs/2604.05417)
  > Speculative decoding (SD) accelerates large language model (LLM) inference by using a smaller model to draft future tokens, which are then verified by the target LLM. This preserves generation quality...

- **NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Optimization** — Enshu Liu, Xuefei Ning, Yu Wang, Zinan Lin
  [arXiv](https://arxiv.org/abs/2604.18471) | [GitHub](https://github.com/imagination-research/NI-Sampling)
  > Discrete diffusion language models (dLLMs) have recently emerged as a promising alternative to traditional autoregressive approaches, offering the flexibility to generate tokens in arbitrary orders an...

- **Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning** — NVIDIA Team
  [arXiv](https://arxiv.org/abs/2604.12374) | [GitHub](https://github.com/https://huggingface.co/nvidia/Nemotron-3-Super)
  > We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the ...

- **RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding** — Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao
  [arXiv](https://arxiv.org/abs/2604.14885) | [GitHub](https://github.com/https://github.com/hkr04/RACER)
  > Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but ...

- **SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation** — Jialiang Kang, Han Shu, Wenshuo Li, Yingjie Zhai, Xinghao Chen
  [arXiv](https://arxiv.org/abs/2603.1)
  > Speculative Jacobi Decoding (SJD) offers a draft-model-free approach to accelerate autoregressive text-to-image synthesis. However, the high-entropy nature of visual generation yields low draft-token ...

- **SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs** — Jiaming Xu, Hong Cao, Yuhan Lin, Jinyang Li, Zheng Liu, Jie Liu, Xingyu Li, Jin Wang, Jingyuan Jia, Ge Li
  [arXiv](https://arxiv.org/abs/2512.00722)
  > In this paper, we point out that the objective of the retrieval algorithms is to align with the LLM, which is similar to the objective of knowledge distillation in LLMs. We analyze the similarity in i...

- **SpecMD: A Comprehensive Study On Speculative Expert Prefetching** — Duc Hoang, Ajay Jaiswal, Mohammad Samragh, Minsik Cho
  [arXiv](https://arxiv.org/abs/2602.03921)
  > Mixture-of-Experts (MoE) models enable sparse expert activation, meaning that only a subset of the model's parameters is used during each inference. However, to translate this sparsity into practical ...

- **Speculating Experts Accelerates Inference for Mixture-of-Experts** — Vivan Madan, Prajwal Singhania, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda
  [arXiv](https://arxiv.org/abs/2603.19289) | [GitHub](https://github.com/axonn-ai/yalis/tree/offload_prefetch)
  > Mixture-of-Experts (MoE) models have gained popularity as a means of scaling the capacity of large language models (LLMs) while maintaining sparse activations and reduced per-token compute. However, i...

- **Speculative Decoding for Autoregressive Video Generation** — Yuezhou Hu, Jintao Zhang
  [arXiv](https://arxiv.org/abs/2604.17397)
  > Autoregressive video diffusion is emerging as a promising paradigm for streaming video synthesis, with step distillation serving as the primary means of accelerating inference. Whether speculative dec...

- **Super Apriel: One Checkpoint, Many Speeds** — SLAM Labs,  :, Oleksiy Ostapenko, Raymond Li, Torsten Scholak
  [arXiv](https://arxiv.org/abs/2604.19877v1) | [GitHub](https://github.com/LMCache/LMCache)
  > We release Super Apriel, a 15B-parameter supernet in which every decoder layer provides four trained mixer choices -- Full Attention (FA), Sliding Window Attention (SWA), Kimi Delta Attention (KDA), a...

- **ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding** — Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li
  [arXiv](https://arxiv.org/abs/2604.13519)
  > Tool calling has greatly expanded the practical utility of large language models (LLMs) by enabling them to interact with external applications. As LLM capabilities advance, effective tool use increas...

- **Training-free Dropout Sampling for Semantic Token Acceptance in Speculative Decoding** — Jeongtae Lee, Minjung Jo, Hyunjoon Jeong, Gunho Park, Sunghyeon Woo, Joonghoon Kim, Se Jung Kwon, Dongsoo Lee
  [arXiv](https://arxiv.org/abs/2602.0)
  > Speculative decoding accelerates large language model inference by proposing tokens with a lightweight draft model and selectively accepting them using a target model. This work introduces DropMatch, ...

- **WISV: Wireless-Informed Semantic Verification for Distributed Speculative Decoding in Device-Edge LLM Inference** — ['Zixuan Liu', 'Zhiyong Chen', 'Nan Xue', 'Shengkang Chen', 'Jiangchao Yao', 'Meixia Tao', 'Wenjun Zhang']
  [arXiv](https://arxiv.org/abs/2604.17701)
  > While distributed device-edge speculative decoding accelerates LLM inference, verification overhead on constrained devices remains significant. We propose WISV, a wireless-informed semantic verificati...

- **[GitHub] BigLittleDecoder: [NeurIPS'23] Speculative Decoding with Big Little Decoder** — kssteven418
  [GitHub](https://github.com/kssteven418/BigLittleDecoder)
  > [NeurIPS'23] Speculative Decoding with Big Little Decoder...

- **[GitHub] Kangaroo: [NeurIPS 2024] The official implementation of "Kangaroo: Lossless Self-Speculative Decoding for Acce** — Equationliu
  [GitHub](https://github.com/Equationliu/Kangaroo)
  > [NeurIPS 2024] The official implementation of "Kangaroo: Lossless Self-Speculative Decoding for Accelerating LLMs via Double Early Exiting"...

- **[GitHub] LLMSpeculativeSampling: Fast inference from large lauguage models via speculative decoding** — feifeibear
  [GitHub](https://github.com/feifeibear/LLMSpeculativeSampling)
  > Fast inference from large lauguage models via speculative decoding...

- **[GitHub] Model-Optimizer: A unified library of SOTA model optimization techniques like quantization, pruning, distillation, sp** — NVIDIA
  [GitHub](https://github.com/NVIDIA/Model-Optimizer)
  > A unified library of SOTA model optimization techniques like quantization, pruning, distillation, speculative decoding, etc. It compresses deep learning models for downstream deployment frameworks lik...

- **[GitHub] Sequoia: scalable and robust tree-based speculative decoding algorithm** — Infini-AI-Lab
  [GitHub](https://github.com/Infini-AI-Lab/Sequoia)
  > scalable and robust tree-based speculative decoding algorithm...

- **[GitHub] Spec-Bench: Spec-Bench: A Comprehensive Benchmark and Unified Evaluation Platform for Speculative Decoding (ACL ** — hemingkx
  [GitHub](https://github.com/hemingkx/Spec-Bench)
  > Spec-Bench: A Comprehensive Benchmark and Unified Evaluation Platform for Speculative Decoding (ACL 2024 Findings)...

- **[GitHub] SpecForge: Train speculative decoding models effortlessly and port them smoothly to SGLang serving.** — sgl-project
  [GitHub](https://github.com/sgl-project/SpecForge)
  > Train speculative decoding models effortlessly and port them smoothly to SGLang serving....

- **[GitHub] SpecTTS-Bench: [ICLR'26] Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time S** — sunshy-1
  [GitHub](https://github.com/sunshy-1/SpecTTS-Bench)
  > [ICLR'26] Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling...

- **[GitHub] Speculative-Decoding: Implementation of the paper Fast Inference from Transformers via Speculative Decoding, Leviathan et ** — romsto
  [GitHub](https://github.com/romsto/Speculative-Decoding)
  > Implementation of the paper Fast Inference from Transformers via Speculative Decoding, Leviathan et al. 2023....

- **[GitHub] SpeculativeDecodingPapers: 📰 Must-read papers and blogs on Speculative Decoding ⚡️** — hemingkx
  [GitHub](https://github.com/hemingkx/SpeculativeDecodingPapers)
  > 📰 Must-read papers and blogs on Speculative Decoding ⚡️...

- **[GitHub] TorchSpec: A PyTorch native library for training speculative decoding models** — lightseekorg
  [GitHub](https://github.com/lightseekorg/TorchSpec)
  > A PyTorch native library for training speculative decoding models...

- **[GitHub] ddtree-mlx: Tree-based speculative decoding for Apple Silicon (MLX). ~10-15% faster than DFlash on code, ~1.5x o** — humanrouter
  [GitHub](https://github.com/humanrouter/ddtree-mlx)
  > Tree-based speculative decoding for Apple Silicon (MLX). ~10-15% faster than DFlash on code, ~1.5x over autoregressive. First MLX port with custom Metal kernels for hybrid model support....

- **[GitHub] dflash-mlx: Exact speculative decoding on Apple Silicon, powered by MLX.** — Aryagm
  [GitHub](https://github.com/Aryagm/dflash-mlx)
  > Exact speculative decoding on Apple Silicon, powered by MLX....

- **[GitHub] dflash: DFlash: Block Diffusion for Flash Speculative Decoding** — z-lab
  [GitHub](https://github.com/z-lab/dflash)
  > DFlash: Block Diffusion for Flash Speculative Decoding...

- **[GitHub] nano-PEARL: Draft-Target Disaggregation LLM Serving System via Parallel Speculative Decoding.** — smart-lty
  [GitHub](https://github.com/smart-lty/nano-PEARL)
  > Draft-Target Disaggregation LLM Serving System via Parallel Speculative Decoding....

- **[GitHub] speculators: A unified library for building, evaluating, and storing speculative decoding algorithms for LLM infe** — vllm-project
  [GitHub](https://github.com/vllm-project/speculators)
  > A unified library for building, evaluating, and storing speculative decoding algorithms for LLM inference in vLLM...

- **[GitHub] ssd: A lightweight inference engine supporting speculative speculative decoding (SSD). ** — tanishqkumar
  [GitHub](https://github.com/tanishqkumar/ssd)
  > A lightweight inference engine supporting speculative speculative decoding (SSD). ...


## 2025


### Distributed Inference

- **Buffer Management for Out-of-GPU LLM Execution** — Jiashen Cao, Joy Arulraj, Hyesoon Kim
  [GitHub](https://github.com/ome-projects/ome)
  > The rapid advancement of large language models (LLMs) has caused their parameter sizes to grow beyond the memory capacity of a single GPU. Although distributed inference across multiple GPUs is a solu...


### Edge Inference

- **MoA-Off: Adaptive Heterogeneous Modality-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference** — Zheming Yang, Qi Guo, Yunqing Hu, Chang Zhao, Chang Zhang
  [arXiv](https://arxiv.org/abs/2509.16995) | [GitHub](https://github.com/kvcache-ai/ktransformers)
  > Multimodal large language models (MLLMs) enable powerful cross-modal inference but impose significant computational and latency burdens, posing severe challenges for deployment in resource-constrained...

- **R-Sparse: Rank-Aware Activation Sparsity for Efficient LLM Inference** — Zhenyu (Allen) Zhang, Zechun Liu, Yuandong Tian, Harshit Khaitan, Zhangyang Wang
  [arXiv](https://arxiv.org/abs/2504.19449)
  > Large Language Models (LLMs), while demonstrating remarkable capabilities across various applications, present significant challenges during inference due to their substantial model size, especially w...


### Inference Kernel

- **DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference** — (待补充)
  [GitHub](https://github.com/LINs-lab/DeFT)
  > DeFT introduces Flash Tree-attention for efficient tree-structured speculative decoding, optimizing the attention computation over tree-shaped token sequences to accelerate LLM inference....

- **DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads** — (待补充)
  [GitHub](https://github.com/mit-han-lab/duo-attention)
  > DuoAttention introduces a dual-head attention mechanism that separates retrieval and streaming workloads, enabling efficient long-context LLM inference with reduced memory and compute requirements....

- **Online Pseudo-average Shifting Attention(PASA) for Robust Low-precision LLM Inference: Algorithms and Numerical Analysis** — Long Cheng, Qichen Liao, Fan Wu, Junlin Mu, Tengfei Han
  [arXiv](https://arxiv.org/abs/2503.01873) | [GitHub](https://github.com/anyscale/e2e-llm-workflows)
  > Attention calculation is extremely time-consuming for long-sequence inference tasks, such as text or image/video generation, in large models. To accelerate this process, we developed a low-precision, ...


### Inference Scheduling

- **AccelGen: Heterogeneous SLO-Guaranteed High-Throughput LLM Inference Serving for Diverse Applications** — Haiying Shen, Tanmoy Sen
  [arXiv](https://arxiv.org/abs/2503.13737) | [GitHub](https://github.com/vllm-project/vllm)
  > In this paper, we consider a mixed-prompt scenario for a large language model (LLM) inference serving system that supports diverse applications with both short prompts and long prompts and heterogeneo...

- **Adaptive Request Scheduling for CodeLLM Serving with SLA Guarantees** — Shi Chang, Boyuan Chen, Kishanthan Thangarajah, Hanan Lutfiyya, Ahmed E. Hassan
  [arXiv](https://arxiv.org/abs/2506.19677)
  > Code Large Language Models (CodeLLMs) are increasingly integrated into modern software development workflows, yet efficiently serving them in resource-constrained, self-hosted environments remains a s...

- **Cloud Native System for LLM Inference Serving** — Minxian Xu, Junhan Liao, Jingfeng Wu, Yiyuan He, Kejiang Ye
  [arXiv](https://arxiv.org/abs/2507.18007) | [GitHub](https://github.com/smart-lty/nano-PEARL)
  > Large Language Models (LLMs) are revolutionizing numerous industries, but their substantial computational demands create challenges for efficient deployment, particularly in cloud environments. Tradit...

- **Equinox: Holistic Fair Scheduling in Serving Large Language Models** — Zhixiang Wei, James Yen, Jingyi Chen, Ziyang Zhang, Zhibai Huang
  [arXiv](https://arxiv.org/abs/2508.16646)
  > We address the limitations of current LLM serving with a dual-counter framework separating user and operator perspectives. The User Fairness Counter measures quality of service via weighted tokens and...

- **Niyama : Breaking the Silos of LLM Inference Serving** — Kanishk Goel, Jayashree Mohan, Nipun Kwatra, R. S. Anupindi, R. Ramjee
  [arXiv](https://arxiv.org/abs/2503.22562)
  > The widespread adoption of Large Language Models (LLMs) has enabled diverse applications with very different latency requirements. Existing LLM serving frameworks rely on siloed infrastructure with co...

- **Past-Future Scheduler for LLM Serving under SLA Guarantees** — Ruihao Gong, Shihao Bai, Siyu Wu, Yunqian Fan, Zaijun Wang
  [arXiv](https://arxiv.org/abs/2507.10150)
  > The exploration and application of Large Language Models (LLMs) is thriving. To reduce deployment costs, continuous batching has become an essential feature in current service frameworks. The effectiv...

- **Prompt-Aware Scheduling for Low-Latency LLM Serving** — Yiheng Tao, Yihe Zhang, M. Dearing, Xin Wang, Yuping Fan
  [arXiv](https://arxiv.org/abs/2510.03243)
  > Efficient scheduling of LLM inference tasks is essential for achieving low latency and high throughput, particularly with the growing use of reasoning-capable LLMs. Traditional strategies like First-C...

- **SCORPIO: Serving the Right Requests at the Right Time for Heterogeneous SLOs in LLM Inference** — Yinghao Tang, Tingfeng Lan, Xiuqi Huang, Hui Lu, Wei Chen
  [arXiv](https://arxiv.org/abs/2505.23022) | [GitHub](https://github.com/sgl-project/SpecForge)
  > Existing Large Language Model (LLM) serving systems prioritize maximum throughput. They often neglect Service Level Objectives (SLOs) such as Time to First Token (TTFT) and Time Per Output Token (TPOT...

- **Serving Heterogeneous LoRA Adapters in Distributed LLM Inference Systems** — Shashwat Jaiswal, Shrikara Arun, Anjaly Parayil, Ankur Mallick, Spyros Mastorakis
  [arXiv](https://arxiv.org/abs/2511.22880) | [GitHub](https://github.com/vllm-project/vllm)
  > Low-Rank Adaptation (LoRA) has become the de facto method for parameter-efficient fine-tuning of large language models (LLMs), enabling rapid adaptation to diverse domains. In production, LoRA-based m...

- **Serving LLM in Distributed GPU Cluster With Fine-Grain Pipeline Constraints** — Yanying Lin, Shijie Peng, Shuaipeng Wu, Yanbo Li, Chengzhi Lu
  [GitHub](https://github.com/vllm-project/vllm)
  > As Large Language Models (LLMs) continue to advance, their parameter sizes are growing exponentially—far outpacing hardware capabilities. This widening gap necessitates distributed computing through p...


### KV Cache

- **APEX: Asynchronous Parallel CPU-GPU Execution for Online LLM Inference on Constrained GPUs** — Jiakun Fan, Yanglin Zhang, Xiangchen Li, Dimitrios S. Nikolopoulos
  [arXiv](https://arxiv.org/abs/2506.03296) | [GitHub](https://github.com/smart-lty/nano-PEARL)
  > Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU executi...

- **AccLLM: Accelerating Long-Context LLM Inference Via Algorithm-Hardware Co-Design** — Yanbiao Liang, Huihong Shi, Haikuo Shao, Zhongfeng Wang
  [arXiv](https://arxiv.org/abs/2505.03745)
  > Recently, large language models (LLMs) have achieved huge success in the natural language processing (NLP) field, driving a growing demand to extend their deployment from the cloud to edge devices. Ho...

- **Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching** — Yanhao Dong, Yubo Miao, Weinan Li, Xiaoyu Zheng, Chao Wang
  [arXiv](https://arxiv.org/abs/2504.06319) | [GitHub](https://github.com/SharpAI/SwiftLM)
  > Large Language Models (LLMs) exhibit pronounced memory-bound characteristics during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In this paper, we propose an L2 Cache-oriented a...

- **Adaptive Cache Pollution Control for Large Language Model Inference Workloads Using Temporal CNN-Based Prediction and Priority-Aware Replacement** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2512.14151)
  > Adaptive Cache Pollution Control for Large Language Model Inference Workloads Using Temporal CNN-Based Prediction and Priority-Aware Replacement, addressing KV cache management through intelligent pre...

- **AlayaDB: The Data Foundation for Efficient and Effective Long-context LLM Inference** — Yangshen Deng, Zhengxin You, Long Xiang, Qilong Li, Peiqi Yuan
  [arXiv](https://arxiv.org/abs/2504.10326)
  > AlayaDB is a cutting-edge vector database system natively architected for efficient and effective long-context inference for Large Language Models (LLMs) at AlayaDB AI. Specifically, it decouples the ...

- **Evaluating CXL Memory Pooling for Scalable LLM Inference** — Sai Krishna Vemuri, Venkata Ravi Shankar Jonnalagadda, Ajay Joshi, Rohit Sindhu, Vijay Kumar Motagi
  [GitHub](https://github.com/vllm-project/vllm)
  > Large-context LLM inference increasingly faces bottlenecks in Key-Value (KV) cache capacity and bandwidth rather than raw compute. While on-package HBM delivers exceptional bandwidth, its capacity is ...

- **FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework** — Jianian Zhu, Hang Wu, Haojie Wang, Yinghui Li, Biao Hou
  [arXiv](https://arxiv.org/abs/2503.08461) | [GitHub](https://github.com/sgl-project/sglang)
  > Multi-modal Large Language Models (MLLMs) serving systems commonly employ KV-cache compression to reduce memory footprint. However, existing compression methods introduce significant processing overhe...

- **FineServe: Precision-Aware KV Slab and Two-Level Scheduling for Heterogeneous Precision LLM Serving** — ['Kyungmin Bin', 'Seungbeom Choi', 'Jimyoung Son', 'Jieun Choi', 'Daseul Bae', 'Daehyeon Baek', 'Kihyo Moon', 'Minsung Jang', 'Hyojung Lee']
  [arXiv](https://arxiv.org/abs/2509.06261) | [GitHub](https://github.com/psmarter/mini-infer)
  > Recent advances in Post-Training Quantization (PTQ) techniques have significantly increased demand for serving quantized large language models (LLMs), enabling higher throughput and substantially redu...

- **First Experimental Demonstration of Disturb-Free 3D Vertical 1T-nC-1T Ferroelectric-based KV Cache with Co-Optimization of Hybrid Analog-Digital CIM and Token-Wise Dynamic Pruning for Efficient Long-Context LLM Inference** — Weikai Xu, Danyun Luo, Minyue Deng, Shuzhang Zhong, Shengjie Cao
  [GitHub](https://github.com/humanrouter/ddtree-mlx)
  > For the first time, a ferroelectric (FE)-based key-value (KV) cache for large language models (LLMs) is proposed and experimentally demonstrated. Through device-architecture-algorithm co-optimization,...

- **FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference** — Guangda Liu, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao
  [arXiv](https://arxiv.org/abs/2505.13109) | [GitHub](https://github.com/psmarter/mini-infer)
  > Large language models (LLMs) are widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, p...

- **HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing for LLMs** — Dongquan Yang, Yifan Yang, Xiaotian Yu, Xianbiao Qi, Rong Xiao
  [arXiv](https://arxiv.org/abs/2507.19823) | [GitHub](https://github.com/jjiantong/Awesome-KV-Cache-Optimization)
  > Processing long-context inputs with large language models presents a significant challenge due to the enormous memory requirements of the Key-Value (KV) cache during inference. Existing KV cache compr...

- **HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing** — Myunghyun Rhee, Joonseop Sim, Taeyoung Ahn, Seungyong Lee, Daegun Yoon
  [arXiv](https://arxiv.org/abs/2504.16112)
  > The attention layer, a core component of Transformer-based LLMs, brings out inefficiencies in current GPU systems due to its low operational intensity and the substantial memory requirements of KV cac...

- **KV Cache Transform Coding for Compact Storage in LLM Inference** — ['Konrad Staniszewski', "Adrian La'ncucki"]
  [arXiv](https://arxiv.org/abs/2511.01815) | [GitHub](https://github.com/psmarter/mini-infer)
  > Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in ite...

- **KV-CAR: KV Cache Compression using Autoencoders and KV Reuse in Large Language Models** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2512.06727)
  > KV-CAR proposes KV Cache Compression using Autoencoders and KV Reuse in Large Language Models, targeting efficient inference by reducing the memory footprint of KV caches through compression and reuse...

- **KVO-LLM: Boosting Long-Context Generation Throughput for Batched LLM Inference** — Zhenyu Li, Dongxu Lyu, Gang Wang, Yuzhou Chen, Liyan Chen
  [GitHub](https://github.com/cuckoo-network/cuckoo)
  > With the widespread deployment of long-context large language models (LLMs), efficient and high-quality generation is becoming increasingly important. Modern LLMs employ batching and key-value (KV) ca...

- **LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference** — Yihua Cheng, Yuhan Liu, Jiayi Yao, Yuwei An, Xiaokun Chen
  [arXiv](https://arxiv.org/abs/2510.09665) | [GitHub](https://github.com/vllm-project/vllm)
  > KV cache has traditionally been stored in GPU memory to accelerate the decoding phase of large language model (LLM) inference. However, it is increasingly necessary to move KV caches outside GPU devic...

- **MCaM : Efficient LLM Inference with Multi-tier KV Cache Management** — Kexin Chu, Zixu Shen, Shengxun Cheng, Dawei Xiang, Ziqin Liu
  [GitHub](https://github.com/vllm-project/vllm)
  > The KV cache in current LLM serving system is primarily used to accelerate processing within a single request and is aggressively deleted once the response is generated. However, in scenarios like vir...

- **MILLION: MasterIng Long-Context LLM Inference Via Outlier-Immunized KV Product QuaNtization** — Zongwu Wang, Peng Xu, Fangxin Liu, Yiwei Hu, Qingxiao Sun
  [arXiv](https://arxiv.org/abs/2504.03661)
  > Large language models (LLMs) are increasingly utilized for complex tasks requiring longer context lengths, with some models supporting up to 128 K or 1 M tokens. This trend, however, presents signific...

- **Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference** — Donghyeon Joo, Helya Hosseini, Ramyad Hadidi, Bahar Asgari
  [arXiv](https://arxiv.org/abs/2505.22913)
  > We demonstrate that unstructured sparsity significantly improves KV cache compression for LLMs, enabling sparsity levels up to 70% without compromising accuracy or requiring fine-tuning. We conduct a ...

- **Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization** — Minsu Kim, Seongmin Hong, Ryeowook Ko, Soongyu Choi, Hunjong Lee
  [arXiv](https://arxiv.org/abs/2503.18599) | [GitHub](https://github.com/jkanalakis/deep-recall)
  > Modern Large Language Model (LLM) serving system batches multiple requests to achieve high throughput, while batching attention operations is challenging, rendering memory bandwidth a critical bottlen...

- **Optimizing Distributed LLM Serving through Request Scheduling and Key-Value Cache Sharing** — Hongye Jiang, Mu Wang, Su Yao, Cui Ting, Ziwei Li
  [GitHub](https://github.com/vllm-project/vllm)
  > The widespread deployment of Large Language Models (LLMs) is often constrained by the significant computational and memory demands of the inference process. A critical bottleneck in distributed servin...

- **PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving** — A. C. Yuzuguler, Jiawei Zhuang, Lukas Cavigelli
  [arXiv](https://arxiv.org/abs/2501.08192) | [GitHub](https://github.com/sgl-project/sglang)
  > Large language models (LLMs) are typically served from clusters of GPUs/NPUs that consist of large number of devices. Unfortunately, communication between these devices incurs significant overhead, in...

- **Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference** — Thomas Joshi, Herman Saini, Neil Dhillon, Antoni Viros i Martin, Kaoutar El Maghraoui
  [arXiv](https://arxiv.org/abs/2506.07311)
  > Large Language Models (LLMs) encounter severe memory inefficiencies during long-context inference due to conventional handling of key-value (KV) caches. In this work, we introduce a novel integration ...

- **VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference** — Zihan Liu, Xinhao Luo, Junxian Guo, Wentao Ni, Yangjie Zhou
  [arXiv](https://arxiv.org/abs/2503.02236) | [GitHub](https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache)
  > Vector quantization (VQ), which treats a vector as a compression unit, gains increasing research interests for its potential to accelerate large language models (LLMs). Compared to conventional elemen...

- **VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization** — Dingyu Yao, Chenxu Yang, Zhengyang Tong, Zheng Lin, Wei Liu
  [arXiv](https://arxiv.org/abs/2510.06175) | [GitHub](https://github.com/skyzh/tiny-llm)
  > The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexib...


### LLM Serving

- **Acceleration Multiple Heads Decoding for LLM via Dynamic Tree Attention** — Zhendong Zhang
  [arXiv](https://arxiv.org/abs/2502.05947)
  > Multiple heads decoding accelerates the inference of Large Language Models (LLMs) by predicting next several tokens simultaneously. It generates and verifies multiple candidate sequences in parallel v...

- **Area- and Utilization-Efficient LLM Accelerator With Fused Speculative Decoding for Edge-Side Inference** — Kaiqi Chen, Zikang Zhou, Yaqi Chen, Jun Han

- **CompAir: Synergizing Complementary PIMs and In-Transit NoC Computation for Efficient LLM Acceleration** — Hongyi Li, Songchen Ma, Huanyu Qu, Weihao Zhang, Jia Chen, Junfeng Lin, Fengbin Tu, Rong Zhao
  [arXiv](https://arxiv.org/abs/2509.13710)
  > The rapid advancement of Large Language Models (LLMs) has revolutionized various aspects of human life, yet their immense computational and energy demands pose significant challenges for efficient inf...

- **Context-Aware Autoscaling for Cost-Efficient Large Language Model Inference With Prefix Cache Integration** — Seyed Hossein Ahmadpanah, A. Sahafi, S. H. Erfani
  > Although granular resource management has been made possible by the architectural shift to Prefill-Decode (PD) disaggregation in Large Language Model (LLM) serving, it is still difficult to maintain s...

- **DisHelis: Optimizing Deployment of Disaggregated LLMs Inference Serving Over Heterogeneous Environments via Hierarchical Max-Flow** — Tao Zhang, Huihuang Qin, Dong Jin, Shuangwu Chen, Huasen He, Xiaobin Tan, Shiyin Zhu, Jian Yang
  > Disaggregated LLM inference service (DLIS), which decouples the compute-intensive prefill phase and the memory-intensive decode phase, enables more flexible and efficient resource usage. Existing solu...

- **EasySpec: Layer-Parallel Speculative Decoding for Efficient Multi-GPU Utilization** — Yize Wu, Ke Gao, Yanjun Wu
  [arXiv](https://arxiv.org/abs/2502.02493) | [GitHub](https://github.com/Yize-Wu/EasySpec)
  > Speculative decoding is an effective and lossless method for Large Language Model (LLM) inference acceleration. It employs a smaller model to generate a draft token sequence, which is then verified by...

- **EdgeSD: Efficient Speculative Decoding with Vision-Decoding Disaggregation for MLLM Inference in Edge-Cloud Networks** — Hualong Huang, Wenhan Zhan, Hancong Duan, Kai Peng, Geyong Min, Zijia Zhao, Zitian Zhao, Yalan Ye
  > The deployment of multimodal large language models (MLLMs) in edge-cloud networks faces critical challenges, including computational resource heterogeneity, memory bottlenecks, and bandwidth constrain...

- **Efficient In-Memory Acceleration of Sparse Block Diagonal LLMs** — J. Lima, Marc Dietrich, J. Castrillón, Asif Ali Khan
  [arXiv](https://arxiv.org/abs/2510.11192)
  > Structured sparsity enables deploying large language models (LLMs) on resource-constrained systems. Approaches like dense-to-sparse fine-tuning are particularly compelling, achieving remarkable struct...

- **Efficient Kernel Mapping and Comprehensive System Evaluation of LLM Acceleration on a CGLA** — Takuto Ando, Yu Eto, Ayumu Takeuchi, Yasuhiko Nakashima
  [arXiv](https://arxiv.org/abs/2512.00335)
  > Large Language Models (LLMs) demand substantial computational resources, resulting in high energy consumption on GPUs. To address this challenge, we focus on Coarse-Grained Reconfigurable Arrays (CGRA...

- **EfficientEdit: Accelerating Code Editing via Edit-Oriented Speculative Decoding** — Peiding Wang, Li Zhang, Fang Liu, Yinghao Zhu, Wang Xu, Lin Shi, Xiaoli Lian, Minxiao Li, Bo Shen, An Fu
  [arXiv](https://arxiv.org/abs/2506.02780) | [GitHub](https://github.com/zhu-zhu-ding/EfficientEdit)
  > Large Language Models (LLMs) have demonstrated remarkable capabilities in code editing, substantially enhancing software development productivity. However, the inherent complexity of code editing task...

- **FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction** — Yuxuan Cai, Xiaozhuan Liang, Xinghua Wang, Jin Ma, Haijin Liang, Jinwen Luo, Xinyu Zuo, Lisheng Duan, Yuyang Yin, Xi Chen
  [arXiv](https://arxiv.org/abs/2509.18362)
  > As large language models (LLMs) become increasingly powerful, the sequential nature of autoregressive generation creates a fundamental throughput bottleneck that limits the practical deployment. While...

- **FlashInfer: Kernel Library for LLM Serving** — Unknown
  [arXiv](https://arxiv.org/abs/2501.01005) | [GitHub](https://github.com/flashinfer-ai/flashinfer)
  > Transformers, driven by attention mechanisms, form the foundation of large language models (LLMs). As these models scale up, efficient GPU attention kernels become essential for high-throughput and lo...

- **FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design** — Hao Zhang, Aining Jia, Weifeng Bu, Yu Cai, Kai Sheng, Hao Chen, Xin He
  [arXiv](https://arxiv.org/abs/2508.04405) | [GitHub](https://github.com/FlyFoxPlayer/FlexQ)
  > Large Language Models (LLMs) demonstrate exceptional performance but entail significant memory and computational costs, restricting their practical deployment. While existing INT4/INT8 quantization re...

- **Glinthawk: A Two-Tiered Architecture for Offline LLM Inference** — Pouya Hamadanian, Sadjad Fouladi
  [arXiv](https://arxiv.org/abs/2501.11779) | [GitHub](https://github.com/https://github.com/microsoft/glinthawk)
  > We introduce Glinthawk, an architecture for offline Large Language Model (LLM) inference. By leveraging a two-tiered structure, Glinthawk optimizes the utilization of the high-end accelerators ("Tier ...

- **La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation** — Kai Liu, Bowen Xu, Shaoyu Wu, Xin Chen, Hao Zhou, Yongliang Tao, Lulu Hu
  [arXiv](https://arxiv.org/abs/2507.01299)
  > Activation sparsity can reduce the computational overhead and memory transfers during the forward pass of Large Language Model (LLM) inference. Existing methods face limitations, either demanding time...

- **LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design** — Renjie Wei, Songqiang Xu, Linfeng Zhong, Zebin Yang, Qingyu Guo, Yuan Wang, Runsheng Wang, Meng Li
  [arXiv](https://arxiv.org/abs/2502.15260)
  > State space models (SSMs) like Mamba have recently attracted much attention. Compared to Transformer-based large language models (LLMs), Mamba achieves linear computation complexity with the sequence ...

- **MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures** — Jiayu Qin, Jianchao Tan, Kefeng Zhang, Xunliang Cai, Wei Wang
  [arXiv](https://arxiv.org/abs/2502.14008)
  > The remarkable performance of large language models (LLMs) in various language tasks has attracted considerable attention. However, the ever-increasing size of these models presents growing challenges...

- **MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching** — Tairan Xu, Leyang Xue, Zhan Lu, Adrian Jackson, Luo Mai
  [arXiv](https://arxiv.org/abs/2503.09716) | [GitHub](https://github.com/EfficientMoE/MoE-Gen)
  > This paper presents MoE-Gen, a high-throughput MoE inference system optimized for single-GPU execution. Existing inference systems rely on model-based or continuous batching strategies, originally des...

- **Optimizing LLM inference for FPGAs** — J. R. de Freitas, J. G. Coutinho, Ce Guo, S. Demirsoy, Wayne Luk, Zhiqiang Que
  [GitHub](https://github.com/custom-computing-ic/llm-oneapi-fpga)
  > Large Language Models (LLMs) deliver state-of-the-art performance but demand high computation and memory, making deployment in resource-limited settings challenging. Field-Programmable Gate Arrays (FP...

- **P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference Using Hybrid Numerical Formats** — Yuzong Chen, Chao Fang, Xilai Dai, Yuheng Wu, Thierry Tambe, Marian Verhelst, Mohamed S. Abdelfattah
  [arXiv](https://arxiv.org/abs/2511.06838) | [GitHub](https://github.com/yc2367/P3-LLM)
  > The substantial memory bandwidth and computational demands of large language models (LLMs) present critical challenges for efficient inference. To tackle this, the literature has explored heterogeneou...

- **PICNIC: Silicon Photonic Interconnected Chiplets with Computational Network and In-memory Computing for LLM Inference Acceleration** — Yue Jiet Chong, Yimin Wang, Zhen Wu, Xuanyao Fong
  [arXiv](https://arxiv.org/abs/2511.04036)
  > This paper presents a 3D-stacked chiplets based large language model (LLM) inference accelerator, consisting of non-volatile in-memory-computing processing elements (PEs) and Inter-PE Computational Ne...

- **Research on Low-Latency Inference and Training Efficiency Optimization for Graph Neural Network and Large Language Model-Based Recommendation Systems** — Yushang Zhao, Haotian Lyu, Yike Peng, Aijia Sun, Feng Jiang, Xinyue Han
  [arXiv](https://arxiv.org/abs/2507.01035)
  > The incessant advent of online services demands high speed and efficient recommender systems (ReS) that can maintain real-time performance along with processing very complex user-item interactions. Th...

- **Reward-Shifted Speculative Sampling Is An Efficient Test-Time Weak-to-Strong Aligner** — Bolian Li, Yanran Wu, Xinyu Luo, Ruqi Zhang
  [arXiv](https://arxiv.org/abs/2508.15044)
  > Aligning large language models (LLMs) with human preferences has become a critical step in their development. Recent research has increasingly focused on test-time alignment, where additional compute ...

- **SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving** — Xiangchen Li, Dimitrios Spatharakis, Saeid Ghafouri, Jiakun Fan, Hans Vandierendonck, Deepu John, Bo Ji, Dimitrios S. Nikolopoulos
  [arXiv](https://arxiv.org/abs/2506.09397)
  > The growing gap between the increasing complexity of large language models (LLMs) and the limited computational budgets of edge devices poses a key challenge for efficient on-device inference, despite...

- **ScaleLLM: A Technique for Scalable LLM-augmented Data Systems** — Ashwin Alaparthi, P. Loh, Ryan Marcus
  [GitHub](https://github.com/NVIDIA/Model-Optimizer)
  > Large language models (LLMs) offer powerful semantic insights for data analytics, but row-by-row LLM calls quickly become prohibitively expensive in large datasets. We introduce ScaleLLM, a novel syst...

- **The Anatomy of a Triton Attention Kernel** — Burkhard Ringlein, Jan van Lunteren, Radu Stoica, Thomas Parnell
  [arXiv](https://arxiv.org/abs/2511.11581)
  > A long-standing goal in both industry and academia is to develop an LLM inference platform that is portable across hardware architectures, eliminates the need for low-level hand-tuning, and still deli...

- **Trinity: Disaggregating Vector Search from Prefill-Decode Disaggregation in LLM Serving** — Yi Liu, Chen Qian
  [arXiv](https://arxiv.org/abs/2512.02281)
  > Trinity consolidates all retrieval into a single shared vector-search GPU pool working with PD disaggregated LLM serving. Introduces: (1) novel architecture for GPU-based vector search in PD disaggreg...

- **UniCAIM: A Unified CAM/CIM Architecture with Static-Dynamic KV Cache Pruning for Efficient Long-Context LLM Inference** — Weikai Xu, Wenxuan Zeng, Qianqian Huang, Meng Li, Ruei-Hao Huang
  [arXiv](https://arxiv.org/abs/2504.07479)
  > Transformer-based large language models (LLMs) have achieved impressive performance in various natural language processing (NLP) applications. However, the high memory and computation cost induced by ...

- **Variation-aware Vision Token Dropping for Faster Large Vision-Language Models** — Junjie Chen, Xuyang Liu, Zichen Wen, Yiyu Wang, Siteng Huang, Honggang Chen
  [arXiv](https://arxiv.org/abs/2509.01552)
  > Large vision-language models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding tasks. However, the increasing demand for high-resolution image and long-video understanding ...


### MoE Inference

- **Faster MoE LLM Inference for Extremely Large Models** — Haoqi Yang, Luohe Shi, Qiwei Li, Zuchao Li, Ping Wang
  [arXiv](https://arxiv.org/abs/2505.03531) | [GitHub](https://github.com/ByteDance-Seed/ShadowKV)
  > Sparse Mixture of Experts (MoE) large language models (LLMs) are gradually becoming the mainstream approach for ultra-large-scale models. Existing optimization efforts for MoE models have focused prim...

- **Frontier: Simulating the Next Generation of LLM Inference Systems** — Yicheng Feng, Xin Tan, Kin Hang Sew, Yimin Jiang, Yibo Zhu
  [arXiv](https://arxiv.org/abs/2508.03148)
  > Large Language Model (LLM) inference is growing increasingly complex with the rise of Mixture-of-Experts (MoE) models and disaggregated architectures that decouple components like prefill/decode (PD) ...

- **MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints** — Yichao Yuan, Lin Ma, Nishil Talati
  [arXiv](https://arxiv.org/abs/2504.09345) | [GitHub](https://github.com/jjiantong/Awesome-KV-Cache-Optimization)
  > Mixture of Experts (MoE) LLMs, characterized by their sparse activation patterns, offer a promising approach to scaling language models while avoiding proportionally increasing the inference cost. How...

- **Patterns behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference** — Zhongkai Yu, Yue Guan, Zihao Yu, Chenyang Zhou, Zhengding Hu
  [arXiv](https://arxiv.org/abs/2510.05497)
  > Large-scale Mixture of Experts (MoE) Large Language Models (LLMs) have recently become the frontier open weight models, achieving remarkable model capability similar to proprietary ones. But their ran...


### Prefill/Disaggregation

- **ADOR: A Design Exploration Framework for LLM Serving with Enhanced Latency and Throughput** — Junsoo Kim, Hunjong Lee, Geonwoo Ko, Gyubin Choi, Seri Ham
  [arXiv](https://arxiv.org/abs/2503.04253) | [GitHub](https://github.com/lucidrains/speculative-decoding)
  > The growing adoption of Large Language Models (LLMs) across various domains has driven the demand for efficient and scalable AI-serving solutions. Deploying LLMs requires optimizations to manage their...

- **Argus: Token Aware Distributed LLM Inference Optimization** — Panlong Wu, Yifei Zhong, Danyang Chen, Ting Wang, Fangxin Wang
  [arXiv](https://arxiv.org/abs/2512.22925) | [GitHub](https://github.com/jjiantong/Awesome-KV-Cache-Optimization)
  > Large Language Models (LLMs) are rapidly being integrated into real-world applications, yet their autoregressive architectures introduce significant inference time variability, especially when deploye...

- **DOPD: A Dynamic PD-Disaggregation Architecture for Maximizing Goodput in LLM Inference Serving** — Junhan Liao, Minxian Xu, Wanyi Zheng, Yan Wang, Kejiang Ye
  [arXiv](https://arxiv.org/abs/2511.20982)
  > To meet strict Service-Level Objectives (SLO), contemporary Large Language Models (LLMs) decouple the prefill and decoding stages and place them on separate GPUs to mitigate the distinct bottlenecks i...

- **Dynamic Offloading Optimization for Multi-Pim LLM Inference** — Jeonghoon Kang, Jae Hyung Ko, Taeho Hwang, Kyu Hyun Choi
  [GitHub](https://github.com/kvcache-ai/ktransformers)
  > The growing demand for energy-efficient on-device AI in consumer appliances is drawing significant attention to Processing-In-Memory (PIM) architectures. This trend is largely driven by the proliferat...

- **Efficient LLM Inference via Chunked Prefills** — Arney Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra
  [GitHub](https://github.com/SharpAI/SwiftLM)
  > Large Language Model (LLM) inference serving faces a fundamental challenge due to the distinct characteristics of its two phases: compute-intensive pre fill and memory-intensive decode. Existing sched...

- **LLMShare: Optimizing LLM Inference Serving with Hardware Architecture Exploration** — Hongduo Liu, Chen Bai, Peng Xu, Lihao Yin, Xianzhi Yu
  [GitHub](https://github.com/vllm-project/vllm)
  > Large Language Models (LLMs) have revolutionized language tasks but pose significant deployment challenges due to their substantial computational demands during inference. The hardware configurations ...

- **Nova: Real-Time Agentic Vision-Language Model Serving With Adaptive Cross-Stage Parallelization** — Yuhang Xu, Shengzhong Liu, Dong Zhang, Bingheng Yan, Fan Wu
  [arXiv](https://arxiv.org/abs/2509.21301)
  > This paper presents Nova, a real-time scheduling framework for serving agentic vision-language models (VLMs) on a single GPU with balanced per-request latency and overall request process throughput. O...

- **SwiftServe: Efficient Disaggregated LLM Inference Serving via Hierarchical Max-Flow in Heterogeneous GPUs and Network** — Tao Zhang, Yan Hu, Shuangwu Chen, Zian Wang, Huihuang Qin
  [GitHub](https://github.com/vllm-project/vllm)
  > Large language models (LLMs) have achieved remarkable performance across a variety of tasks. Disaggregated LLM inference serving (DLIS), which separates the compute-intensive prefill phase and the mem...


### Quantization

- **BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook** — Hao Gu, Lujun Li, Zheyu Wang, Beisong Liu, Qiyuan Zhu
  [arXiv](https://arxiv.org/abs/2506.12040) | [GitHub](https://github.com/uccl-project/uccl)
  > Binary quantization represents the most extreme form of compression, reducing weights to +/-1 for maximal memory and computational efficiency. While recent sparsity-aware binarization achieves sub-1-b...

- **Bench360: Benchmarking Local LLM Inference from 360°** — ['Linus Stuhlmann', 'Mauricio Fadel Argerich', 'Jonathan Furst']
  [arXiv](https://arxiv.org/abs/2511.16682) | [GitHub](https://github.com/EricLBuehler/candle-vllm)
  > Running LLMs locally has become increasingly common, but users face a complex design space across models, quantization levels, inference engines, and serving scenarios. Existing inference benchmarks a...

- **Coruscant: Co-Designing GPU Kernel and Sparse Tensor Core to Advocate Unstructured Sparsity in Efficient LLM Inference** — Donghyeon Joo, Helya Hosseini, Ramyad Hadidi, Bahar Asgari
  [GitHub](https://github.com/0xSero/turboquant)
  > In the era of large language models (LLMs) and long-context generation, model compression techniques such as pruning, quantization, and distillation offer effective ways to reduce memory usage. Among ...

- **D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving** — ['Haodong Wang', 'Qihua Zhou', 'Zicong Hong', 'Song Guo']
  [arXiv](https://arxiv.org/abs/2504.15299)
  > The mixture of experts (MoE) model is a sparse variant of large language models (LLMs), designed to hold a better balance between intelligent capability and computational overhead. Despite its benefit...

- **DILEMMA: Joint LLM Quantization and Distributed LLM Inference Over Edge Computing Systems** — ['Minoo Hosseinzadeh', 'Hana Khamfroush']
  [arXiv](https://arxiv.org/abs/2503.01704) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > With a recent trend of using Large Language Models (LLMs) for different applications within smart cities, there is a need for pushing these models toward the edge of network while still preserving the...

- **Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference** — ['Kexin Chu', 'Dawei Xiang', 'Zixu Shen', 'Yiwei Yang', 'Zecheng Liu', 'Wei Zhang']
  [arXiv](https://arxiv.org/abs/2511.15015) | [GitHub](https://github.com/aerlabsAI/ai-inference-resources)
  > Mixture-of-Experts (MoE) has become a practical architecture for scaling LLM capacity while keeping per-token compute modest, but deploying MoE models on a single, memory-limited GPU remains difficult...

- **EdgeShard: Efficient LLM Inference via Collaborative Edge Computing** — Mingjin Zhang, Xiaoming Shen, Jiannong Cao, Zeyang Cui, Shan Jiang
  [GitHub](https://github.com/skyzh/tiny-llm)
  > Large language models (LLMs) have shown great success in content generation and intelligent intelligent decision making for IoT systems. Traditionally, LLMs are deployed on the cloud, incurring prolon...

- **Energy-Efficient Cloud Infrastructure Design For Large Language Model Training And Inference** — G. Kathiresan
  [GitHub](https://github.com/ModelTC/LightLLM)
  > The rapidly increasing development of Large Language Models (LLMs) has rapidly placed a tremendous burden on cloud computing in terms of energy requirements, cost of operation, and environmental impac...

- **FPGA Co-Design for Efficient N:M Sparse and Quantized Model Inference** — F. Hsieh, Yun-Chang Teng, Ding-Yong Hong, Jan-Jan Wu
  [arXiv](https://arxiv.org/abs/2512.24713) | [GitHub](https://github.com/uccl-project/uccl)
  > Large language models (LLMs) have demonstrated remarkable performance across a wide range of language processing tasks. However, this success comes at the cost of substantial computation and memory re...

- **FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration** — Daehyeon Baek, Jieun Choi, Jimyoung Son, Kyungmin Bin, Seungbeom Choi
  [arXiv](https://arxiv.org/abs/2505.20839) | [GitHub](https://github.com/aerlabsAI/ai-inference-resources)
  > As large language models become increasingly prevalent, memory bandwidth constraints significantly limit inference throughput, motivating post-training quantization (PTQ). In this paper, we propose Fi...

- **GPU-Centric Memory Tiering for LLM Serving With NVIDIA Grace Hopper Superchip** — ['Woohyun Choi', 'Jinwoo Jeong', 'Hanhwi Jang', 'Jeongseob Ahn']
  [GitHub](https://github.com/dipampaul17/KVSplit)
  > This study investigates the performance of serving large language models (LLMs) with a focus on the high-bandwidth interconnect between GPU and CPU using a real NVIDIA Grace Hopper Superchip. This arc...

- **How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference** — Nidhal Jegham, Marwen Abdelatti, Lassad Elmoubarki, Abdeltawab M. Hendawi
  [arXiv](https://arxiv.org/abs/2505.09598)
  > This paper introduces an infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models in commercial datacenters. The frame...

- **Huff-LLM: End-to-End Lossless Compression for Efficient LLM Inference** — Patrick Yubeaton, Tareq Mahmoud, S. Naga, Pooria Taheri, Tianhua Xia
  [arXiv](https://arxiv.org/abs/2502.00922) | [GitHub](https://github.com/bstnxbt/dflash-mlx)
  > As they become more capable, large language models (LLMs) have continued to rapidly increase in size. This has exacerbated the difficulty in running state of the art LLMs on small, edge devices. Stand...

- **LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving** — ['Huanqi Hu', 'Bowen Xiao', 'Shixuan Sun', 'Jianian Yin', 'Zhexi Zhang', 'Xiangzhong Luo', 'Chengquan Jiang', 'Weiqi Xu', 'Xiaoying Jia', 'Xin Liu', 'Minyi Guo']
  [arXiv](https://arxiv.org/abs/2509.01229)
  > Quantization is a critical technique for accelerating LLM inference by reducing memory footprint and improving computational efficiency. Among various schemes, 4-bit weight and 8-bit activation quanti...

- **Optimizing Attention for Efficient LLM Inference: A Review** — ['Siyuan Sun', 'Jinling Yu', 'Han Liu', 'Hanyu Guo', 'Yang Cao', 'Shouhua Zhang', 'Jiehan Zhou']
  [GitHub](https://github.com/sgl-project/sglang)
  > The rapid advancement of deep learning has led to significant progress in large language models (LLMs), with the Attention mechanism serving as a core component of their success. However, the computat...

- **Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization** — Bowei He, Lihao Yin, Huiling Zhen, Shuqi Liu, Han Wu
  [arXiv](https://arxiv.org/abs/2510.10618) | [GitHub](https://github.com/containers/ramalama)
  > Post-training compression has been a widely employed approach to scale down large language model (LLM) and facilitate efficient inference. In various proposed compression methods, including pruning an...

- **QLLMS: Quantization-Adaptive LLM Scheduling for Partially Informed Edge Serving Systems** — ['Miao Hu', 'Q. He', 'Di Wu']
  [GitHub](https://github.com/kvcache-ai/Mooncake)

- **Recursive Offloading for LLM Serving in Multi-tier Networks** — Zhiyuan Wu, Sheng Sun, Yuwei Wang, Min Liu, Bo Gao
  [arXiv](https://arxiv.org/abs/2505.16502)
  > Heterogeneous device-edge-cloud computing infrastructures have become widely adopted in telecommunication operators and Wide Area Networks (WANs), offering multi-tier computational support for emergin...

- **TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms** — Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Esha Choukse, Haoran Qiu
  [arXiv](https://arxiv.org/abs/2501.02600)
  > The rising demand for generative large language models (LLMs) poses challenges for thermal and power management in cloud datacenters. Traditional techniques are often inadequate for LLM inference due ...

- **Toward Sustainable AI: A Review of Energy-Efficient Large Language Models** — Bhanu Kaushik, Aman Taneja, Sonika Dahiya
  [GitHub](https://github.com/SqueezeAILab/KVQuant)
  > The rapid development of Large Language Models (LLMs) has brought significant advancements in natural language processing, but their high demands for computational resources, memory, and energy pose s...

- **When Compression Meets Model Compression: Memory-Efficient Double Compression for Large Language Models** — Weilan Wang, Yu Mao, Dongdong Tang, Hongchao Du, Nan Guan
  [arXiv](https://arxiv.org/abs/2502.15443) | [GitHub](https://github.com/containers/ramalama)
  > Large language models (LLMs) exhibit excellent performance in various tasks. However, the memory requirements of LLMs present a great challenge when deploying on memory-limited devices, even for quant...


### Speculative Decoding

- **Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous Vocabularies** — ['Nadav Timor', 'J. Mamou', 'Daniel Korat', 'Moshe Berchansky', 'Oren Pereg', 'Gaurav Jain', 'Roy Schwartz', 'Moshe Wasserblat', 'David Harel']
  [arXiv](https://arxiv.org/abs/2502.05202) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > Accelerating the inference of large language models (LLMs) is a critical challenge in generative AI. Speculative decoding (SD) methods offer substantial efficiency gains by generating multiple tokens ...

- **AdaServe: Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding** — ['Zikun Li', 'Zhuofu Chen', 'Rémi Delacourt', 'Gabriele Oliaro', 'Zeyu Wang', 'Qinghan Chen', 'Shuhuai Lin', 'April Yang', 'Zhihao Zhang', 'Zhuoming Chen', 'Sean Lai', 'Xinhao Cheng', 'Xupeng Miao', 'Zhihao Jia']
  [arXiv](https://arxiv.org/abs/2501.12162)
  > Modern large language model (LLM) applications exhibit diverse service-level objectives (SLOs), from low-latency requirements in interactive coding assistants to more relaxed constraints in data wrang...

- **CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving** — Dong Liu, Yanxuan Yu
  [arXiv](https://arxiv.org/abs/2512.11920)
  > Large Language Models (LLMs) have revolutionized natural language processing tasks, but their deployment in datacenter environments faces significant challenges due to the massive memory requirements ...

- **Collaborative Speculative Inference for Efficient LLM Inference Serving** — Luyao Gao, Jianchun Liu, Hong-Ze Xu, Liusheng Huang
  [arXiv](https://arxiv.org/abs/2503.10325) | [GitHub](https://github.com/vllm-project/vllm)
  > Speculative inference is a promising paradigm employing small speculative models (SSMs) as drafters to generate draft tokens, which are subsequently verified in parallel by the target large language m...

- **DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding** — ['Jiahong Ning', 'Ce Zheng', 'Tingting Yang']
  [arXiv](https://arxiv.org/abs/2507.12000) | [GitHub](https://github.com/NLPOptimize/flash-tokenizer)
  > Large language models (LLMs) have transformed natural language processing but face critical deployment challenges in device-edge systems due to resource limitations and communication overhead. To addr...

- **EdgeLLM: Fast On-Device LLM Inference With Speculative Decoding** — ['Daliang Xu', 'Wangsong Yin', 'Hao Zhang', 'Xin Jin', 'Ying Zhang', 'Shiyun Wei', 'Mengwei Xu', 'Xuanzhe Liu']
  [GitHub](https://github.com/asprenger/ray_vllm_inference)
  > Generative tasks, such as text generation and question answering, are essential for mobile applications. Given their inherent privacy sensitivity, executing them on devices is demanded. Nowadays, the ...

- **Efficient LLM Inference over Heterogeneous Edge Networks with Speculative Decoding** — ['Bingjie Zhu', 'Zhixiong Chen', 'Liqiang Zhao', 'Hyundong Shin', 'Arumugam Nallanathan']
  [arXiv](https://arxiv.org/abs/2510.11331) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > Large language model (LLM) inference at the network edge is a promising serving paradigm that leverages distributed edge resources to run inference near users and enhance privacy. Existing edge-based ...

- **Entropy-Aware Speculative Decoding Toward Improved LLM Reasoning** — ['Tiancheng Su', 'Meicong Zhang', 'Guoxiu He']
  [arXiv](https://arxiv.org/abs/2512.23765) | [GitHub](https://github.com/psmarter/mini-infer)
  > Speculative decoding (SD) accelerates large language model (LLM) reasoning by using a small draft model to generate candidate tokens, which the target LLM either accepts directly or regenerates upon r...

- **FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference** — ['Xing Liu', 'Lizhuo Luo', 'Ming Tang', 'Chao Huang']
  [arXiv](https://arxiv.org/abs/2507.02620) | [GitHub](https://github.com/psmarter/mini-infer)
  > Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure t...

- **LLMs on a Budget? Say HOLA** — Z. Siddiqui, Jiechao Gao, Ebad Shabbir, M. Azeez, Rafiq Ali
  [arXiv](https://arxiv.org/abs/2506.18952) | [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
  > Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded s...

- **LP-Spec: Leveraging LPDDR PIM for Efficient LLM Mobile Speculative Inference with Architecture-Dataflow Co-Optimization** — Siyuan He, Zhantong Zhu, Yandong He, Tianyu Jia
  [arXiv](https://arxiv.org/abs/2508.07227)
  > LLM inference on mobile devices faces extraneous challenges due to limited memory bandwidth and computational resources. To address these issues, speculative inference and processing-in-memory (PIM) t...

- **Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference** — ['Nikhil Bhendawade', 'Kumari Nishu', 'Arnav Kundu', 'Chris Bartels', 'Minsik Cho', 'Irina Belousova']
  [arXiv](https://arxiv.org/abs/2510.13161) | [GitHub](https://github.com/psmarter/mini-infer)
  > Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rate...

- **Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters** — Zonghang Li, Tao Li, Wenjiao Feng, Rongxing Xiao, Jianshu She
  [arXiv](https://arxiv.org/abs/2504.08791) | [GitHub](https://github.com/friendliai/friendli-client)
  > On-device inference offers privacy, offline use, and instant response, but consumer hardware restricts large language models (LLMs) to low throughput and capability. To overcome this challenge, we pre...

- **Quantize-Sample-and-Verify: LLM Acceleration via Adaptive Edge-Cloud Speculative Decoding** — ['Guangyi Zhang', 'Yunlong Cai', 'Guanding Yu', 'P. Popovski', 'Osvaldo Simeone']
  [arXiv](https://arxiv.org/abs/2507.00605) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > In edge-cloud speculative decoding (SD), edge devices equipped with small language models (SLMs) generate draft tokens that are verified by large language models (LLMs) in the cloud. A key bottleneck ...

- **ReSpec: Towards Optimizing Speculative Decoding in Reinforcement Learning Systems** — Qiaoling Chen, Zijun Liu, Peng Sun, Shenggui Li, Guoteng Wang, Ziming Liu, Yonggang Wen, Siyuan Feng, Tianwei Zhang
  [arXiv](https://arxiv.org/abs/2510.26475)
  > Adapting large language models (LLMs) via reinforcement learning (RL) is often bottlenecked by the generation stage, which can consume over 75% of the training time. Speculative decoding (SD) accelera...

- **Reward-Guided Speculative Decoding for Efficient LLM Reasoning** — ['Baohao Liao', 'Yuhui Xu', 'Hanze Dong', 'Junnan Li', 'C. Monz', 'Silvio Savarese', 'Doyen Sahoo', 'Caiming Xiong']
  [arXiv](https://arxiv.org/abs/2501.19324) | [GitHub](https://github.com/psmarter/mini-infer)
  > We introduce Reward-Guided Speculative Decoding (RSD), a novel framework aimed at improving the efficiency of inference in large language models (LLMs). RSD synergistically combines a lightweight draf...

- **SIMPLE: Disaggregating Sampling from GPU Inference into a Decision Plane for Faster Distributed LLM Serving** — Bohan Zhao, Zane Cao, Yongchao He
  [arXiv](https://arxiv.org/abs/2512.00719) | [GitHub](https://github.com/containers/ramalama)
  > As large language models (LLMs) scale out with tensor parallelism (TP) and pipeline parallelism (PP) and production stacks have aggressively optimized the data plane (attention/GEMM and KV cache), sam...

- **SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration** — (待补充)
  [GitHub](https://github.com/hemingkx/SWIFT)
  > SWIFT presents an on-the-fly self-speculative decoding approach that allows a single LLM to generate speculative drafts without requiring a separate draft model, reducing latency while maintaining out...

- **Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling** — ['Shengyin Sun', 'Yiming Li', 'Xing Li', 'Yingzhao Lian', 'Weizhe Lin', 'Hui-Ling Zhen', 'Zhiyuan Yang', 'Chen Chen', 'Xianzhi Yu', 'Mingxuan Yuan', 'Chen Ma']
  [arXiv](https://arxiv.org/abs/2509.04474)
  > Test-time scaling has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs) by allocating additional computational resources during inference. However...

- **SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding** — Shenggui Li, Chao Wang, Yikai Zhu, Yubo Wang, Fan Yin, Shuai Shi, Yefei Chen, Xiaomin Dong, Qiaoling Chen, Jin Pan, Ji Li, Laixin Xie, Yineng Zhang, Lei Yu, Yonggang Wen, Ivor Tsang, Tianwei Zhang
  [GitHub](https://github.com/SpecForge)

- **Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput** — ['Jingwei Song', 'Wanyi Chen', 'Xinyuan Song', 'Chris Tong', 'Gufeng Chen', 'Tianyi Zhao', 'Eric Yang', 'Bill Shi', 'Lynn Ai', 'Gradient Network']
  [arXiv](https://arxiv.org/abs/2511.11733) | [GitHub](https://github.com/sgl-project/sglang)
  > Speculative decoding accelerates large language model (LLM) inference by using a lightweight draft model to propose tokens that are later verified by a stronger target model. While effective in centra...

- **SwiftSpec: Ultra-Low Latency LLM Decoding by Scaling Asynchronous Speculative Decoding** — ['Ziyi Zhang', 'Ziheng Jiang', 'Chengquan Jiang', 'Menghan Yu', 'Size Zheng', 'Haibin Lin', 'Henry Hoffmann', 'Xin Liu']
  [arXiv](https://arxiv.org/abs/2506.11309)
  > Low-latency decoding for large language models (LLMs) is crucial for applications like chatbots and code assistants, yet generating long outputs remains slow in single-query settings. Prior work on sp...

- **Towards Efficient LLM Inference via Collective and Adaptive Speculative Decoding** — ['Siqi Wang', 'Hailong Yang', 'Xuezhu Wang', 'Tongxuan Liu', 'Pengbo Wang', 'Yufan Xu', 'Xuning Liang', 'Kejie Ma', 'Tianyu Feng', 'Xin You', 'Ruihao Gong', 'Rui Wang', 'Zhongzhi Luan', 'Yi Liu', 'Depei Qian']
  [GitHub](https://github.com/kvcache-ai/Mooncake)
  > Large language models (LLMs) have gained considerable attention for their remarkable performance across a wide range of tasks. However, efficient LLM inference remains challenging because of the autor...

- **Tutorial Proposal: Speculative Decoding for Efficient LLM Inference** — ['Heming Xia', 'Cunxiao Du', 'Yongqing Li', 'Qian Liu', 'Wenjie Li']
  [arXiv](https://arxiv.org/abs/2503.00491) | [GitHub](https://github.com/psmarter/mini-infer)
  > This tutorial presents a comprehensive introduction to Speculative Decoding (SD), an advanced technique for LLM inference acceleration that has garnered significant research interest in recent years. ...

- **Utility-Driven Speculative Decoding for Mixture-of-Experts** — Anish Saxena, Po-An Tsai, Hritvik Taneja, Aamer Jaleel, Moinuddin Qureshi
  [arXiv](https://arxiv.org/abs/2506.20675)
  > GPU memory bandwidth is the main bottleneck for low-latency Large Language Model (LLM) inference. Speculative decoding leverages idle GPU compute by using a lightweight drafter to propose K tokens, wh...


## 2024


### KV Cache

- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** — (待补充)
  [GitHub](https://github.com/jy-yuan/KIVI)
  > KV cache quantization is crucial for reducing memory footprint in LLM inference. This paper presents KIVI, a tuning-free asymmetric 2bit quantization method for KV cache that achieves minimal accuracy...

- **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** — (待补充)
  [GitHub](https://github.com/SqueezeAILab/KVQuant)
  > This paper presents KVQuant, a KV cache quantization method that enables LLM inference with context lengths up to 10 million tokens. The method uses per-channel scaling and asymmetric quantization to ...


### LLM Serving

- **Lookahead Decoding: Break the Sequential Dependency of LLM Inference** — (待补充)
  [GitHub](https://github.com/hao-ai-lab/LookaheadDecoding)
  > Large Language Models (LLMs) generate tokens auto-regressively, which creates sequential dependency that limits parallelization during inference. This paper presents Lookahead Decoding, a technique th...

- **PowerInfer-2: High-Speed LLM Inference for Smartphones** — 来自上海交通大学IPADS实验室
  [GitHub](https://github.com/Tiiny-AI/PowerInfer)

- **Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference** — (待补充)
  [GitHub](https://github.com/mit-han-lab/Quest)
  > Long-context LLM inference requires significant memory and compute resources. Quest proposes a query-aware sparsity approach that selectively attends to relevant context, reducing computation while ma...

- **vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving** — Jiale Xu, Rui Zhang, Cong Guo, Weiming Hu, Zihan Liu, Feiyang Wu, Yu Feng, Shixuan Sun, Changxu Shao, Yuhong Guo, Junping Zhao, Ke Zhang, Minyi Guo, Jingwen Leng
  [arXiv](https://arxiv.org/abs/2407.15309)
  > Large Language Models (LLMs) are widely used across various domains, processing millions of daily requests. This surge in demand poses significant challenges in optimizing throughput and latency while...


### Prefill/Disaggregation

- **P/D-Serve: Serving Disaggregated Large Language Model at Scale** — Yibo Jin, Tao Wang, Huimin Lin, Mingyang Song, Peiyang Li, Yipeng Ma, Yicheng Shan, Zhengfan Yuan, Cailong Li, Yajing Sun, Tiandeng Wu, Xing Chu, Ruizhi Huan, Li Ma, Xiao You, Wenting Zhou, Yunpeng Ye, Wen Liu, Xiangkun Xu, Yongsheng Zhang, Tiantian Dong, Jiawei Zhu, Zhe Wang, Xijian Ju, Jianxun Song 等
  [arXiv](https://arxiv.org/abs/2408.08147)
  > Serving disaggregated large language models (LLMs) over tens of thousands of xPU devices (GPUs or NPUs) with reliable performance faces multiple challenges. 1) Ignoring the diversity (various prefixes...


### Speculative Decoding

- **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding** — (待补充)
  [GitHub](https://github.com/facebookresearch/LayerSkip)
  > LayerSkip combines early exit inference with self-speculative decoding, allowing LLMs to dynamically skip layers during inference based on sample difficulty, while using the same model for both draft ...

- **TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding** — (待补充)
  [GitHub](https://github.com/Infini-AI-Lab/TriForce)
  > TriForce presents a hierarchical speculative decoding approach that uses multiple levels of draft models to achieve lossless acceleration of long sequence generation, addressing the verification bottl...


## 2023


### Inference Kernel

- **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention** — Unknown
  [arXiv](https://arxiv.org/abs/2309.06180) | [GitHub](https://github.com/vllm-project/vllm)
  > High throughput serving of large language models (LLMs) requires batching sufficiently many requests at a time. However, existing systems struggle because the key-value cache (KV cache) memory for eac...


### KV Cache

- **LLMLingua: Prompt Compression for LLM Inference** — Unknown
  [arXiv](https://arxiv.org/abs/2310.05736) | [GitHub](https://github.com/microsoft/LLMLingua)
  > Large language models (LLMs) have been applied in various applications due to their astonishing capabilities. With advancements in technologies such as chain-of-thought (CoT) prompting and in-context ...
