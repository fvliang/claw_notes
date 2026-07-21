# LLM Serving Papers Index

_Generated: 2026-07-22 06:17_


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

- **Attention Distribution-Aware Softmax for NPU-Accelerated On-Device Inference of LLMs: An Edge-Oriented Approximation Design** — Sanoop Sadheerthan, Min-Jie Hsu, Chihan Huang, Yin-Tien Wang
  > Low-power NPUs enable on-device LLM inference through efficient integer and fixed-point algebra, yet their lack of native exponential support makes Transformer softmax a critical performance bottlenec...

- **Balancing Latency and Accuracy of Code Completion via Local-Cloud Model Cascading** — Multiple Authors
  > We present a model cascading framework for code completion that balances latency and accuracy by routing queries between local small models and cloud large models. The framework uses confidence-based ...

- **Black-Box Skill Stealing Attack from Proprietary LLM Agents: An Empirical Study** — Zihan Wang, Rui Zhang, Yu Liu, Chi Liu, Qingchuan Zhao
  [arXiv](https://arxiv.org/abs/2604.21829v1)
  > LLM agents increasingly rely on skills to encapsulate reusable capabilities via progressively disclosed instructions. High-quality skills inject expert knowledge into general-purpose models, improving...

- **Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices** — Multiple Authors
  > We present a compiler-assisted speculative sampling framework that optimizes draft model deployment on heterogeneous edge devices with varying compute capabilities. The compiler automatically partitio...

- **ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge--Cloud Speculative LLM Serving** — Xiangchen Li, Saeid Ghafouri, Jiakun Fan, Babar Ali, Hans Vandierendonck, Dimitrios S. Nikolopoulos
  > Speculative decoding enables collaborative Large Language Model (LLM) inference across cloud and edge by separating lightweight token drafting from heavyweight verification. While prior work focuses o...

- **ConsRoute: Consistency-Aware Adaptive Query Routing for Cloud-Edge-Device LLMs** — Haoyu Qiao, Hao Zhang, Shanwen Mao, Siyao (待完善)
  > > Large Language Models (LLMs) are equipped with profound semantic knowledge, making them a natural choice for injecting semantic generalization into personalized search systems......

- **EdgeFlow: Fast Cold Starts for LLMs on Mobile Devices** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2604.09083)
  > EdgeFlow targets fast cold starts for LLMs on mobile devices, addressing the challenge of deploying large language models on resource-constrained edge devices by optimizing initialization and inferenc...

- **HaS: Accelerating RAG through Homology-Aware Speculative Retrieval** — Peng Peng, Weiwei Lin, Wentai Wu, Xinyang Wang, Yongheng Liu
  [arXiv](https://arxiv.org/abs/2604.20452v1)
  > Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) at inference by retrieving external documents as context. However, retrieval becomes increasingly ti...

- **Privacy-Aware Split Inference with Speculative Decoding for Large Language Models over Wide-Area Networks** — Multiple Authors
  > We propose a privacy-aware split inference framework that combines speculative decoding with model splitting across wide-area networks. The framework ensures sensitive user data remains on local edge ...

- **WANSpec: Leveraging Global Compute Capacity for LLM Inference** — Multiple Authors
  > We present WANSpec, a system that leverages geographically distributed compute resources for speculative LLM inference over wide-area networks. WANSpec places draft models on edge or regional compute ...


### Edge LLM Serving

- **Unlocking the Edge Deployment and On-Device Acceleration of Multi-LoRA Enabled One-for-All Foundational LLM** — Sravanth Kodavanti, Sowmya Vajrala, Srinivas Miriyala
  [arXiv](https://arxiv.org/abs/2604.18655)
  > Deploying large language models (LLMs) on smartphones poses significant engineering challenges due to stringent constraints on memory, latency, and runtime. This paper explores edge deployment and on-...


### Hardware Acceleration

- **A Full-Stack Performance Evaluation Infrastructure for 3D-DRAM-based LLM Accelerators** — Cong Li, Chenhao Xue, Yi Ren, Xiping Dong, Yu Cheng, Yinbo Hu, Fujun Bai, Yixin Guo, Xiping Jiang, Qiang Wu, Zhi Yang, Zhe Cheng, Yuan Xie, Guangyu Sun
  [arXiv](https://arxiv.org/abs/2604.08044)
  > Large language models (LLMs) exhibit memory-intensive behavior during decoding, making it a key bottleneck in LLM inference. To accelerate decoding execution, hybrid-bonding-based 3D-DRAM has been ado...

- **A-IO: Adaptive Inference Orchestration for Memory-Bound NPUs** — Multiple Authors
  > We present A-IO, an adaptive inference orchestration system for LLM inference on Neural Processing Units (NPUs) that are memory-bound. A-IO dynamically adjusts memory management, computation schedulin...

- **Ouroboros: Wafer-Scale SRAM CIM with Token-Grained Pipelining for Large Language Model Inference** — Yiqi Liu, Cheng Liu, Zhen Gu, Tianchen Ding, Zongyue Zhao, Ziyu Yang, Yufei Ding, Yibo Lin, Mingjie Lin, Xiaowei Li, Zidong Du, Chen Liu, Yunji Chen
  [arXiv](https://arxiv.org/abs/2603.02737)
  > Conventional LLM inference architectures suffer from high energy and latency due to frequent data movement across memory hierarchies. We propose Ouroboros, a wafer-scale SRAM-based Computing-in-Memory...

- **ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression** — Ruibo Fan, Xiangrui Yu, Xinglin Pan, Zeyu Li, Weile Luo, Qiang Wang, Wei Wang, Xiaowen Chu
  > ZipServ提出了一种硬件感知的无损压缩方法，用于加速LLM推理并降低内存占用。该方法针对现代GPU架构进行了优化，实现了显著的性能提升。...


### Inference Kernel

- **CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference** — Chuxu Song, Zhencan Peng, Jiuqi Wei, Chuanhui Yang
  > Long-context LLMs increasingly rely on extended, reusable prefill prompts for agents and domain Q&A, pushing attention and KV-cache to become the dominant decode-time bottlenecks. While sparse attenti...

- **FastTree: Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference** — Zaifeng Pan, Yitong Ding, Yue Guan, Zheng Wang, Zhongkai Yu
  [GitHub](https://github.com/aerlabsAI/ai-inference-resources)

- **NOSA: Native and Offloadable Sparse Attention** — Yuxiang Huang, Pengjie Wang, Jicheng Han, Weilin Zhao, Zhou Su, Ao Sun, Hongya Lyu, Hengyu Zhao, Yudong Wang, Chaojun Xiao, Xu Han, Zhiyuan Liu
  > Decoding throughput improvements from larger batch sizes are limited by the quadratic complexity of attention. NOSA proposes a native and offloadable sparse attention mechanism that reduces computatio...

- **PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel** — Jinjun Yi, Zhixin Zhao, Yitao Hu, Ke Yan, Weiwei Sun, Hao Wang, Laiping Zhao, Yuhao Zhang, Wenxin Li, Keqiu Li
  > PAT（前缀感知注意力）提出了一种新的注意力机制优化，通过资源高效的多tile内核加速LLM解码。该方法显著降低了注意力计算的内存和计算开销。...

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

- **Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving** — Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen
  > Existing LLM serving systems typically configure each instance for worst-case context length, leading to substantial KV-cache over-allocation and under-utilized concurrency. In practice, 80-95% of req...

- **From Research Question to Scientific Workflow: Leveraging Agentic AI for Science Automation** — Bartosz Balis, Michal Orzechowski, Piotr Kica, Michal Dygas, Michal Kuszewski
  [arXiv](https://arxiv.org/abs/2604.21910v1) | [GitHub](https://github.com/containers/ramalama)
  > Scientific workflow systems automate execution -- scheduling, fault tolerance, resource management -- but not the semantic translation that precedes it. Scientists still manually convert research ques...

- **Hive: A Multi-Agent Infrastructure for Algorithm- and Task-Level Scaling** — Zizhang Luo, Yuhao Luo, Youwei Xiao, Yansong Xu, Runlin Guo, Yun Liang
  [arXiv](https://arxiv.org/abs/2604.17353)
  > Large language models are increasingly deployed as complex agentic systems that scale with task complexity. While prior work has extensively explored model- and system-level scaling, algorithm- and ta...

- **Rocks Pebbles and Sand: Modality-aware Scheduling for Multimodal Large Language Model Inference** — Konstantinos Papaioannou, Thaleia Dimitra Doudali
  > Multimodal Large Language Models (MLLMs) power platforms like ChatGPT, Gemini, and Copilot, enabling richer interactions with text, images, and videos. These heterogeneous workloads introduce addition...

- **RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment** — Yingfeng Luo, Hongyu Liu, Dingyang Lin, Kaiyan Chang, Chenglong Wang
  [arXiv](https://arxiv.org/abs/2604.22520v1)
  > Large Language Models (LLMs) have achieved remarkable performance in Machine Translation (MT), but deploying them at scale remains prohibitively expensive. A widely adopted remedy is the hybrid system...

- **Thinking with Reasoning Skills: Fewer Tokens, More Accuracy** — Guangxiang Zhao, Qilong Shi, Xusen Xiao, Xiangzheng Zhang, Tong Yang
  [arXiv](https://arxiv.org/abs/2604.21764v1) | [GitHub](https://github.com/dilab-zju/self-speculative-decoding)
  > Reasoning LLMs often spend substantial tokens on long intermediate reasoning traces (e.g., chain-of-thought) when solving new problems. We propose to summarize and store reusable reasoning skills dist...

- **TingIS: Real-time Risk Event Discovery from Noisy Customer Incidents at Enterprise Scale** — Jun Wang, Ziyin Zhang, Rui Wang, Hang Yu, Peng Di
  [arXiv](https://arxiv.org/abs/2604.21889v1)
  > Real-time detection and mitigation of technical anomalies are critical for large-scale cloud-native services, where even minutes of downtime can result in massive financial losses and diminished user ...

- **Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference** — Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu
  > We present token-budget routing, a simple yet effective approach that reduces GPU costs for LLM inference by routing requests to appropriately-sized serving pools. Our theoretical analysis shows that ...

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

- **CacheSolidarity: Preventing Prefix Caching Side Channels in Multi-tenant LLM Serving Systems** — Panagiotis Georgios Pennas, Konstantinos Papaioannou, Marco Guarnieri, Thaleia Dimitra Doudali
  > Large Language Models (LLMs) rely on optimizations like Automatic Prefix Caching (APC) to accelerate inference. APC works by reusing previously computed states for the beginning part of a request (pre...

- **CodeComp: Structural KV Cache Compression for Agentic Coding** — ['Qiujiang Chen', 'Jing Xiong', 'Chenyang Zhao', 'Sidi Yang', 'Ngai Wong']
  [arXiv](https://arxiv.org/abs/2604.10235)

- **Comparative Characterization of KV Cache Management Strategies for LLM Inference** — ['Oteo Mamo', 'Olga Kogiou', 'Hyunjin Yi', 'Weikuan Yu']
  [arXiv](https://arxiv.org/abs/2604.05012)

- **DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing** — Jinyu Guo, Zhihan Zhang, Yutong Li, Jiehui Xie, Md. Tamim Iqbal, Dongshen Han, Lik-Hang Lee, Sung-Ho Bae, Jie Zou, Yang Yang, Chaoning Zhang
  [arXiv](https://arxiv.org/abs/2604.19351)
  > The quadratic computational complexity of the standard attention mechanism constitutes a fundamental bottleneck for large language models in long-context scenarios. While KV cache compression methods ...

- **Don't Waste Bits! Adaptive KV-Cache Quantization for Lightweight On-Device LLMs** — ['Sayed Pedram Haeri Boroujeni', 'Niloufar Mehrabi', 'Patrick Woods', 'Gabriel Hillesheim', 'Abolfazl Razi']
  [arXiv](https://arxiv.org/abs/2604.04722)

- **Graph-Guided Adaptive Channel Elimination for KV Cache Compression (GRACE)** — ['Enwei Tong', 'Yao Zhu', 'Yuanchao Bai', 'Kai Wang', 'Xianming Liu', 'Xiangyang Ji']
  [arXiv](https://arxiv.org/abs/2604.17164)
  > We introduce GRACE (Graph-guided Adaptive Channel Elimination), a novel framework that reframes KV cache compression as a graph-based optimization problem. GRACE models channels as nodes and their int...

- **HillInfer: Efficient Long-Context LLM Inference on the Edge with Hierarchical KV Eviction using Smar** — He Sun, Shinan Liu, Li Li, Mingjun Xiao
  > Deploying Large Language Models (LLMs) on memory-constrained AI Personal Computers (AIPCs) enables low-latency, privacy-preserving AI applications. However, the massive memory footprint of long-contex...

- **ICaRus: Identical Cache Reuse for Efficient Multi Model Inference** — Sunghyeon Woo, Jaeeun Kil, Hoseung Kim, Minsub Kim, Joonghoon Kim, Ahreum Seo, Sungjae Lee, Minjung Jo, Jiwon Ryu, Baeseong Park, Se Jung Kwon, Dongsoo Lee
  > Multi model inference has recently emerged as a prominent paradigm, particularly in the development of agentic AI systems. However, existing systems lack efficient cache sharing mechanisms across diff...

- **KV Cache Offloading for Context-Intensive Tasks** — Andrey Bocharnikov, Ivan Ermakov, Denis Kuznedelev, Vyacheslav Zhdanovskiy, Yegor Yershov
  > With the growing demand for long-context LLMs across a wide range of applications, the key-value (KV) cache has become a critical bottleneck for both latency and memory usage. Recently, KV-cache offlo...

- **KV Cache Optimization Strategies for Scalable and Efficient LLM Inference** — Yichun Xu, Navjot K. Khaira, Tejinder Singh
  > The key-value (KV) cache is a foundational optimization in Transformer-based large language models (LLMs), eliminating redundant recomputation of past token representations during autoregressive gener...

- **KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs** — Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Bing Li, Ulf Schlichtmann
  [arXiv](https://arxiv.org/abs/2604.13226)
  > Large Language Models (LLMs) rely heavily on Key-Value (KV) caching to minimize inference latency. However, standard KV caches are context-dependent: reusing a cached document in a new context require...

- **KVSculpt: KV Cache Compression as Distillation** — ['Bo Jiang', 'Sian Jin']
  [arXiv](https://arxiv.org/abs/2603.27819)

- **LLMLingua: Prompt Compression for LLM Inference** — Unknown
  [GitHub](https://github.com/microsoft/LLMLingua)

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

- **Reducing Peak Memory Usage for Modern Multimodal Large Language Model Pipelines** — Junwan Kim, Hyunkyung Bae
  [arXiv](https://arxiv.org/abs/2604.16734)
  > Multimodal large language models (MLLMs) have recently demonstrated strong capabilities in understanding and generating responses from diverse visual inputs, including high-resolution images and long ...

- **RelayCaching: Accelerating LLM Collaboration via Decoding KV Cache Reuse** — Yingsheng Geng, Yuchong Gao, Weihong Wu, Guyue Liu, Jiang Liu
  > The increasing complexity of AI tasks has shifted the paradigm from monolithic models toward multi-agent large language model (LLM) systems. However, these collaborative architectures introduce a crit...

- **SAW-INT4: System-Aware 4-Bit KV-Cache Quantization for Real-World LLM Serving** — Jinda Jia, Jisen Li, Zhongzhu Zhou, Jung Hwan Heo, Jue Wang, Tri Dao, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, Tianyi Zhang, Xiaoxia Wu
  [arXiv](https://arxiv.org/abs/2604.19157)
  > KV-cache memory is a major bottleneck in real-world LLM serving, where systems must simultaneously support latency-sensitive small-batch requests and high-throughput concurrent workloads. Although man...

- **ScoutAttention: Efficient KV Cache Offloading via Layer-Ahead CPU Pre-computation for LLM Inference** — ['Qiuyang Zhang', 'Kai Zhou', 'Ding Tang', 'Kai Lu', 'Cheng Li', 'Zhenyu Yang']
  [arXiv](https://arxiv.org/abs/2603.27138)

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

- **TokenDance: Scaling Multi-Agent LLM Serving via Collective KV Cache Sharing** — Zhuohang Bian, Feiyang Wu, Chengrui Zhang, Hangcheng Dong, Yun Liang, Youwei Zhuo
  > Multi-agent LLM applications organize execution in synchronized rounds where a central scheduler gathers outputs from all agents and redistributes the combined context. This All-Gather communication p...

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
  [GitHub](https://github.com/bytedance/InfiniStore)
  > KV cache store for distributed LLM inference...

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
  [GitHub](https://github.com/NVIDIA/kvpress)
  > LLM KV cache compression made easy...

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

- **A Generative Partially Specified Finite State Machine Approach to Complex Behaviour Planning** — Kalana Ratnayake, Michael Pritchard, David Hinwood, Maleen Jayasuriya, Damith Herath
  [arXiv](https://arxiv.org/abs/2607.15674v1)
  > Autonomous robots operating in dynamic environments require behaviour planning systems that combine reactivity, interpretability, and adaptability. While Large Language Models have been successfully i...

- **A Hybrid Online and Offline Requests Inference Serving System for LLM in Private Computer Environment** — Yuchen Shen, Yuning Zhang, Dong Yuan
  > While advancements in Large Language Models (LLMs) have broadened their applications, performing multitask LLM inference on a single GPU remains challenging due to insufficient GPU memory to load all ...

- **A Metamorphic Testing Approach to Diagnosing Memorization in LLM-Based Program Repair** — Milan De Koning, Ali Asgari, Pouria Derakhshanfar, Annibale Panichella
  [arXiv](https://arxiv.org/abs/2604.21579v1)
  > LLM-based automated program repair (APR) techniques have shown promising results in reducing debugging costs. However, prior results can be affected by data leakage: large language models (LLMs) may m...

- **AE-LLM: Adaptive Efficiency Optimization for Large Language Models** — Kaito Tanaka, Masato Ito, Yuji Nishimura, Keisuke Matsuda, Aya Nakayama
  > Large Language Models (LLMs) have achieved remarkable success across diverse applications, yet their deployment remains challenging due to substantial computational costs, memory requirements, and ene...

- **AGG: Jacobian-Aggregated Group Gradient for Efficient GRPO Training of Diffusion Models** — Ruiyi Ding, Jie Li, He Kang, Ziyan Liu, Chengru Song, Yuan chen
  [arXiv](https://arxiv.org/abs/2607.17572v1)
  > Group Relative Policy Optimization (GRPO) is a powerful reinforcement learning algorithm for aligning generative models with human preferences. While successful in large language models~\cite{shao2024...

- **AIGB-R1: Self-Evolving Generative Auto-Bidding via Hierarchical Planner-Executor Optimization** — Yuejia Dou, Hesong Wang, Xinyu Zhang, Tianyu Wang, Zhilin Zhang, Chuan Yu, Jian Xu, Bo Zheng, Qi Qi
  [arXiv](https://arxiv.org/abs/2607.17281v1)
  > Auto-bidding plays an essential role in online advertising, automatically adjusting bids for advertisers to optimize their commercial goals. The emerging AI-Generated Bidding (AIGB) paradigm widely ad...

- **ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants** — Haohui Mai, Xiaoyan Guo, Xiangyun Ding, Daifeng Li, Qiuchu Yu, Chenzhun Guo, Cong Wang, Jiacheng Zhao, Christos Kozyrakis, Binhang Yuan
  [arXiv](https://arxiv.org/abs/2604.18616)
  > LLM-based coding agents can generate functionally correct GPU kernels, yet their performance remains far below hand-optimized libraries on critical computations such as matrix multiplication, attentio...

- **Accelerating Speculative Decoding with Block Diffusion Draft Trees** — Liran Ringel
  [arXiv](https://arxiv.org/abs/2604.12989)
  > Speculative decoding accelerates autoregressive language models by using a lightweight drafter to propose multiple future tokens, which the target model then verifies in parallel. DFlash shows that a ...

- **Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving** — Takeshi Yoshimura, Valentijn Dymphnus van de Beek, Tatsuhiro Chiba
  [arXiv](https://arxiv.org/abs/2604.15732)
  > Distributed LLM serving systems optimize per-request latency and throughput. However, under long-context workloads, inference accuracy becomes more variable. When incorrect responses trigger retries, ...

- **AdaGen: Workload-Adaptive Cluster Scheduler for Latency-Optimal LLM Inference Serving** — Sudipta Saha Shubha, Ayush Goel, D. Z. Tootaghaj, Khaled Diab, Hardik Soni

- **AdaHome: An Adaptive Smart Home Assistant using Local Small Language Models** — Eu Jin Lim, Zhaoxing Li, Sebastian Stein
  [arXiv](https://arxiv.org/abs/2607.18034v1)
  > Smart home assistants interpret a wide range of user commands, from explicit device control to underspecified and preference dependent requests. While recent systems based on Large Language Models (LL...

- **AdaSpec: Adaptive Multilingual Speculative Decoding with Self-Synthesized Language-Aware Training and Vocabulary Simplification** — Dinh-Truong Do, Nguyen-Khang Le, Le-Minh Nguyen
  > Speculative decoding accelerates large language model (LLM) inference by using a lightweight drafter to propose multiple tokens, which are then verified in parallel by the base model. While effective ...

- **Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration** — Zhuofan Wen
  [arXiv](https://arxiv.org/abs/2604.12247)
  > Speculative decoding has emerged as a promising approach to accelerate autoregressive inference in LLMs. Self-draft methods leverage the base LLM itself for speculation, but shallow layers often produ...

- **Adelia: A 4-nm LLM Processing Unit With Streamlined Dataflow and Dual-Mode Parallelism for Maximizing Hardware Efficiency** — Sukbin Lim, Jung-Hoon Kim, Seungjae Moon, Junseo Cha, Dongjin Seo, Jongho Kim, Hunjong Lee, Jinwon Lee, Joo-Young Kim
  > The proliferation of large language models (LLMs) as cross-domain foundation models is fueled by aggressive scaling in both parameter counts and inference-time computation. The emergence of sophistica...

- **AgentServe: Algorithm-System Co-Design for Efficient Agentic AI Serving on a Consumer-Grade GPU** — Yuning Zhang et al.
  [arXiv](https://arxiv.org/abs/2603.10342)
  > AgentServe presents a single-GPU serving system that ensures stable multi-agent execution by isolating prefills from decodes, applying dynamic budgeting to resume prefills, and allocating GPU resource...

- **An MLIR-Based Compilation Method for Large Language Models** — Pengchao Hu, Zhibin Xin, Yifan Chen, Yangyang Zhou, Liang Wang
  [arXiv](https://arxiv.org/abs/2607.15865v1)
  > Large Language Models (LLMs) have become the dominant workload on modern AI accelerators, yet deploying them on specialized hardware still faces two core challenges: how to import a trained model into...

- **AnovaX: A Local, Multi-Agent Voice Assistant with LLM Planning, Typed Executors, and Adaptive Recovery** — Raunak B Sinha
  [arXiv](https://arxiv.org/abs/2607.15367v1)
  > Desktop voice assistants are still dominated by cloud pipelines that ship raw audio off the machine and expose a fixed set of skills. We describe AnovaX, a small local-first assistant that runs entire...

- **AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention** — Yuxuan Hu, Jianchao Tan, Jiaqi Zhang, Wen Zan, Pingwei Sun, Yifan Lu, Yerui Sun, Yuchen Xie, Xunliang Cai, Jing Zhang
  [arXiv](https://arxiv.org/abs/2604.07815)
  > Long-context inference in LLMs faces quadratic attention complexity and prohibitive KV cache memory. AsyncTLS proposes a hierarchical sparse attention system combining coarse-grained block filtering w...

- **Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics** — Youhe Jiang, Ran Yan, You Peng, Wenshuang Li, Taiyi Wang, Fangcheng Fu, Binhang Yuan
  > Modern Large Language Model (LLM) serving systems face highly dynamic runtime conditions including fluctuating request rates, varying input lengths, and changing hardware availability. Existing system...

- **B-PASTE: Beam-Aware Pattern-Guided Speculative Execution for Resource-Constrained LLM Agents** — Yanfei Song
  [arXiv](https://arxiv.org/abs/2604.16469)
  > LLM agents execute in an interleaved reasoning-and-action loop, where future tool calls cannot be launched until the current reasoning step completes. This serial dependency inflates end-to-end latenc...

- **Benchmarking Compound AI Applications for Hardware-Software Co-Design** — Paramuth Samuthrsindh, Angel Cervantes, Varun Gohil, Gohar Irfan Chaudhry, Christina Delimitrou, Adam Belay
  [arXiv](https://arxiv.org/abs/2604.09593)
  > Compound AI applications, composed from interactions between Large Language Models (LLMs), Machine Learning (ML) models, external tools and data sources are quickly becoming an integral workload in da...

- **Beyond Test-Time Compute Strategies: Advocating Energy-per-Token in LLM Inference** — Patrick Wilhelm, Thorsten Wittkopp, Odej Kao
  > Large Language Models (LLMs) demonstrate exceptional performance across diverse tasks but come with substantial energy and computational costs, particularly in request-heavy scenarios. In many real-wo...

- **Bit-Serial Acceleration of LLM Inference With Mixture-of-Datatype Quantization** — Yuzong Chen, Chi-Chih Chang, Xilai Dai, Ahmed Abouelhamayed, Marta Andronic, George A. Constantinides, Mohamed S. Abdelfattah
  > Large language models (LLMs) have achieved significant breakthroughs on machine learning tasks. Yet the substantial memory footprint of LLMs significantly hinders their wide deployment. In this paper,...

- **BlendServe: Optimizing Offline Inference with Resource-Aware Batching** — Yilong Zhao, Shuo Yang, Kan Zhu, Lianmin Zheng, Baris Kasikci, Yifan Qiao, Yang Zhou, Jiarong Xing, Ion Stoica
  > BlendServe是UC Berkeley离子·斯托伊卡团队提出的离线推理优化系统，通过资源感知的批处理策略最大化离线场景下的推理效率。...

- **Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC** — Mohammad Siavashi, Mariano Scazzariello, Gerald Q. Maguire Jr., Dejan Kostić, Marco Chiesa
  > Large Language Model (LLM) inference is rapidly becoming a core datacenter service, yet current serving systems heavily rely on the CPU for orchestrating the inference pipeline. We present Blink, a sy...

- **Break the Optimization Barrier of LLM-Enhanced Recommenders: A Theoretical Analysis and Practical Framework** — Zhangchi Zhu, Wei Zhang
  [arXiv](https://arxiv.org/abs/2604.20490v1) | [GitHub](https://github.com/kvcache-ai/Mooncake)
  > Large language model (LLM)-enhanced recommendation models inject LLM representations into backbone recommenders to exploit rich item text without inference-time LLM cost. However, we find that existin...

- **Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration** — Zejia Lin, Hongxin Xu, Guanyi Chen, Zhiguang Chen, Yutong Lu, Xianwei Zhang
  > 本文提出Bullet系统，通过动态空间-时间协调来提升LLM serving的GPU利用率。传统的LLM serving系统存在GPU计算资源浪费的问题，Bullet通过创新的调度策略实现了更高效的GPU资源利用。...

- **C$^2$KV: Compressed and Composable KV Cache Reuse for Efficient LLM Inference** — Chuheng Du, Junyi Chen, Hanlin Tang, Kan Liu, Tao Lan, Lin Qu, Chaoyue Niu, Shengzhong Liu, Guihai Chen, Fan Wu
  [arXiv](https://arxiv.org/abs/2607.17715v1)
  > Long-context inference is central to modern large language model (LLM) applications such as retrieval-augmented generation and multi-document reasoning. To mitigate the growing inference cost, recent ...

- **CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands** — Weiye Wang, Chen Chen, Junxue Zhang, Zhusheng Wang, Hui Yuan, Zixuan Guan, Xiaolong Zheng, Qizhen Weng, Yin Chen, Minyi Guo
  [arXiv](https://arxiv.org/abs/2603.21257)
  > Distributed prefix caching has become a core technique for efficient LLM serving. However, for long-context requests with high cache hit ratios, retrieving reusable KVCache blocks from remote servers ...

- **CCCL: In-GPU Compression-Coupled Collective Communication** — Chon Lam Lao, Zhiying Xu, Zhuang Wang, Ziming Mao, Delong Meng, Jia Zhen, Jun Wu, Ion Stoica, Yida Wang, Yang Zhou
  [arXiv](https://arxiv.org/abs/2604.17172)
  > Collective communication incurs significant overhead in LLM workloads. Although overlapping communication with computation in application-level is a common strategy, it often requires substantial code...

- **CHESS: Context-aware Hierarchical Efficient Semantic Selection for Long-Context LLM Inference** — Chao Fei, Guozhong Li, Chenxi Liu, Panos Kalnis
  > Long-context LLMs demand accurate and efficient context selection mechanisms. We propose CHESS, a hierarchical semantic selection framework that efficiently identifies the most relevant context segmen...

- **Cache-Aware Prompt Compression:A Two-Tier Cost Model for LLM API Caching** — Yan Song
  [arXiv](https://arxiv.org/abs/2607.15516v1)
  > Production LLM deployments combine two cost-reduction primitives: prompt caching (a discounted rate for re-used token prefixes) and prompt compression (fewer tokens sent). The compression literature h...

- **Can We Break LLMs Out of Self-Loops? Fine-Grained Reasoning Control with Activation Steering** — Sheldon Yu, Tong Yu, Xunyi Jiang, Rohan Surana, Gagan Mundada, Sungchul Kim, Lina Yao, Julian McAuley, Junda Wu
  [arXiv](https://arxiv.org/abs/2607.18100v1)
  > Extended reasoning has become standard for frontier Large Language Models (LLMs), yet the trajectories these models produce remain largely uncontrollable. Existing methods for shaping how a model reas...

- **Characterizing CPU-Induced Slowdowns in Multi-GPU LLM Inference** — Euijun Chung, Yuxiao Jia, Aaron Jezghani, Hyesoon Kim
  > > ...increasingly rely on multi-GPU systems, yet their performance is often limited by an overlooked component: the CPU. Through a detailed study of modern large language model (LLM)......

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

- **CodecSight: Leveraging Video Codec Signals for Efficient Streaming VLM Inference** — Yulin Zou, Yan Chen, Wenyan Chen, JooYoung Park, Shivaraman Nitin, Luo Tao, Francisco Romero, Dmitrii Ustiugov
  > Video streaming analytics is a crucial workload for vision-language model serving, but the high cost of multimodal token generation creates significant inference overhead. We present CodecSight, a sys...

- **Communication-Efficient Collaborative LLM Inference over LEO Satellite Networks** — ['Songge Zhang', 'Wen Wu', 'Liang Li', 'Ye Wang', 'Xuemin', 'Shen']
  [arXiv](https://arxiv.org/abs/2604.04654)

- **ContinuityBench: A Benchmark and Systems Study of Stateful Failover in Multi-Provider LLM Routing** — Vishal Pandey, Gopal Singh
  [arXiv](https://arxiv.org/abs/2607.15899v1)
  > In production large language model (LLM) deployments, high API availability guarantees do not equate to conversational continuity. When a primary provider experiences an outage or strict rate-limiting...

- **Continuous Semantic Caching for Low-Cost LLM Serving** — Baran Atalar, Xutong Liu, Jinhang Zuo, Siwei Wang, Wei Chen, Carlee Joe-Wong
  [arXiv](https://arxiv.org/abs/2604.15873)
  > [Abstract待从arxiv页面获取]...

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

- **DFVG: A Heterogeneous Architecture for Speculative Decoding with Draft-on-FPGA and Verify-on-GPU** — Shaoqiang Lu, Yangbo Wei, Junhong Qian, Dongge Qin, Shiji Gao, Yizhi Ding, Qifan Wang, Chen Wu, Xiao Shi, Lei He
  [GitHub](https://github.com/ShaoqiangLu/DFVG)
  > Speculative decoding is a promising paradigm that accelerates LLM inference by generating drafts and performing verification. However, such systems still face three major challenges: (1) The imbalance...

- **DIAA: A Decoding-Efficient Inference Acceleration Approach for On-Device Large Language Models** — Hao Tian, Sheng Lu, Fuwen Tian, Guangming Cui, Zheng Li, Xuyun Zhang, Quan Z. Sheng, Wanchun Dou
  > Large Language Models (LLMs) have revolutionized intelligent interactions, enabling mobile applications such as personal assistants on edge devices for local execution. Speculative decoding (SD) has e...

- **DUET: Disaggregated Hybrid Mamba-Transformer LLMs with Prefill and Decode-Specific Packages** — Alish Kanani, Sangwan Lee, Han Lyu, Jiahao Lin, Jaehyun Park, Umit Y. Ogras
  [arXiv](https://arxiv.org/abs/2603.15530)
  > DUET introduces a disaggregated accelerator that assigns prefill and decode phases to specialized packages. The Prefill package utilizes systolic array chiplets with off-package memory. The Decode pac...

- **DWDP: Distributed Weight Data Parallelism for High-Performance LLM Inference on NVL72** — Wanqian Li, Jintao Peng, Zongfei Jing, Tianyu Zhang, Ze Long, Xianjie Qiao, Xiaoming Chen, Dongxu Yang, Kefeng Duan, June Yang
  > ### English Large language model (LLM) inference increasingly depends on multi-GPU execution, yet existing inference parallelization strategies require layer-wise inter-rank synchronization, making en...

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

- **DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving** — Ying Yuan, Pengfei Zuo, Bo Wang, Zhangyu Chen, Zhipeng Tan, Zhou Yu
  [arXiv](https://arxiv.org/abs/2602.06502)
  > In LLM serving, reusing the KV cache of prompts across requests is critical for reducing TTFT and serving costs. Cache-affinity scheduling, which co-locates requests with the same prompt prefix to max...

- **DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference** — Yongtong Wu, Shaoyuan Chen, Yinmin Zhong, Rilin Huang, Yixuan Tan, Wentao Zhang, Liyue Zhang, Shangyan Zhou, Yuxuan Liu, Shunfeng Zhou, Mingxing Zhang, Xin Jin, Panpan Huang
  > ## 摘要 (中文) 多轮、agentic LLM推理的性能越来越受KV-Cache存储I/O而非计算支配。在流行的解聚架构中，从外部存储加载大量KV-Cache造成了一个根本的不平衡：预填充引擎上的存储NIC变得带宽饱和，而解码引擎上的存储NIC却保持空闲。这种不对称性严重限制了整体系统吞吐量。我们提出了DualPath，这是一种通过引入双路径KV-Cache加载来打破这一瓶颈的推理系统。除了传...

- **DualScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS** — Omar Basit, Yunzhao Liu, Z. Jonny Kong, Y. Charlie Hu
  [arXiv](https://arxiv.org/abs/2602.18755)
  > Prefill/decode disaggregation is increasingly adopted in LLM serving to improve the latency-throughput tradeoff and meet strict TTFT and TPOT SLOs. However, LLM inference remains energy-hungry: autosc...

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

- **Efficient Multi-round LLM Inference over Disaggregated Serving (AMPD)** — Wenhao He, Youhe Jiang, Penghao Zhao, Quanqing Xu, Eiko Yoneki, Bin Cui, Fangcheng Fu
  [arXiv](https://arxiv.org/abs/2602.14516)
  > Multi-round workflows raise hurdles for PD disaggregation — existing systems overlook interleaved prefill-decode workload patterns. AMPD adaptively coordinates prefill workloads based on real-time con...

- **Efficiently Aligning Draft Models via Parameter- and Data-Efficient Adaptation** — Luxi Lin, Zhihang Lin, Zhanpeng Zeng, Yuhao Chen, Qingyu Zhang, Jixiang Luo, Xuelong Li, Rongrong Ji
  [arXiv](https://arxiv.org/abs/2603.09527) | [GitHub](https://github.com/https://github.com/Lyn-Lucy/Efficient-Draft-Adaptation)
  > Speculative decoding accelerates LLM inference but suffers from performance degradation when target models are fine-tuned for specific domains. We introduce EDA (Efficient Draft Adaptation), a paramet...

- **Enhancing Rubric-based RL via Self-Distillation** — Mingxuan Xia, Yuhang Yang, Chao Ye, Shuai Zhu, Shenzhi Yang, Guangcheng Zhu, Yuhang Zhang, Cheng Peng, Haobo Wang, Siqing Wang
  [arXiv](https://arxiv.org/abs/2607.18082v1)
  > Rubric-based RL has recently shown promise in improving LLMs on open-ended tasks. A widely recognized limitation of rubric-based RL is limited exploration: criteria that no rollout manages to satisfy ...

- **Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel** — Hongyi Jin, Bohan Hou, Guanjie Wang, Ruihang Lai, Jinqi Chen, Zihao Ye, Yaxing Cai, Yixin Dong, Xinhao Cheng, Zhihao Zhang, Yilong Zhao, Yingyi Huang, Lijie Yang, Jinchen Jiang, Gabriele Oliaro, Jianan Ji, Xupeng Miao, Vinod Grover, Todd C. Mowry, Zhihao Jia, Tianqi Chen
  [arXiv](https://arxiv.org/abs/2604.13327)
  > Modern GPU workloads, especially large language model (LLM) inference, suffer from kernel launch overheads and coarse synchronization that limit inter-kernel parallelism. Recent megakernel techniques ...

- **Every Microsecond Matters: Achieving Near Speed-of-Light Latency in GPU Collectives** — Siyuan Shen, Anton Korzh, John Bachan, Tiancheng Chen, Arnav Goel, Ludwig Schneider, Pouya Kousha, Zhenhao He, Sylvain Jeaugey, Kamil Iskra, Nishank Chandawala, Jeff R. Hammond, Torsten Hoefler
  [arXiv](https://arxiv.org/abs/2607.16100v1)
  > GPU collective communication is typically optimized for bandwidth, yet many emerging workloads are increasingly limited by latency. Long-context decode-heavy large language model (LLM) inference is a ...

- **ExpertPlex: A High-Goodput Disaggregated Serving System for MoE LLMs with Adaptive Persistent Kernels** — Bingyang Wu, Chao Jin, Zili Zhang, Xinming Wei, Yinmin Zhong, Ruidong Zhu, Chengxu Yang, Xin Jin, Yuliang Liu
  [arXiv](https://arxiv.org/abs/2607.18002v1)
  > LLMs scale Mixture-of-Experts (MoE) parameters for superior intelligence, but massive weights and dynamic computation impede efficient serving. Existing instance-level prefill-decode disaggregation is...

- **FLYING SERVING: On-the-Fly Parallelism Switching for Large Language Model Serving** — Shouwei Gao, Junqi Yin, Feiyi Wang, Wenqian Dong
  > ## 摘要 (中文) 生产级LLM服务必须在非平稳流量和混合请求需求下同时提供高吞吐量、低延迟和足够的上下文容量。数据并行（DP）通过运行独立副本来最大化吞吐量，而张量并行（TP）减少每请求延迟并聚合内存用于长上下文推理。然而，现有服务堆栈通常在部署时静态配置并行性；适应突发、优先级或长上下文请求通常具有破坏性且缓慢。我们提出了Flying Serving，这是一个基于vLLM的系统，可以在不重启...

- **Fast Forward: Accelerating LLM Prefill with Predictive FFN Sparsity** — Aayush Gautam, Mukul Gagrani, Junyoung Park, Mingu Lee, Chiris Lott, Narasimha Reddy
  [arXiv](https://arxiv.org/abs/2602.00397)
  > The prefill stage of large language model (LLM) inference is a key computational bottleneck for long-context workloads. At short-to-moderate context lengths (1K--16K tokens), Feed-Forward Networks (FF...

- **Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference** — Jiaming Cheng, Duong Tung Nguyen
  > Deploying large language model (LLM) services in heterogeneous GPU clusters presents significant challenges in model allocation and resource management. Different GPU types have varying memory capacit...

- **Faster LLM Inference via Sequential Monte Carlo** — Yahya Emara et al.
  [arXiv](https://arxiv.org/abs/2604.15672)
  > Speculative decoding (SD) accelerates language model inference by drafting tokens from a cheap proposal model and verifying them against an expensive target model via rejection sampling. Because rejec...

- **FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications** — Krish Agarwal, Zhuoming Chen, Yanyuan Qin, Zhenyu Gu, Atri Rudra, Beidi Chen
  [arXiv](https://arxiv.org/abs/2607.18171v1)
  > Real-time multimodal applications, including voice agents and interactive video generation, compose heterogeneous models into pipelines whose efficient deployment requires application-specific decisio...

- **Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs** — N/A
  [arXiv](https://arxiv.org/abs/2604.15379)

- **FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design** — Jiahao Zhang, Zifan He, Nicholas Fraser, M. Blott, Yizhou Sun, Jason Cong
  [arXiv](https://arxiv.org/abs/2601.15710)
  > We present FlexLLM, a composable High-Level Synthesis (HLS) library for rapid development of domain-specific LLM accelerators. FlexLLM exposes key architectural degrees of freedom for stage-customized...

- **FlexServe: A Fast and Secure LLM Serving System for Mobile Devices with Flexible Resource Isolation** — Yinpeng Wu, Yitong Chen, Lixiang Wang, Jinyu Gu, Zhichao Hua, Yubin Xia
  > Device-side Large Language Models (LLMs) have witnessed explosive growth, offering higher privacy and availability compared to cloud-side LLMs. During LLM inference, both model weights and user data a...

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

- **ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache** — Shao Wang, Rui Ren, Lin Gui
  [arXiv](https://arxiv.org/abs/2604.06370)
  > The serving paradigm of LLMs is rapidly shifting towards complex multi-agent workflows. While LoRA enables efficient co-hosting of specialized agents, it introduces a critical memory bottleneck — uniq...

- **Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start** — Xueshen Liu, Yongji Wu, Yuncheng Yao, Danyang Zhuo, Ion Stoica, Z. Morley Mao
  > Modern LLM service providers increasingly rely on autoscaling and parallelism reconfiguration to respond to rapidly changing workloads, but cold-start latency remains a critical bottleneck. CUDA graph...

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

- **GPU Acceleration of TFHE-Based High-Precision Nonlinear Layers for Encrypted LLM Inference** — ['Guoci Chen', 'Xiurui Pan', 'Qiao Li', 'Bo Mao', 'Congming Gao', 'Chengying Huan']
  [arXiv](https://arxiv.org/abs/2604.04783)

- **Generalizing Test Cases for Comprehensive Test Scenario Coverage** — Binhang Qi, Yun Lin, Xinyi Weng, Chenyan Liu, Hailong Sun
  [arXiv](https://arxiv.org/abs/2604.21771v1) | [GitHub](https://github.com/LMCache/LMCache)
  > Test cases are essential for software development and maintenance. In practice, developers derive multiple test cases from an implicit pattern based on their understanding of requirements and inferenc...

- **GreenScheduler: Coordinated Two-Tier Energy Optimization for Disaggregated LLM Serving** — Waled Milad Abulgasem Alashheb, Mabruka Khlifa Ali Karkeb, Sabria AbdulGader Ali Elmusrati, Sumia Abdussalam Milad Elagtel
  > Large Language Model (LLM) inference has become a dominant consumer of en- ergy in modern AI data centers, often accounting for over 90% of total operational power [1].Recent architectural shifts towa...

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

- **ITQ3_S: High-Fidelity 3-bit LLM Inference via Interleaved Ternary Quantization with Rotation-Domain Smoothing** — ['Edward J. Yoon']
  [arXiv](https://arxiv.org/abs/2603.27914)

- **IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs** — Yuzhen Mao et al.
  [arXiv](https://arxiv.org/abs/2604.10539)
  > Key-Value (KV) cache plays a crucial role in accelerating inference in large language models (LLMs) by storing intermediate attention states and avoiding redundant computation during autoregressive ge...

- **InnerQ: Hardware-aware Tuning-free Quantization of KV Cache for Large Language Models** — Sayed Mohammadreza Tayaranian Hosseini, Amir Ardakani, W. Gross
  [arXiv](https://arxiv.org/abs/2602.23200)
  > Reducing the hardware footprint of large language models (LLMs) during decoding is critical for efficient long-sequence generation. A key bottleneck is the key-value (KV) cache, whose size scales with...

- **IoUPD: IoU-Aware Privileged Distillation for Visual Grounding with Multimodal Large Language Models** — Xiuyuan Zhu, Ke Lu, Hao Wu, Zijin Du, Dongming Zhang, Jian Xue
  [arXiv](https://arxiv.org/abs/2607.15732v1)
  > Visual grounding with multimodal large language models is commonly formulated as autoregressive coordinate generation, where a model outputs bounding-box coordinates as text given an image and a refer...

- **Kernelized Linear Attention: Breaking the Capacity Wall with Symmetric Cones** — Ayoub Ghriss, Sourav Chakraborty
  [arXiv](https://arxiv.org/abs/2607.17419v1)
  > Linear attention promises constant-time recurrent inference but degrades sharply on associative recall. We formulate attention recall as a spherical-packing problem and introduce Kernelized Linear Att...

- **LAMARS: Large Language Model-Based Anticipation Mechanism Acceleration in Real-Time Robotic Systems** — Yifang Gao, Wei Luo, Xuye Wang, Shunshun Zhang, Patrick Goh
  > Large language models (LLMs) have assumed an increasingly crucial role in robotic systems because of their ability to leverage the extensive knowledge they possess in robotic inference and task handli...

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

- **LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure** — Jaehong Cho, Hyunmin Choi, Guseul Heo, Jongse Park
  [arXiv](https://arxiv.org/abs/2602.23036)
  > Large language model (LLM) serving infrastructures are undergoing a shift toward heterogeneity and disaggregation. Modern deployments increasingly integrate diverse accelerators and near-memory proces...

- **LLMs and Agentic AI Systems for Smart Grids: A Tutorial on Architectures and Applications** — Daniela Rojas, Abdulwahab Albassam, Aidan G. Leung, Jett Ngo, Ryan Luo, Peter R. Quawas, Junpyung Kim, Kangkai Liang, Mansi Nanavati, Jonathan Mai, Meng-Chi Tsai, Yun-Tong Tsai, Yize Chen, Yuanyuan Shi
  [arXiv](https://arxiv.org/abs/2607.18147v1)
  > Large language models (LLMs) and agentic AI systems have evolved from natural language tasks to using external tools to plan, retrieve, and act in technical domains. In smart grids, recent work applie...

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

- **MARS: Enabling Autoregressive Models Multi-Token Generation** — Ziqi Jin, Lei Wang, Ziwei Luo, Aixin Sun
  > Autoregressive (AR) language models generate text one token at a time, even when consecutive tokens are highly predictable given earlier context. We introduce MARS (Mask AutoRegreSsion), a lightweight...

- **MARS: Unleashing the Power of Speculative Decoding via Margin-Aware Verification** — Jingwei Song, Xinyu Wang, Hanbin Wang, Xiaoxuan Lei, Bill Shi, Shixin Han, Eric Yang, Xiao-Wen Chang, Lynn Ai
  [arXiv](https://arxiv.org/abs/2601.15498) | [GitHub](https://github.com/5SSjw/MARS)
  > Speculative Decoding (SD) accelerates autoregressive large language model (LLM) inference by decoupling generation and verification. While recent methods improve draft quality by tightly coupling the ...

- **MSAO: Adaptive Modality Sparsity-Aware Offloading with Edge-Cloud Collaboration for Efficient Multimodal LLM Inference** — Zheming Yang et al.
  [arXiv](https://arxiv.org/abs/2604.02945)
  > Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities but impose substantial computational and latency burdens, posing critical challenges for deployment on resou...

- **MemBoost: A Memory-Boosted Framework for Cost-Aware LLM Inference** — Joris Köster, Zixuan Liu, Siavash Khajavi, Zizhan Zheng
  [arXiv](https://arxiv.org/abs/2603.26557)
  > Large Language Models (LLMs) deliver strong performance but incur high inference cost in real-world services, especially under workloads with repeated or near-duplicate queries across users and sessio...

- **MemExplorer: Navigating the Heterogeneous Memory Design Space for Agentic Inference NPUs** — Haoran Wu, Zeyu Cao, Yao Lai, Binglei Lou, Jiayi Nie, Can Xiao, T. Adeniran, Przemyslaw Forys, Kauser Johar, Catriona R Wright, Junyi Liu, Kai Shi, Nicholas D. Lane, R. Antonova, Jianyi Cheng, Timothy Jones, Aaron Zhao, Robert Mullins
  [arXiv](https://arxiv.org/abs/2604.16007)
  > Emerging agentic LLM workloads are driving rapidly growing demand on both memory capacity and bandwidth, with different phases of inference (e.g., prefill and decode) imposing distinct requirements. I...

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

- **Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving** — Zongze Li, Jingyu Liu, Zach Xu, Yineng Zhang, Tahseen Rabbani, Ce Zhang
  > Prefill-Decode (PD) disaggregation has become the standard architecture for modern LLM serving systems. However, we observe that for multi-turn LLM serving workloads, the performance is suboptimal due...

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

- **P-EAGLE: Parallel-Drafting EAGLE with Scalable Training** — Mude Hui, Xin Huang, Jaime Campos Salas, Yue Sun, Nathan Pemberton, Xiang Song, Ashish Khetan, George Karypis
  > ## 摘要 (中文) 推理LLM产生更长的输出，需要在长序列上训练的投机解码drafters。并行drafting——每次前向传播预测多个tokens——比顺序生成提供延迟优势，但训练复杂度随序列长度和并行位置的乘积呈二次方增长，使得长上下文训练不切实际。我们提出了P(arallel)-EAGLE，它通过可学习的共享隐藏状态将EAGLE从自回归转变为并行多token预测。为了将训练扩展到长上下文，...

- **PAM: Processing Across Memory Hierarchy for Efficient KV-centric LLM Serving System** — Lian Liu, Shixin Zhao, Yutian Zhou, Yutian Zhou, Yintao He, Mengdi Wang, Yinhe Han, Ying Wang
  [arXiv](https://arxiv.org/abs/2602.11521)
  > The widespread adoption of Large Language Models (LLMs) has exponentially increased the demand for efficient serving systems. With growing requests and context lengths, key-value (KV)-related operatio...

- **PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models** — Eunyeong Cho, Jehyeon Bang, Ranggi Hwang, Minsoo Rhu
  [arXiv](https://arxiv.org/abs/2602.11530)
  > The emergence of reasoning-based LLMs leveraging Chain-of-Thought (CoT) inference introduces new serving challenges, as their extended reasoning phases delay user-visible output and inflate Time-To-Fi...

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

- **PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems** — Amit Singh Bhatti, Vishal Vaddina, Dagnachew Birru
  > Production LLM deployments increasingly leverage multiple specialized models to handle diverse query types, necessitating intelligent routing mechanisms that direct requests to appropriate backend mod...

- **PackInfer: Compute- and I/O-Efficient Attention for Batched LLM Inference** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2602.06072)
  > Attention efficiency is critical to large language model (LLM) inference. While prior advances optimize attention execution for individual requests (e.g., FlashAttention), production LLM serving relie...

- **PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization** — Yuchen Yang, Yifan Zhao, Anisha Dasgupta, Sasa Misailovic
  [arXiv](https://arxiv.org/abs/2607.16184v1)
  > Mixture-of-Experts (MoE) is a popular class of large language models (LLMs), offering high efficiency and accuracy. However, in KV-cache-intensive serving scenarios, MoEs often exhibit a tension betwe...

- **Pancake: Hierarchical Memory System for Multi-Agent LLM Serving** — Zhengding Hu, Zaifeng Pan, Prabhleen Kaur, Vibha Murthy, Zhongkai Yu, Yue Guan 等
  [arXiv](https://arxiv.org/abs/2602.21477)
  > In this work, we identify and address the core challenges of agentic memory management in LLM serving, where large-scale storage, frequent updates, and multiple coexisting agents jointly introduce com...

- **ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving** — Annette Taberner-Miller
  > ### English Production LLM serving often relies on multi-model portfolios spanning a ~530x cost range, where routing decisions trade off quality against cost. This trade-off is non-stationary: provide...

- **PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving** — Xu Bai
  [arXiv](https://arxiv.org/abs/2604.12171)
  > Pipeline parallelism (PP) is widely used to partition LLM layers across GPUs. However, existing systems rely on static PP configurations that fail to adapt to dynamic settings. PipeLive enables live i...

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

- **ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems** — Daeyeon Son
  > An OS kernel that runs LLM inference internally can read logit distributions before any text is generated and act on them as a governance primitive. We present ProbeLogits, a kernel-level operation th...

- **QCFuse: Query-Centric Cache Fusion for Efficient RAG Inference** — ['Jianxin Yan', 'Zeheng Qian', 'Wangze Ni', 'Zhitao Shen', 'Zhiping Wang', 'Haoyang Li']
  [arXiv](https://arxiv.org/abs/2604.08585)

- **QUADS: Stabilizing NVFP4 Reinforcement Learning for MoE via QUantization-error Alignment across Dual Sides** — Zhengyang Zhuge, Hao Yu, Xin Wang, Zheng Li, Yizhong Cao, Dayiheng Liu, Jianwei Zhang
  [arXiv](https://arxiv.org/abs/2607.15810v1)
  > Rollout generation is a major bottleneck in Reinforcement Learning (RL) for Mixture-of-Experts (MoE) Large Language Models, motivating low-precision rollout acceleration such as FP8. As an emerging lo...

- **QoServe: Breaking the Silos of LLM Inference Serving** — Kanishk Goel, Jayashree Mohan, Nipun Kwatra, Ravi Anupindi, Ram Ramjee
  > QoServe是一个统一的LLM推理服务框架，打破了传统系统中的"孤岛"设计。传统LLM serving系统针对特定场景优化，导致系统碎片化。QoServe通过创新的架构设计，实现了跨场景的高质量服务。...

- **RAP: KV-Cache Compression via RoPE-Aligned Pruning** — Jihao Xin, Tian Lyu, David E. Keyes, H. Ltaief, Marco Canini
  [arXiv](https://arxiv.org/abs/2602.02599)
  > Long-context inference in large language models is increasingly bottlenecked by the memory and compute cost of the KV-Cache. Low-rank factorization compresses KV projections by writing $W \approx A * ...

- **RAPID-Serve: Resource-efficient and Accelerated P/D Intra-GPU Disaggregation** — Amna Masood, Pratishtha Gaur, N. Jayasena
  [arXiv](https://arxiv.org/abs/2601.11822)
  > Two widely adopted techniques for LLM inference serving systems today are hybrid batching and disaggregated serving. A hybrid batch combines prefill and decode tokens of different requests in the same...

- **Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU** — N/A
  [arXiv](https://arxiv.org/abs/2604.15464)

- **Rarity-Aware Discrete Diffusion with Spatially Consistent Decoding for Photo-Realistic Image Super-Resolution** — Ao Li, Yapeng Du, Yi Xin, Lei Zhu, Le Zhang, Guangtao Zhai, Ce Zhu, Xiaohong Liu
  [arXiv](https://arxiv.org/abs/2607.17612v1)
  > Continuous diffusion models have become the dominant paradigm for photo-realistic image Super-Resolution (SR), but they typically formulate reconstruction as continuous signal-level denoising and inco...

- **RecGPT-V3 Technical Report** — Bowen Zheng, Chao Yi, Dian Chen, Gaoyang Guo, Han Zhu, Jiakai Tang, Jian Wu, Mao Zhang, Wen Chen, Yifan Lu, Yujie Luo, Yuning Jiang, Zhujin Gao, Bo Zheng, Dixuan Wang, Hao Fang, Jiancai Liu, Jing Yu, Ke Chen, Kewei Zhu, Mingke Xu, Wenjun Yang, Xunke Xi, Zile Zhou
  [arXiv](https://arxiv.org/abs/2607.15591v1)
  > Large language models (LLMs) are transforming recommender systems from matching co-occurrence patterns in historical behavior toward reasoning about the intent that drives it. RecGPT-V1 pioneered this...

- **RedFuser: An Automatic Operator Fusion Framework for Cascaded Reductions on AI Accelerators** — Xinsheng Tang, Yuhui Zhao, Jintao Li, Jiaming Xu, Shuo Li, Jiansong Chen, Chen Zhang, Yong Li, Xiaoyong Liu, Ji Liu, Jin Wang, Wei Lin
  [arXiv](https://arxiv.org/abs/2603.10026)
  > Operator fusion, as a key performance optimization technique in the deployment of AI models, significantly improves execution efficiency and has been widely adopted in modern AI compilers. However, fo...

- **Resource Multiplexing in Tuning and Serving Large Language Models** — Yongjun He, Hao Yang, Yao Lu, Ana Klimovic, Gustavo Alonso
  [GitHub](https://github.com/aerlabsAI/ai-inference-resources)

- **Rethinking Latency Denial-of-Service: Attacking the LLM Serving Framework, Not the Model** — Tianyi Wang, Huawei Fan, Yuanchao Shu, Peng Cheng, Cong Wang
  [arXiv](https://arxiv.org/abs/2602.07878)
  > Large Language Models face an emerging and critical threat known as latency attacks. Because LLM inference is inherently expensive, even modest slowdowns can translate into substantial operating costs...

- **ReviveMoE: Fast Recovery for Hardware Failures in Large-Scale MoE LLM Inference Deployments** — Haley Li, Xinglu Wang, Cong Feng, Chunxu Zuo, Yanan Wang, Hei Lo, Yufei Cui, Bingji Wang, Duo Cui, Shuming Jing, Yizhou Shan, Ying Xiong, Jiannan Wang, Yong Zhang, Zhenan Fan
  [arXiv](https://arxiv.org/abs/2602.21140)
  > As LLM deployments scale over more hardware, the probability of a single failure increases significantly. A common recovery approach is to restart the LLM serving instance; however, this is costly in ...

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

- **SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference** — Luchang Li, Dongfang Li, Bozhao Gong, Yu Zhang
  [arXiv](https://arxiv.org/abs/2603.04716)
  > Prefill-Decode (P/D) disaggregation has emerged as a widely adopted optimization strategy for Large Language Model (LLM) inference. However, there currently exists no well-established methodology for ...

- **SLO-Guard: Crash-Aware, Budget-Consistent Autotuning for SLO-Constrained LLM Serving** — ['Christian Lysenstøen']
  [arXiv](https://arxiv.org/abs/2604.17627) | [GitHub](https://github.com/Chrislysen/SLO-Guard)
  > Serving large language models under latency service-level objectives (SLOs) is a configuration-heavy systems problem with an unusually failure-prone search space. We present SLO-Guard, a crash-aware a...

- **SMART: When is it Actually Worth Expanding a Speculative Tree?** — Lifu Wang, Pan Zhou
  [arXiv](https://arxiv.org/abs/2604.09731)
  > Tree-based speculative decoding accelerates autoregressive generation by verifying a branching tree of draft tokens in a single target-model forward pass. However, existing methods prioritize maximizi...

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

- **Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving** — Tingyang Sun, Ting He, I-Hong Hou
  [arXiv](https://arxiv.org/abs/2604.14993)
  > As a current trend in Artificial Intelligence (AI), large foundation models are increasingly employed as the core of AI services. However, even after training, serving such models at scale remains a c...

- **Serving Compound Inference Systems on Datacenter GPUs** — Sriram Devata, Rahul Sukthankar, Saurabh Adya
  [arXiv](https://arxiv.org/abs/2603.08797)
  > Applications in emerging domains such as XR are being built as compound inference systems, where multiple ML models are composed in the form of a task graph to service each request. Serving these comp...

- **Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking** — Zizhao Mo, Junlin Chen, Huanle Xu, Chengzhong Xu
  [arXiv](https://arxiv.org/abs/2603.12831)
  > Nowadays, service providers often deploy multiple types of LLM services within shared clusters. While the service colocation improves resource utilization, it introduces significant interference risks...

- **Shift Parallelism: Low-Latency, High-Throughput LLM Inference for Dynamic Workloads** — Mert Hidayetoglu, Aurick Qiao, Michael Wyatt, Jeff Rasley, Yuxiong He, Samyam Rajbhandari
  > Shift Parallelism提出了一种新的并行策略，用于在动态工作负载下实现低延迟和高吞吐量的LLM推理。该方法通过创新的请求调度和计算分配，实现了在变化负载下的稳定性能。...

- **Slot Machines: How LLMs Keep Track of Multiple Entities** — Paul C. Bogdan, Jack Lindsey
  [arXiv](https://arxiv.org/abs/2604.21139v1) | [GitHub](https://github.com/turboderp-org/exllamav3)
  > Language models must bind entities to the attributes they possess and maintain several such binding relationships within a context. We study how multiple entities are represented across token position...

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

- **TENT: A Declarative Slice Spraying Engine for Performant and Resilient Data Movement in Disaggregate** — Feng Ren, Ruoyu Qin, Teng Ma, Shangming Cai, Zheng Liu, Chao Lei, Dejiang Zhu, Ke Yang, Zheming Li, Jialei Cui, Weixiao Huang, Yikai Zhao, Yineng Zhang, Hao Wu, Xiang Gao, Yuhao Fu, Jinlei Jiang, Yong
  > ### English Modern LLM serving systems increasingly adopt disaggregated architectures that separate prefill and decode stages onto different GPU clusters. However, orchestrating diverse interconnects ...

- **TIDE: Temporal Incremental Draft Engine for Self-Improving LLM Inference** — Jiyoung Park, Hankyu Jang, Changseok Song, Wookeun Jung
  > Speculative decoding has emerged as a promising solution to accelerate large language model inference by leveraging a small draft model to propose candidate tokens in parallel and a large target model...

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

- **The Workload-Router-Pool Architecture for LLM Inference Optimization: A Vision Paper from the vLLM S** — Huamin Chen, Xunzhuo Liu, Bowei He, Fuyuan Lyu, Yankai Chen, Xue Liu, Yuhan Liu, Junchen Jiang
  > > ...caching, user-feedback-driven routing adaptation, hallucination detection, and hierarchical content-safety classification for privacy and jailbreak protection; (2) fleet optimization -- fleet pro...

- **The xPU-athalon: Quantifying the Competition of AI Acceleration** — ['Alicia Golden', 'Carole-Jean Wu', 'Gu-Yeon Wei', 'David Brooks']
  [arXiv](https://arxiv.org/abs/2604.10852)

- **Token Coherence: Adapting MESI Cache Protocols to Minimize Synchronization Overhead in Multi-Agent LLM Systems** — Vladyslav Parakhin
  [arXiv](https://arxiv.org/abs/2603.15183) | [GitHub](https://github.com/hipvlady/agent-coherence)
  > Multi-agent LLM orchestration incurs synchronization costs scaling as O(n x S x |D|). This work maps synchronization cost explosion onto the cache coherence problem and adapts MESI-protocol invalidati...

- **Towards High-Goodput LLM Serving with Prefill-decode Multiplexing** — Weihao Cui, Yukang Chen, Han Zhao, Ziyi Xu, Xiaoze Fan, Xusheng Chen, Yangjie Zhou, Shixuan Sun, Bingsheng He, Quan Chen
  > 现代大型语言模型(LLM)的部署需要同时优化延迟和吞吐量。传统的LLM serving系统通常将prefill阶段和解码阶段分开处理，但这导致了资源利用不均衡的问题。本文提出了Prefill-Decode Multiplexing (PDM) 框架，通过创新的请求调度和资源分配策略，实现了更高的goodput（服务质量感知的吞吐量）。...

- **Transformer-Based Resource and Stage-Aware Scheduling for Model-Parallel LLM Inference** — Rami Naeem, Tengis Buyantogtokh, Hamada Rizk, Tatsuya Amano, Hirozumi Yamaguchi
  > Current large language model (LLM) serving systems face three key limitations in distributed scheduling. First, most parallelization strategies are not stage-aware: they treat prefill and decode as un...

- **Transition-Aware Backend Dispatch for Edge LLM Inference** — Alaaddin Goktug Ayar, Martin Margala
  [arXiv](https://arxiv.org/abs/2607.17415v1)
  > Efficient large language model (LLM) inference on edge platforms is limited not only by model size, but also by shape-dependent performance differences across execution backends. Static backend assign...

- **Understand and Accelerate Memory Processing Pipeline for Disaggregated LLM Inference** — ['Zifan He', 'Rui Ma', 'Yizhou Sun', 'Jason Cong']
  [arXiv](https://arxiv.org/abs/2603.29002)

- **VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation** — Qijun Han, Haoqin Tu, Zijun Wang, Haoyue Dai, Yiyang Zhou
  [arXiv](https://arxiv.org/abs/2604.21375v1)
  > Autonomous GUI agents face two fundamental challenges: early stopping, where agents prematurely declare success without verifiable evidence, and repetitive loops, where agents cycle through the same f...

- **Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate** — Fangyue Liu, Hua Liu, Xinyuan Lyu, Shuo Ai, Hao Liang, Lingpeng Chen, Ziqian Hu, Chong Zha, Xin Jin, Hanmei Luo, Peng Chen
  [arXiv](https://arxiv.org/abs/2604.07874)
  > LLM inference powers latency-critical production services. Valve is a production-friendly colocation system that jointly bounds preemption latency and preemption rate. It enables sub-millisecond compu...

- **VarRate: Training-Free Variable-Rate KV Cache Compression for Long-Context LLMs** — Shahrzad Esmat, Dhawal Shah, Ali Jannesari
  [arXiv](https://arxiv.org/abs/2607.15498v1)
  > The key-value (KV) cache is the main memory bottleneck in long-context large language model (LLM) inference. Two leading training-free families are both structurally limited: token-selection methods (...

- **WWW.Serve: Interconnecting Global LLM Services through Decentralization** — Huanyu Wang, Ziyu Xia, Zhuoming Chen, Beidi Chen
  [arXiv](https://arxiv.org/abs/2603.20661)
  > Large language model (LLM) services are mostly centralized, leading to scalability bottlenecks and underutilization of substantial scattered GPU resources. While decentralization offers a promising al...

- **Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures** — ['Mauricio Fadel Argerich', 'Jonathan Fürst', 'Marta Patiño-Martínez']
  [arXiv](https://arxiv.org/abs/2604.09048)

- **WaveTune: Wave-aware Bilinear Modeling for Efficient GPU Kernel Auto-tuning** — ['Kaixuan Zhang', 'Chutong Ding', 'Shiyou Qian', 'Luping Wang', 'Jian Cao', 'Guangtao Xue']
  [arXiv](https://arxiv.org/abs/2604.10187)

- **WebGen-R1: Incentivizing Large Language Models to Generate Functional and Aesthetic Websites with Reinforcement Learning** — Juyong Jiang, Chenglin Cai, Chansung Park, Jiasi Shen, Sunghun Kim
  [arXiv](https://arxiv.org/abs/2604.20398v1) | [GitHub](https://github.com/sgl-project/sglang)
  > While Large Language Models (LLMs) excel at function-level code generation, project-level tasks such as generating functional and visually aesthetic multi-page websites remain highly challenging. Exis...

- **XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads** — Mingcong Song, Xinru Tang, Fengfan Hou, Jing Li, Wei Wei, Yipeng Ma, Runqiu Xiao, Hongjie Si, Dingcheng Jiang, Shouyi Yin, Yang Hu, Guoping Long
  > XY-Serve是华为提出的端到端LLM服务系统，专门针对动态工作负载进行优化。该系统提供了生产级别的LLM serving能力，支持多种模型和场景。...

- **ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval** — David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen
  [arXiv](https://arxiv.org/abs/2604.10898)
  > Large language models (LLMs) have shown great performance on complex reasoning tasks but often require generating long intermediate thoughts before reaching a final answer. During generation, LLMs rel...

- **[GitHub] BitNet: Official inference framework for 1-bit LLMs** — microsoft
  [GitHub](https://github.com/microsoft/BitNet)
  > Official inference framework for 1-bit LLMs...

- **[GitHub] FastDeploy: High-performance Inference and Deployment Toolkit for LLMs and VLMs based on PaddlePaddle** — PaddlePaddle
  [GitHub](https://github.com/PaddlePaddle/FastDeploy)
  > High-performance Inference and Deployment Toolkit for LLMs and VLMs based on PaddlePaddle...

- **[GitHub] InferLLM: a lightweight LLM model inference framework** — MegEngine
  [GitHub](https://github.com/MegEngine/InferLLM)
  > a lightweight LLM model inference framework...

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

- **Toppings: CPU-Assisted, Rank-Aware Adapter Serving for LLM Inference** — Suyi Li, Hanfeng Lu, Tianyuan Wu, Minchen Yu, Qizhen Weng


### MoE Inference

- **DuoServe-MoE: Dual-Phase Expert Prefetch and Caching for LLM Inference QoS Assurance** — Yuning Zhang, Grant Pinkert, Nan Yang, Yanli Li, Dong Yuan
  [arXiv](https://arxiv.org/abs/2509.07379)
  > Large Language Models (LLMs) are increasingly deployed as Internet/Web services (LLM-as-a-Service) with strict latency Service-Level Objectives (SLOs) under tight GPU memory budgets. Mixture-of-Expert...

- **Efficient Mixture-of-Experts LLM Inference with Apple Silicon NPUs** — Afsara Benazir, Felix Xiaozhu Lin
  [arXiv](https://arxiv.org/abs/2604.18788)
  > Apple Neural Engine (ANE) is a dedicated neural processing unit (NPU) present in every Apple Silicon chip. Mixture-of-Experts (MoE) LLMs improve inference efficiency by activating only a sparse subset...

- **EvoESAP: Non-Uniform Expert Pruning for Sparse MoE** — Multiple Authors
  > Mixture-of-Experts (MoE) models achieve high quality with efficient compute by activating only a subset of experts per token. However, the total number of experts still consumes significant memory. We...

- **FluxMoE: Decoupling Expert Residency for High-Performance MoE Serving** — Qingxiu Liu, Cyril Y. He, Hanser Jiang
  [arXiv](https://arxiv.org/abs/2604.02715)
  > Mixture-of-Experts (MoE) models have become a dominant paradigm for scaling large language models, but their rapidly growing parameter sizes introduce severe challenges for efficient serving. FluxMoE ...

- **From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill** — Gunjun Lee, Jiwon Kim, Jaiyoung Park, Younjoo Lee, Jung Ho Ahn
  [arXiv](https://arxiv.org/abs/2510.08055)
  > Large Language Model (LLM) inference in production must meet stringent service-level objectives for both time-to-first-token (TTFT) and time-between-token (TBT) while maximizing throughput under fixed...

- **LAER-MoE: Load-Adaptive Expert Re-layout for Efficient Mixture-of-Experts Training** — Xinyi Liu, Zijian Zhang, YongLi Zhu, Jiale Zhang, Peng Sun, XuanWang, Qi Qi, Jingren Zhou, Tong Yang, Bin Cui
  [arXiv](https://arxiv.org/abs/2602.11686)
  > Expert parallelism is vital for effectively training Mixture-of-Experts (MoE) models, enabling different devices to host distinct experts, with each device processing different input data. However, du...

- **MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading** — Peng Tang, Jiacheng Liu, Xiaofeng Hou, Yifei Pu, Jing Wang, Pheng-Ann Heng, Chao Li, Minyi Guo
  > MoE-APEX是针对MoE（混合专家）模型的高效推理系统，通过自适应精度的专家卸载策略，在保持模型质量的同时显著降低推理成本。...

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

- **InfiniLoRA: Disaggregated Multi-LoRA Serving for Large Language Models** — Hongyu Chen, Letian Ruan, Zilin Xu, Yuchen Li, Xinyu Chen, Jingwen Leng, Bingsheng He, Minyi Guo, Shixuan Sun
  > LoRA enables efficient customization of LLMs and is widely used in multi-tenant and multi-task serving scenarios. However, serving many LoRA adapters simultaneously introduces significant memory and s...

- **Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter** — Ruoyu Qin, Weiran He, Yaoyu Wang, Zheming Li, Xinran Xu, Yongwei Wu, Weimin Zheng, Mingxing Zhang
  [arXiv](https://arxiv.org/abs/2604.15039)
  > Prefill-decode (PD) disaggregation has become the standard architecture for large-scale LLM serving, but in practice its deployment boundary is still determined by KVCache transfer. In conventional de...

- **Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token (TTFT)** — Rajveer Bachkaniwala, Chengqi Luo, Richard So, Divya Mahajan, Kexin Rong
  [arXiv](https://arxiv.org/abs/2603.19458)
  > [Abstract待从arxiv页面获取]...


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

- **A Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference** — Yida Zhang, Zhiyong Gao, Shuaibing Yue, Jie Li, Rui Wang
  > Recent advancements and widespread adoption of Large Language Models (LLMs) in both industry and academia have catalyzed significant demand for efficient LLM inference systems. This paper presents a p...

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
  [GitHub](https://github.com/kvcache-ai/Mooncake)

- **Benchmarking the Energy Savings with Speculative Decoding Strategies** — Multiple Authors
  > While speculative decoding is primarily evaluated for latency reduction, its impact on energy consumption is equally important for sustainable LLM deployment. We present the first comprehensive benchm...

- **Cactus: Accelerating Auto-Regressive Decoding with Constrained Acceptance Speculative Sampling** — Yongchang Hao, Lili Mou
  > Speculative sampling (SpS) has been successful in accelerating the decoding throughput of auto-regressive large language models by leveraging smaller draft models. SpS strictly enforces the generated ...

- **Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference** — Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan
  [arXiv](https://arxiv.org/abs/2604.13634)
  > Speculative decoding accelerates autoregressive generation by letting draft tokens bypass full verification, but conventional frameworks suffer from frequent false rejections, particularly when draft ...

- **ConFu: Contemplate the Future for Better Speculative Sampling** — Zongyue Qin, Raghavv Goel, Mukul Gagrani, Risheek Garrepalli, Mingu Lee, Yizhou Sun
  > ### English Speculative sampling aims to accelerate autoregressive generation by proposing multiple tokens in parallel and verifying them together. However, existing methods treat token proposal and v...

- **ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding** — Walaa Amer, Uday Das, Fadi Kurdahi
  [arXiv](https://arxiv.org/abs/2604.14612)
  > Self-speculative decoding is an inference technique for large language models designed to speed up generation without sacrificing output quality. It combines fast, approximate decoding using a compact...

- **Cross-Family Speculative Prefill: Training-Free Long-Context Compression with Small Draft Models** — Shubhangi Upasani, Ravi Shanker Raju, Bo Li, Mengmeng Ji, John Long, Chen Wu, Urmish Thakker, Guangtao Wang
  > Prompt length is a major bottleneck in agentic large language model (LLM) workloads, where repeated inference steps and multi-call loops incur substantial prefill cost. Recent work on speculative deco...

- **DFlash: Block Diffusion for Flash Speculative Decoding** — Jian Chen, Yesheng Liang, Zhijian Liu
  > ## 摘要 (中文) 自回归大型语言模型（LLM）虽然性能强大，但需要固有的顺序解码，导致高推理延迟和GPU利用率低。投机解码通过使用快速draft模型来缓解这一瓶颈，该模型的输出由目标LLM并行验证；然而，现有方法仍然依赖于自回归drafting，这仍然是顺序的，限制了实际的加速。扩散LLM通过实现并行生成提供了另一种有前途的方案，但当前的扩散模型通常比自回归模型性能差。在本文中，我们引入了DF...

- **DIVERSED: Relaxed Speculative Decoding via Dynamic Ensemble Verification** — Ziyi Wang, Siva Rajesh Kasa, Ankith M S, Santhosh Kumar Kasa, Jiaru Zou, Sumit Negi, Ruqi Zhang, Nan Jiang, Qifan Song
  > Speculative decoding is an effective technique for accelerating large language model inference by drafting multiple tokens in parallel. In practice, its speedup is often bottlenecked by a rigid verifi...

- **DiP-SD: Distributed Pipelined Speculative Decoding for Efficient LLM Inference at the Edge** — N/A
  [arXiv](https://arxiv.org/abs/2604.20919)
  > Speculative decoding has emerged as a promising technique for large language model (LLM) inference by accelerating autoregressive decoding via draft-then-verify. This paper studies a new edge scenario...

- **Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization** — Jiu Chen, Shuangyan Yang, Xu Xiong, Hexiao Duan, Xinran Zhang
  [arXiv](https://arxiv.org/abs/2604.21072v1) | [GitHub](https://github.com/SharpAI/SwiftLM)
  > Decentralized LLM inference distributes computation among heterogeneous nodes across the internet, offering a performant and cost-efficient solution, alternative to traditional centralized inference. ...

- **EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs** — Chang Han, Yijie Hu, Jingling Liu
  > ### English Autoregressive decoding remains a primary bottleneck in large language model (LLM) serving, motivating the adoption of speculative decoding techniques. However, most existing speculative d...

- **ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios** — Xinyi Hu, Yuhao Shen, Baolin Zhang, Hengxin Zhang, Jun Dai, Shuang Ge, Lei Chen, Yue Li, Mingcheng Wan
  > This paper presents ECHO, an elastic speculative decoding method with sparse gating for high-concurrency scenarios....

- **ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving** — Yuseon Choi, Jingu Lee, Jungjun Oh, Sunjoo Whang, Byeongcheol Kim, Minsung Kim, Hoi-Jun Yoo, Sangjin Kim
  [arXiv](https://arxiv.org/abs/2604.14626)
  > Mixture-of-Experts (MoE) models have become the dominant architecture for large-scale language models, yet on-premises serving remains fundamentally memory-bound as batching turns sparse per-token com...

- **Efficient LLM Serving for Agentic Workflows: A Data Systems Perspective** — (待补充)
  > Agentic workflows are composed of sequences of interdependent Large Language Model (LLM) calls, and they have become a dominant workload in modern AI systems. This paper examines LLM serving from a da...

- **Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing** — Multiple Authors
  > We propose a training-free approach for multi-token prediction in speculative decoding by probing the embedding space of the target model. Instead of training a separate draft model, our method uses t...

- **FASER: Fine-Grained Phase Management for Speculative Decoding in Dynamic LLM Serving** — Wenyan Chen, Chengzhi Lu, Yanying Lin, Dmitrii Ustiugov
  [arXiv](https://arxiv.org/abs/2604.20503v1)
  > Speculative decoding (SD) is a widely used approach for accelerating decode-heavy LLM inference workloads. While online inference workloads are highly dynamic, existing SD systems are rigid and take a...

- **From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning (SpecGuard)** — Authors from arXiv:2604.15244
  [arXiv](https://arxiv.org/abs/2604.15244)
  > Speculative decoding (SD) accelerates large language model inference by allowing a lightweight draft model to propose outputs that a stronger target model verifies. However, its token-centric nature a...

- **Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding** — Unknown
  > ### English Speculative decoding accelerates large language model inference by drafting multiple candidate tokens and verifying them in a single forward pass. Candidates are organized as a tree: deepe...

- **KnapSpec: Self-Speculative Decoding via Adaptive Layer Selection as a Knapsack Problem** — Multiple Authors
  > Self-speculative decoding skips intermediate model layers to generate draft tokens, but selecting which layers to skip is challenging. We formulate layer selection as a knapsack optimization problem, ...

- **LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding** — Multiple Authors
  > Training draft models for speculative decoding typically uses language modeling losses, but these losses do not directly optimize the acceptance rate—the metric that determines acceleration. We propos...

- **Learning to Draft: Adaptive Speculative Decoding with Reinforcement Learning** — Jiebin Zhang, Zhenghan Yu, Liang Wang, Nan Yang, Eugene J. Yu, Zheng Li, Yifan Song, Dawei Zhu, Xingxing Zhang, Furu Wei, Sujian Li
  > ### English Speculative decoding accelerates LLM inference by using a draft model to propose candidate tokens. However, existing methods use fixed drafting strategies that don't adapt to different inp...

- **Make Every Draft Count: Hidden State based Speculative Decoding** — Multiple Authors
  > We propose a hidden state-based speculative decoding approach that leverages the intermediate representations of the target model to guide draft token generation. By analyzing the hidden state traject...

- **MineDraft: A Framework for Batch Parallel Speculative Decoding** — Zhenwei Tang, Arun Verma, Zijian Zhou, Zhaoxuan Wu, Alok Prakash, Daniela Rus, Bryan Kian Hsiang Low
  > 投机解码 (Speculative Decoding) 是一种通过使用draft模型预测token序列，然后由target模型验证来加速LLM推理的技术。  然而，现有的投机解码框架主要针对单序列推理进行优化，无法有效处理批处理场景。  **MineDraft** 是一个**批量并行投机解码框架**，专门针对批处理推理场景进行优化。...

- **Minimizing Response Latency in LLM-Based Agent Systems: A Comprehensive Survey** — G. Park, Seonghyeon Lee, Yeonsu Park
  > The advent of Large Language Model (LLM)-based agent systems represents a significant paradigm shift in Artificial Intelligence, enabling unprecedented capabilities in autonomous reasoning, planning, ...

- **MoE-Spec: Expert Budgeting for Efficient Speculative Decoding** — Bradley McDanel, Steven Li, Sruthikesh Surineni, Harshit Khaitan
  > ## 摘要 (中文) 投机解码通过并行验证多个draft tokens来加速大型语言模型（LLM）推理。然而，对于混合专家（MoE）模型，这种并行性引入了严重的瓶颈：大型draft树激活许多独特的专家，大大增加了内存压力，并减少了相对于自回归解码的投机解码加速。之前的方法在MoE验证变得昂贵时减少投机深度。我们提出了MoE-Spec，这是一种无需训练的验证时专家预算方法，通过在每一层强制执行固定的...

- **Multi-Drafter Speculative Decoding with Alignment Feedback** — ['Taehyeon Kim', 'Hojung Jung', 'Se-Young Yun']
  [arXiv](https://arxiv.org/abs/2604.05417)

- **NI Sampling: Accelerating Discrete Diffusion Sampling by Token Order Optimization** — Enshu Liu, Xuefei Ning, Yu Wang, Zinan Lin
  [arXiv](https://arxiv.org/abs/2604.18471) | [GitHub](https://github.com/imagination-research/NI-Sampling)
  > Discrete diffusion language models (dLLMs) have recently emerged as a promising alternative to traditional autoregressive approaches, offering the flexibility to generate tokens in arbitrary orders an...

- **Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning** — NVIDIA Team
  [arXiv](https://arxiv.org/abs/2604.12374) | [GitHub](https://github.com/https://huggingface.co/nvidia/Nemotron-3-Super)
  > We describe the pre-training, post-training, and quantization of Nemotron 3 Super, a 120 billion (active 12 billion) parameter hybrid Mamba-Attention Mixture-of-Experts model. Nemotron 3 Super is the ...

- **Nightjar: Dynamic Adaptive Speculative Decoding for Large Language Models Serving** — Rui Li, Zhaoning Zhang, Libo Zhang, et al.
  > Speculative decoding has emerged as a promising technique to accelerate large language model (LLM) inference by leveraging a small draft model to propose candidate tokens and a large target model to v...

- **Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verificatio** — Guang Huang, Zeyi Wen
  > ### English We present Quasar, a quantized self-speculative acceleration framework that combines model quantization with self-speculation for efficient LLM inference. Unlike traditional speculative de...

- **RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding** — Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao
  [arXiv](https://arxiv.org/abs/2604.14885) | [GitHub](https://github.com/https://github.com/hkr04/RACER)
  > Autoregressive decoding in Large Language Models (LLMs) generates one token per step, causing high inference latency. Speculative decoding (SD) mitigates this through a guess-and-verify strategy, but ...

- **S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation** — Ligong Wang, Hao Wang, Kai Xu, Akash Srivastava
  > > Block-diffusion language models offer a promising path toward faster-than-autoregressive generation by combining block-wise autoregressive decoding with within-block parallel denoising. However, in ...

- **SDFP: Speculative Decoding with FIT-Pruned Models for Training-Free and Plug-and-Play LLM Acceleration** — Multiple Authors
  > We present SDFP, a training-free speculative decoding method that uses FIT-pruned (feature-importance-tree pruned) versions of the target model as draft models. By pruning less important layers and at...

- **SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation** — Jialiang Kang, Han Shu, Wenshuo Li, Yingjie Zhai, Xinghao Chen
  [arXiv](https://arxiv.org/abs/2603.1)
  > Speculative Jacobi Decoding (SJD) offers a draft-model-free approach to accelerate autoregressive text-to-image synthesis. However, the high-entropy nature of visual generation yields low draft-token ...

- **SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding** — Talor Abramovich, Maor Ashkenazi, Carl Putterman, Benjamin Chislett, Tiyasa Mitra, Bita Darvish Rouhani, Ran Zilberstein, Yonatan Geifman
  > Speculative Decoding (SD) has emerged as a critical technique for accelerating Large Language Model (LLM) inference. Unlike deterministic approximation methods, SD guarantees exact output distribution...

- **See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs** — Multiple Authors
  > We propose loosely speculative decoding for video LLMs that uses visual-semantic guidance from higher-level scene understanding to draft tokens. Rather than requiring exact distribution matching, our ...

- **Sparrow: Text-Anchored Window Attention with Visual-Semantic Glimpsing for Speculative Decoding in Video LLMs** — Multiple Authors
  > We propose Sparrow, a speculative decoding method for video LLMs that uses text-anchored window attention to select relevant visual tokens for drafting. By glimpsing visual semantics anchored to the t...

- **SpeContext: Enabling Efficient Long-context Reasoning with Speculative Context Sparsity in LLMs** — Jiaming Xu, Hong Cao, Yuhan Lin, Jinyang Li, Zheng Liu, Jie Liu, Xingyu Li, Jin Wang, Jingyuan Jia, Ge Li
  [arXiv](https://arxiv.org/abs/2512.00722)
  > In this paper, we point out that the objective of the retrieval algorithms is to align with the LLM, which is similar to the objective of knowledge distillation in LLMs. We analyze the similarity in i...

- **SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding** — Multiple Authors
  > We propose SpecAttn, a co-design framework that jointly optimizes sparse attention and self-speculative decoding for LLM inference acceleration. By sharing computation between sparse attention selecti...

- **SpecEyes: Accelerating Agentic Multimodal LLMs via Speculative Perception and Planning** — Haoyu Huang, Jinfa Huang, Zhongwei Wan, Xiawu Zheng, Rongrong Ji, Jiebo Luo
  > Agentic multimodal large language models (MLLMs) (e.g., OpenAI o3 and Gemini Agentic Vision) achieve remarkable reasoning capabilities through iterative visual tool invocation. However, the cascaded p...

- **SpecMD: A Comprehensive Study On Speculative Expert Prefetching** — Duc Hoang, Ajay Jaiswal, Mohammad Samragh, Minsik Cho
  [arXiv](https://arxiv.org/abs/2602.03921)
  > Mixture-of-Experts (MoE) models enable sparse expert activation, meaning that only a subset of the model's parameters is used during each inference. However, to translate this sparsity into practical ...

- **Speculating Experts Accelerates Inference for Mixture-of-Experts** — Vivan Madan, Prajwal Singhania, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda
  [arXiv](https://arxiv.org/abs/2603.19289) | [GitHub](https://github.com/axonn-ai/yalis/tree/offload_prefetch)
  > Mixture-of-Experts (MoE) models have gained popularity as a means of scaling the capacity of large language models (LLMs) while maintaining sparse activations and reduced per-token compute. However, i...

- **Speculating Experts: Accelerates Inference for Mixture-of-Experts** — Vivan Madan, Prajwal Singhania, Abhinav Bhatele, Tom Goldstein, Ashwinee Panda
  > > ...per-token compute. However, in memory-constrained inference settings, expert权重必须卸载到CPU, creating a performance bottleneck from CPU-GPU transfers during decoding. We propose an expert prefetching ...

- **Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple** — Unknown
  > ### English We present Speculative Decoding Scaling Laws (SDSL), a systematic framework for optimizing speculative decoding throughput. Unlike prior work that focuses on individual components, SDSL pr...

- **Speculative Decoding for Autoregressive Video Generation** — Yuezhou Hu, Jintao Zhang
  [arXiv](https://arxiv.org/abs/2604.17397)
  > Autoregressive video diffusion is emerging as a promising paradigm for streaming video synthesis, with step distillation serving as the primary means of accelerating inference. Whether speculative dec...

- **Speculative Speculative Decoding** — Tanishq Kumar, Tri Dao, Avner May
  > ### English Autoregressive decoding is bottlenecked by its sequential nature. Speculative decoding addresses this by using a draft model to propose tokens in parallel. We go one step further with Spec...

- **StarSD: One-for-Many Speculative Decoding** — Unknown
  > ## 关键词 - Speculative Decoding - One-for-Many - Multi-Target...

- **Super Apriel: One Checkpoint, Many Speeds** — SLAM Labs,  :, Oleksiy Ostapenko, Raymond Li, Torsten Scholak
  [arXiv](https://arxiv.org/abs/2604.19877v1) | [GitHub](https://github.com/LMCache/LMCache)
  > We release Super Apriel, a 15B-parameter supernet in which every decoder layer provides four trained mixer choices -- Full Attention (FA), Sliding Window Attention (SWA), Kimi Delta Attention (KDA), a...

- **ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding** — Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li
  [arXiv](https://arxiv.org/abs/2604.13519)
  > Tool calling has greatly expanded the practical utility of large language models (LLMs) by enabling them to interact with external applications. As LLM capabilities advance, effective tool use increas...

- **Training-free Dropout Sampling for Semantic Token Acceptance in Speculative Decoding** — Jeongtae Lee, Minjung Jo, Hyunjoon Jeong, Gunho Park, Sunghyeon Woo, Joonghoon Kim, Se Jung Kwon, Dongsoo Lee
  [arXiv](https://arxiv.org/abs/2602.0)
  > Speculative decoding accelerates large language model inference by proposing tokens with a lightweight draft model and selectively accepting them using a target model. This work introduces DropMatch, ...

- **WISP: Waste- and Interference-Suppressed Distributed Speculative LLM Serving at the Edge** — Unknown
  > ## 关键词 - Distributed Speculative Decoding - Edge Computing - SLO-Aware Batching - Dynamic Drafting...

- **WISV: Wireless-Informed Semantic Verification for Distributed Speculative Decoding in Device-Edge LLM Inference** — ['Zixuan Liu', 'Zhiyong Chen', 'Nan Xue', 'Shengkang Chen', 'Jiangchao Yao', 'Meixia Tao', 'Wenjun Zhang']
  [arXiv](https://arxiv.org/abs/2604.17701)
  > While distributed device-edge speculative decoding accelerates LLM inference, verification overhead on constrained devices remains significant. We propose WISV, a wireless-informed semantic verificati...

- **When RL Meets Adaptive Speculative Training: A Unified Training-Serving System** — Junxiong Wang, Fengxiang Bie, Jisen Li, et al.
  > Speculative decoding has emerged as a prominent technique for accelerating large language model (LLM) inference. However, existing works primarily focus on optimizing inference efficiency, overlooking...

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

- **Birds in Cages: Edge Inference Allocation for Distributed LLM Deployment** — Jiahao Zhu, Lu Zhao, Fu Xiao, Lingjie Duan

- **Buffer Management for Out-of-GPU LLM Execution** — Jiashen Cao, Joy Arulraj, Hyesoon Kim
  [GitHub](https://github.com/ome-projects/ome)
  > The rapid advancement of large language models (LLMs) has caused their parameter sizes to grow beyond the memory capacity of a single GPU. Although distributed inference across multiple GPUs is a solu...


### Edge Inference

- **Database as Runtime: Compiling LLMs to SQL for In-database Model Serving** — Wenbo Sun, Ziyu Li, Rihan Hai
  > Deploying large language models (LLMs) often requires specialized hardware and complex frameworks, creating barriers for CPU-based environments with resource constraints. These systems, common in air-...

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

- **POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference** — Aditya Kamath 等
  > Each request in LLM inference goes through two phases: compute-bound prefill and memory-bandwidth-bound decode. To improve GPU utilization, recent systems use hybrid batching that combines the prefill...

- **vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention** — Ashish Panwar, Rishabh Prabhu, et al.
  > PagedAttention is a popular approach for dynamic memory allocation in LLM serving systems. It enables on-demand allocation of GPU memory to mitigate KV cache fragmentation -- a phenomenon that cripple...


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

- **LLM-Driven Offloading Decisions for Edge Object Detection in Smart City Deployments** — Xingyu Yuan, He Li
  > Object detection is a critical technology for smart city development. As request volumes surge, inference is increasingly offloaded from centralized clouds to user-proximal edge sites to reduce latenc...

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

- **Fast State Restoration in LLM Serving with HCache** — Shiwei Gao 等
  > The growing complexity of LLM usage today, e.g., multi-round conversation and retrieval-augmented generation (RAG), makes contextual states (i.e., KV cache) reusable across user requests. Given the ca...

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

- **KV Admission: Learning What to Write for Efficient Long-Context Inference** — Yen-Chieh Huang, Pi-Cheng Hsiu, Rui Fang, Ming-Syan Chen
  > Long-context LLM inference requires efficient KV cache management. KV Admission proposes a learning-based approach to determine which tokens should be stored in the KV cache, balancing memory usage an...

- **KV Cache Transform Coding for Compact Storage in LLM Inference** — ['Konrad Staniszewski', "Adrian La'ncucki"]
  [arXiv](https://arxiv.org/abs/2511.01815) | [GitHub](https://github.com/psmarter/mini-infer)
  > Serving large language models (LLMs) at scale necessitates efficient key-value (KV) cache management. KV caches can be reused across conversation turns via shared-prefix prompts that are common in ite...

- **KV-CAR: KV Cache Compression using Autoencoders and KV Reuse in Large Language Models** — Authors from arxiv (see full paper)
  [arXiv](https://arxiv.org/abs/2512.06727)
  > KV-CAR proposes KV Cache Compression using Autoencoders and KV Reuse in Large Language Models, targeting efficient inference by reducing the memory footprint of KV caches through compression and reuse...

- **KVO-LLM: Boosting Long-Context Generation Throughput for Batched LLM Inference** — Zhenyu Li, Dongxu Lyu, Gang Wang, Yuzhou Chen, Liyan Chen
  [GitHub](https://github.com/cuckoo-network/cuckoo)
  > With the widespread deployment of long-context large language models (LLMs), efficient and high-quality generation is becoming increasingly important. Modern LLMs employ batching and key-value (KV) ca...

- **LLaMCAT: Optimizing Large Language Model Inference with Cache Arbitration and Throttling** — Zhongchun Zhou, Chengtao Lai, Wei Zhang
  > Large Language Models (LLMs) have achieved unprecedented success but their substantial memory requirements pose significant challenges. LLaMCAT proposes a cache arbitration and throttling mechanism to...

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

- **PCR: A Prefetch-Enhanced Cache Reuse System for Low-Latency RAG Serving** — Wenfeng Wang, Xiaofeng Hou, Peng Tang, Hengyi Zhou, Jing Wang, Xinkai Wang, Chao Li, Minyi Guo
  > 检索增强生成 (Retrieval-Augmented Generation, RAG) 系统通过整合检索到的外部文档来增强大语言模型 (LLMs) 的性能，从而实现更准确和上下文感知的响应。  然而，集成这些外部文档通常会导致**非常长的输入序列**，这显著增加了预填充 (prefill) 阶段的计算成本。...

- **PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving** — A. C. Yuzuguler, Jiawei Zhuang, Lukas Cavigelli
  [arXiv](https://arxiv.org/abs/2501.08192) | [GitHub](https://github.com/sgl-project/sglang)
  > Large language models (LLMs) are typically served from clusters of GPUs/NPUs that consist of large number of devices. Unfortunately, communication between these devices incurs significant overhead, in...

- **Paged Attention Meets FlexAttention: Unlocking Long-Context Efficiency in Deployed Inference** — Thomas Joshi, Herman Saini, Neil Dhillon, Antoni Viros i Martin, Kaoutar El Maghraoui
  [arXiv](https://arxiv.org/abs/2506.07311)
  > Large Language Models (LLMs) encounter severe memory inefficiencies during long-context inference due to conventional handling of key-value (KV) caches. In this work, we introduce a novel integration ...

- **TARDIS: A GPU-Centric KV Cache Service for Efficient LLM Inference** — Yifan Hu, Shi Qiu, Jianqin Yan, Hao Chen, Xintao Wang
  > Key-value (KV) cache is a crucial optimization for large language model (LLM) serving, particularly in long-context inference scenarios. While existing KV stores suffer from a fundamental mismatch bet...

- **TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale** — Dongha Yoon, Younghoon Min, Hoshik Kim, Sam H. Noh, Jongryool Kim
  > Disaggregated LLM serving with CXL shared memory enables efficient KV cache sharing across GPU nodes. TraCT leverages rack-scale CXL memory to provide high-bandwidth, low-latency access to KV cache, e...

- **VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference** — Zihan Liu, Xinhao Luo, Junxian Guo, Wentao Ni, Yangjie Zhou
  [arXiv](https://arxiv.org/abs/2503.02236) | [GitHub](https://github.com/Zefan-Cai/Awesome-LLM-KV-Cache)
  > Vector quantization (VQ), which treats a vector as a compression unit, gains increasing research interests for its potential to accelerate large language models (LLMs). Compared to conventional elemen...

- **VecInfer: Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization** — Dingyu Yao, Chenxu Yang, Zhengyang Tong, Zheng Lin, Wei Liu
  [arXiv](https://arxiv.org/abs/2510.06175) | [GitHub](https://github.com/skyzh/tiny-llm)
  > The Key-Value (KV) cache introduces substantial memory overhead during large language model (LLM) inference. Although existing vector quantization (VQ) methods reduce KV cache usage and provide flexib...


### LLM Serving

- **3D-CIMlet: A Chiplet Co-Design Framework for Heterogeneous In-Memory Acceleration of Edge LLM Inference and Continual Learning** — Shuting Du, Luqi Zheng, A. M. Parvathy, Feifan Xie, Tiwei Wei, Anand Raghunathan, Haitong Li
  > The design space for edge AI hardware supporting large language model (LLM) inference and continual learning is underexplored. We present 3D-CIMlet, a thermal-aware modeling and co-design framework fo...

- **Acceleration Multiple Heads Decoding for LLM via Dynamic Tree Attention** — Zhendong Zhang
  [arXiv](https://arxiv.org/abs/2502.05947)
  > Multiple heads decoding accelerates the inference of Large Language Models (LLMs) by predicting next several tokens simultaneously. It generates and verifies multiple candidate sequences in parallel v...

- **Area- and Utilization-Efficient LLM Accelerator With Fused Speculative Decoding for Edge-Side Inference** — Kaiqi Chen, Zikang Zhou, Yaqi Chen, Jun Han

- **CompAir: Synergizing Complementary PIMs and In-Transit NoC Computation for Efficient LLM Acceleration** — Hongyi Li, Songchen Ma, Huanyu Qu, Weihao Zhang, Jia Chen, Junfeng Lin, Fengbin Tu, Rong Zhao
  [arXiv](https://arxiv.org/abs/2509.13710)
  > The rapid advancement of Large Language Models (LLMs) has revolutionized various aspects of human life, yet their immense computational and energy demands pose significant challenges for efficient inf...

- **Context-Aware Autoscaling for Cost-Efficient Large Language Model Inference With Prefix Cache Integration** — Seyed Hossein Ahmadpanah, A. Sahafi, S. H. Erfani
  > Although granular resource management has been made possible by the architectural shift to Prefill-Decode (PD) disaggregation in Large Language Model (LLM) serving, it is still difficult to maintain s...

- **Corsair: An In-Memory Computing Chiplet Architecture for Inference-Time Compute Acceleration** — S. Srivastava, Akhil Arunkumar, Nithesh kurella, A. Panda, Gaurav Jain, Purushotham Kamath, Mark Wutzke, Arun Tiruvur, M. Gupta, Ilya Soloveychik, Vamsi Darsi, M. Dalal, Vinayak Patankar, Sasidhar Dudyala, S. Duraisamy, Santhosh Ramchandran, R. Venkatasubramanian, Yuwei Qin, Xin Wang, Jayaprakash Balachandran, A. Gok, Piotr Wojciechowski, S. Ekanayake, Chris Ng, Ranju Sarma, Shubhankit Rathore, Tristan Trouwen, Siwei Zhuang, Chris Nicol, Sudeep Bhoja
  > Advances in generative AI (GenAI) have reinvigorated research into novel computing architectures such as Transformer. Transformer, characterized by low arithmetic intensity during most of the inferenc...

- **DisHelis: Optimizing Deployment of Disaggregated LLMs Inference Serving Over Heterogeneous Environments via Hierarchical Max-Flow** — Tao Zhang, Huihuang Qin, Dong Jin, Shuangwu Chen, Huasen He, Xiaobin Tan, Shiyin Zhu, Jian Yang
  > Disaggregated LLM inference service (DLIS), which decouples the compute-intensive prefill phase and the memory-intensive decode phase, enables more flexible and efficient resource usage. Existing solu...

- **DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing** — Lei Gao, Chaoyi Jiang, Hossein Entezari Zarch, Daniel Wong, Murali Annavaram
  > Modern LLM serving systems struggle to balance prefill and decode workloads. DuetServe introduces adaptive GPU multiplexing to dynamically allocate resources between prefill and decode stages, improvi...

- **Dynamically Reconfigurable NPU Acceleration for Knowledge Loading in LLM Retrieval-Augmented Generation** — Peidong Lin, Jintao Li, Hui Deng, Shihong Li, Shui Yu, Yun Li
  > Retrieval-Augmented Generation (RAG) provides large language models (LLMs) a means of retrieving relevant external knowledge, but its document parsing leads to increased latency and energy consumption...

- **EasySpec: Layer-Parallel Speculative Decoding for Efficient Multi-GPU Utilization** — Yize Wu, Ke Gao, Yanjun Wu
  [arXiv](https://arxiv.org/abs/2502.02493) | [GitHub](https://github.com/Yize-Wu/EasySpec)
  > Speculative decoding is an effective and lossless method for Large Language Model (LLM) inference acceleration. It employs a smaller model to generate a draft token sequence, which is then verified by...

- **EdgeSD: Efficient Speculative Decoding with Vision-Decoding Disaggregation for MLLM Inference in Edge-Cloud Networks** — Hualong Huang, Wenhan Zhan, Hancong Duan, Kai Peng, Geyong Min, Zijia Zhao, Zitian Zhao, Yalan Ye

- **Efficient In-Memory Acceleration of Sparse Block Diagonal LLMs** — J. Lima, Marc Dietrich, J. Castrillón, Asif Ali Khan
  [arXiv](https://arxiv.org/abs/2510.11192)
  > Structured sparsity enables deploying large language models (LLMs) on resource-constrained systems. Approaches like dense-to-sparse fine-tuning are particularly compelling, achieving remarkable struct...

- **Efficient Kernel Mapping and Comprehensive System Evaluation of LLM Acceleration on a CGLA** — Takuto Ando, Yu Eto, Ayumu Takeuchi, Yasuhiko Nakashima
  [arXiv](https://arxiv.org/abs/2512.00335)
  > Large Language Models (LLMs) demand substantial computational resources, resulting in high energy consumption on GPUs. To address this challenge, we focus on Coarse-Grained Reconfigurable Arrays (CGRA...

- **Efficient Pruning and Acceleration of Encoder-Based LLM Transformers on eFPGAs** — Omar Elayat, Vincent Gaudet, M. Elmasry
  > Transformer encoders such as Bidirectional Encoder Representations from Transformers (BERT) are widely adopted for Natural Language Processing (NLP) tasks, yet their computational and memory requireme...

- **EfficientEdit: Accelerating Code Editing via Edit-Oriented Speculative Decoding** — Peiding Wang, Li Zhang, Fang Liu, Yinghao Zhu, Wang Xu, Lin Shi, Xiaoli Lian, Minxiao Li, Bo Shen, An Fu
  [arXiv](https://arxiv.org/abs/2506.02780) | [GitHub](https://github.com/zhu-zhu-ding/EfficientEdit)
  > Large Language Models (LLMs) have demonstrated remarkable capabilities in code editing, substantially enhancing software development productivity. However, the inherent complexity of code editing task...

- **FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction** — Yuxuan Cai, Xiaozhuan Liang, Xinghua Wang, Jin Ma, Haijin Liang, Jinwen Luo, Xinyu Zuo, Lisheng Duan, Yuyang Yin, Xi Chen
  [arXiv](https://arxiv.org/abs/2509.18362)
  > As large language models (LLMs) become increasingly powerful, the sequential nature of autoregressive generation creates a fundamental throughput bottleneck that limits the practical deployment. While...

- **FlashInfer: Kernel Library for LLM Serving** — Unknown
  [arXiv](https://arxiv.org/abs/2501.01005) | [GitHub](https://github.com/flashinfer-ai/flashinfer)

- **FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design** — Hao Zhang, Aining Jia, Weifeng Bu, Yu Cai, Kai Sheng, Hao Chen, Xin He
  [arXiv](https://arxiv.org/abs/2508.04405) | [GitHub](https://github.com/FlyFoxPlayer/FlexQ)
  > Large Language Models (LLMs) demonstrate exceptional performance but entail significant memory and computational costs, restricting their practical deployment. While existing INT4/INT8 quantization re...

- **Glinthawk: A Two-Tiered Architecture for Offline LLM Inference** — Pouya Hamadanian, Sadjad Fouladi
  [arXiv](https://arxiv.org/abs/2501.11779) | [GitHub](https://github.com/https://github.com/microsoft/glinthawk)
  > We introduce Glinthawk, an architecture for offline Large Language Model (LLM) inference. By leveraging a two-tiered structure, Glinthawk optimizes the utilization of the high-end accelerators ("Tier ...

- **HpT: Hybrid Acceleration of Spatio-Temporal Attention Model Training on Heterogeneous Manycore Architectures** — S. Dahal, Pratyush Dhingra, Krishu K. Thapa, P. Pande, Ananth Kalyanaraman
  > Transformer models have become widely popular in numerous applications, and especially for building foundation large language models (LLMs). Recently, there has been a surge in the exploration of tran...

- **La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation** — Kai Liu, Bowen Xu, Shaoyu Wu, Xin Chen, Hao Zhou, Yongliang Tao, Lulu Hu
  [arXiv](https://arxiv.org/abs/2507.01299)
  > Activation sparsity can reduce the computational overhead and memory transfers during the forward pass of Large Language Model (LLM) inference. Existing methods face limitations, either demanding time...

- **LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design** — Renjie Wei, Songqiang Xu, Linfeng Zhong, Zebin Yang, Qingyu Guo, Yuan Wang, Runsheng Wang, Meng Li
  [arXiv](https://arxiv.org/abs/2502.15260)
  > State space models (SSMs) like Mamba have recently attracted much attention. Compared to Transformer-based large language models (LLMs), Mamba achieves linear computation complexity with the sequence ...

- **Llama Stack** — Unknown
  > Llama Stack是一个开源的AI应用代理API服务器，提供OpenAI兼容的API，可以在任何地方运行——笔记本电脑、数据中心或云端。使用任何OpenAI兼容的客户端或代理框架。可以在不更改应用代码的情况下在Llama、GPT、Gemini、Mistral或任何模型之间切换。...

- **MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures** — Jiayu Qin, Jianchao Tan, Kefeng Zhang, Xunliang Cai, Wei Wang
  [arXiv](https://arxiv.org/abs/2502.14008)
  > The remarkable performance of large language models (LLMs) in various language tasks has attracted considerable attention. However, the ever-increasing size of these models presents growing challenges...

- **MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching** — Tairan Xu, Leyang Xue, Zhan Lu, Adrian Jackson, Luo Mai
  [arXiv](https://arxiv.org/abs/2503.09716) | [GitHub](https://github.com/EfficientMoE/MoE-Gen)
  > This paper presents MoE-Gen, a high-throughput MoE inference system optimized for single-GPU execution. Existing inference systems rely on model-based or continuous batching strategies, originally des...

- **NIXL: NVIDIA Inference Xfer Library** — Unknown
  > NIXL (NVIDIA Inference Xfer Library) 是用于加速AI推理框架中点对点通信的库，特别是为NVIDIA Dynamo等推理框架设计。NIXL提供了对各种类型内存（CPU和GPU）和存储（文件、块和对象存储）的抽象，通过模块化插件架构实现。...

- **NanoFlow: Towards Optimal Large Language Model Serving Throughput** — Kan Zhu, Yufei Gao, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Tian Tang, Qinyu Xu, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Ziren Wang, Stephanie Wang, Arvind Krishnamurthy, Baris Ka
  > Large Language Models (LLMs) have resulted in a surging demand for planet-scale serving. Despite significant advancements in LLM inference systems, achieving optimal throughput remains challenging due...

- **Optimizing LLM inference for FPGAs** — J. R. de Freitas, J. G. Coutinho, Ce Guo, S. Demirsoy, Wayne Luk, Zhiqiang Que
  [GitHub](https://github.com/custom-computing-ic/llm-oneapi-fpga)
  > Large Language Models (LLMs) deliver state-of-the-art performance but demand high computation and memory, making deployment in resource-limited settings challenging. Field-Programmable Gate Arrays (FP...

- **P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference Using Hybrid Numerical Formats** — Yuzong Chen, Chao Fang, Xilai Dai, Yuheng Wu, Thierry Tambe, Marian Verhelst, Mohamed S. Abdelfattah
  [arXiv](https://arxiv.org/abs/2511.06838) | [GitHub](https://github.com/yc2367/P3-LLM)
  > The substantial memory bandwidth and computational demands of large language models (LLMs) present critical challenges for efficient inference. To tackle this, the literature has explored heterogeneou...

- **PICNIC: Silicon Photonic Interconnected Chiplets with Computational Network and In-memory Computing for LLM Inference Acceleration** — Yue Jiet Chong, Yimin Wang, Zhen Wu, Xuanyao Fong
  [arXiv](https://arxiv.org/abs/2511.04036)
  > This paper presents a 3D-stacked chiplets based large language model (LLM) inference accelerator, consisting of non-volatile in-memory-computing processing elements (PEs) and Inter-PE Computational Ne...

- **Pie: A Programmable Serving System for Emerging LLM Applications** — In Gim 等
  > Emerging large language model (LLM) applications involve diverse reasoning strategies and agentic workflows, straining the capabilities of existing serving systems built on a monolithic token generati...

- **Reasoning Language Model Inference Serving Unveiled: An Empirical Study** — Qi Li, Junpan Wu, Xiang Liu, et al.
  > The reasoning large language model (RLLM) has been proven competitive in solving complex reasoning tasks such as mathematics, coding, compared to general LLM. However, the unique inference patterns of...

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

- **Survey on Efficient Large Language Models: Principles, Algorithms, Applications, and Open Issues.** — Jian Cheng, Haidong Kang, Yuxin Shao, Nan Li, Pengjun Chen, Rui Wang, Saiqin Long, Xiaochun Yang, Lianbo Ma
  > With the rapid advancement of large language models (LLMs) in both academia and industry, their growing size and complexity have introduced significant challenges in terms of computational cost and de...

- **The Anatomy of a Triton Attention Kernel** — Burkhard Ringlein, Jan van Lunteren, Radu Stoica, Thomas Parnell
  [arXiv](https://arxiv.org/abs/2511.11581)
  > A long-standing goal in both industry and academia is to develop an LLM inference platform that is portable across hardware architectures, eliminates the need for low-level hand-tuning, and still deli...

- **TokenSwift: Ultra Long Sequence Generation** — bigai-nlco
  > 1. **Hierarchical speculation**: Multi-level draft generation pipeline 2. **Long context optimization**: Specifically designed for 10K+ token sequences 3. **Memory efficiency**: Optimized KVCache mana...

- **Trinity: Disaggregating Vector Search from Prefill-Decode Disaggregation in LLM Serving** — Yi Liu, Chen Qian
  [arXiv](https://arxiv.org/abs/2512.02281)
  > Trinity consolidates all retrieval into a single shared vector-search GPU pool working with PD disaggregated LLM serving. Introduces: (1) novel architecture for GPU-based vector search in PD disaggreg...

- **UniCAIM: A Unified CAM/CIM Architecture with Static-Dynamic KV Cache Pruning for Efficient Long-Context LLM Inference** — Weikai Xu, Wenxuan Zeng, Qianqian Huang, Meng Li, Ruei-Hao Huang
  [arXiv](https://arxiv.org/abs/2504.07479)
  > Transformer-based large language models (LLMs) have achieved impressive performance in various natural language processing (NLP) applications. However, the high memory and computation cost induced by ...

- **Variation-aware Vision Token Dropping for Faster Large Vision-Language Models** — Junjie Chen, Xuyang Liu, Zichen Wen, Yiyu Wang, Siteng Huang, Honggang Chen
  [arXiv](https://arxiv.org/abs/2509.01552)
  > Large vision-language models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding tasks. However, the increasing demand for high-resolution image and long-video understanding ...

- **xLLM Technical Report** — Tongxuan Liu, Tao Peng, Peijun Yang, et al.
  > We introduce xLLM, an intelligent and efficient Large Language Model (LLM) inference framework designed for high-performance, large-scale enterprise-grade deployments. xLLM addresses the critical chal...


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

- **Idle Consumer GPUs as a Complement to Enterprise Hardware for LLM Inference: Performance, Cost and Carbon Analysis** — ['A. Almeida']
  > We examine the cost-performance landscape of Large Language Model (LLM) inference across two GPU tiers: Nvidia's enterprise-class H100 and the widely available consumer-grade RTX 4090. We benchmark la...

- **LLM-Optimized Cloud Architectures: Evaluating Infrastructure Patterns For Fine-Tuning And Serving Large Models** — ['Satya Teja Muddada']
  > Large Language Models have ignited a paradigm shift in the field of artificial intelligence, but their implementation comes with daunting infrastructure issues that traditional cloud architectures can...

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

- **DSD: A Distributed Speculative Decoding Solution for Edge-Cloud Agile Large Model Serving** — Fengze Yu, Leshu Li, Brad McDanel, Sai Qian Zhang
  > Large language model (LLM) inference often suffers from high latency, which limits its practical applicability in real-time applications. Speculative decoding has emerged as a promising technique to r...

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

- **HeiSD: Hybrid Speculative Decoding for Embodied Vision-Language-Action Models with Kinematic Awarene** — Zihao Zheng, Zhihao Mao, Sicheng Tian, Maoliang Li, Jiayu Chen, Xinhao Sun, Zhaobo Zhang, Xuanzhe Liu, Donggang Cao, Hong Mei, Xiang Chen
  > 视觉语言动作模型 (Vision-Language-Action, VLA) 已成为机器人控制的主流解决方案，但推理速度较慢。  投机解码 (Speculative Decoding, SD) 是一种有前景的加速方法，可分为两类： - 基于draft的SD - 基于检索的SD  现有方法未能分析VLA模型的独特优势。  **HeiSD** 提出了一种**具有运动学感知的混合投机解码**方法。...

- **LLMs on a Budget? Say HOLA** — Z. Siddiqui, Jiechao Gao, Ebad Shabbir, M. Azeez, Rafiq Ali
  [arXiv](https://arxiv.org/abs/2506.18952) | [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
  > Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded s...

- **LP-Spec: Leveraging LPDDR PIM for Efficient LLM Mobile Speculative Inference with Architecture-Dataflow Co-Optimization** — Siyuan He, Zhantong Zhu, Yandong He, Tianyu Jia
  [arXiv](https://arxiv.org/abs/2508.07227)
  > LLM inference on mobile devices faces extraneous challenges due to limited memory bandwidth and computational resources. To address these issues, speculative inference and processing-in-memory (PIM) t...

- **MMSpec: Benchmarking Speculative Decoding for Vision-Language Models** — Hui Shen, Xin Wang, Ping Zhang, Yunta Hsieh, Qi Han, Zhongwei Wan, Ziheng Zhang, Jingxuan Zhang, Jing Xiong, Ziyuan Liu, Yifan Zhang, Hangrui Cao, Chenyang Zhao, Mi Zhang
  > 视觉语言模型 (Vision-Language Models, VLMs) 在多模态任务上表现出色，但由于模型规模大、上下文长，推理延迟很高。  投机解码 (Speculative Decoding, SD) 是一种有前景的加速方法，但现有工作主要集中在纯语言模型上，VLM上的投机解码缺乏系统研究。  **MMSpec** 是第一个专门针对**视觉语言模型投机解码**的基准测试框架。...

- **Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference** — ['Nikhil Bhendawade', 'Kumari Nishu', 'Arnav Kundu', 'Chris Bartels', 'Minsik Cho', 'Irina Belousova']
  [arXiv](https://arxiv.org/abs/2510.13161) | [GitHub](https://github.com/psmarter/mini-infer)
  > Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rate...

- **ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decodi** — Quan Kong, Yuhao Shen, Yicheng Ji, Huan Li, Cong Wang
  > 尽管当前的视频语言模型 (Video-LLMs) 在视频理解任务上取得了令人印象深刻的性能，但它们的自回归解码效率仍然受到大量视频token的限制。  视觉token剪枝可以部分缓解这一瓶颈，但现有方法仍存在信息丢失问题，且加速效果有限。  **ParallelVLM** 提出了一种**视觉对齐感知的并行投机解码**方法，实现无损加速。...

- **Pipelined Collaborative Speculative Decoding Framework for Efficient Edge-Cloud LLM Inference** — Yida Zhang, Zhiyong Gao, Shuaibing Yue, Jie Li, Rui Wang
  > 边缘-云协作推理已成为平衡设备端计算能力和云端强大模型能力的重要范式。然而，如何有效平衡边缘设备的计算能力与云端能力仍是一个开放问题。  **Pipelined Collaborative Speculative Decoding** 是一个高效的**边缘-云LLM推理的流水线协作投机解码框架**。...

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
  > 大型语言模型由于顺序自回归解码而产生高推理延迟。Speculative Decoding (投机解码) 通过使用draft模型预测多个token，然后使用target模型并行验证，是加速自回归LLM推理的一种有前景的方法。  然而，现有的投机解码框架主要关注推理阶段，缺乏对draft模型训练的支持，导致部署效率受限。  **SpecForge** 是一个灵活高效的**开源训练框架**，专门用于投机...

- **SpecSteer: Synergizing Local Context and Global Reasoning for Efficient Personalized Generation** — Hang Lv, Sheng Liang, Hao Wang, Yongyue Zhang, Hongchao Gu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen
  > 个性化生成需要结合用户的本地上下文和云端的大规模推理能力。本地设备上的模型可以快速响应，但能力有限；云端模型能力强，但延迟高。  **SpecSteer** 是一个将私有设备端上下文与云端规模推理协同的框架。...

- **Speculative Decoding in Decentralized LLM Inference: Turning Communication Latency into Computation Throughput** — ['Jingwei Song', 'Wanyi Chen', 'Xinyuan Song', 'Chris Tong', 'Gufeng Chen', 'Tianyi Zhao', 'Eric Yang', 'Bill Shi', 'Lynn Ai', 'Gradient Network']
  [arXiv](https://arxiv.org/abs/2511.11733) | [GitHub](https://github.com/sgl-project/sglang)
  > Speculative decoding accelerates large language model (LLM) inference by using a lightweight draft model to propose tokens that are later verified by a stronger target model. While effective in centra...

- **Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding** — Sudhanshu Agrawal, Risheek Garrepalli, Raghavv Goel, Mingu Lee, Christopher Lott, Fatih Porikli
  > Diffusion-based LLMs offer an alternative to autoregressive generation. Spiffy applies speculative decoding techniques to diffusion models, achieving lossless acceleration of the generation process....

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


### Inference Kernel

- **FlashAttention-3: Fast and Accurate Attention** — TriDao et al.
  > 1. **Asynchrony**: Overlaps computation and memory operations 2. **Low-precision**: FP8 support with minimal accuracy loss 3. **Hardware optimization**: Better utilization of modern GPU tensor cores 4...

- **Star-Attention: Efficient LLM Inference over Long Sequences** — NVIDIA
  > 1. **Reduces attention complexity** from O(n²) to O(n) 2. **Maintains model quality** with minimal accuracy loss 3. **Achieves 11x speedup** on long sequence benchmarks  This is achieved by dividing t...


### KV Cache

- **Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache** — Bin Lin, Chen Zhang, Tao Peng, et al.
  > Large Language Models (LLMs) demonstrate substantial potential across a diverse array of domains via request serving. However, as trends continue to push for expanding context sizes, the autoregressiv...

- **InstCache: Predictive Cache for LLM Serving** — Various
  > 1. **Prefix prediction**: Predicts common prompt prefixes 2. **Intent anticipation**: Anticipates user intent from context 3. **Smart pre-caching**: Loads likely-needed KV cache in advance 4. **High h...

- **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache** — (待补充)
  [GitHub](https://github.com/jy-yuan/KIVI)
  > KV cache quantization is crucial for reducing memory footprint in LLM inference. This paper presents KIVI, a tuning-free asymmetric 2bit quantization method for KV cache that achieves minimal accuracy...

- **KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head** — Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, Ashish Panwar
  > Context lengths of Large Language Models (LLMs) have exploded in recent years, with 128k-token context becoming a standard and million-token context becoming a reality. Efficiently supporting long-con...

- **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization** — (待补充)
  [GitHub](https://github.com/SqueezeAILab/KVQuant)
  > This paper presents KVQuant, a KV cache quantization method that enables LLM inference with context lengths up to 10 million tokens. The method uses per-channel scaling and asymmetric quantization to ...

- **MiniKV: Layer-Discriminative KV Cache** — Various (Microsoft?)
  > 1. **Layer-aware quantization**: Different layers get different precision 2. **2-bit compression**: Aggressive KV cache compression 3. **Minimal accuracy loss**: Maintains model quality through carefu...

- **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** — Moonshot AI
  > The system implements a novel architecture that separates the prefill and decode stages while focusing on maximizing KVCache reuse across requests. Mooncake achieves significantly higher throughput co...

- **ShadowKV: KV Cache in Shadows for Long-Context Inference** — Various
  > 1. **Shadow cache**: Secondary cache layer for efficiency 2. **Hierarchical caching**: Multi-level cache hierarchy 3. **Selective computation**: Only compute when necessary 4. **3-4x speedup**: Signif...


### LLM Serving

- **Chameleon: Adaptive Caching for Multi-Adapter LLM Inference** — Various
  > 1. **Adapter-aware caching**: Different adapters have different cache needs 2. **Adaptive scheduling**: Dynamic adjustment based on workload 3. **Memory optimization**: Efficient sharing of cache acro...

- **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** — Y. Sheng, L. Zheng, Y. Zhu, et al. (PKU)
  > We propose DistServe, a system that disaggregates prefilling and decoding to different GPUs. The key insight is that isolating the two phases eliminates interference and enables tailoring the hardware...

- **Lookahead Decoding: Break the Sequential Dependency of LLM Inference** — (待补充)
  [GitHub](https://github.com/hao-ai-lab/LookaheadDecoding)
  > Large Language Models (LLMs) generate tokens auto-regressively, which creates sequential dependency that limits parallelization during inference. This paper presents Lookahead Decoding, a technique th...

- **PowerInfer-2: High-Speed LLM Inference for Smartphones** — 来自上海交通大学IPADS实验室
  [GitHub](https://github.com/Tiiny-AI/PowerInfer)
  > PowerInfer-2是一个专为智能手机设计的高度优化推理框架。  使用TurboSparse-Mixtral-47B，PowerInfer-2在智能手机上实现了**11.68 tokens/s**的推理速度，比其他SOTA框架快达**22倍**。...

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

- **ALISE: Speculative Scheduling for LLM Serving** — Various
  > 1. **Speculative scheduling**: Intelligently schedules speculation 2. **Workload-aware**: Adapts to request characteristics 3. **Batch optimization**: Better batching decisions with speculation 4. **I...

- **DART: Diffusion-Inspired Speculative Decoding** — Various
  > 1. **Multiple draft candidates**: Generate several possible token sequences 2. **Tree-based verification**: Efficiently verify multiple candidates 3. **Quality-aware selection**: Choose best among mul...

- **EAGLE: Early Exit and Speculative Decoding** — Various (SafeAILab)
  > 1. **EAGLE-1 (ICML'24)**: Uses early-exit techniques to speed up draft token generation 2. **EAGLE-2 (EMNLP'24)**: Improved speculative decoding with better acceptance rates 3. **EAGLE-3 (NeurIPS'25)*...

- **LayerSkip: Early Exit and Self-Speculative Decoding** — Facebook Research
  > 1. **Early exit at variable layers**: Different tokens exit at different layers based on confidence 2. **Self-speculative decoding**: Model drafts tokens and verifies them in a unified framework 3. **...

- **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding** — (待补充)
  [GitHub](https://github.com/facebookresearch/LayerSkip)
  > LayerSkip combines early exit inference with self-speculative decoding, allowing LLMs to dynamically skip layers during inference based on sample difficulty, while using the same model for both draft ...

- **PipeInfer: Asynchronous Pipelined Speculation** — AutonomicPerfectionist
  > 1. **Pipelining**: Overlap multiple requests' computation 2. **Speculation**: Use draft tokens for acceleration 3. **Asynchronous execution**: Maximize GPU utilization  This approach achieves better t...

- **REST: Retrieval-Based Speculative Decoding** — FasterDecoding
  > 1. **Retrieves similar contexts** from recent history 2. **Uses retrieved tokens as drafts** for verification 3. **Achieves high acceptance rates** when prompts share common patterns  This approach is...

- **Sequoia: Tree-Based Speculative Decoding** — Infini-AI-Lab
  > 1. **Efficient tree construction**: Optimal draft tree structure 2. **Robust verification**: Handles various acceptance scenarios 3. **Adaptive strategy**: Adjusts to different workload characteristic...

- **TriForce: Hierarchical Speculative Decoding** — Infini-AI-Lab
  > 1. **Coarse-level draft**: Fast, smaller model generates draft 2. **Medium-level verification**: Intermediate model verifies 3. **Fine-level verification**: Full model confirms final tokens  This hier...

- **TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding** — (待补充)
  [GitHub](https://github.com/Infini-AI-Lab/TriForce)
  > TriForce presents a hierarchical speculative decoding approach that uses multiple levels of draft models to achieve lossless acceleration of long sequence generation, addressing the verification bottl...

- **speculative_decoding_survey** — Unknown


## 2023


### Inference Kernel

- **vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention** — Unknown
  [arXiv](https://arxiv.org/abs/2309.06180) | [GitHub](https://github.com/vllm-project/vllm)
  > ## 摘要 (中文) 大型语言模型（LLM）的高吞吐量服务需要同时批处理足够多的请求。然而，现有系统面临挑战，因为每个请求的键值缓存（KV缓存）内存很大，且会动态增长和收缩。当管理不高效时，这些内存会因碎片化和冗余复制而严重浪费，限制了批处理大小。为解决这一问题，我们提出了PagedAttention，这是一种受操作系统中经典虚拟内存和分页技术启发的注意力算法。在此基础上，我们构建了vLLM，一个...
