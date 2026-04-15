#!/usr/bin/env python3
"""Add 7 new LLM Serving papers to the database and create markdown files."""

import json
import os
import re
from datetime import datetime

DB_PATH = "/home/admin/claw_notes/database.json"
ARXIV_DIR = "/home/admin/claw_notes/arxiv/2026"

# Load existing database
with open(DB_PATH, 'r') as f:
    db = json.load(f)

existing_titles = set(p['title'].lower().strip() for p in db['papers'])

new_papers = [
    {
        "title": "PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving",
        "topic": "LLM Serving",
        "category": "LLM Serving",
        "source": "arxiv",
        "arxiv_id": "2604.12171",
        "url": "https://arxiv.org/abs/2604.12171",
        "pdf_url": "https://arxiv.org/pdf/2604.12171",
        "date": "2026-04-14",
        "authors": "Xu Bai, Muhammed Tawfiqul Islam, Chen Wang, Adel N. Toosi",
        "summary_en": "Pipeline parallelism (PP) is widely used to partition layers of LLMs across GPUs for scalable inference. However, existing systems rely on static PP configurations that fail to adapt to dynamic settings. Reconfiguring PP by stopping and redeploying service incurs prohibitive downtime, so reconfiguration must proceed live and in-place without interrupting inference. PipeLive enables live in-place PP reconfiguration with minimal disruption. It introduces a redesigned KV cache layout together with a co-designed extension to PageAttention, forming a unified mechanism for live KV resizing. It adopts an incremental KV patching mechanism, inspired by live VM migration, to synchronize KV states between source and target configurations. PipeLive achieves 2.5X TTFT reduction without KV cache overflow, reduces reconfiguration overhead from seconds to under 10ms, and improves TTFT and TPOT by up to 54.7% and 14.7% respectively.",
        "summary_cn": "Pipeline并行（PP）广泛用于将LLM层分区到多个GPU以实现可扩展推理。但现有系统依赖静态PP配置，无法适应动态环境。通过停止和重新部署服务来重新配置PP会产生不可接受的停机时间，因此重新配置必须在不中断推理的情况下在线原地进行。PipeLive实现了最小干扰的在线原地PP重配置。引入重新设计的KV缓存布局和PageAttention扩展，形成统一的在线KV调整机制。采用增量KV补丁机制（受虚拟机在线迁移启发），在源和目标配置间同步KV状态。PipeLive在不发生KV缓存溢出的情况下实现2.5倍TTFT降低，将重配置开销从数秒降至10ms以下，TTFT和TPOT分别改善54.7%和14.7%。",
        "tags": ["pipeline-parallelism", "dynamic-serving", "kv-cache-resizing", "live-reconfiguration", "pipelive"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
    {
        "title": "RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving",
        "topic": "LLM Serving",
        "category": "LLM Serving",
        "source": "arxiv",
        "arxiv_id": "2604.10907",
        "url": "https://arxiv.org/abs/2604.10907",
        "pdf_url": "https://arxiv.org/pdf/2604.10907",
        "date": "2026-04-12",
        "authors": "Hossein Hosseini Kasnavieh, Christopher Leckie, Adel N. Toosi",
        "summary_en": "Multi-model LLM routing has emerged as an effective approach for reducing serving cost and latency while maintaining output quality. However, prior routing methods typically assume each model has fixed latency. In real deployments, multiple models often share limited GPU resources, and a model's latency depends strongly on both its allocated resources and the request load induced by the routing policy. Consequently, routing and resource allocation are tightly coupled. RouterWise formalizes this as a constrained joint optimization over deployment setup and routing fractions, combining a dual-price formulation for score-maximizing routing with setup-specific latency models derived from system profiling. Results show that achievable output-quality score can vary by up to 87% across retained setups on the same GPU cluster.",
        "summary_cn": "多模型LLM路由已成为降低服务成本和延迟同时保持输出质量的有效方法。但先前路由方法假设每个模型有固定延迟，这在实际部署中不准确：多个模型往往共享有限的GPU资源，模型的延迟很大程度上取决于其分配资源和路由策略引起的请求负载。路由和资源分配紧密耦合。RouterWise将此问题形式化为部署设置和路由比例的约束联合优化，结合双价格公式进行分数最大化路由和基于系统分析的设置特定延迟模型。结果显示，在同一GPU集群上，可实现的输出质量分数在不同设置间变化高达87%。",
        "tags": ["multi-model-routing", "resource-allocation", "latency-aware", "joint-optimization", "routerwise"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
    {
        "title": "StepCache: Step-Level Reuse with Lightweight Verification and Selective Patching for LLM Serving",
        "topic": "LLM Serving",
        "category": "LLM Serving",
        "source": "arxiv",
        "arxiv_id": "2603.28795",
        "url": "https://arxiv.org/abs/2603.28795",
        "pdf_url": "https://arxiv.org/pdf/2603.28795",
        "date": "2026-03-24",
        "authors": "Azam Nouri",
        "summary_en": "StepCache addresses LLM serving workloads where repeated requests share common solution structure but differ in localized constraints. Prior caching approaches reuse either full responses (semantic caching) or model-internal KV/prefix states, which are respectively brittle under partial changes or tightly coupled to specific backends. StepCache is a backend-agnostic step-level reuse layer that segments outputs into ordered steps, retrieves the best-matching cached request, verifies steps using lightweight task-aware checks, and regenerates only failing regions via selective patching. It supports strict structured-output enforcement for JSON, including single-step extraction and required-key constraints. In perturbation-heavy micro-benchmarks, StepCache reduces mean latency from 2.13s to 0.67s, median latency from 2.42s to 0.01s, and improves end-to-end correctness from 72.5% to 100%. 79.7% of requests take the reuse-only fast path.",
        "summary_cn": "StepCache处理LLM服务中重复请求共享解决方案结构但局部约束不同的场景。先前缓存方法要么重用完整响应（语义缓存），要么重用模型内部KV/前缀状态，前者在部分变更下脆弱，后者与特定后端紧密耦合。StepCache是后端无关的步骤级重用层，将输出分段为有序步骤，检索最佳匹配的缓存请求，使用轻量任务感知检查验证步骤，并通过选择性补丁仅重新生成失败区域。支持JSON的严格结构化输出强制执行。在扰动密集的微基准测试中，StepCache将平均延迟从2.13秒降至0.67秒，中位延迟从2.42秒降至0.01秒，端到端正确性从72.5%提升至100%。79.7%的请求走重用快速路径。",
        "tags": ["step-level-caching", "selective-patching", "structured-output", "reuse", "stepcache"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
    {
        "title": "Chimera: Latency- and Performance-Aware Multi-agent Serving for Heterogeneous LLMs",
        "topic": "LLM Serving",
        "category": "LLM Serving",
        "source": "arxiv",
        "arxiv_id": "2603.22206",
        "url": "https://arxiv.org/abs/2603.22206",
        "pdf_url": "https://arxiv.org/pdf/2603.22206",
        "date": "2026-03-23",
        "authors": "Kangqi Ni, Wenyue Hua, Xiaoxiang Shi, Jiang Guo, Shiyu Chang, Tianlong Chen",
        "summary_en": "Multi-agent applications execute complex tasks as multi-stage workflows where each stage is an LLM call. Existing LLM serving systems largely assume homogeneous clusters with identical model replicas, overlooking the potential of heterogeneous deployments where models of different sizes enable finer latency-performance trade-offs. Chimera is a predictive scheduling system for multi-agent workflow serving on heterogeneous LLM clusters. It applies semantic routing to estimate per-model confidence scores, predicts total remaining output length, and estimates per-model congestion using in-flight predicted token volumes for load balancing. Evaluated on code generation and math reasoning workflows, Chimera traces the best latency-performance frontier, reducing end-to-end latency by 1.2-2.4x and improving task performance by 8.0-9.5 percentage points over competitive baselines including vLLM.",
        "summary_cn": "多智能体应用将复杂任务作为多阶段工作流执行，每个阶段是一个LLM调用。现有LLM服务系统大多假设同构集群（相同模型副本），忽略了异构部署的潜力——不同大小和能力模型可以实现更精细的延迟-性能权衡。Chimera是异构LLM集群上多智能体工作流服务的预测调度系统。应用语义路由估计每个模型的置信度分数，预测工作流总剩余输出长度，并使用在途预测token量估计每模型拥塞以进行负载均衡。在代码生成和数学推理工作流上评估，Chimera追踪最佳延迟-性能前沿，端到端延迟降低1.2-2.4倍，任务性能比vLLM等基线改善8.0-9.5个百分点。",
        "tags": ["multi-agent-serving", "heterogeneous-llm", "semantic-routing", "load-balancing", "chimera"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
    {
        "title": "SMART: When is it Actually Worth Expanding a Speculative Tree?",
        "topic": "LLM Serving",
        "category": "Speculative Decoding",
        "source": "arxiv",
        "arxiv_id": "2604.09731",
        "url": "https://arxiv.org/abs/2604.09731",
        "pdf_url": "https://arxiv.org/pdf/2604.09731",
        "date": "2026-04-09",
        "authors": "Lifu Wang, Pan Zhou",
        "summary_en": "Tree-based speculative decoding accelerates autoregressive generation by verifying a branching tree of draft tokens in a single target-model forward pass. However, existing methods prioritize maximizing token-level likelihood or accepted tokens while ignoring a critical efficiency paradox: computational overhead of drafting and verifying big trees can grow super-linearly, leading to negative wall-clock speedup when batch sizes increase or hardware saturation limits are reached. SMART is a system-aware marginal analysis framework for runtime tree construction. It reformulates tree expansion as a hardware-aware optimization problem directly maximizing end-to-end speedup. By applying a principled marginal benefit-cost rule at inference time, SMART expands a node only when its marginal benefit-cost ratio exceeds the tree-level speedup. SMART is training-free and plug-and-play for existing frameworks like MSD and EAGLE. It delivers average additional speedup of 20.0% for MLLMs and 15.4% for LLMs across compute-bound batching regimes.",
        "summary_cn": "基于树的投机解码通过在单次目标模型前向传播中验证分支草稿token树来加速自回归生成。但现有方法优先最大化token级似然或接受token数，忽略了关键的效率悖论：草拟和验证大树的计算开销可能超线性增长，导致批次增大或硬件饱和时出现负墙钟加速。SMART是运行时树构建的系统感知边际分析框架。将树扩展重新表述为直接最大化端到端加速的硬件感知优化问题。在推理时应用边际收益-成本规则，仅当边际收益-成本比超过树级加速时才扩展节点。SMART无需训练，可作为MSD和EAGLE等框架的即插即用控制器。在计算受限批次场景下，MLLM平均额外加速20.0%，LLM平均额外加速15.4%。",
        "tags": ["speculative-decoding", "tree-construction", "marginal-analysis", "hardware-aware", "smart"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
    {
        "title": "CALVO: Improve Serving Efficiency for LLM Inferences with Intense Network Demands",
        "topic": "LLM Serving",
        "category": "LLM Serving",
        "source": "arxiv",
        "arxiv_id": "2603.21257",
        "url": "https://arxiv.org/abs/2603.21257",
        "pdf_url": "https://arxiv.org/pdf/2603.21257",
        "date": "2026-03-22",
        "authors": "Weiye Wang, Chen Chen, Junxue Zhang, Zhusheng Wang, Hui Yuan, Zixuan Guan, Xiaolong Zheng, Qizhen Weng, Yin Chen, Minyi Guo",
        "summary_en": "Distributed prefix caching has become a core technique for efficient LLM serving. However, for long-context requests with high cache hit ratios, retrieving reusable KVCache blocks from remote servers has emerged as a new performance bottleneck. Such network-intensive LLM inference is expected to become increasingly common as agentic AI workloads grow. Existing LLM inference engines remain compute-centric: they treat KVCache loading as a subordinate phase to GPU execution and fail to account for its delay explicitly during scheduling. CALVO is an LLM serving engine that treats KVCache loading as a first-class concern. It decouples KVCache loading and GPU computation into independently managed, asynchronously progressing stages, enabling better utilization of network, PCIe, and computation resources. CALVO incorporates KVCache loading delay as an explicit component of per-request service cost, achieving up to 61.67% higher SLO attainment than the baseline.",
        "summary_cn": "分布式前缀缓存已成为高效LLM服务的核心技术。但对于长上下文请求的高缓存命中比，从远程服务器检索可重用KVCache块已成为新的性能瓶颈。随着智能体AI工作负载增长，这种网络密集型LLM推理将越来越常见。现有LLM推理引擎仍以计算为中心：将KVCache加载作为GPU执行的附属阶段，未在调度中显式考虑其延迟。CALVO是将KVCache加载作为首要关注的LLM服务引擎。将KVCache加载和GPU计算解耦为独立管理的异步推进阶段，更好利用网络、PCIe和计算资源。将KVCache加载延迟作为每请求服务成本的显式组成部分，SLO达成率比基线高61.67%。",
        "tags": ["kv-cache-loading", "network-intensive", "distributed-prefix-caching", "scheduling", "calvo"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
    {
        "title": "Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate",
        "topic": "LLM Serving",
        "category": "LLM Serving",
        "source": "arxiv",
        "arxiv_id": "2604.07874",
        "url": "https://arxiv.org/abs/2604.07874",
        "pdf_url": "https://arxiv.org/pdf/2604.07874",
        "date": "2026-04-09",
        "authors": "Fangyue Liu, Hua Liu, Xinyuan Lyu, Shuo Ai, Hao Liang, Lingpeng Chen, Ziqian Hu, Chong Zha, Xin Jin, Hanmei Luo, Peng Chen",
        "summary_en": "LLM inference powers latency-critical production services. The bursty nature of inference traffic results in over-provisioning, which leads to resource underutilization. While online-offline colocation promises to utilize idle capacity, broad production deployment must overcome two challenges: (i) large online interference due to slow or frequent preemptions, and (ii) extensive framework and driver modifications. Valve is a production-friendly colocation system that jointly bounds preemption latency and rate. It enables sub-millisecond compute preemption at most once per online request, and rate-limited sub-layer memory reclamation. These guarantees are provided by a GPU runtime combining channel-controlled compute isolation, page-fault-free memory reclamation, and dynamic memory reservation. Valve requires only one line of driver modification and 20 lines of framework patch. Deployed on 8,054 GPUs in production, Valve improves cluster utilization by 34.6% (2,170 GPU savings) with <5% TTFT increase and <2% TPOT increase.",
        "summary_cn": "LLM推理驱动延迟关键的生产服务。推理流量的突发特性导致过度配置，进而导致资源利用率不足。虽然在线-离线混部有望利用闲置容量，但大规模生产部署需要克服两大挑战：(i)由于慢或频繁抢占导致的大在线干扰，(ii)大量框架和驱动修改。Valve是生产友好的混部系统，联合约束抢占延迟和抢占率。实现在线请求最多一次的亚毫秒计算抢占，以及速率受限的子层内存回收。通过GPU运行时结合通道控制计算隔离、无页错内存回收和动态内存预留来提供这些保证。仅需1行驱动修改和20行框架补丁。在8,054个GPU的生产部署中，集群利用率提升34.6%（节省2,170个GPU），TTFT增加<5%，TPOT增加<2%。",
        "tags": ["online-offline-colocation", "preemption", "production-deployment", "gpu-isolation", "valve"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
    {
        "title": "Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions",
        "topic": "LLM Serving",
        "category": "LLM Inference",
        "source": "arxiv",
        "arxiv_id": "2604.07931",
        "url": "https://arxiv.org/abs/2604.07931",
        "pdf_url": "https://arxiv.org/pdf/2604.07931",
        "date": "2026-04-09",
        "authors": "Jing Wang, Yu-Yang Qian, Ke Xue, Chao Qian, Peng Zhao, Zhi-Hua Zhou",
        "summary_en": "Output-length prediction is important for efficient LLM serving, as it directly affects batching, memory reservation, and scheduling. Most existing methods use a one-shot sampled length as the label, implicitly treating each prompt as having one true target length. This is unreliable: even under a fixed model and decoding setup, the same prompt induces a prompt-conditioned output length distribution, not a deterministic scalar, and this distribution exhibits heavy-tailed behavior. Robust Length Prediction casts length prediction as robust estimation from heavy-tailed prompt-conditioned length distributions. It proposes ProD methods, which construct training targets from multiple independent generations of the same prompt. ProD-M uses a median-based target for robust point prediction; ProD-D uses a distributional target preserving prompt-conditioned uncertainty. Experiments across diverse scenarios show consistent gains in prediction quality.",
        "summary_cn": "输出长度预测对高效LLM服务很重要，直接影响批处理、内存预留和调度。大多数现有方法使用一次性采样长度作为标签，隐含地将每个提示视为有一个真实目标长度。这是不可靠的：即使在固定模型和解码设置下，同一提示诱导的是提示条件输出长度分布，而非确定性标量，且该分布表现出重尾行为。鲁棒长度预测将长度预测重新表述为重尾提示条件长度分布的鲁棒估计。提出ProD方法，从同一提示的多次独立生成构建训练目标。ProD-M使用基于中位数的目标进行鲁棒点预测；ProD-D使用分布目标保留提示条件不确定性。在各种场景下实验显示预测质量的持续提升。",
        "tags": ["length-prediction", "heavy-tailed", "prompt-conditioned", "robust-estimation", "prod"],
        "conference": "arxiv",
        "year": "2026",
        "added_date": "2026-04-16",
    },
]

added_count = 0
for paper in new_papers:
    if paper["title"].lower().strip() not in existing_titles:
        # Generate ID
        timestamp = int(datetime.now().timestamp())
        idx = len(db["papers"]) + 1
        paper["id"] = f"paper_{timestamp}_{idx}"
        
        # Add to database
        db["papers"].append(paper)
        existing_titles.add(paper["title"].lower().strip())
        added_count += 1
        
        # Create markdown file
        slug = paper["title"].split(":")[0].strip().lower()
        slug = re.sub(r'[^\w\s-]', '', slug).replace(' ', '_').strip()
        md_filename = f"{slug}.md"
        md_path = os.path.join(ARXIV_DIR, md_filename)
        
        md_content = f"""# {paper['title']}

**Source:** {paper['source']} | **Category:** {paper['category']} | **Date:** {paper['date']}
**ArXiv ID:** {paper['arxiv_id']}
**Authors:** {paper['authors']}
**Tags:** {', '.join(paper['tags'])}

## Links

- 📄 [Paper (PDF)]({paper['pdf_url']})
- 🌐 [ArXiv Page]({paper['url']})

## Abstract (English)

{paper['summary_en']}

## Abstract (Chinese)

{paper['summary_cn']}

## Key Contributions

1. **{paper['title'].split(':')[0].strip()}** — {paper['summary_en'][:100]}...
2. Addresses core challenges in {paper['category']} systems
3. Demonstrates significant improvements over existing baselines

## Notes

- Added on {paper['added_date']}
- Paper published on {paper['date']}
"""
        
        with open(md_path, 'w') as f:
            f.write(md_content)
        print(f"Created: {md_path}")

# Save updated database
with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2, ensure_ascii=False)

print(f"\nTotal papers in database: {len(db['papers'])}")
print(f"New papers added: {added_count}")