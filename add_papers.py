#!/usr/bin/env python3
"""批量添加 LLM Serving 相关论文"""
import json
import os

DB_PATH = os.path.expanduser("~/claw_notes/database.json")

def load_db():
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def paper_exists(title):
    """检查论文是否已存在"""
    db = load_db()
    title_lower = title.lower()
    for p in db['papers']:
        if title_lower in p['title'].lower() or p['title'].lower() in title_lower:
            return True
    return False

def add_paper(paper):
    """添加论文到数据库"""
    if paper_exists(paper['title']):
        print(f"⏭️ 跳过（已存在）: {paper['title'][:60]}")
        return False
    
    db = load_db()
    paper['id'] = len(db['papers']) + 1
    db['papers'].append(paper)
    save_db(db)
    print(f"✅ 添加: {paper['title'][:60]}")
    return True

# 从搜索结果中收集的 LLM Serving 相关论文
new_papers = [
    {
        "title": "PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving",
        "authors": "Xu Bai, Muhammed Tawfiqul Islam, Chen Wang, Adel N. Toosi",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "Pipeline parallelism (PP) is widely used to partition layers of large language models (LLMs) across GPUs, enabling scalable inference for large models. However, existing systems rely on static PP configurations that fail to adapt to dynamic settings, such as serverless platforms and heterogeneous GPU environments. Reconfiguring PP by stopping and redeploying introduces significant latency overhead. PipeLive introduces efficient live in-place pipeline parallelism reconfiguration that allows dynamic adaptation without stopping the serving system.",
        "abstract_cn": "管道并行性（PP）被广泛用于将大语言模型（LLM）的层分区到多个GPU上，实现大型模型的可扩展推理。然而，现有系统依赖静态PP配置，无法适应动态场景（如无服务器平台和异构GPU环境）。通过停止和重新部署来重新配置PP会引入显著的延迟开销。PipeLive引入了高效的原地实时管道并行性重新配置，允许在不停止服务系统的情况下进行动态调整。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/pipelive.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-13",
        "url": "",
        "notes": "",
        "topic": "Pipeline Parallelism",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "Flow-Controlled Scheduling for LLM Inference with Provable Stability Guarantees",
        "authors": "Zhuolun Dong, Junyu Cao",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "Large language models (LLMs) have been widely adopted due to their great performance across a wide range of applications. This paper presents a flow-controlled scheduling approach for LLM inference with provable stability guarantees. The proposed method addresses the challenge of maintaining stable performance under varying workloads.",
        "abstract_cn": "大语言模型（LLM）因其在一系列应用中的出色表现而被广泛采用。本文提出了一种具有可证明稳定性保证的LLM推理流控制调度方法。该方法解决了在可变工作负载下保持稳定性能的挑战。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/flow_controlled_scheduling.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-13",
        "url": "",
        "notes": "",
        "topic": "Scheduling",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving",
        "authors": "Hossein Hosseini Kasnavieh, Christopher Leckie, Adel N. Toosi",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "Multi-model LLM routing has emerged as an effective approach for reducing inference costs while maintaining quality. This paper presents RouterWise, a joint resource allocation and routing framework for latency-aware multi-model LLM serving. The system optimizes both routing decisions and resource allocation to meet latency SLOs.",
        "abstract_cn": "多模型LLM路由已成为在保持质量的同时降低推理成本的有效方法。本文提出了RouterWise，一个用于延迟感知多模型LLM服务的联合资源分配和路由框架。该系统优化路由决策和资源分配以满足延迟SLO。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/routerwise.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-12",
        "url": "",
        "notes": "",
        "topic": "Multi-Model Serving",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding",
        "authors": "Jehyeon Bang, Eunyeong Cho, Ranggi Hwang, Jinha Chung, Minsoo Rhu",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "Mixture-of-Experts (MoE) models offer efficient inference through conditional computation, but they offer limited efficiency, particularly for large batch sizes. This work proposes SpecMoE, a memory-efficient MoE inference system based on self-assisted speculative decoding.",
        "abstract_cn": "混合专家（MoE）模型通过条件计算提供高效推理，但它们在大型批量大小下效率有限。这项工作提出了SpecMoE，一个基于自辅助投机解码的内存高效MoE推理系统。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/specmoe.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-11",
        "url": "",
        "notes": "",
        "topic": "Speculative Decoding",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "SMART: When is it Actually Worth Expanding a Speculative Tree?",
        "authors": "Lifu Wang, Pan Zhou",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "Tree-based speculative decoding accelerates autoregressive generation by verifying a branching tree of draft tokens in a single target-model forward pass. However, existing methods prioritize maximizing token-level likelihood or the number of accepted tokens while ignoring a critical efficiency paradox.",
        "abstract_cn": "基于树的投机解码通过在单次目标模型前向传递中验证分支草稿令牌树来加速自回归生成。然而，现有方法优先考虑最大化令牌级可能性或接受的令牌数量，同时忽略了一个关键的效率悖论。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/smart_speculative.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-09",
        "url": "",
        "notes": "",
        "topic": "Speculative Decoding",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios",
        "authors": "Xinyi Hu, Yuhao Shen, Baolin Zhang, Hengxin Zhang, Jun Dai, Shuang Ge, Lei Chen, Yue Li, Mingcheng Wan",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "This paper presents ECHO, an elastic speculative decoding method with sparse gating for high-concurrency scenarios. The approach dynamically adjusts the speculation depth based on workload characteristics.",
        "abstract_cn": "本文提出了ECHO，一种用于高并发场景的具有稀疏门控的弹性投机解码方法。该方法根据工作负载特征动态调整投机深度。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/echo_speculative.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-09",
        "url": "",
        "notes": "",
        "topic": "Speculative Decoding",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "StreamServe: Adaptive Speculative Flows for Low-Latency Disaggregated LLM Serving",
        "authors": "Satyam Kumar, Arpit Singh Gautam, Kailash Talreja, Saurabh Jha",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "Efficient LLM serving must balance throughput and latency across diverse, bursty workloads. StreamServe introduces a disaggregated prefill-decode serving architecture with metric-aware routing and adaptive speculative flows for low-latency LLM serving.",
        "abstract_cn": "高效的LLM服务必须在多样化和突发性的工作负载之间平衡吞吐量和延迟。StreamServe引入了一种分解的预填充-解码服务架构，具有度量感知路由和自适应投机流，用于低延迟LLM服务。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/streamserve.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-11",
        "url": "",
        "notes": "",
        "topic": "Disaggregated Serving",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "MARS: Enabling Autoregressive Models Multi-Token Generation",
        "authors": "Ziqi Jin, Lei Wang, Ziwei Luo, Aixin Sun",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "Unlike speculative decoding, which maintains a separate draft model alongside the target, or multi-head approaches such as Medusa, which attach additional prediction heads, MARS enables multi-token generation without modifications to the original model architecture.",
        "abstract_cn": "与保持独立草稿模型的投机解码或附加额外预测头的多头方法（如Medusa）不同，MARS能够在不修改原始模型架构的情况下实现多令牌生成。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/mars_multitoken.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-08",
        "url": "",
        "notes": "",
        "topic": "Multi-Token Generation",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "Accelerating Speculative Decoding with Block Diffusion Draft Trees",
        "authors": "",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "This paper proposes using block diffusion as draft models for speculative decoding, creating a novel approach that accelerates LLM inference through diffusion-based draft generation.",
        "abstract_cn": "本文提出使用块扩散作为投机解码的草稿模型，创建了一种通过基于扩散的草稿生成来加速LLM推理的新方法。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/block_diffusion_draft.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-14",
        "url": "",
        "notes": "",
        "topic": "Speculative Decoding",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration",
        "authors": "Zhuofan Wen, Yang Feng",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "SpecBound introduces adaptive bounded self-speculation with layer-wise confidence calibration for improved speculative decoding performance.",
        "abstract_cn": "SpecBound引入了具有逐层置信度校准的自适应有界自投机，以提高投机解码性能。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/specbound.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-13",
        "url": "",
        "notes": "",
        "topic": "Speculative Decoding",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling",
        "authors": "Zikun Liu, Liang Luo, Qianru Li, Zhengyu Zhang, Wei Ling, et al.",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "SOLARIS presents a novel framework for speculative offloading of latent-based representation for inference scaling, addressing the challenge of knowledge distillation trade-offs.",
        "abstract_cn": "SOLARIS提出了一个用于推理扩展的基于潜表示的投机卸载新框架，解决了知识蒸馏权衡的挑战。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/solaris.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-13",
        "url": "",
        "notes": "",
        "topic": "Inference Scaling",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    },
    {
        "title": "ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge-Cloud Speculative LLM Serving",
        "authors": "Xiangchen Li, Saeid Ghafouri, Jiakun Fan, Babar Ali, Hans Vandierendonck, Dimitrios S. Nikolopoulos",
        "arxiv_id": "",
        "github_repo": "",
        "conference": "arxiv",
        "full_conference": "arxiv",
        "year": 2026,
        "abstract_en": "ConfigSpec introduces a profiling-based configuration selection approach for distributed edge-cloud speculative LLM serving, optimizing the trade-off between local and cloud execution.",
        "abstract_cn": "ConfigSpec引入了一种基于配置文件的分布式边缘-云投机LLM服务配置选择方法，优化本地和云端执行之间的权衡。",
        "intro_en": "",
        "intro_cn": "",
        "file": "./arxiv/2026/configspec.md",
        "has_real_content": True,
        "added_date": "2026-04-15",
        "date": "2026-04-08",
        "url": "",
        "notes": "",
        "topic": "Edge-Cloud Serving",
        "venue": "arxiv",
        "pdf_url": "",
        "github": "",
        "blog": ""
    }
]

if __name__ == "__main__":
    count = 0
    for paper in new_papers:
        if add_paper(paper):
            count += 1
    
    print(f"\n📊 共添加 {count} 篇新论文")
