#!/usr/bin/env python3
"""Batch add new LLM serving papers to database.json"""
import json, os, re, urllib.request, urllib.error, time, sys

DB_PATH = "database.json"

def load_db():
    with open(DB_PATH, 'r') as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, 'w') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

def get_existing_titles(db):
    return set(p['title'].strip().lower() for p in db['papers'])

def next_id(db):
    numeric_ids = [int(p['id']) for p in db['papers'] if str(p['id']).isdigit()]
    return max(numeric_ids) + 1 if numeric_ids else 1

def fetch_arxiv_page(arxiv_id):
    """Fetch abstract from arxiv abs page"""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode('utf-8', errors='replace')
        # Extract abstract
        m = re.search(r'<blockquote class="abstract math-scroll">(.*?)</blockquote>', html, re.DOTALL)
        if m:
            abstract = m.group(1).strip()
            abstract = re.sub(r'<[^>]+>', '', abstract)
            abstract = abstract.replace('\n', ' ').strip()
            # Remove "Abstract:" prefix
            if abstract.lower().startswith('abstract'):
                abstract = abstract[8:].strip()
            return abstract
        # Try alternative pattern
        m = re.search(r'<div class="abstract">(.*?)</div>', html, re.DOTALL)
        if m:
            abstract = m.group(1).strip()
            abstract = re.sub(r'<[^>]+>', '', abstract)
            abstract = abstract.replace('\n', ' ').strip()
            if abstract.lower().startswith('abstract'):
                abstract = abstract[8:].strip()
            return abstract
        return ""
    except Exception as e:
        print(f"  Error fetching {arxiv_id}: {e}")
        return ""

# New papers to add - LLM serving related only
NEW_PAPERS = [
    # LLM Serving (April 2026)
    {"title": "Prefill-as-a-Service: KVCache of Next-Generation Models Could Go Cross-Datacenter", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Serving Chain-structured Jobs with Large Memory Footprints with Application to Large Foundation Model Serving", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "RouterWise: Joint Resource Allocation and Routing for Latency-Aware Multi-Model LLM Serving", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge--Cloud Speculative LLM Serving", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "StreamServe: Adaptive Speculative Flows for Low-Latency Disaggregated LLM Serving", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Robust Length Prediction: A Perspective from Heavy-Tailed Prompt-Conditioned Distributions", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Autopoiesis: A Self-Evolving System Paradigm for LLM Serving Under Runtime Dynamics", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Foundry: Template-Based CUDA Graph Context Materialization for Fast LLM Serving Cold Start", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    # LLM Inference (April 2026)
    {"title": "Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for SLO-Constrained Inference", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "SHIELD: A Segmented Hierarchical Memory Architecture for Energy-Efficient LLM Inference on Edge NPUs", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "AsyncTLS: Efficient Generative LLM Inference with Asynchronous Two-level Sparse Attention", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Valve: Production Online-Offline Inference Colocation with Jointly-Bounded Preemption Latency and Rate", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Flow-Controlled Scheduling for LLM Inference with Provable Stability Guarantees", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Scheduling the Unschedulable: Taming Black-Box LLM Inference at Scale", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Flux Attention: Context-Aware Hybrid Attention for Efficient LLMs Inference", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "ProbeLogits: Kernel-Level LLM Inference Primitives for AI-Native Operating Systems", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    # Speculative Decoding (April 2026)
    {"title": "From Tokens to Steps: Verification-Aware Speculative Decoding for Efficient Multi-Step Reasoning", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Acceptance Dynamics Across Cognitive Domains in Speculative Decoding", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "ELMoE-3D: Leveraging Intrinsic Elasticity of MoE for Hybrid-Bonding-Enabled Self-Speculative Decoding in On-Premises Serving", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Accelerating Speculative Decoding with Block Diffusion Draft Trees", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "SOLARIS: Speculative Offloading of Latent-bAsed Representation for Inference Scaling", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "SMART: When is it Actually Worth Expanding a Speculative Tree?", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "ECHO: Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Multi-Drafter Speculative Decoding with Alignment Feedback", "arxiv_id": "2604.0xx", "conference": "arXiv 2026", "year": 2026},
    # Earlier 2026 papers from initial search
    {"title": "WWW.Serve: Interconnecting Global LLM Services through Decentralization", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Multi-stage Flow Scheduling for LLM Serving", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Orla: A Library for Serving LLM-Based Multi-Agent Systems", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "SUN: Shared Use of Next-token Prediction for Efficient Multi-LLM Disaggregated Serving", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "LLMServingSim 2.0: A Unified Simulator for Heterogeneous and Disaggregated LLM Serving Infrastructure", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Pancake: Hierarchical Memory System for Multi-Agent LLM Serving", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "DualScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "FlowPrefill: Decoupling Preemption from Prefill Scheduling Granularity to Mitigate Head-of-Line Blocking in LLM Serving", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "PrefillShare: A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "PASCAL: A Phase-Aware Scheduling Algorithm for Serving Reasoning-based Large Language Models", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Serving Hybrid LLM Loads with SLO Guarantees Using CPU-GPU Attention Piggybacking", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Multi-Layer Scheduling for MoE-Based LLM Reasoning", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "S-HPLB: Efficient LLM Attention Serving via Sparsity-Aware Head Parallelism Load Balance", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "MoE-SpAc: Efficient MoE Inference Based on Speculative Activation Utility in Heterogeneous Edge Scenarios", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "TAPS: Task Aware Proposal Distributions for Speculative Sampling", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "MineDraft: A Framework for Batch Parallel Speculative Decoding", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "ConFu: Contemplate the Future for Better Speculative Sampling", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "EAGLE-Pangu: Accelerator-Safe Tree Speculative Decoding on Ascend NPUs", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
    {"title": "When Drafts Evolve: Speculative Decoding Meets Online Learning", "arxiv_id": "2603.0xx", "conference": "arXiv 2026", "year": 2026},
]

# Actually, we need the real arxiv IDs. Let me search for them properly.
# I'll just add papers with placeholder arxiv_ids and mark them as needing update.

def main():
    db = load_db()
    existing = get_existing_titles(db)
    nid = next_id(db)
    
    added = 0
    for paper in NEW_PAPERS:
        title_lower = paper["title"].strip().lower()
        if title_lower in existing:
            print(f"SKIP (exists): {paper['title']}")
            continue
        
        # Only add LLM serving / inference / speculative decoding related
        entry = {
            "id": nid,
            "title": paper["title"],
            "authors": "",
            "arxiv_id": paper.get("arxiv_id", ""),
            "github_repo": "",
            "conference": paper.get("conference", "arXiv 2026"),
            "year": paper.get("year", 2026),
            "abstract_en": "",
            "abstract_cn": "",
            "intro_en": "",
            "intro_cn": "",
            "topic": "llm_serving",
            "has_content": False,
            "is_placeholder_arxiv": True,
            "is_github_project": False,
            "added_date": "2026-04-18",
            "file": ""
        }
        db["papers"].append(entry)
        existing.add(title_lower)
        nid += 1
        added += 1
        print(f"ADD [{entry['id']}]: {paper['title']}")
    
    save_db(db)
    print(f"\nAdded {added} new papers. Total: {len(db['papers'])}")

if __name__ == "__main__":
    main()