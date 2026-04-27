#!/usr/bin/env python3
"""LLM serving paper search - 2026-04-27 evening v6 (from web_fetch results)"""
import json, urllib.request, urllib.parse, time, re, os

DB_PATH = '/home/admin/claw_notes/database.json'
CLAW_DIR = '/home/admin/claw_notes'

with open(DB_PATH) as f:
    db = json.load(f)

existing_titles = set()
existing_arxiv = set()
for p in db['papers']:
    t = p.get('title', '').lower().strip().replace('\n', ' ').replace('{', '').replace('}', '').replace('$', '').replace('\\', '')
    existing_titles.add(t[:80])
    existing_titles.add(t[:60])
    existing_titles.add(t[:50])
    existing_titles.add(t[:40])
    if p.get('arxiv_id'):
        existing_arxiv.add(p['arxiv_id'].strip())

print(f"📊 DB: {len(db['papers'])} papers")

# Papers found from arXiv web search - manually curated LLM serving papers
new_paper_candidates = [
    # Speculative Decoding
    {"title": "FASER: Fine-Grained Phase Management for Speculative Decoding in Dynamic LLM Serving", "authors": "Wenyan Chen, Chengzhi Lu, Yanying Lin, Dmitrii Ustiugov", "date": "2026-04-22", "topic": "Speculative Decoding"},
    {"title": "DiP-SD: Distributed Pipelined Speculative Decoding for Efficient LLM Inference at the Edge", "date": "2026-04-22", "topic": "Speculative Decoding"},
    {"title": "WISV: Wireless-Informed Semantic Verification for Distributed Speculative Decoding in Device-Edge LLM Inference", "authors": "Zixuan Liu, Zhiyong Chen, Nan Xue, Shengkang Chen, Jiangchao Yao, Meixia Tao, Wenjun Zhang", "date": "2026-04-19", "topic": "Speculative Decoding"},
    {"title": "Accelerating PayPal's Commerce Agent with Speculative Decoding: An Empirical Study on EAGLE3 with Fine-Tuned Nemotron Models", "authors": "Ally Qin, Jian Wan, Sarat Mudunuri, Srinivasan Manoharan", "date": "2026-03-26", "topic": "Speculative Decoding"},
    {"title": "Faster LLM Inference via Sequential Monte Carlo", "authors": "Yahya Emara, Mauricio Barba da Costa, Chi-Chih Chang, Cameron Freer, Tim Vieira, Ryan Cotterell, Mohamed S. Abdelfattah", "date": "2026-04-16", "topic": "Speculative Decoding"},
    {"title": "RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding", "authors": "Zihong Zhang, Zuchao Li, Lefei Zhang, Ping Wang, Hai Zhao", "date": "2026-04-16", "topic": "Speculative Decoding"},
    {"title": "ConfLayers: Adaptive Confidence-based Layer Skipping for Self-Speculative Decoding", "authors": "Walaa Amer, Uday das, Fadi Kurdahi", "date": "2026-04-16", "topic": "Early Exit"},
    {"title": "Calibrated Speculative Decoding: Frequency-Guided Candidate Selection for Efficient Inference", "authors": "Xuwen Zhou, Fangxin Liu, Chao Wang, Xiao Zheng, Hao Zheng, Min He, Li Jiang, Haibing Guan", "date": "2026-04-15", "topic": "Speculative Decoding"},
    {"title": "SpecBound: Adaptive Bounded Self-Speculation with Layer-wise Confidence Calibration", "authors": "Zhuofan Wen, Yang Feng", "date": "2026-04-13", "topic": "Speculative Decoding"},
    {"title": "SMART: When is it Actually Worth Expanding a Speculative Tree?", "authors": "Lifu Wang, Pan Zhou", "date": "2026-04-09", "topic": "Speculative Decoding"},
    {"title": "SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding", "authors": "Jehyeon Bang, Eunyeong Cho, Ranggi Hwang, Jinha Chung, Minsoo Rhu", "date": "2026-04-11", "topic": "Speculative Decoding"},
    {"title": "ConfigSpec: Profiling-Based Configuration Selection for Distributed Edge--Cloud Speculative LLM Serving", "authors": "Xiangchen Li, Saeid Ghafouri, Jiakun Fan, Babar Ali, Hans Vandierendonck, Dimitrios S. Nikolopoulos", "date": "2026-04-08", "topic": "Speculative Decoding"},
    {"title": "SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding", "date": "2026-02-10", "topic": "Speculative Decoding"},
    {"title": "ToolSpec: Accelerating Tool Calling via Schema-Aware and Retrieval-Augmented Speculative Decoding", "authors": "Heming Xia, Yongqi Li, Cunxiao Du, Mingbo Song, Wenjie Li", "date": "2026-04-15", "topic": "Speculative Decoding"},
    
    # KV Cache
    {"title": "SparKV: Overhead-Aware KV Cache Loading for Efficient On-Device LLM Inference", "authors": "Hongyao Liu, Liuqun Zhai, Junyi Wang, Zhengru Fang", "date": "2026-04-22", "topic": "KV Cache"},
    {"title": "TTKV: Temporal-Tiered KV Cache for Long-Context LLM Inference", "authors": "Gradwell Dzikanyanga, Weihao Yang, Hao Huang, Donglei Wu, Shihao Wang, Wen Xia, Sanjeeb K C", "date": "2026-03-27", "topic": "KV Cache"},
    {"title": "DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing", "authors": "Jinyu Guo, Zhihan Zhang, Yutong Li, Jiehui Xie, Md. Tamim Iqbal, Dongshen Han, Lik-Hang Lee, Sung-Ho Bae, Jie Zou, Yang Yang, Chaoning Zhang", "date": "2026-04-21", "topic": "KV Cache"},
    {"title": "River-LLM: Large Language Model Seamless Exit Based on KV Share", "date": "2026-04-20", "topic": "KV Cache"},
    {"title": "MoE-nD: Per-Layer Mixture-of-Experts Routing for Multi-Axis KV Cache Compression", "authors": "Libo Sun, Peixiong He, Po-Wei Harn, Xiao Qin", "date": "2026-04-19", "topic": "KV Cache"},
    {"title": "YOCO++: Enhancing YOCO with KV Residual Connections for Efficient LLM Inference", "authors": "You Wu, Ziheng Chen, Yizhen Zhang, Haoyi Wu, Chengting Yu, Yuchi Xu, Wenbo Su, Bo Zheng, Kewei Tu", "date": "2026-04-15", "topic": "KV Cache"},
    {"title": "KV Packet: Recomputation-Free Context-Independent KV Caching for LLMs", "authors": "Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Bing Li, Ulf Schlichtmann", "date": "2026-04-14", "topic": "KV Cache"},
    {"title": "IceCache: Memory-efficient KV-cache Management for Long-Sequence LLMs", "authors": "Yuzhen Mao, Qitong Wang, Martin Ester, Ke Li", "date": "2026-04-12", "topic": "KV Cache"},
    {"title": "CodeComp: Structural KV Cache Compression for Agentic Coding", "authors": "Qiujiang Chen, Jing Xiong, Chenyang Zhao, Sidi Yang, Ngai Wong", "date": "2026-04-11", "topic": "KV Cache"},
    {"title": "ZoomR: Memory Efficient Reasoning through Multi-Granularity Key Value Retrieval", "authors": "David H. Yang, Yuxuan Zhu, Mohammad Mohammadi Amiri, Keerthiram Murugesan, Tejaswini Pedapati, Subhajit Chaudhury, Pin-Yu Chen", "date": "2026-04-12", "topic": "KV Cache"},
    {"title": "KV Cache Offloading for Context-Intensive Tasks", "authors": "Andrey Bocharnikov, Ivan Ermakov, Denis Kuznedelev, Vyacheslav Zhdanovskiy, Yegor Yershov", "date": "2026-04-09", "topic": "KV Cache"},
    {"title": "CSAttention: Centroid-Scoring Attention for Accelerating LLM Inference", "authors": "Chuxu Song, Zhencan Peng, Jiuqi Wei, Chuanhui Yang", "date": "2026-03-29", "topic": "KV Cache"},
    {"title": "The Illusion of Equivalence: Systematic FP16 Divergence in KV-Cached Autoregressive Inference", "authors": "Ranjith Chodavarapu, Lei Xu", "date": "2026-04-16", "topic": "KV Cache"},
    
    # LLM Serving System
    {"title": "Continuous Semantic Caching for Low-Cost LLM Serving", "authors": "Baran Atalar, Xutong Liu, Jinhang Zuo, Siwei Wang, Wei Chen, Carlee Joe-Wong", "date": "2026-04-21", "topic": "LLM Serving"},
    {"title": "KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving", "date": "2026-04-17", "topic": "LLM Serving"},
    {"title": "Stream2LLM: Overlap Context Streaming and Prefill for Reduced Time-to-First-Token (TTFT)", "authors": "Rajveer Bachkaniwala, Chengqi Luo, Richard So, Divya Mahajan, Kexin Rong", "date": "2026-03-29", "topic": "Prefill/Disaggregation"},
    {"title": "Accuracy Is Speed: Towards Long-Context-Aware Routing for Distributed LLM Serving", "authors": "Takeshi Yoshimura, Valentijn Dymphnus van de Beek, Tatsuhiro Chiba", "date": "2026-04-17", "topic": "Inference Scheduling"},
    {"title": "PipeLive: Efficient Live In-place Pipeline Parallelism Reconfiguration for Dynamic LLM Serving", "authors": "Xu Bai, Muhammed Tawfiqul Islam, Chen Wang, Adel N. Toosi", "date": "2026-04-13", "topic": "Parallelism"},
    {"title": "Flow-Controlled Scheduling for LLM Inference with Provable Stability Guarantees", "authors": "Zhuolun Dong, Junyu Cao", "date": "2026-04-13", "topic": "Inference Scheduling"},
    {"title": "Dual-Pool Token-Budget Routing for Cost-Efficient and Reliable LLM Serving", "authors": "Xunzhuo Liu, Bowei He, Xue Liu, Andy Luo, Haichen Zhang, Huamin Chen", "date": "2026-04-09", "topic": "Inference Scheduling"},
    {"title": "Token-Budget-Aware Pool Routing for Cost-Efficient LLM Inference", "authors": "Huamin Chen, Xunzhuo Liu, Junchen Jiang, Bowei He, Xue Liu", "date": "2026-03-13", "topic": "Inference Scheduling"},
    {"title": "Blink: CPU-Free LLM Inference by Delegating the Serving Stack to GPU and SmartNIC", "authors": "Mohammad Siavashi, Mariano Scazzariello, Gerald Q. Maguire Jr., Dejan Kostić, Marco Chiesa", "date": "2026-04-08", "topic": "LLM Serving"},
    {"title": "InfiniLoRA: Disaggregated Multi-LoRA Serving for Large Language Models", "authors": "Hongyu Chen, Letian Ruan, Zilin Xu, Yuchen Li, Xinyu Chen, Jingwen Leng, Bingsheng He, Minyi Guo, Shixuan Sun", "date": "2026-04-08", "topic": "LoRA/Adapter Serving"},
    {"title": "HadAgent: Harness-Aware Decentralized Agentic AI Serving with Proof-of-Inference Blockchain Consensus", "date": "2026-04-15", "topic": "LLM Serving"},
    
    # Inference Kernel
    {"title": "Ragged Paged Attention: A High-Performance and Flexible LLM Inference Kernel for TPU", "authors": "Jevin Jiang, Ying Chen, Blake A. Hechtman, Fenghui Zhang, Yarong Mu", "date": "2026-04-16", "topic": "Inference Kernel"},
    {"title": "Guess-Verify-Refine: Data-Aware Top-K for Sparse-Attention Decoding on Blackwell via Temporal Correlation", "authors": "Long Cheng, Ritchie Zhao, Timmy Liu, Mindy Li, Xianjie Qiao, Kefeng Duan, Yu-Jung Chen, Xiaoming Chen, Bita Darvish Rouhani, June Yang", "date": "2026-04-24", "topic": "Inference Kernel"},
    {"title": "Open-TQ-Metal: Fused Compressed-Domain Attention for Long-Context LLM Inference on Apple Silicon", "authors": "Sai Vegasena", "date": "2026-04-18", "topic": "Inference Kernel"},
    {"title": "Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel", "authors": "Hongyi Jin, Bohan Hou, Guanjie Wang, Ruihang Lai, Jinqi Chen, Zihao Ye, Yaxing Cai, Yixin Dong, Xinhao Cheng, Zhihao Zhang, Yilong Zhao, Yingyi Huang, Lijie Yang, Jinchen Jiang, Gabriele Oliaro, Jianan Ji, Xupeng Miao, Vinod Grover, Todd C. Mowry, Zhihao Jia, Tianqi Chen", "date": "2026-04-14", "topic": "Inference Kernel"},
    
    # Distributed Inference
    {"title": "BloomBee: Distributed Generative Inference of LLM at Internet Scales with Multi-Dimensional Communication Optimization", "authors": "Jiu Chen, Shuangyan Yang, Xu Xiong, Hexiao Duan, Xinran Zhang, Jie Ren, Dong Li", "date": "2026-04-22", "topic": "Distributed Inference"},
    {"title": "A-IO: Adaptive Inference Orchestration for Memory-Bound NPUs", "authors": "Chen Zhang, Yan Ding, Haotian Wang, Chubo Liu, Keqin Li, Kenli Li", "date": "2026-04-10", "topic": "LLM Serving"},
    
    # Other serving-related
    {"title": "Super Apriel: One Checkpoint, Many Speeds", "authors": "SLAM Labs, Oleksiy Ostapenko, Raymond Li, Torsten Scholak, Alireza Mousavi-Hosseini, Aman Tiwari, Denis Kocetkov, Joel Lamy Poirier, Kelechi Ogueji, Nanda H Krishna, Rafael Pardinas, Sathwik Tejaswi Madhusudhan, Shruthan Radhakrishna, Srinivas Sunkara, Valerie Becaert", "date": "2026-04-21", "topic": "LLM Serving"},
    {"title": "Unlocking the Edge deployment and ondevice acceleration of multi-LoRA enabled one-for-all foundational LLM", "authors": "Sravanth Kodavanti, Sowmya Vajrala, Srinivas Miriyala, Utsav Tiwari, Uttam Kumar, Utkarsh Kumar Mahawar, Achal Pratap Singh, Arya D, Narendra Mutyala, Vikram Nelvoy Rajendiran, Sharan Kumar Allur, Euntaik Lee, Dohyoung Kim, HyeonSu Lee, Gyusung Cho, JungBae Kim", "date": "2026-04-20", "topic": "LoRA/Adapter Serving"},
    {"title": "Copy-as-Decode: Grammar-Constrained Parallel Prefill for LLM Editing", "authors": "Ziyang Liu", "date": "2026-04-20", "topic": "Prefill/Disaggregation"},
    {"title": "HybridGen: Efficient LLM Generative Inference via CPU-GPU Hybrid Computing", "authors": "Mao Lin, Xi Wang, Guilherme Cox, Dong Li, Hyeran Jeon", "date": "2026-04-20", "topic": "Offloading/Heterogeneous"},
    {"title": "GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models", "authors": "Ziyang Wang, Jiangfeng Xiao, Chuan Xiao, Ruoxiang Li, Rui Mao, Jianbin Qin", "date": "2026-04-21", "topic": "LLM Pruning/Serving"},
    {"title": "MemoSight: Unifying Context Compression and Multi Token Prediction for Reasoning Acceleration", "authors": "Xinyu Liu, Xin Liu, Bo Jin, Runsong Zhao, Pengcheng Huang, Junhao Ruan, Bei Li, Chunyang Xiao, Tong Xiao, Jingbo Zhu", "date": "2026-04-16", "topic": "LLM Serving"},
]

# Fetch arxiv IDs for papers that don't have them yet
def get_arxiv_id_from_title(title):
    """Search arxiv for a paper by title to get its ID"""
    q = title.replace(':', '').replace('"', '').strip()
    url = f"https://arxiv.org/search/?query={urllib.parse.quote(q)}&searchtype=all&order=-announced_date_first"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
        # Find arxiv IDs in the page
        ids = re.findall(r'arXiv:(\d{4}\.\d{4,5})', html)
        if ids:
            return ids[0]
    except:
        pass
    return ''

def get_arxiv_abstract(arxiv_id):
    """Fetch abstract from arxiv abs page"""
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8')
        # Extract abstract
        m = re.search(r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)</blockquote>', html, re.DOTALL)
        if m:
            abs_text = m.group(1).strip()
            abs_text = re.sub(r'<[^>]+>', '', abs_text)
            abs_text = abs_text.replace('\n', ' ').strip()
            return abs_text
    except:
        pass
    return ''

# Filter out existing papers
new_papers = []
for p in new_paper_candidates:
    tkey = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '').replace('$', '').replace('\\', '')
    is_existing = False
    for l in [80, 60, 50, 40]:
        if tkey[:l] in existing_titles:
            is_existing = True
            break
    if not is_existing:
        new_papers.append(p)

print(f"📊 {len(new_paper_candidates)} candidates, {len(new_papers)} new (not in DB)")
print(f"\n🆕 New papers:")
for p in new_papers:
    print(f"  • {p['title'][:70]} | {p['topic']}")

# Fetch arxiv IDs and abstracts
print("\n🔍 Fetching arxiv IDs and abstracts...")
for i, p in enumerate(new_papers):
    title = p['title']
    # Try to find arxiv ID
    aid = get_arxiv_id_from_title(title)
    p['arxiv_id'] = aid
    if aid:
        print(f"  [{i+1}] Found arxiv ID: {aid} for '{title[:50]}'")
        abs_text = get_arxiv_abstract(aid)
        p['abstract_en'] = abs_text
        time.sleep(2)
    else:
        print(f"  [{i+1}] No arxiv ID found for '{title[:50]}'")
        p['abstract_en'] = ''
    time.sleep(2)

# Add to DB and create markdown notes
def get_conf_dir(conf, year):
    c = conf.lower()
    if c == 'arxiv':
        return os.path.join(CLAW_DIR, 'arXiv', str(year))
    m = {'osdi':'osdi','sosp':'sosp','nsdi':'nsdi','sigcomm':'sigcomm','sigmod':'sigmod',
         'atc':'atc','eurosys':'eurosys','dac':'dac','asplos':'asplos','sc':'sc',
         'neurips':'nips','iclr':'iclr','icml':'icml','acl':'acl','emnlp':'emnlp',
         'mlsys':'mlsys','isca':'isca','euromlsys':'euromlsys','ispass':'ispass'}
    return os.path.join(CLAW_DIR, m.get(c, c), str(year))

def sanitize_fn(title, aid=''):
    n = title.replace('\n',' ').strip()
    n = re.sub(r'[^\w\s-]', '', n)
    n = re.sub(r'\s+', '_', n)
    if aid:
        n = f"{aid}_{n[:60]}"
    else:
        n = n[:70]
    return n + '.md'

max_id = max(int(p['id']) for p in db['papers']) if db['papers'] else 0
added = 0
files = []

for p in new_papers:
    date = p.get('date', '2026-04-27')
    try: year = int(date[:4])
    except: year = 2026
    
    conf = 'arXiv'
    topic = p.get('topic', 'LLM Serving')
    aid = p.get('arxiv_id', '')
    title = p['title']
    abs_en = p.get('abstract_en', '') or '[Abstract待从arxiv页面获取]'
    abs_cn = "[中文翻译待补充] " + abs_en[:200] + "..." if abs_en and abs_en != '[Abstract待从arxiv页面获取]' else "[中文翻译待补充]"
    authors = p.get('authors', '')
    
    d = get_conf_dir(conf, year)
    os.makedirs(d, exist_ok=True)
    fn = sanitize_fn(title, aid)
    fp = os.path.join(d, fn)
    
    arxiv_url = f"https://arxiv.org/abs/{aid}" if aid else '[arxiv链接待补充]'
    pdf_url = f"https://arxiv.org/pdf/{aid}" if aid else '[PDF链接待补充]'
    
    if not os.path.exists(fp):
        md = f"""# {title}

## Metadata
- **Authors:** {authors}
- **Conference:** {conf} {year}
- **Topic:** {topic}
- **arXiv ID:** {aid}
- **Published:** {date}
- **GitHub:** [待补充]

## 原文链接
- arXiv: {arxiv_url}
- PDF: {pdf_url}

## 摘要 (Abstract)

{abs_en}

## 摘要 (中文)

{abs_cn}

## 引言 (Introduction)

[待补充]

## 博客内容

[待补充]

## GitHub 介绍

[待补充]

---
*Auto-collected on 2026-04-27 evening*
"""
        with open(fp, 'w', encoding='utf-8') as f:
            f.write(md)
        files.append(fp)
    
    entry = {
        'id': max_id + 1 + added,
        'title': title,
        'authors': authors,
        'arxiv_id': aid,
        'github_repo': '',
        'conference': conf,
        'year': str(year),
        'topic': topic,
        'abstract_en': abs_en,
        'abstract_cn': abs_cn,
    }
    db['papers'].append(entry)
    added += 1
    print(f"  ✅ [{added}] {title[:60]} | {aid} | {topic}")

with open(DB_PATH, 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print(f"\n📊 Summary: added {added} papers, {len(files)} md files, total DB: {len(db['papers'])}")