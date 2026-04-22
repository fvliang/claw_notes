#!/usr/bin/env python3
"""Fetch recent arXiv paper abstracts via web and filter for LLM serving"""
import json, time, os, re, sys

DB_PATH = '/home/admin/claw_notes/database.json'
PAPERS_DIR = '/home/admin/claw_notes/papers'

with open(DB_PATH) as f:
    db = json.load(f)

existing_titles = set()
for p in db['papers']:
    t = p.get('title', '').lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')
    existing_titles.add(t[:80])
    existing_titles.add(t[:60])

print(f"Existing: {len(db['papers'])} papers, {len(existing_titles)} unique titles")

# List of recent arXiv IDs to check from cs.CL and cs.DC recent listings
# These are from the April 22-21 listings
recent_ids = [
    # cs.DC Apr 22
    "2604.19654", "2604.19503", "2604.19494", "2604.19454", "2604.19363",
    "2604.19337", "2604.19243", "2604.19241", "2604.19181", "2604.19004",
    "2604.18655", "2604.18616", "2604.18615", "2604.18614",
    "2604.19705", "2604.19399", "2604.19286", "2604.19219", "2604.19057",
    "2604.19053", "2604.18801",
    # cs.DC Apr 21
    "2604.18098", "2604.18049", "2604.18043", "2604.18029", "2604.17861",
    "2604.17834", "2604.17640", "2604.17635", "2604.17550", "2604.17373",
    "2604.17227", "2604.17172", "2604.17111", "2604.17104", "2604.17064",
    "2604.17063", "2604.16898", "2604.16864", "2604.16715", "2604.16682",
    "2604.16469", "2604.16457", "2604.16409", "2604.16400",
    "2604.18529", "2604.18020", "2604.17709", "2604.17627", "2604.17353",
    # cs.CL Apr 22 (most recent)
    "2604.19716", "2604.19699", "2604.19685", "2604.19678", "2604.19667",
    "2604.19656", "2604.19645", "2604.19642", "2604.19620", "2604.19598",
    "2604.19593", "2604.19584", "2604.19578", "2604.19572", "2604.19565",
    "2604.19548", "2604.19547", "2604.19508", "2604.19502", "2604.19499",
    "2604.19464", "2604.19447", "2604.19440", "2604.19405", "2604.19395",
    "2604.19394", "2604.19351", "2604.19342", "2604.19331", "2604.19299",
    "2604.19298", "2604.19292", "2604.19274", "2604.19262", "2604.19261",
    "2604.19254", "2604.19245", "2604.19189", "2604.19185", "2604.19162",
    "2604.19151", "2604.19149", "2604.19144", "2604.19139", "2604.19137",
    "2604.19125", "2604.19124", "2604.19098", "2604.19071", "2604.19070",
]

# Also check cs.AR recent
recent_ids.extend([
    # cs.AR recent papers
    "2604.19241", "2604.19004", "2604.18655",
])

SERVING_KEYWORDS = [
    'serving', 'inference', 'speculative decoding', 'kv cache', 'kv-cache',
    'prefill', 'decode', 'batching', 'scheduling', 'throughput', 'latency',
    'memory management', 'paged attention', 'vllm', 'continuous batching',
    'disaggregation', 'offloading', 'acceleration', 'ttft', 'tpot',
    'inference speedup', 'inference latency', 'inference throughput',
    'inference optimization', 'efficient inference', 'inference system',
    'inference framework', 'inference engine', 'inference acceleration',
    'self-speculative', 'early exit', 'layer skipping',
    'parallel decoding', 'batched inference', 'speculative execution',
    'distributed inference', 'edge inference', 'on-device inference',
    'kernel optimization', 'cuda kernel', 'metal kernel',
    'lora adapter serving', 'adapter routing', 'adapter caching',
    'agentic inference', 'inference scaling', 'decode phase',
    'kv compression', 'kv share', 'cache compression',
    'prefix caching', 'token eviction',
    'fp4 inference', 'fp8 inference', 'low-bit inference',
    'tpu inference', 'pim inference', 'processing-in-memory',
    'speculative offloading', 'draft tree',
    'long-context inference', 'long-context serving',
    'moe inference', 'moe serving', 'moe routing',
    'power-efficient inference', 'energy-efficient inference',
    'multi-gpu inference', 'multi-die',
    'attention sink', 'context-aware scheduling',
    'request scheduling', 'batch scheduling',
    'memory footprint', 'gpu memory',
    'cost-efficient inference', 'inference cost',
]

TRAINING_EXCLUSIONS = [
    'training system', 'distributed training', 'fine-tuning',
    'pre-training', 'gradient', 'optimizer', 'learning rate',
    'backpropagation', 'loss function', 'data parallelism',
    'model parallel training', 'pipeline parallel training',
    'tensor parallel training', 'checkpoint',
]

def is_llm_serving_paper(title, abstract):
    text = (title + ' ' + abstract).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 'language model serving', 'transformer inference', 'transformer serving'])
    if not has_llm:
        return False
    serving_score = sum(1 for k in SERVING_KEYWORDS if k in text)
    training_score = sum(1 for k in TRAINING_EXCLUSIONS if k in text)
    # Must have at least 1 serving keyword and serving > training
    if serving_score >= 1 and serving_score > training_score:
        return True
    # Also accept if clearly about inference/serving
    if any(k in text for k in ['inference system', 'serving system', 'speculative decoding', 'kv cache', 'inference efficiency', 'inference latency']):
        return True
    return False

# Fetch papers using arXiv API with delay to avoid rate limiting
import urllib.request, xml.etree.ElementTree as ET

def fetch_paper(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (compatible; PaperCollector/1.0)')
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode('utf-8')
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        root = ET.fromstring(data)
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            arxiv_id_full = entry.find('atom:id', ns).text.strip()
            aid = arxiv_id_full.replace('http://arxiv.org/abs/', '')
            published = entry.find('atom:published', ns).text.strip()[:10]
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
            return {
                'title': title,
                'abstract_en': summary,
                'arxiv_id': aid,
                'published': published,
                'authors': authors,
                'categories': categories,
            }
    except Exception as e:
        print(f"  Error fetching {arxiv_id}: {e}")
        return None

# Batch fetch - use id_list with multiple IDs at once (up to 20)
print("\n🔍 Fetching recent arXiv papers...")

all_papers = []
batch_size = 15
for i in range(0, len(recent_ids), batch_size):
    batch = recent_ids[i:i+batch_size]
    id_list = ','.join(batch)
    url = f"http://export.arxiv.org/api/query?id_list={id_list}&max_results={len(batch)}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (compatible; PaperCollector/1.0)')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read().decode('utf-8')
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        root = ET.fromstring(data)
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            arxiv_id_full = entry.find('atom:id', ns).text.strip()
            aid = arxiv_id_full.replace('http://arxiv.org/abs/', '').split('v')[0]  # remove version
            published = entry.find('atom:published', ns).text.strip()[:10]
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
            all_papers.append({
                'title': title,
                'abstract_en': summary,
                'arxiv_id': aid,
                'published': published,
                'authors': authors,
                'categories': categories,
            })
        print(f"  Batch {i//batch_size+1}: fetched {len(batch)} IDs")
    except Exception as e:
        print(f"  Batch {i//batch_size+1} error: {e}")
    time.sleep(5)  # Longer delay between batches

print(f"\n📊 Fetched {len(all_papers)} papers total")

# Filter for LLM serving
serving_papers = []
for p in all_papers:
    if is_llm_serving_paper(p['title'], p['abstract_en']):
        serving_papers.append(p)

print(f"  LLM serving relevant: {len(serving_papers)}")

# Deduplicate against existing
new_papers = []
for p in serving_papers:
    key = p['title'].lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')[:80]
    short_key = key[:60]
    if key not in existing_titles and short_key not in existing_titles:
        new_papers.append(p)

print(f"  Already in DB: {len(serving_papers) - len(new_papers)}")
print(f"  New papers: {len(new_papers)}")

print("\n📋 New LLM serving papers:")
for p in new_papers:
    print(f"  [{p['arxiv_id']}] {p['title'][:80]} | {p['published']}")

# Add to database and create markdown files
added = 0
for p in new_papers:
    conf = 'arxiv'
    year = 2026 if p['published'].startswith('2026') else 2025
    
    dir_path = os.path.join(PAPERS_DIR, conf, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    safe_title = re.sub(r'[^\w\s-]', '', p['title'][:60]).strip().replace(' ', '_')
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(filepath):
        continue
    
    # Find GitHub
    github = ""
    abstract = p.get('abstract_en', '')
    gh_match = re.search(r'github\.com/([^\s\)\.]+)', abstract)
    if gh_match:
        github = gh_match.group(1)
    
    md_content = f"""# {p['title']}

**ArXiv ID:** {p['arxiv_id']}
**Published:** {p['published']}
**Authors:** {', '.join(p.get('authors', []))}
**URL:** https://arxiv.org/abs/{p['arxiv_id']}
**PDF:** https://arxiv.org/pdf/{p['arxiv_id']}
**GitHub:** {github if github else '暂无'}
**Categories:** {', '.join(p.get('categories', []))}

## Abstract (English)

{abstract}

## 摘要 (中文)

*(待翻译)*

## Introduction (English)

*(需要阅读原文PDF补充)*

## 引言 (中文)

*(需要阅读原文PDF补充)*

## 博客内容

*(待补充)*

## GitHub 介绍

{github if github else '暂无 GitHub 仓库'}

---
*注: 此文件由自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
"""
    
    with open(filepath, 'w') as f:
        f.write(md_content)
    
    db_entry = {
        "id": f"paper_{int(time.time())}_{added}",
        "title": p['title'],
        "authors": ', '.join(p.get('authors', [])),
        "conference": conf,
        "year": year,
        "url": f"https://arxiv.org/abs/{p['arxiv_id']}",
        "github_repo": github,
        "arxiv_id": p['arxiv_id'],
        "keywords": p.get('categories', []),
        "published": p['published'],
        "abstract_en": abstract[:500],
        "abstract_cn": "",
        "introduction_en": "",
        "introduction_cn": "",
        "markdown_path": os.path.join(conf, str(year), filename),
        "topic": "LLM Serving",
    }
    db['papers'].append(db_entry)
    added += 1
    print(f"  ✅ Added: [{p['arxiv_id']}] {p['title'][:80]}")

with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\n✅ Total papers added: {added}")
print(f"📊 Total papers in database: {len(db['papers'])}")