#!/usr/bin/env python3
"""Targeted search for recent LLM serving papers - April 23, 2026"""
import json, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, os, re, sys, hashlib

DB_PATH = '/home/admin/claw_notes/database.json'
PAPERS_DIR = '/home/admin/claw_notes/papers'

with open(DB_PATH) as f:
    db = json.load(f)

existing_titles = set()
for p in db['papers']:
    t = p.get('title', '').lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')
    existing_titles.add(t[:80])
    existing_titles.add(t[:60])

print(f"Existing papers: {len(db['papers'])}, unique titles: {len(existing_titles)}")

def search_arxiv(query, max_results=50):
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        return parse_arxiv_xml(data)
    except Exception as e:
        print(f"  Error: {e}")
        return []

def parse_arxiv_xml(xml_data):
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        arxiv_id_full = entry.find('atom:id', ns).text.strip()
        arxiv_id = arxiv_id_full.replace('http://arxiv.org/abs/', '')
        published = entry.find('atom:published', ns).text.strip()[:10]
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
        papers.append({
            'title': title,
            'abstract_en': summary,
            'arxiv_id': arxiv_id,
            'url': f"https://arxiv.org/abs/{arxiv_id}",
            'published': published,
            'authors': authors,
            'categories': categories,
        })
    return papers

# Focus on papers from April 20-23, 2026 specifically
# Use very specific queries targeting recent work
queries = [
    # Very specific LLM serving topics
    "ti:speculative AND ti:decoding AND abs:llm",
    "ti:kv AND ti:cache AND abs:llm AND abs:inference",
    "ti:serving AND abs:llm AND abs:latency",
    "abs:llm AND abs:inference AND abs:serving AND abs:system",
    "abs:speculative AND abs:decoding AND abs:acceleration",
    "abs:kv AND abs:cache AND abs:compression AND abs:llm",
    "abs:prefill AND abs:decode AND abs:llm",
    "abs:llm AND abs:inference AND abs:gpu AND abs:memory",
    "abs:llm AND abs:inference AND abs:throughput",
    "abs:llm AND abs:inference AND abs:quantization AND abs:speed",
    # New categories  
    "cat:cs.DC AND abs:llm AND abs:inference",
    "cat:cs.AR AND abs:llm AND abs:inference",
]

SERVING_KEYWORDS_LOWER = [
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
    'disaggregated serving', 'prefill-decode disaggregation',
    'long-context inference', 'long-context serving',
    'moe inference', 'moe serving', 'moe routing',
    'power-efficient inference', 'energy-efficient inference',
    'multi-gpu inference', 'multi-die',
    'attention sink', 'context-aware scheduling',
    'request scheduling', 'batch scheduling',
    'memory footprint', 'gpu memory',
    'cost-efficient inference', 'inference cost',
    'token-level speculation', 'verification step',
]

TRAINING_EXCLUSIONS = [
    'training only', 'fine-tuning only', 'pre-training only',
    'rlhf', 'alignment only', 'distillation training',
    'model training system', 'distributed training',
    'gradient', 'optimizer', 'learning rate',
    'backpropagation', 'loss function',
]

def is_llm_serving_paper(title, abstract):
    text = (title + ' ' + abstract).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 'language model serving', 'transformer inference'])
    if not has_llm:
        return False
    serving_score = sum(1 for k in SERVING_KEYWORDS_LOWER if k in text)
    # Exclude pure training papers
    training_score = sum(1 for k in TRAINING_EXCLUSIONS if k in text)
    if serving_score >= 1 and serving_score > training_score:
        return True
    return False

all_found = []
seen_keys = set()

for q in queries:
    print(f"  Query: {q[:60]}...")
    results = search_arxiv(q, max_results=30)
    count = 0
    for p in results:
        if not p['published'].startswith('2026'):
            continue
        key = p['title'].lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')[:80]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if is_llm_serving_paper(p['title'], p['abstract_en']):
            all_found.append(p)
            count += 1
    print(f"    Found {count} serving papers from this query")
    time.sleep(3)  # Rate limit

# Deduplicate against existing
new_papers = []
for p in all_found:
    key = p['title'].lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')[:80]
    short_key = key[:60]
    if key not in existing_titles and short_key not in existing_titles:
        new_papers.append(p)

print(f"\n📊 Results:")
print(f"  Total found (relevant, 2026): {len(all_found)}")
print(f"  Already in DB: {len(all_found) - len(new_papers)}")
print(f"  New papers: {len(new_papers)}")

# Also try to search for papers from specific conferences (accepted papers)
# These won't be on arXiv API directly, so we'll try Semantic Scholar or web search
print("\n📋 New paper titles from arXiv:")
for p in new_papers:
    print(f"  [{p['arxiv_id']}] {p['title'][:80]} | {p['published']}")

# Add papers to DB and create markdown
added = 0
for p in new_papers:
    conf = 'arxiv'
    year = 2026
    
    dir_path = os.path.join(PAPERS_DIR, conf, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    safe_title = re.sub(r'[^\w\s-]', '', p['title'][:60]).strip().replace(' ', '_')
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(filepath):
        continue
    
    # Try to find GitHub repo in abstract
    github = ""
    abstract = p.get('abstract_en', '')
    gh_match = re.search(r'github\.com/([^\s\)\.]+)', abstract)
    if gh_match:
        github = gh_match.group(1)
    
    # Write markdown
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
    
    # Add to database
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

# Save database
with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\n✅ Total papers added from arXiv: {added}")
print(f"📊 Total papers in database: {len(db['papers'])}")

# Save new_papers list for reference
with open('/tmp/arxiv_new_papers_20260423.json', 'w') as f:
    json.dump(new_papers, f, indent=2)