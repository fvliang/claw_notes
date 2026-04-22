#!/usr/bin/env python3
"""Search Semantic Scholar for recent LLM serving papers"""
import json, urllib.request, time, os, re

DB_PATH = '/home/admin/claw_notes/database.json'
PAPERS_DIR = '/home/admin/claw_notes/papers'

with open(DB_PATH) as f:
    db = json.load(f)

existing_titles = set()
for p in db['papers']:
    t = p.get('title', '').lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')
    existing_titles.add(t[:80])
    existing_titles.add(t[:60])

print(f"Existing: {len(db['papers'])} papers")

# Semantic Scholar API
def search_semantic_scholar(query, limit=100, year_from=2026):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={limit}&year={year_from}-2026&fields=title,abstract,url,externalIds,authors,publicationDate,fieldsOfStudy&sort=relevance"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        return data.get('data', [])
    except Exception as e:
        print(f"  Error: {e}")
        return []

import urllib.parse

queries = [
    "LLM serving system",
    "speculative decoding LLM",
    "KV cache LLM inference",
    "LLM inference optimization",
    "LLM inference efficiency acceleration",
    "distributed LLM inference serving",
    "LLM inference latency throughput",
    "paged attention continuous batching",
    "prefill decode disaggregation",
]

SERVING_KEYWORDS = [
    'serving', 'inference', 'speculative decoding', 'kv cache',
    'prefill', 'decode', 'batching', 'scheduling', 'throughput', 'latency',
    'paged attention', 'vllm', 'continuous batching',
    'disaggregation', 'offloading', 'acceleration', 'ttft', 'tpot',
    'inference speedup', 'inference latency', 'inference throughput',
    'inference optimization', 'efficient inference', 'inference system',
    'inference framework', 'inference engine', 'inference acceleration',
    'self-speculative', 'early exit', 'layer skipping',
    'parallel decoding', 'speculative execution',
    'distributed inference', 'edge inference', 'on-device inference',
    'lora adapter serving', 'adapter routing',
    'agentic inference', 'inference scaling',
    'kv compression', 'cache compression',
    'prefix caching', 'token eviction',
    'moe inference', 'moe serving',
    'multi-gpu inference', 'memory footprint',
]

def is_llm_serving_paper(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 'language model serving'])
    if not has_llm:
        return False
    serving_score = sum(1 for k in SERVING_KEYWORDS if k in text)
    if serving_score >= 1:
        return True
    return False

all_found = []
seen = set()

for q in queries:
    print(f"  Searching S2: {q[:50]}...")
    results = search_semantic_scholar(q, limit=50, year_from=2026)
    count = 0
    for p in results:
        title = p.get('title', '')
        abstract = p.get('abstract', '') or ''
        key = title.lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')[:80]
        if key in seen:
            continue
        seen.add(key)
        if is_llm_serving_paper(title, abstract):
            all_found.append(p)
            count += 1
    print(f"    Found {count} serving papers")
    time.sleep(1)

# Deduplicate against existing DB
new_papers = []
for p in all_found:
    title = p.get('title', '')
    key = title.lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')[:80]
    short_key = key[:60]
    if key not in existing_titles and short_key not in existing_titles:
        new_papers.append(p)

print(f"\n📊 Results:")
print(f"  Total S2 found (relevant, 2026): {len(all_found)}")
print(f"  Already in DB: {len(all_found) - len(new_papers)}")
print(f"  New papers: {len(new_papers)}")

print("\n📋 New papers from S2:")
for p in new_papers[:30]:
    title = p.get('title', '')[:80]
    ext_ids = p.get('externalIds', {})
    arxiv_id = ext_ids.get('ArXiv', '')
    pub_date = p.get('publicationDate', '')
    print(f"  [{arxiv_id}] {title} | {pub_date}")

# Add papers
added = 0
for p in new_papers:
    title = p.get('title', '')
    abstract = p.get('abstract', '') or ''
    ext_ids = p.get('externalIds', {})
    arxiv_id = ext_ids.get('ArXiv', '')
    pub_date = p.get('publicationDate', '')
    authors_list = p.get('authors', [])
    author_names = [a.get('name', '') for a in authors_list] if isinstance(authors_list, list) else []
    fields = p.get('fieldsOfStudy', []) or []
    
    # Skip if no arXiv ID (can't link properly)
    if not arxiv_id:
        # Try to still add with URL
        pass
    
    year = 2026 if pub_date and pub_date.startswith('2026') else 2025
    conf = 'arxiv' if arxiv_id else 'semantic_scholar'
    
    dir_path = os.path.join(PAPERS_DIR, conf, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    safe_title = re.sub(r'[^\w\s-]', '', title[:60]).strip().replace(' ', '_')
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(filepath):
        continue
    
    # GitHub
    github = ""
    gh_match = re.search(r'github\.com/([^\s\)\.]+)', abstract)
    if gh_match:
        github = gh_match.group(1)
    
    url_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else p.get('url', '')
    
    md_content = f"""# {title}

**ArXiv ID:** {arxiv_id or 'N/A'}
**Published:** {pub_date or 'N/A'}
**Authors:** {', '.join(author_names)}
**URL:** {url_link}
**PDF:** {f'https://arxiv.org/pdf/{arxiv_id}' if arxiv_id else 'N/A'}
**GitHub:** {github if github else '暂无'}
**Fields:** {', '.join(fields) if fields else 'N/A'}

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
        "title": title,
        "authors": ', '.join(author_names),
        "conference": conf,
        "year": year,
        "url": url_link,
        "github_repo": github,
        "arxiv_id": arxiv_id,
        "keywords": fields,
        "published": pub_date,
        "abstract_en": abstract[:500],
        "abstract_cn": "",
        "introduction_en": "",
        "introduction_cn": "",
        "markdown_path": os.path.join(conf, str(year), filename),
        "topic": "LLM Serving",
    }
    db['papers'].append(db_entry)
    added += 1
    print(f"  ✅ Added: [{arxiv_id}] {title[:80]}")

with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\n✅ Total new papers from S2: {added}")
print(f"📊 Total papers in database: {len(db['papers'])}")