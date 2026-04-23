#!/usr/bin/env python3
"""Search Semantic Scholar for specific conference papers - 2026-04-23 evening"""
import json, urllib.request, urllib.parse, time, re, sys, os

DB_PATH = '/home/admin/claw_notes/database.json'
PAPERS_DIR = '/home/admin/claw_notes/papers'

with open(DB_PATH) as f:
    db = json.load(f)

existing_titles = set()
for p in db['papers']:
    t = p.get('title', '').lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')
    existing_titles.add(t[:80])
    existing_titles.add(t[:60])

print(f"Existing: {len(db['papers'])} papers")

SERVING_KEYWORDS = [
    'serving', 'inference', 'speculative decoding', 'kv cache', 'kv-cache',
    'prefill', 'decode', 'batching', 'scheduling', 'throughput', 'latency',
    'memory management', 'paged attention', 'vllm', 'continuous batching',
    'disaggregation', 'offloading', 'acceleration', 'ttft', 'tpot',
    'inference speedup', 'inference latency', 'inference throughput',
    'inference optimization', 'efficient inference', 'inference system',
    'flash attention', 'draft model', 'moe inference', 'moe serving',
    'gpu memory', 'generation latency', 'cost-efficient inference',
    'token eviction', 'prefix caching', 'kv compression',
    'request scheduling', 'batch scheduling', 'multi-gpu inference',
    'speculative sampling', 'medusa', 'eagle', 'long-context inference',
]
TRAINING_EXCLUSIONS = [
    'training system', 'distributed training', 'fine-tuning system',
    'pre-training', 'gradient accumulation', 'optimizer',
    'training efficiency', 'training acceleration', 'training framework',
]

def is_llm_serving(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 
                                       'language model serving', 'transformer inference', 
                                       'transformer serving', 'foundation model inference',
                                       'autoregressive inference'])
    if not has_llm: return False
    ss = sum(1 for k in SERVING_KEYWORDS if k in text)
    ts = sum(1 for k in TRAINING_EXCLUSIONS if k in text)
    return (ss >= 1 and ss > ts) or ss >= 2

def search_s2(query, limit=20, year_from=2025):
    encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&year={year_from}-2026&fields=title,abstract,url,externalIds,authors,publicationDate,fieldsOfStudy,venue&sort=relevance"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (PaperCollector/2.0)')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = []
        for p in data.get('data', []):
            results.append({
                'title': p.get('title', ''),
                'abstract_en': p.get('abstract', '') or '',
                'arxiv_id': p.get('externalIds', {}).get('ArXiv', ''),
                'published': p.get('publicationDate', ''),
                'authors': [a.get('name', '') for a in p.get('authors', [])] if isinstance(p.get('authors'), list) else [],
                'venue': p.get('venue', '') or '',
                'categories': p.get('fieldsOfStudy', []) or [],
            })
        return results
    except Exception as e:
        print(f"  S2 error: {e}")
        return []

# Conference-specific searches
queries = [
    "LLM inference OSDI SOSP",
    "LLM serving NSDI SIGCOMM",
    "LLM inference EuroSys ATC",
    "speculative decoding NeurIPS ICLR ICML",
    "LLM inference ACL EMNLP",
    "kv cache serving ASPLOS DAC SC",
    "LLM inference SIGMOD",
]

all_found = []
seen = set()

for q in queries:
    print(f"Searching: {q}")
    results = search_s2(q, limit=20)
    added_count = 0
    for p in results:
        key = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
        if key not in seen:
            seen.add(key)
            if is_llm_serving(p['title'], p['abstract_en']):
                all_found.append(p)
                added_count += 1
    print(f"  Found {added_count} serving papers")
    time.sleep(3)  # Longer delays to avoid 429

# Also try individual recent arXiv paper pages
recent_arxiv_ids = [
    "2604.20503", "2604.20410", "2604.20342", "2604.20081", "2604.20032",
    "2604.19958", "2604.20825", "2604.20819",
]

for aid in recent_arxiv_ids:
    print(f"Fetching arXiv paper: {aid}")
    url = f"https://api.semanticscholar.org/graph/v1/paper/ArXiv:{aid}?fields=title,abstract,url,externalIds,authors,publicationDate,fieldsOfStudy,venue"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=15) as resp:
            p = json.loads(resp.read().decode('utf-8'))
        title = p.get('title', '')
        abstract = p.get('abstract', '') or ''
        key = title.lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
        if key not in seen:
            seen.add(key)
            if is_llm_serving(title, abstract):
                entry = {
                    'title': title,
                    'abstract_en': abstract,
                    'arxiv_id': aid,
                    'published': p.get('publicationDate', ''),
                    'authors': [a.get('name', '') for a in p.get('authors', [])] if isinstance(p.get('authors'), list) else [],
                    'venue': p.get('venue', '') or '',
                    'categories': p.get('fieldsOfStudy', []) or [],
                }
                all_found.append(entry)
                print(f"  ✅ Relevant: [{aid}] {title[:60]}")
            else:
                print(f"  ❌ Not serving: {title[:60]}")
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(2)

print(f"\n=== Results ===")
print(f"Total found (relevant): {len(all_found)}")

# Deduplicate against DB
new_papers = []
for p in all_found:
    key = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
    short = key[:60]
    if key not in existing_titles and short not in existing_titles:
        new_papers.append(p)

print(f"Already in DB: {len(all_found) - len(new_papers)}")
print(f"New papers: {len(new_papers)}")

for i, p in enumerate(new_papers, 1):
    aid = p.get('arxiv_id', '')
    print(f"  {i}. [{aid}] {p['title'][:70]} | {p.get('published', '')} | {p.get('venue', '')}")

CONFERENCE_MAP = {
    'osdi': 'OSDI', 'sosp': 'SOSP', 'nsdi': 'NSDI', 'sigcomm': 'SIGCOMM',
    'sigmod': 'SIGMOD', 'atc': 'ATC', 'eurosys': 'EuroSys', 'dac': 'DAC',
    'asplos': 'ASPLOS', 'sc': 'SC', 'nips': 'NeurIPS', 'neurips': 'NeurIPS',
    'iclr': 'ICLR', 'icml': 'ICML', 'acl': 'ACL', 'emnlp': 'EMNLP',
}

def detect_conference(venue, categories, title, abstract):
    text = (venue + ' ' + ' '.join(categories if categories else []) + ' ' + title + ' ' + (abstract or '')).lower()
    for key, conf_name in CONFERENCE_MAP.items():
        if key in text:
            return conf_name
    if venue:
        return venue
    return 'arxiv'

added = 0
for p in new_papers:
    title = p.get('title', '')
    abstract = p.get('abstract_en', '') or ''
    arxiv_id = p.get('arxiv_id', '')
    authors = p.get('authors', [])
    author_str = ', '.join(authors) if isinstance(authors, list) else str(authors)
    published = p.get('published', '')
    categories = p.get('categories', [])
    venue = p.get('venue', '')
    
    year = 2026
    if published:
        try:
            year = int(published[:4])
        except:
            year = 2026
    
    conf = detect_conference(venue, categories, title, abstract)
    
    github = ""
    gh_matches = re.findall(r'github\.com/([^\s\)\.\,]+)', abstract)
    if gh_matches:
        github = gh_matches[0]
    
    dir_path = os.path.join(PAPERS_DIR, conf, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    safe_title = re.sub(r'[^\w\s-]', '', title[:60]).strip().replace(' ', '_')
    if not safe_title:
        safe_title = f"paper_{arxiv_id}"
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(filepath):
        continue
    
    url_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ''
    pdf_link = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ''
    
    md_content = f"""# {title}

**ArXiv ID:** {arxiv_id or 'N/A'}
**Published:** {published or 'N/A'}
**Authors:** {author_str}
**Conference/Venue:** {venue or conf}
**URL:** {url_link or 'N/A'}
**PDF:** {pdf_link or 'N/A'}
**GitHub:** {github if github else '暂无'}
**Categories:** {', '.join(categories) if categories else 'N/A'}

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

{'https://github.com/' + github if github else '暂无 GitHub 仓库'}

---
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
"""
    
    with open(filepath, 'w') as f:
        f.write(md_content)
    
    db_entry = {
        "id": f"paper_{int(time.time())}_{added}",
        "title": title,
        "authors": author_str,
        "conference": conf,
        "year": year,
        "url": url_link,
        "github_repo": github,
        "arxiv_id": arxiv_id,
        "keywords": categories,
        "published": published,
        "abstract_en": abstract[:1000],
        "abstract_cn": "",
        "introduction_en": "",
        "introduction_cn": "",
        "markdown_path": os.path.join(conf, str(year), filename),
        "topic": "LLM Serving",
        "venue": venue,
    }
    db['papers'].append(db_entry)
    added += 1
    print(f"  ✅ Added: [{arxiv_id or conf}] {title[:70]}")

with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\n=== Conference/GitHub Search Summary ===")
print(f"New papers added: {added}")
print(f"Total in database: {len(db['papers'])}")