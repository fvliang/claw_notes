#!/usr/bin/env python3
"""Fetch recent arXiv papers from cs.DC, cs.CL, cs.AR and filter for LLM serving - 2026-04-23"""
import json, time, os, re, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET

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
    'inference framework', 'inference engine', 'inference acceleration',
    'self-speculative', 'early exit', 'layer skipping',
    'parallel decoding', 'batched inference', 'speculative execution',
    'distributed inference', 'edge inference', 'on-device inference',
    'lora adapter serving', 'agentic inference', 'inference scaling',
    'kv compression', 'cache compression', 'prefix caching', 'token eviction',
    'moe inference', 'moe serving', 'multi-gpu inference', 'memory footprint',
    'flash attention', 'draft model', 'tensor parallel inference',
    'speculative sampling', 'medusa', 'eagle', 'long-context inference',
    'generation latency', 'generation speedup', 'gpu memory',
    'request scheduling', 'batch scheduling', 'weight quantization inference',
    'cost-efficient inference',
]
TRAINING_EXCLUSIONS = [
    'training system', 'distributed training', 'fine-tuning system',
    'pre-training', 'gradient accumulation', 'optimizer',
    'training efficiency', 'training acceleration', 'training framework',
    'training infrastructure', 'distributed training system',
]

def is_llm_serving(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 
                                       'language model serving', 'transformer inference', 
                                       'transformer serving', 'foundation model inference',
                                       'autoregressive inference'])
    if not has_llm:
        return False
    ss = sum(1 for k in SERVING_KEYWORDS if k in text)
    ts = sum(1 for k in TRAINING_EXCLUSIONS if k in text)
    return (ss >= 1 and ss > ts) or ss >= 2

# Recent arXiv IDs from cs.DC (Apr 23), cs.CL (Apr 23), and cs.AR
# Fetch in small batches with long delays to avoid 429
recent_ids = [
    # cs.DC Apr 23
    "2604.20503", "2604.20410", "2604.20342", "2604.20081", "2604.20032",
    "2604.19958", "2604.20825", "2604.20819", "2604.20639", "2604.20599",
    "2604.20129", "2604.20062", "2604.19792",
    # cs.DC Apr 22 (some already in DB, but check anyway)
    "2604.19654", "2604.19503", "2604.19494", "2604.19454", "2604.19363",
    "2604.19337", "2604.19243", "2604.19241", "2604.19181", "2604.19004",
    "2604.18655", "2604.18616", "2604.18615", "2604.18614",
    # cs.CL Apr 23 (first batch)
    "2604.20842", "2604.20835", "2604.20817", "2604.20791", "2604.20789",
    "2604.20738", "2604.20726", "2604.20677", "2604.20666", "2604.20658",
    "2604.20572", "2604.20564", "2604.20560", "2604.20556", "2604.20549",
    "2604.20548", "2604.20535", "2604.20531", "2604.20487", "2604.20454",
    "2604.20447", "2604.20443", "2604.20398", "2604.20382", "2604.20331",
    "2604.20283", "2604.20256", "2604.20244", "2604.20241", "2604.20225",
]

def fetch_paper_batch(ids):
    """Fetch paper metadata from arXiv API in a batch"""
    id_list = ','.join(ids)
    url = f"http://export.arxiv.org/api/query?id_list={id_list}&max_results={len(ids)}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (compatible; PaperCollector/2.0; mailto:research@example.com)')
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        root = ET.fromstring(data)
        papers = []
        for entry in root.findall('atom:entry', ns):
            try:
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ').replace('{', '').replace('}', '')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ').replace('{', '').replace('}', '')
                aid = entry.find('atom:id', ns).text.strip().replace('http://arxiv.org/abs/', '').split('v')[0]
                published = entry.find('atom:published', ns).text.strip()[:10]
                authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
                papers.append({
                    'title': title,
                    'abstract_en': summary,
                    'arxiv_id': aid,
                    'published': published,
                    'authors': authors,
                    'categories': categories,
                    'source': 'arxiv',
                })
            except Exception as e:
                continue
        return papers
    except Exception as e:
        print(f"  Batch error: {e}")
        return []

all_arxiv_papers = []
# Fetch in batches of 5 with 10 second delays
batch_size = 5
for i in range(0, len(recent_ids), batch_size):
    batch = recent_ids[i:i+batch_size]
    print(f"Fetching batch {i//batch_size + 1}/{len(recent_ids)//batch_size + 1}: {len(batch)} IDs")
    results = fetch_paper_batch(batch)
    for p in results:
        if is_llm_serving(p['title'], p['abstract_en']):
            all_arxiv_papers.append(p)
            print(f"  ✅ Relevant: [{p['arxiv_id']}] {p['title'][:60]}")
        else:
            # Skip non-serving papers quietly
            pass
    print(f"  Batch returned {len(results)} papers, {len(all_arxiv_papers)} total serving")
    time.sleep(10)  # Long delay to avoid 429

print(f"\narXiv total serving papers found: {len(all_arxiv_papers)}")

# Deduplicate against existing DB
new_arxiv_papers = []
for p in all_arxiv_papers:
    key = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
    short = key[:60]
    if key not in existing_titles and short not in existing_titles:
        new_arxiv_papers.append(p)

print(f"Already in DB: {len(all_arxiv_papers) - len(new_arxiv_papers)}")
print(f"New arxiv papers: {len(new_arxiv_papers)}")

for i, p in enumerate(new_arxiv_papers, 1):
    print(f"  {i}. [{p['arxiv_id']}] {p['title'][:70]} | {p['published']}")

# Add papers to database
CONFERENCE_MAP = {
    'osdi': 'OSDI', 'sosp': 'SOSP', 'nsdi': 'NSDI', 'sigcomm': 'SIGCOMM',
    'sigmod': 'SIGMOD', 'atc': 'ATC', 'eurosys': 'EuroSys', 'dac': 'DAC',
    'asplos': 'ASPLOS', 'sc': 'SC', 'nips': 'NeurIPS', 'neurips': 'NeurIPS',
    'iclr': 'ICLR', 'icml': 'ICML', 'acl': 'ACL', 'emnlp': 'EMNLP',
}

added = 0
for p in new_arxiv_papers:
    title = p.get('title', '')
    abstract = p.get('abstract_en', '') or ''
    arxiv_id = p.get('arxiv_id', '')
    authors = p.get('authors', [])
    author_str = ', '.join(authors) if isinstance(authors, list) else str(authors)
    published = p.get('published', '')
    categories = p.get('categories', [])
    
    year = 2026
    if published:
        try:
            year = int(published[:4])
        except:
            year = 2026
    
    conf = 'arxiv'
    # Check categories for conference hints
    
    # Find GitHub
    github = ""
    gh_matches = re.findall(r'github\.com/([^\s\)\.\,]+)', abstract)
    if gh_matches:
        github = gh_matches[0]
    
    # Create directory
    dir_path = os.path.join(PAPERS_DIR, conf, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    safe_title = re.sub(r'[^\w\s-]', '', title[:60]).strip().replace(' ', '_')
    if not safe_title:
        safe_title = f"paper_{arxiv_id}"
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(filepath):
        continue
    
    url_link = f"https://arxiv.org/abs/{arxiv_id}"
    pdf_link = f"https://arxiv.org/pdf/{arxiv_id}"
    
    md_content = f"""# {title}

**ArXiv ID:** {arxiv_id}
**Published:** {published}
**Authors:** {author_str}
**URL:** {url_link}
**PDF:** {pdf_link}
**GitHub:** {github if github else '暂无'}
**Categories:** {', '.join(categories)}

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
    }
    db['papers'].append(db_entry)
    added += 1
    print(f"  ✅ Added: [{arxiv_id}] {title[:70]}")

with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\n=== arXiv Summary ===")
print(f"New arxiv papers: {added}")
print(f"Total in database: {len(db['papers'])}")