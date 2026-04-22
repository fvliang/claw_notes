#!/usr/bin/env python3
"""Enhanced search for LLM serving papers - April 23, 2026"""
import json, urllib.request, urllib.parse, xml.etree.ElementTree as ET, time, os, re, sys

DB_PATH = '/home/admin/claw_notes/database.json'
PAPERS_DIR = '/home/admin/claw_notes/papers'

# Load existing database
with open(DB_PATH) as f:
    db = json.load(f)

existing_titles = set()
for p in db['papers']:
    existing_titles.add(p.get('title', '').lower().strip().replace('\n', ' ')[:80])
    existing_titles.add(p.get('title', '').lower().strip().replace('\n', ' ')[:60])

print(f"Existing papers in DB: {len(db['papers'])}")
print(f"Unique title prefixes: {len(existing_titles)}")

# Extended keywords
keywords = [
    "llm serving",
    "speculative decoding",
    "llm inference",
    "llm inference efficiency",
    "llm inference optimization",
    "kv cache",
    "kv cache compression",
    "llm inference system",
    "llm serving system",
    "llm inference acceleration",
    "large language model serving",
    "large language model inference",
    "paged attention",
    "continuous batching",
    "llm inference latency",
    "llm inference throughput",
    "distributed llm inference",
    "llm quantization inference",
    "llm memory optimization",
    "llm decoding acceleration",
    "prefill decode disaggregation",
    "llm inference gpu",
    "speculative execution llm",
    "draft model verification",
    "efficient llm inference",
    "llm batch scheduling",
    "llm cost efficiency",
]

def search_arxiv(query, max_results=50):
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        return parse_arxiv_xml(data)
    except Exception as e:
        print(f"  Error fetching {query}: {e}")
        return []

def parse_arxiv_xml(xml_data):
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        arxiv_id = entry.find('atom:id', ns).text.strip()
        arxiv_id_clean = arxiv_id.replace('http://arxiv.org/abs/', '')
        published = entry.find('atom:published', ns).text.strip()[:10]
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
        # Find PDF link
        pdf_link = ""
        for link in entry.findall('atom:link', ns):
            if link.attrib.get('title') == 'pdf':
                pdf_link = link.attrib['href']
        papers.append({
            'title': title,
            'abstract_en': summary,
            'arxiv_id': arxiv_id_clean,
            'url': arxiv_id,
            'pdf': pdf_link,
            'published': published,
            'authors': authors,
            'categories': categories,
        })
    return papers

# LLM serving relevance filter - comprehensive
SERVING_KEYWORDS = [
    'serving', 'inference', 'speculative decoding', 'kv cache', 'kv-cache',
    'prefill', 'decode', 'batching', 'scheduling', 'throughput', 'latency',
    'memory management', 'paged attention', 'vllm', 'continuous batching',
    'disaggregation', 'offloading', 'quantization for inference',
    'compression for inference', 'acceleration', 'ttft', 'tpot',
    'attention kernel', 'flash attention', 'inferencing',
    'token budget', 'draft model', 'verification', 'acceptance rate',
    'load balancing', 'request routing', 'gpu memory', 'memory footprint',
    'cost-efficient inference', 'efficient inference', 'inference speedup',
    'inference latency', 'inference throughput', 'inference optimization',
    'model serving', 'model deployment', 'inference system',
    'inference framework', 'inference engine', 'inference acceleration',
    'self-speculative', 'early exit', 'layer skipping',
    'parallel decoding', 'batched inference', 'speculative execution',
    'distributed inference', 'edge inference', 'on-device inference',
    'megakernel', 'kernel optimization', 'cuda kernel', 'metal kernel',
    'context-aware scheduling', 'request scheduling', 'batch scheduling',
    'lora adapter serving', 'adapter routing', 'adapter caching',
    'agentic inference', 'inference scaling', 'decode phase',
    'kv compression', 'kv share', 'kv residual', 'cache compression',
    'cross-layer kv', 'prefix caching', 'token eviction',
    'fp4 inference', 'fp8 inference', 'low-bit inference',
    'tpu inference', 'pim inference', 'processing-in-memory',
    'speculative offloading', 'draft tree', 'medusa',
    'disagg serving', 'prefill-decode disaggregation',
    'long-context inference', 'long-context serving',
    'moE inference', 'moe serving', 'moe routing',
    'power-efficient inference', 'energy-efficient inference',
    'gpu cluster', 'multi-gpu inference', 'multi-die',
]

NOT_SERVING_KEYWORDS = [
    'training', 'fine-tuning', 'pre-training', 'alignment', 'rlhf',
    'safety', 'guard model', 'phishing', 'question answering benchmark',
    'prompt tuning', 'prompt engineering only',
    'sentiment analysis', 'text classification', 'translation model',
    'code generation model', 'math reasoning model',
    'medical llm', 'clinical', 'health llm', 'drug', 'bioinformatics',
    'drug discovery', 'protein', 'molecule', 'chemistry',
    'education llm', 'survey paper', 'benchmark evaluation only',
    'social media', 'election', 'politics', 'law',
    'music generation', 'art generation', 'creative writing',
    'robotics', 'autonomous driving', 'vision-language model',
    'speech recognition', 'asr', 'tts synthesis',
    'image generation', 'video generation', 'diffusion model',
    'graph neural', 'gnn', 'knowledge graph',
    'embedding model', 'representation learning',
    'data augmentation', 'dataset creation',
    'emotion recognition', 'hate speech',
    'argument mining', 'dialogue system',
    'retrieval augmented generation', 'rag system',
    'retrieval model', 'search engine',
    'world model', 'simulation', 'game ai',
    'multilingual translation',
    'protein folding', 'genomics',
    'climate model', 'weather prediction',
    'financial prediction', 'stock prediction',
    'legal ai', 'court', 'judgment',
    'child safety', 'content moderation',
    'jailbreak', 'red teaming', 'adversarial attack',
    'watermarking', 'membership inference',
]

def is_llm_serving_paper(title, abstract):
    text = (title + ' ' + abstract).lower()
    # Must have LLM-related term
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 'language model serving', 'transformer inference', 'transformer serving', 'gpt inference', 'gpt serving'])
    if not has_llm:
        return False
    # Check serving score
    serving_score = sum(1 for k in SERVING_KEYWORDS if k in text)
    not_serving_score = sum(1 for k in NOT_SERVING_KEYWORDS if k in text)
    if serving_score >= 2 and serving_score > not_serving_score:
        return True
    if serving_score >= 1 and not_serving_score == 0:
        strong_terms = ['serving system', 'inference system', 'inference framework', 'inference engine',
                       'speculative decoding', 'kv cache', 'prefill', 'decode phase',
                       'inference latency', 'inference throughput', 'inference speedup',
                       'inference acceleration', 'inference optimization', 'efficient inference',
                       'parallel decoding', 'batched inference', 'llm serving',
                       'serving framework', 'serving engine', 'serving latency']
        return any(k in text for k in strong_terms)
    return False

# Only search recent papers (2026)
print("\n🔍 Searching arXiv for recent LLM serving papers...")

all_found = []
for kw in keywords:
    print(f"  Searching: {kw}")
    results = search_arxiv(kw, max_results=30)
    for p in results:
        if is_llm_serving_paper(p['title'], p['abstract_en']):
            # Only accept papers from 2026
            if p['published'].startswith('2026'):
                all_found.append(p)
    time.sleep(0.5)

# Deduplicate by title
seen = set()
unique = []
for p in all_found:
    key = p['title'].lower().strip().replace('\n', ' ')[:80]
    if key not in seen:
        seen.add(key)
        unique.append(p)

# Filter out existing
new_papers = []
for p in unique:
    key = p['title'].lower().strip().replace('\n', ' ')[:80]
    short_key = p['title'].lower().strip().replace('\n', ' ')[:60]
    if key not in existing_titles and short_key not in existing_titles:
        new_papers.append(p)

print(f"\n📊 Search results:")
print(f"  Total found (after relevance filter, 2026 only): {len(unique)}")
print(f"  Already in DB: {len(unique) - len(new_papers)}")
print(f"  New papers (not in DB): {len(new_papers)}")

if not new_papers:
    print("\n⚠️ No new papers found from arXiv search. Trying broader search...")

    # Try broader search with different query format
    broad_queries = [
        "cat:cs.CL+AND+abs:serving",
        "cat:cs.DC+AND+abs:inference+AND+abs:llm",
        "cat:cs.AR+AND+abs:inference+AND+abs:language",
        "cat:cs.LG+AND+abs:speculative+AND+abs:decoding",
        "cat:cs.CL+AND+abs:kv+AND+abs:cache",
    ]
    for q in broad_queries:
        print(f"  Broad search: {q}")
        url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&sortBy=submittedDate&sortOrder=descending&max_results=30&start=0"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read().decode('utf-8')
            results = parse_arxiv_xml(data)
            for p in results:
                if p['published'].startswith('2026') and is_llm_serving_paper(p['title'], p['abstract_en']):
                    key = p['title'].lower().strip().replace('\n', ' ')[:80]
                    short_key = p['title'].lower().strip().replace('\n', ' ')[:60]
                    if key not in seen:
                        seen.add(key)
                        if key not in existing_titles and short_key not in existing_titles:
                            new_papers.append(p)
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(0.5)

    # Also search most recent 100 papers in cs.CL and cs.DC
    print("  Searching recent cs.CL papers...")
    recent_cl = search_arxiv("cat:cs.CL", max_results=100)
    for p in recent_cl:
        if p['published'].startswith('2026') and is_llm_serving_paper(p['title'], p['abstract_en']):
            key = p['title'].lower().strip().replace('\n', ' ')[:80]
            short_key = p['title'].lower().strip().replace('\n', ' ')[:60]
            if key not in seen:
                seen.add(key)
                if key not in existing_titles and short_key not in existing_titles:
                    new_papers.append(p)

    print("  Searching recent cs.DC papers...")
    recent_dc = search_arxiv("cat:cs.DC", max_results=100)
    for p in recent_dc:
        if p['published'].startswith('2026') and is_llm_serving_paper(p['title'], p['abstract_en']):
            key = p['title'].lower().strip().replace('\n', ' ')[:80]
            short_key = p['title'].lower().strip().replace('\n', ' ')[:60]
            if key not in seen:
                seen.add(key)
                if key not in existing_titles and short_key not in existing_titles:
                    new_papers.append(p)

    print("  Searching recent cs.AR papers...")
    recent_ar = search_arxiv("cat:cs.AR", max_results=50)
    for p in recent_ar:
        if p['published'].startswith('2026') and is_llm_serving_paper(p['title'], p['abstract_en']):
            key = p['title'].lower().strip().replace('\n', ' ')[:80]
            short_key = p['title'].lower().strip().replace('\n', ' ')[:60]
            if key not in seen:
                seen.add(key)
                if key not in existing_titles and short_key not in existing_titles:
                    new_papers.append(p)

    print(f"\n📊 After broad search:")
    print(f"  New papers: {len(new_papers)}")

print("\n📋 New paper titles:")
for p in new_papers:
    print(f"  [{p['arxiv_id']}] {p['title'][:80]} | {p['published']}")

if not new_papers:
    print("\n❌ No new LLM serving papers found today. Database is up to date.")
    sys.exit(0)

# Now add papers to database and create markdown files
added_count = 0
for p in new_papers:
    # Determine conference
    cats = p.get('categories', [])
    if any('cs.CL' in c for c in cats):
        conf = 'arxiv'
    elif any('cs.DC' in c for c in cats):
        conf = 'arxiv'
    elif any('cs.AR' in c for c in cats):
        conf = 'arxiv'
    else:
        conf = 'arxiv'
    
    year = 2026 if p['published'].startswith('2026') else 2025
    
    # Create directory
    dir_path = os.path.join(PAPERS_DIR, conf, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    # Create filename
    safe_title = re.sub(r'[^\w\s-]', '', p['title'][:60]).strip().replace(' ', '_')
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    # Check if file already exists
    if os.path.exists(filepath):
        continue
    
    # Try to find GitHub repo (search in abstract)
    github = ""
    abstract_text = p.get('abstract_en', '')
    gh_match = re.search(r'github\.com/([^\s\)]+)', abstract_text)
    if gh_match:
        github = gh_match.group(1)
    
    # Write markdown file
    md_content = f"""# {p['title']}

**ArXiv ID:** {p['arxiv_id']}
**Published:** {p['published']}
**Authors:** {', '.join(p.get('authors', []))}
**URL:** https://arxiv.org/abs/{p['arxiv_id']}
**PDF:** https://arxiv.org/pdf/{p['arxiv_id']}
**GitHub:** {github if github else '暂无'}
**Categories:** {', '.join(p.get('categories', []))}

## Abstract (English)

{abstract_text}

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
    
    # Add to database with compatible schema
    db_entry = {
        "id": f"paper_{int(time.time())}_{added_count}",
        "title": p['title'],
        "authors": ', '.join(p.get('authors', [])),
        "conference": conf,
        "year": year,
        "url": f"https://arxiv.org/abs/{p['arxiv_id']}",
        "github_repo": github,
        "arxiv_id": p['arxiv_id'],
        "keywords": p.get('categories', []),
        "published": p['published'],
        "abstract_en": abstract_text[:500],
        "abstract_cn": "",
        "introduction_en": "",
        "introduction_cn": "",
        "markdown_path": os.path.join(conf, str(year), filename),
        "topic": "LLM Serving",
    }
    db['papers'].append(db_entry)
    added_count += 1
    print(f"  ✅ Added: [{p['arxiv_id']}] {p['title'][:80]}")

# Save database
with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\n✅ Total papers added: {added_count}")
print(f"📊 Total papers in database: {len(db['papers'])}")