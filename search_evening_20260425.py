#!/usr/bin/env python3
"""LLM serving paper search - 2026-04-25 evening"""
import json, urllib.request, urllib.parse, time, re, os, xml.etree.ElementTree as ET

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

print(f"📊 DB: {len(db['papers'])} papers, {len(existing_arxiv)} arxiv IDs")

# ── Relevance filter ──
SERVING_KEYWORDS = [
    'serving', 'inference', 'speculative decoding', 'kv cache', 'kv-cache',
    'prefill', 'decode', 'batching', 'scheduling', 'throughput', 'latency',
    'memory management', 'paged attention', 'vllm', 'continuous batching',
    'disaggregation', 'offloading', 'acceleration', 'ttft', 'tpot',
    'inference speedup', 'inference latency', 'inference throughput',
    'inference optimization', 'efficient inference', 'inference system',
    'inference framework', 'inference engine', 'inference acceleration',
    'self-speculative', 'early exit', 'layer skipping',
    'parallel decoding', 'speculative execution',
    'distributed inference', 'edge inference', 'on-device inference',
    'gpu memory', 'draft model', 'verification', 'acceptance rate',
    'load balancing', 'request routing',
    'flash attention', 'attention kernel',
    'moe inference', 'moe serving',
    'long-context inference', 'generation latency',
    'cost-efficient inference', 'model deployment',
    'speculative sampling', 'prefix caching', 'token eviction',
    'kv compression', 'cache compression',
    'lora serving', 'adapter serving',
    'token budget', 'request scheduling',
    'attention sink', 'streaming llm',
    'chunked prefill', 'microbatch',
    'speculative drafting', 'verification head',
    'distillation inference', 'model compression serving',
    'tensor parallel', 'pipeline parallel', 'data parallel',
    'speculative', 'draft', 'medusa', 'eagle',
]

NOT_SERVING_KEYWORDS = [
    'training system', 'distributed training', 'fine-tuning system',
    'pre-training', 'gradient', 'optimizer',
    'safety alignment', 'rlhf', 'dpo',
    'question answering', 'retrieval augmented generation',
    'prompt tuning', 'sentiment analysis',
    'code generation benchmark', 'math reasoning benchmark',
    'medical', 'clinical', 'health', 'drug discovery',
    'protein structure', 'molecule', 'chemistry',
    'education', 'survey review only',
    'robotics', 'autonomous driving',
    'speech recognition', 'image generation', 'video generation',
    'diffusion model training', 'graph neural network',
    'embedding model', 'dataset construction',
    'social media', 'election', 'politics',
    'music', 'art generation', 'creative writing',
    'game playing', 'reinforcement learning training',
    'phishing', 'guard model', 'alignment',
    'reward model', 'human feedback',
]

def is_llm_serving(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference',
                                       'language model serving', 'transformer inference',
                                       'foundation model inference', 'autoregressive inference',
                                       'gpt inference', 'llm system', 'llm workload',
                                       'inference platform', 'serving platform',
                                       'generative inference', 'text generation inference'])
    if not has_llm:
        return False
    ss = sum(1 for k in SERVING_KEYWORDS if k in text)
    ns = sum(1 for k in NOT_SERVING_KEYWORDS if k in text)
    return (ss >= 2 and ss > ns) or (ss >= 1 and ns == 0 and has_llm)

# ── ArXiv search ──
def search_arxiv(query, max_results=30):
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read().decode('utf-8')
        return parse_arxiv_xml(data)
    except Exception as e:
        print(f"  arXiv error: {e}")
        return []

def parse_arxiv_xml(xml_data):
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall('atom:entry', ns):
        try:
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            arxiv_id = entry.find('atom:id', ns).text.strip().replace('http://arxiv.org/abs/', '')
            published = entry.find('atom:published', ns).text.strip()[:10]
            authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
            papers.append({
                'title': title,
                'abstract_en': summary,
                'arxiv_id': arxiv_id,
                'published': published,
                'authors': ', '.join(authors[:5]),
                'source': 'arxiv',
            })
        except:
            continue
    return papers

# ── Semantic Scholar search ──
def search_s2(query, limit=15, year_from=2025):
    encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&year={year_from}-2026&fields=title,abstract,url,externalIds,authors,publicationDate,venue&sort=relevance"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (PaperCollector/3.0)')
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = []
        for p in data.get('data', []):
            arxiv_id = p.get('externalIds', {}).get('ArXiv', '') or ''
            results.append({
                'title': p.get('title', '') or '',
                'abstract_en': p.get('abstract', '') or '',
                'arxiv_id': arxiv_id,
                'published': p.get('publicationDate', '') or '',
                'authors': ', '.join([a.get('name', '') for a in p.get('authors', [])][:5]) if isinstance(p.get('authors'), list) else '',
                'venue': p.get('venue', '') or '',
                'source': 'semantic_scholar',
            })
        return results
    except Exception as e:
        print(f"  S2 error: {e}")
        return []

# ── GitHub search ──
def search_github(query, per_page=15):
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc&per_page={per_page}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (PaperCollector/3.0)')
        req.add_header('Accept', 'application/vnd.github.v3+json')
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        repos = []
        for r in data.get('items', []):
            desc = r.get('description', '') or ''
            repos.append({
                'name': r.get('name', ''),
                'full_name': r.get('full_name', ''),
                'description': desc,
                'url': r.get('html_url', ''),
                'stars': r.get('stargazers_count', 0),
                'language': r.get('language', ''),
                'topics': r.get('topics', []),
                'source': 'github',
            })
        return repos
    except Exception as e:
        print(f"  GitHub error: {e}")
        return []

# ── Conference mapping ──
CONFERENCE_MAP = {
    'osdi': 'OSDI', 'sosp': 'SOSP', 'nsdi': 'NSDI',
    'sigcomm': 'SIGCOMM', 'sigmod': 'SIGMOD',
    'atc': 'ATC', 'eurosys': 'EuroSys', 'dac': 'DAC',
    'asplos': 'ASPLOS', 'sc': 'SC',
    'nips': 'NeurIPS', 'neurips': 'NeurIPS',
    'iclr': 'ICLR', 'icml': 'ICML',
    'acl': 'ACL', 'emnlp': 'EMNLP',
    'mlsys': 'MLSys', 'isca': 'ISCA',
    'euromlsys': 'EuroMLSys', 'ispass': 'ISPASS',
    'naacl': 'NAACL', 'cvpr': 'CVPR',
    'coling': 'COLING', 'colm': 'COLM',
}

def guess_conference(venue, title, abstract):
    if venue:
        v = venue.lower()
        for conf_lower, conf_name in CONFERENCE_MAP.items():
            if conf_lower in v:
                return conf_name
    return 'arXiv'

def guess_topic(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    if 'speculative' in text and ('decoding' in text or 'sampling' in text or 'execution' in text or 'draft' in text):
        return 'Speculative Decoding'
    if 'kv cache' in text or 'kv-cache' in text or 'kv compression' in text or 'prefix caching' in text or 'token eviction' in text or 'attention sink' in text:
        return 'KV Cache'
    if 'quantization' in text or 'quant' in text:
        return 'Quantization'
    if 'moe' in text or 'mixture of expert' in text:
        return 'MoE Inference'
    if 'disaggregation' in text or 'prefill' in text:
        return 'Prefill/Disaggregation'
    if 'scheduling' in text or 'routing' in text or 'load balancing' in text or 'request' in text:
        return 'Inference Scheduling'
    if 'early exit' in text or 'layer skip' in text:
        return 'Early Exit'
    if 'distributed' in text:
        return 'Distributed Inference'
    if 'edge' in text or 'on-device' in text or 'mobile' in text:
        return 'Edge Inference'
    if 'kernel' in text or 'flash attention' in text or 'megakernel' in text:
        return 'Inference Kernel'
    if 'pruning' in text:
        return 'LLM Pruning/Serving'
    if 'lora' in text or 'adapter' in text:
        return 'LoRA/Adapter Serving'
    if 'streaming' in text:
        return 'Streaming Inference'
    if 'offloading' in text or 'cpu' in text or 'heterogeneous' in text:
        return 'Offloading/Heterogeneous'
    if 'tensor parallel' in text or 'pipeline parallel' in text:
        return 'Parallelism'
    return 'LLM Serving'

# ── MAIN SEARCH ──
all_papers = []
all_repos = []
seen_titles = set()

# 1. ArXiv search - evening queries (more targeted, recent)
arxiv_queries = [
    "llm serving system 2025 2026",
    "speculative decoding llm efficient",
    "llm inference optimization serving",
    "kv cache management llm serving",
    "llm inference latency throughput scheduling",
    "continuous batching llm serving",
    "llm serving disaggregation prefill decode",
    "moe inference serving llm",
    "llm inference acceleration draft verification",
    "efficient llm inference quantization pruning",
    "llm inference gpu kernel flash attention",
    "llm serving heterogeneous offloading",
]

print("\n🔍 Phase 1: ArXiv Search (Evening)")
for q in arxiv_queries:
    print(f"  Searching arXiv: {q}")
    results = search_arxiv(q, max_results=25)
    found = 0
    for p in results:
        key = p['title'].lower().strip()[:80]
        if key not in seen_titles and is_llm_serving(p['title'], p['abstract_en']):
            seen_titles.add(key)
            p['conference'] = 'arXiv'
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
            found += 1
    print(f"    → found {found} relevant papers")
    time.sleep(4)

# 2. Semantic Scholar for conference papers (focus on 2025-2026 conferences)
s2_queries = [
    "LLM inference serving system OSDI SOSP NSDI ATC",
    "speculative decoding LLM verification",
    "KV cache management LLM serving efficient",
    "LLM serving scheduling batching throughput",
    "LLM inference disaggregation prefill decode",
    "efficient LLM inference kernel optimization",
    "MoE LLM inference serving routing",
    "LLM inference heterogeneous offloading",
    "distributed LLM inference serving",
    "LLM inference quantization serving",
    "LLM serving EuroSys ASPLOS DAC SC",
    "speculative decoding NeurIPS ICLR ICML",
]

print("\n🔍 Phase 2: Semantic Scholar Search (Evening)")
for q in s2_queries:
    print(f"  Searching S2: {q}")
    results = search_s2(q, limit=15, year_from=2025)
    found = 0
    for p in results:
        key = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
        if key not in seen_titles and is_llm_serving(p['title'], p['abstract_en']):
            seen_titles.add(key)
            p['conference'] = guess_conference(p.get('venue', ''), p['title'], p['abstract_en'])
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
            found += 1
    print(f"    → found {found} relevant papers")
    time.sleep(3.5)

# 3. GitHub search
print("\n🔍 Phase 3: GitHub Search (Evening)")
gh_queries = [
    "llm serving inference engine",
    "speculative decoding llm",
    "llm inference framework",
    "kv cache llm serving",
    "llm inference gpu optimization kernel",
    "vllm speculative decoding",
]
for q in gh_queries:
    print(f"  Searching GitHub: {q}")
    repos = search_github(q, per_page=15)
    for r in repos:
        if r['stars'] >= 30:
            all_repos.append(r)
    time.sleep(3)

# ── Filter new papers ──
new_papers = []
for p in all_papers:
    tkey = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '').replace('$', '').replace('\\', '')
    arxiv_id = p.get('arxiv_id', '').strip()
    
    is_existing = False
    for length in [80, 60, 50, 40]:
        if tkey[:length] in existing_titles:
            is_existing = True
            break
    if is_existing:
        continue
    if arxiv_id and arxiv_id in existing_arxiv:
        continue
    
    # Validate arxiv ID date range (2025-2026)
    if arxiv_id:
        base = arxiv_id.split('v')[0]
        try:
            yymm = base[:4]
            mm = int(yymm[2:4])
            yy = int(yymm[:2])
            if not ((yy == 25 and 1 <= mm <= 12) or (yy == 26 and 1 <= mm <= 4)):
                continue
        except:
            pass
    new_papers.append(p)

# Deduplicate repos
new_repos = []
seen_repos = set()
for r in all_repos:
    full_name = r.get('full_name', '')
    if full_name in seen_repos:
        continue
    seen_repos.add(full_name)
    already = any(full_name == (p.get('github_repo', '') or '') for p in db['papers'])
    if not already:
        new_repos.append(r)

print(f"\n📈 Search results:")
print(f"  Total found: {len(all_papers)} papers, {len(all_repos)} repos")
print(f"  New papers (not in DB): {len(new_papers)}")
print(f"  New repos (not in DB): {len(new_repos)}")

print(f"\n🆕 New papers to add:")
for p in new_papers:
    conf = p.get('conference', 'arXiv')
    topic = p.get('topic', 'LLM Serving')
    arxiv = p.get('arxiv_id', '')
    print(f"  [{arxiv}] {p['title'][:70]} | {conf} | {topic}")

# ── Write markdown notes and add to DB ──
def get_conference_dir(conference, year):
    conf_lower = conference.lower().strip()
    if conf_lower in ('arxiv', 'arXiv'):
        return os.path.join(CLAW_DIR, 'arXiv', str(year))
    dir_map = {
        'osdi': 'osdi', 'sosp': 'sosp', 'nsdi': 'nsdi',
        'sigcomm': 'sigcomm', 'sigmod': 'sigmod',
        'atc': 'atc', 'eurosys': 'eurosys', 'dac': 'dac',
        'asplos': 'asplos', 'sc': 'sc',
        'neurips': 'nips', 'nips': 'nips',
        'iclr': 'iclr', 'icml': 'icml',
        'acl': 'acl', 'emnlp': 'emnlp',
        'mlsys': 'mlsys', 'isca': 'isca',
        'euromlsys': 'euromlsys', 'ispass': 'ispass',
        'naacl': 'naacl', 'cvpr': 'cvpr',
    }
    dirname = dir_map.get(conf_lower, conf_lower)
    return os.path.join(CLAW_DIR, dirname, str(year))

def sanitize_filename(title, arxiv_id=''):
    name = title.replace('\n', ' ').strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name)
    if arxiv_id:
        base = arxiv_id.split('v')[0]
        name = f"{base}_{name[:60]}"
    else:
        name = name[:70]
    return name + '.md'

def translate_abstract_cn(abstract_en):
    """Stub for Chinese translation."""
    return "[中文翻译待补充] " + abstract_en[:200] + "..."

max_id = max((int(p.get('id', 0)) if str(p.get('id', '0')).isdigit() else 0) for p in db['papers']) if db['papers'] else 0

added_count = 0
added_files = []

for p in new_papers:
    published = p.get('published', '')
    try:
        year = int(published[:4]) if published else 2026
    except:
        year = 2026
    
    conference = p.get('conference', '') or 'arXiv'
    topic = p.get('topic', '') or guess_topic(p['title'], p.get('abstract_en', ''))
    arxiv_id = p.get('arxiv_id', '').strip()
    
    # Try to match GitHub repo
    github_repo = ''
    title_lower = p['title'].lower()
    for r in new_repos + all_repos:
        desc = (r.get('description', '') or '').lower()
        repo_name = r.get('name', '').lower()
        title_words = [w for w in title_lower.split()[:4] if len(w) > 3]
        if any(w in desc for w in title_words) and r.get('stars', 0) >= 50:
            github_repo = r.get('full_name', '')
            break
    
    # Write markdown note
    dir_path = get_conference_dir(conference, year)
    os.makedirs(dir_path, exist_ok=True)
    fname = sanitize_filename(p['title'], arxiv_id)
    filepath = os.path.join(dir_path, fname)
    
    if not os.path.exists(filepath):
        title = p.get('title', '').strip()
        abstract_en = p.get('abstract_en', '').strip()
        abstract_cn = translate_abstract_cn(abstract_en)
        authors = p.get('authors', '')
        
        md = f"""# {title}

## Metadata
- **Authors:** {authors}
- **Conference:** {conference} {year}
- **Topic:** {topic}
- **arXiv ID:** {arxiv_id}
- **Published:** {published}
- **GitHub:** {github_repo}

## 原文链接
- arXiv: https://arxiv.org/abs/{arxiv_id}
- PDF: https://arxiv.org/pdf/{arxiv_id}

## 摘要 (Abstract)

{abstract_en}

## 摘要 (中文)

{abstract_cn}

## 引言 (Introduction)

[引言内容待补充 - 需阅读全文]

## 博客内容

[相关博客内容待搜索补充]

## GitHub 介绍

{github_repo if github_repo else '[GitHub仓库待搜索补充]'}

---
*Auto-collected on 2026-04-25 evening*
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
        added_files.append(filepath)
    
    # Add to DB
    paper_entry = {
        'id': max_id + 1 + added_count,
        'title': p['title'].strip(),
        'authors': p.get('authors', '') or '',
        'arxiv_id': arxiv_id,
        'github_repo': github_repo,
        'conference': conference,
        'year': str(year),
        'topic': topic,
        'abstract_en': p.get('abstract_en', '').strip(),
        'abstract_cn': translate_abstract_cn(p.get('abstract_en', '').strip()),
    }
    db['papers'].append(paper_entry)
    added_count += 1
    print(f"  ✅ [{added_count}] {p['title'][:60]} | {conference} {year} | {topic}")

# ── Add top GitHub repos ──
gh_added = 0
for r in sorted(new_repos, key=lambda x: x.get('stars', 0), reverse=True)[:10]:
    if r.get('stars', 0) < 50:
        continue
    full_name = r.get('full_name', '')
    
    desc = r.get('description', '') or ''
    topic = guess_topic(r.get('name', ''), desc)
    
    gh_title = f"[GitHub] {r.get('name', '')}: {desc[:100]}"
    gh_key = gh_title.lower().strip()[:80]
    if gh_key in existing_titles:
        continue
    
    paper_entry = {
        'id': max_id + 1 + added_count + gh_added,
        'title': gh_title,
        'authors': full_name.split('/')[0] if '/' in full_name else '',
        'arxiv_id': '',
        'github_repo': full_name,
        'conference': 'GitHub',
        'year': '2026',
        'topic': topic,
        'abstract_en': desc,
        'abstract_cn': '[中文翻译待补充]',
    }
    db['papers'].append(paper_entry)
    gh_added += 1
    
    gh_dir = os.path.join(CLAW_DIR, 'github', '2026')
    os.makedirs(gh_dir, exist_ok=True)
    fname = sanitize_filename(r.get('name', ''))
    filepath = os.path.join(gh_dir, fname)
    if not os.path.exists(filepath):
        md = f"""# {r.get('name', '')}

## Metadata
- **GitHub:** https://github.com/{full_name}
- **Stars:** {r.get('stars', 0)}
- **Language:** {r.get('language', '')}
- **Topic:** {topic}

## GitHub 介绍

{desc}

## README Highlights

[需从GitHub README补充]

---
*Auto-collected on 2026-04-25 evening*
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md)
        added_files.append(filepath)
        print(f"  ✅ [GH-{gh_added}] ⭐{r.get('stars',0)} {full_name} | {topic}")

# Save database
with open(DB_PATH, 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print(f"\n📊 Summary:")
print(f"  Papers added to DB: {added_count}")
print(f"  GitHub repos added: {gh_added}")
print(f"  Markdown notes created: {len(added_files)}")
print(f"  Total DB papers now: {len(db['papers'])}")