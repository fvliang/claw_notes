#!/usr/bin/env python3
"""Comprehensive LLM serving paper search - 2026-04-24 morning"""
import json, urllib.request, urllib.parse, time, re, sys, os, xml.etree.ElementTree as ET

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
    if p.get('arxiv_id'):
        existing_arxiv.add(p['arxiv_id'])

print(f"📊 Existing: {len(db['papers'])} papers, {len(existing_arxiv)} arxiv IDs")

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
    'lora adapter serving', 'adapter routing',
    'agentic inference', 'inference scaling',
    'kv compression', 'cache compression',
    'prefix caching', 'token eviction',
    'gpu memory', 'memory footprint',
    'draft model', 'verification', 'acceptance rate',
    'load balancing', 'request routing',
    'flash attention', 'attention kernel',
    'megakernel', 'kernel optimization',
    'moe inference', 'moe serving',
    'long-context inference', 'context window',
    'generation latency', 'cost-efficient inference',
    'speculative sampling', 'medusa', 'eagle',
    'token budget', 'model deployment',
]

NOT_SERVING_KEYWORDS = [
    'training system', 'distributed training', 'fine-tuning system',
    'pre-training', 'gradient', 'optimizer',
    'safety', 'guard model', 'phishing',
    'question answering', 'retrieval augmented', 'rag',
    'prompt tuning', 'prompt engineering',
    'sentiment analysis', 'text classification',
    'translation', 'summarization task',
    'code generation model', 'math reasoning',
    'medical', 'clinical', 'health', 'drug',
    'protein', 'molecule', 'chemistry',
    'education', 'survey only',
    'social media', 'election',
    'music', 'art generation', 'creative',
    'robotics', 'autonomous driving',
    'speech recognition', 'asr', 'tts synthesis',
    'image generation', 'video generation', 'diffusion model',
    'graph neural', 'gnn', 'knowledge graph construction',
    'embedding model', 'representation learning',
    'data augmentation', 'dataset construction',
]

def is_llm_serving(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference',
                                       'language model serving', 'transformer inference',
                                       'transformer serving', 'foundation model inference',
                                       'autoregressive inference', 'generative model inference',
                                       'gpt inference', 'gpt serving', 'llm system',
                                       'inference platform', 'serving platform'])
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
        with urllib.request.urlopen(req, timeout=30) as resp:
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
                'authors': authors,
                'source': 'arxiv',
            })
        except:
            continue
    return papers

# ── Semantic Scholar search ──
def search_s2(query, limit=20, year_from=2025):
    encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&year={year_from}-2026&fields=title,abstract,url,externalIds,authors,publicationDate,fieldsOfStudy,venue&sort=relevance"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (PaperCollector/3.0)')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = []
        for p in data.get('data', []):
            arxiv_id = p.get('externalIds', {}).get('ArXiv', '') or ''
            results.append({
                'title': p.get('title', '') or '',
                'abstract_en': p.get('abstract', '') or '',
                'arxiv_id': arxiv_id,
                'published': p.get('publicationDate', '') or '',
                'authors': [a.get('name', '') for a in p.get('authors', [])] if isinstance(p.get('authors'), list) else [],
                'venue': p.get('venue', '') or '',
                'source': 'semantic_scholar',
            })
        return results
    except Exception as e:
        print(f"  S2 error: {e}")
        return []

# ── GitHub search for LLM serving repos ──
def search_github(query, per_page=30):
    url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}&sort=stars&order=desc&per_page={per_page}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (PaperCollector/3.0)')
        req.add_header('Accept', 'application/vnd.github.v3+json')
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        repos = []
        for r in data.get('items', []):
            desc = r.get('description', '') or ''
            if is_llm_serving(r.get('name', ''), desc):
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
}

def guess_conference(venue, title, abstract):
    """Guess conference from venue string or paper content."""
    if venue:
        v = venue.lower()
        for conf_lower, conf_name in CONFERENCE_MAP.items():
            if conf_lower in v:
                return conf_name
    return 'arXiv'

def guess_topic(title, abstract):
    """Classify into topic categories."""
    text = (title + ' ' + (abstract or '')).lower()
    if 'speculative' in text and ('decoding' in text or 'sampling' in text or 'execution' in text):
        return 'Speculative Decoding'
    if 'kv cache' in text or 'kv-cache' in text or 'kv compression' in text or 'prefix caching' in text or 'token eviction' in text:
        return 'KV Cache'
    if 'quantization' in text or 'quant' in text:
        return 'Quantization'
    if 'moe' in text or 'mixture of expert' in text:
        return 'MoE Inference'
    if 'disaggregation' in text or 'prefill' in text:
        return 'Prefill/Disaggregation'
    if 'scheduling' in text or 'routing' in text or 'load balancing' in text:
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
    return 'LLM Serving'

# ── MAIN SEARCH ──
all_papers = []  # {title, abstract_en, arxiv_id, published, authors, conference, topic, source}
all_repos = []
seen_titles = set()

# 1. ArXiv search with main keywords
arxiv_queries = [
    "llm serving",
    "speculative decoding",
    "llm inference optimization",
    "kv cache llm",
    "llm inference latency",
]

print("\n🔍 Phase 1: ArXiv Search")
for q in arxiv_queries:
    print(f"  Searching arXiv: {q}")
    results = search_arxiv(q, max_results=30)
    for p in results:
        key = p['title'].lower().strip()[:80]
        if key not in seen_titles and is_llm_serving(p['title'], p['abstract_en']):
            seen_titles.add(key)
            p['conference'] = 'arXiv'
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
    time.sleep(2)

# 2. Semantic Scholar for conference papers
s2_queries = [
    "LLM inference serving system",
    "speculative decoding LLM",
    "KV cache LLM inference efficiency",
    "LLM serving scheduling batching",
    "LLM inference disaggregation prefill",
    "LLM inference kernel acceleration",
    "quantization LLM inference serving",
]

print("\n🔍 Phase 2: Semantic Scholar Search")
for q in s2_queries:
    print(f"  Searching S2: {q}")
    results = search_s2(q, limit=20, year_from=2025)
    for p in results:
        key = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
        if key not in seen_titles and is_llm_serving(p['title'], p['abstract_en']):
            seen_titles.add(key)
            p['conference'] = guess_conference(p.get('venue', ''), p['title'], p['abstract_en'])
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
    time.sleep(3)

# 3. GitHub search for LLM serving repos
print("\n🔍 Phase 3: GitHub Search")
gh_queries = [
    "llm serving inference",
    "speculative decoding",
    "llm inference engine",
    "kv cache llm",
]
for q in gh_queries:
    print(f"  Searching GitHub: {q}")
    repos = search_github(q, per_page=20)
    for r in repos:
        if r['stars'] >= 10:  # Only repos with decent stars
            all_repos.append(r)
    time.sleep(2)

# ── Filter new papers (not in DB) ──
new_papers = []
for p in all_papers:
    key = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '').replace('$', '').replace('\\', '')[:80]
    arxiv_id = p.get('arxiv_id', '')
    if key not in existing_titles and arxiv_id not in existing_arxiv:
        new_papers.append(p)

new_repos = []
for r in all_repos:
    # Check if repo is already linked in DB
    already = False
    for p in db['papers']:
        if r['full_name'] in (p.get('github_repo', '') or ''):
            already = True
            break
    if not already:
        new_repos.append(r)

print(f"\n📈 Results:")
print(f"  Found via search: {len(all_papers)} papers")
print(f"  New papers (not in DB): {len(new_papers)}")
print(f"  GitHub repos found: {len(all_repos)}, new: {len(new_repos)}")

# Print new papers
print(f"\n🆕 New papers to add:")
for p in new_papers:
    conf = p.get('conference', 'arXiv')
    topic = p.get('topic', 'LLM Serving')
    arxiv = p.get('arxiv_id', '')
    print(f"  [{arxiv}] {p['title'][:70]} | {conf} | {topic}")

print(f"\n🆕 New GitHub repos:")
for r in new_repos:
    print(f"  ⭐{r['stars']} {r['full_name']}: {r['description'][:60]}")

# Save results
with open('/tmp/search_results_20260424.json', 'w') as f:
    json.dump({'new_papers': new_papers, 'new_repos': new_repos}, f, indent=2, ensure_ascii=False)

print(f"\n✅ Results saved to /tmp/search_results_20260424.json")