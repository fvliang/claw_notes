#!/usr/bin/env python3
"""LLM serving paper search - 2026-04-27 evening v3 (fast, fewer queries)"""
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

SERVING_KEYWORDS = [
    'serving', 'inference', 'speculative decoding', 'kv cache', 'kv-cache',
    'prefill', 'decode', 'batching', 'scheduling', 'throughput', 'latency',
    'paged attention', 'vllm', 'continuous batching',
    'disaggregation', 'offloading', 'acceleration', 'ttft', 'tpot',
    'inference speedup', 'inference latency', 'inference throughput',
    'inference optimization', 'efficient inference', 'inference system',
    'inference framework', 'inference engine', 'inference acceleration',
    'draft model', 'verification', 'acceptance rate',
    'load balancing', 'request routing',
    'moe inference', 'moe serving',
    'long-context inference', 'generation latency',
    'speculative sampling', 'prefix caching', 'token eviction',
    'kv compression', 'lora serving', 'adapter serving',
    'attention sink', 'streaming llm',
    'chunked prefill', 'tensor parallel', 'pipeline parallel',
    'speculative', 'draft', 'medusa', 'eagle', 'flash attention',
]

NOT_SERVING_KEYWORDS = [
    'training system', 'distributed training', 'fine-tuning',
    'pre-training', 'gradient', 'optimizer',
    'safety alignment', 'rlhf', 'dpo',
    'retrieval augmented', 'retriever', 'dense retrieval',
    'code verification', 'math evaluation', 'judge framework',
    'medical', 'drug discovery', 'protein',
    'robotics', 'autonomous driving',
    'image generation', 'video generation', 'diffusion',
    'speech recognition', 'sentiment analysis',
    'social media', 'election', 'music', 'art generation',
    'game playing', 'reward model', 'guard model',
]

def is_llm_serving(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference',
                                       'language model serving', 'transformer inference',
                                       'foundation model inference', 'autoregressive',
                                       'gpt inference', 'llm system', 'llm workload',
                                       'inference platform', 'serving platform',
                                       'generative inference', 'text generation inference'])
    if not has_llm:
        return False
    ss = sum(1 for k in SERVING_KEYWORDS if k in text)
    ns = sum(1 for k in NOT_SERVING_KEYWORDS if k in text)
    if ns >= 2:
        return False
    return (ss >= 2) or (ss >= 1 and ns == 0 and has_llm)

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
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
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
                'title': title, 'abstract_en': summary, 'arxiv_id': arxiv_id,
                'published': published, 'authors': ', '.join(authors[:5]), 'source': 'arxiv',
            })
        except:
            continue
    return papers

CONFERENCE_MAP = {
    'osdi': 'OSDI', 'sosp': 'SOSP', 'nsdi': 'NSDI', 'sigcomm': 'SIGCOMM', 'sigmod': 'SIGMOD',
    'atc': 'ATC', 'eurosys': 'EuroSys', 'dac': 'DAC', 'asplos': 'ASPLOS', 'sc': 'SC',
    'neurips': 'NeurIPS', 'iclr': 'ICLR', 'icml': 'ICML', 'acl': 'ACL', 'emnlp': 'EMNLP',
    'mlsys': 'MLSys', 'isca': 'ISCA', 'euromlsys': 'EuroMLSys', 'ispass': 'ISPASS',
    'naacl': 'NAACL', 'cvpr': 'CVPR',
}

def guess_conference(venue):
    if venue:
        v = venue.lower()
        for k, name in CONFERENCE_MAP.items():
            if k in v:
                return name
    return 'arXiv'

def guess_topic(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    if 'speculative' in text and ('decoding' in text or 'draft' in text or 'sampling' in text):
        return 'Speculative Decoding'
    if 'kv cache' in text or 'kv-cache' in text or 'kv compression' in text or 'prefix caching' in text:
        return 'KV Cache'
    if 'quantization' in text:
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
    if 'kernel' in text or 'flash attention' in text:
        return 'Inference Kernel'
    if 'lora' in text or 'adapter' in text:
        return 'LoRA/Adapter Serving'
    if 'streaming' in text:
        return 'Streaming Inference'
    if 'offloading' in text or 'heterogeneous' in text:
        return 'Offloading/Heterogeneous'
    return 'LLM Serving'

# ── MAIN ──
all_papers = []
seen = set()

queries = [
    "llm serving inference system",
    "speculative decoding llm",
    "llm inference optimization efficient",
    "kv cache llm serving memory",
    "llm inference batching scheduling throughput",
]

print("\n🔍 ArXiv Search")
for q in queries:
    print(f"  Query: {q}")
    results = search_arxiv(q, max_results=30)
    found = 0
    for p in results:
        k = p['title'].lower().strip()[:80]
        if k not in seen and is_llm_serving(p['title'], p['abstract_en']):
            seen.add(k)
            p['conference'] = 'arXiv'
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
            found += 1
    print(f"    → {found} relevant")
    time.sleep(3)

# S2 with longer delays
def search_s2(query, limit=10):
    encoded = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded}&limit={limit}&year=2025-2026&fields=title,abstract,url,externalIds,authors,publicationDate,venue&sort=relevance"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (PaperBot/4.0)')
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
                'authors': ', '.join([a.get('name', '') for a in (p.get('authors') or [])[:5]]),
                'venue': p.get('venue', '') or '',
                'source': 'semantic_scholar',
            })
        return results
    except Exception as e:
        print(f"  S2 error: {e}")
        return []

s2_queries = [
    "LLM inference serving system optimization",
    "speculative decoding LLM verification",
]

print("\n🔍 Semantic Scholar")
for q in s2_queries:
    print(f"  S2: {q}")
    results = search_s2(q, limit=15)
    found = 0
    for p in results:
        k = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
        if k not in seen and is_llm_serving(p['title'], p['abstract_en']):
            seen.add(k)
            p['conference'] = guess_conference(p.get('venue', ''))
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
            found += 1
    print(f"    → {found} relevant")
    time.sleep(12)

# Filter new
new_papers = []
for p in all_papers:
    tkey = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '').replace('$', '').replace('\\', '')
    arxiv_id = p.get('arxiv_id', '').strip()
    
    is_existing = False
    for l in [80, 60, 50, 40]:
        if tkey[:l] in existing_titles:
            is_existing = True
            break
    if is_existing:
        continue
    if arxiv_id and arxiv_id in existing_arxiv:
        continue
    if arxiv_id:
        base = arxiv_id.split('v')[0]
        try:
            mm = int(base[2:4]); yy = int(base[:2])
            if not ((yy == 25 and 1 <= mm <= 12) or (yy == 26 and 1 <= mm <= 4)):
                continue
        except:
            pass
    new_papers.append(p)

print(f"\n📈 Found {len(all_papers)} total, {len(new_papers)} new")
for p in new_papers:
    print(f"  🆕 [{p.get('arxiv_id','')}] {p['title'][:70]} | {p.get('conference','arXiv')} | {p.get('topic','LLM Serving')}")

# Add to DB
def get_conf_dir(conf, year):
    c = conf.lower().strip()
    if c in ('arxiv',):
        return os.path.join(CLAW_DIR, 'arXiv', str(year))
    m = {'osdi':'osdi','sosp':'sosp','nsdi':'nsdi','sigcomm':'sigcomm','sigmod':'sigmod',
         'atc':'atc','eurosys':'eurosys','dac':'dac','asplos':'asplos','sc':'sc',
         'neurips':'nips','iclr':'iclr','icml':'icml','acl':'acl','emnlp':'emnlp',
         'mlsys':'mlsys','isca':'isca','euromlsys':'euromlsys','ispass':'ispass'}
    return os.path.join(CLAW_DIR, m.get(c, c), str(year))

def sanitize_fn(title, arxiv_id=''):
    n = title.replace('\n',' ').strip()
    n = re.sub(r'[^\w\s-]', '', n)
    n = re.sub(r'\s+', '_', n)
    if arxiv_id:
        n = f"{arxiv_id.split('v')[0]}_{n[:60]}"
    else:
        n = n[:70]
    return n + '.md'

max_id = max(int(p['id']) for p in db['papers']) if db['papers'] else 0
added = 0
files = []

for p in new_papers:
    pub = p.get('published', '')
    try: year = int(pub[:4]) if pub else 2026
    except: year = 2026
    conf = p.get('conference', '') or 'arXiv'
    topic = p.get('topic', '') or guess_topic(p['title'], p.get('abstract_en',''))
    arxiv_id = p.get('arxiv_id', '').strip()
    
    d = get_conf_dir(conf, year)
    os.makedirs(d, exist_ok=True)
    fn = sanitize_fn(p['title'], arxiv_id)
    fp = os.path.join(d, fn)
    
    if not os.path.exists(fp):
        title = p['title'].strip()
        abs_en = p.get('abstract_en', '').strip()
        abs_cn = "[中文翻译待补充] " + abs_en[:200] + "..."
        md = f"""# {title}

## Metadata
- **Authors:** {p.get('authors','')}
- **Conference:** {conf} {year}
- **Topic:** {topic}
- **arXiv ID:** {arxiv_id}
- **Published:** {pub}
- **GitHub:** [待补充]

## 原文链接
- arXiv: https://arxiv.org/abs/{arxiv_id}
- PDF: https://arxiv.org/pdf/{arxiv_id}

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
        'authors': p.get('authors', '') or '',
        'arxiv_id': arxiv_id,
        'github_repo': '',
        'conference': conf,
        'year': str(year),
        'topic': topic,
        'abstract_en': abs_en,
        'abstract_cn': abs_cn,
    }
    db['papers'].append(entry)
    added += 1
    print(f"  ✅ [{added}] {title[:60]} | {conf} {year} | {topic}")

with open(DB_PATH, 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print(f"\n📊 Summary: added {added} papers, {len(files)} md files, total DB: {len(db['papers'])}")