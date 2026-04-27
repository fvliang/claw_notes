#!/usr/bin/env python3
"""LLM serving paper search - 2026-04-27 evening v5 (correct arXiv syntax, delayed S2)"""
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

print(f"📊 DB: {len(db['papers'])} papers")

def is_strictly_llm_serving(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    llm_terms = ['llm', 'large language model', 'language model serving',
                 'language model inference', 'transformer inference',
                 'foundation model inference', 'autoregressive inference',
                 'llm system', 'generative inference', 'gpt inference']
    has_llm = any(k in text for k in llm_terms)
    if not has_llm:
        return False
    
    serving_terms = ['serving', 'inference system', 'inference engine',
                     'inference framework', 'inference optimization',
                     'inference acceleration', 'inference latency',
                     'inference throughput', 'efficient inference',
                     'speculative decoding', 'speculative execution',
                     'draft model', 'draft token', 'verification',
                     'kv cache', 'kv-cache', 'paged attention',
                     'continuous batching', 'chunked prefill',
                     'prefill', 'decode', 'disaggregation',
                     'batching', 'scheduling', 'throughput',
                     'ttft', 'time to first token',
                     'memory management', 'gpu memory',
                     'offloading', 'acceleration',
                     'vllm', 'tensor parallel', 'pipeline parallel',
                     'moe inference', 'moe serving', 'expert routing',
                     'load balancing', 'request routing',
                     'prefix caching', 'token eviction',
                     'kv compression', 'flash attention',
                     'lora serving', 'adapter serving',
                     'early exit', 'layer skipping',
                     'medusa', 'eagle',
                     'acceptance rate', 'verification step',
                     'inference cost', 'serving cost',
                     'attention sink', 'streaming llm',
                     'model serving', 'model deployment',
                     'inference kernel', 'attention kernel',
                     'generation latency', 'generation throughput']
    
    not_serving = ['training', 'fine-tuning', 'pre-training', 'gradient',
                   'alignment', 'rlhf', 'dpo', 'retrieval', 'retriever',
                   'code verification', 'math evaluation', 'medical',
                   'drug', 'protein', 'robotics', 'image generation',
                   'video generation', 'diffusion', 'speech',
                   'sentiment', 'social media', 'election',
                   'game playing', 'reward model', 'guard model',
                   'education', 'hackathon', 'atomistic',
                   'dense retrieval', 'natural language to verified',
                   'math reasoning', 'judge framework']
    
    ss = sum(1 for k in serving_terms if k in text)
    ns = sum(1 for k in not_serving if k in text)
    if ns >= 2:
        return False
    if ss >= 2:
        return True
    if ss >= 1 and ns == 0:
        return True
    return False

def search_arxiv(query, max_results=30):
    # Use 'all:' for full-text search, which arXiv API supports
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=45) as resp:
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

def search_s2(query, limit=15):
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

CONFERENCE_MAP = {
    'osdi': 'OSDI', 'sosp': 'SOSP', 'nsdi': 'NSDI', 'sigcomm': 'SIGCOMM', 'sigmod': 'SIGMOD',
    'atc': 'ATC', 'eurosys': 'EuroSys', 'dac': 'DAC', 'asplos': 'ASPLOS', 'sc': 'SC',
    'neurips': 'NeurIPS', 'iclr': 'ICLR', 'icml': 'ICML', 'acl': 'ACL', 'emnlp': 'EMNLP',
    'mlsys': 'MLSys', 'isca': 'ISCA', 'euromlsys': 'EuroMLSys', 'ispass': 'ISPASS',
}

def guess_conference(venue):
    if venue:
        for k, name in CONFERENCE_MAP.items():
            if k in venue.lower():
                return name
    return 'arXiv'

def guess_topic(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    if 'speculative' in text and ('decoding' in text or 'draft' in text):
        return 'Speculative Decoding'
    if 'kv cache' in text or 'kv-cache' in text:
        return 'KV Cache'
    if 'quantization' in text:
        return 'Quantization'
    if 'moe' in text or 'mixture of expert' in text:
        return 'MoE Inference'
    if 'disaggregation' in text or 'prefill' in text:
        return 'Prefill/Disaggregation'
    if 'scheduling' in text or 'routing' in text or 'load balancing' in text:
        return 'Inference Scheduling'
    if 'early exit' in text:
        return 'Early Exit'
    if 'kernel' in text or 'flash attention' in text:
        return 'Inference Kernel'
    if 'lora' in text or 'adapter' in text:
        return 'LoRA/Adapter Serving'
    if 'streaming' in text:
        return 'Streaming Inference'
    if 'offloading' in text or 'heterogeneous' in text:
        return 'Offloading/Heterogeneous'
    return 'LLM Serving'

# ── SEARCH ──
all_papers = []
seen = set()

arxiv_queries = [
    # Combined terms for LLM serving/inference
    "llm serving inference system",
    "speculative decoding llm draft verification",
    "kv cache llm inference serving",
    "llm inference batching scheduling throughput latency",
    "llm inference disaggregation prefill decode",
    "efficient llm inference acceleration",
    "continuous batching paged attention vllm",
    "llm serving moe expert routing",
    "llm inference memory management gpu",
    "llm inference quantization serving deployment",
]

print("\n🔍 ArXiv Search")
for q in arxiv_queries:
    print(f"  Query: {q}")
    results = search_arxiv(q, max_results=30)
    found = 0
    for p in results:
        k = p['title'].lower().strip()[:80]
        if k not in seen and is_strictly_llm_serving(p['title'], p['abstract_en']):
            seen.add(k)
            p['conference'] = 'arXiv'
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
            found += 1
    print(f"    → {found} strictly relevant")
    time.sleep(5)

# S2 with 30s delays
print("\n🔍 Semantic Scholar (30s delays)")
s2_queries = [
    "LLM serving inference system",
    "speculative decoding LLM",
]
for i, q in enumerate(s2_queries):
    print(f"  S2 [{i+1}]: {q}")
    results = search_s2(q, limit=15)
    found = 0
    for p in results:
        k = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '')[:80]
        if k not in seen and is_strictly_llm_serving(p['title'], p['abstract_en']):
            seen.add(k)
            p['conference'] = guess_conference(p.get('venue', ''))
            p['topic'] = guess_topic(p['title'], p['abstract_en'])
            all_papers.append(p)
            found += 1
    print(f"    → {found} relevant")
    if i < len(s2_queries) - 1:
        print("  Waiting 30s...")
        time.sleep(30)

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
    print(f"  🆕 [{p.get('arxiv_id','')}] {p['title'][:80]} | {p.get('conference','arXiv')} | {p.get('topic','')}")

# Add to DB
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
        n = f"{aid.split('v')[0]}_{n[:60]}"
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
    aid = p.get('arxiv_id', '').strip()
    title = p['title'].strip()
    abs_en = p.get('abstract_en', '').strip()
    abs_cn = "[中文翻译待补充] " + abs_en[:200] + "..."
    
    d = get_conf_dir(conf, year)
    os.makedirs(d, exist_ok=True)
    fn = sanitize_fn(title, aid)
    fp = os.path.join(d, fn)
    
    if not os.path.exists(fp):
        md = f"""# {title}

## Metadata
- **Authors:** {p.get('authors','')}
- **Conference:** {conf} {year}
- **Topic:** {topic}
- **arXiv ID:** {aid}
- **Published:** {pub}

## 原文链接
- arXiv: https://arxiv.org/abs/{aid}
- PDF: https://arxiv.org/pdf/{aid}

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
    print(f"  ✅ [{added}] {title[:60]} | {conf} {year} | {topic}")

with open(DB_PATH, 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print(f"\n📊 Summary: added {added} papers, {len(files)} md files, total DB: {len(db['papers'])}")