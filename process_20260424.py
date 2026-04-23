#!/usr/bin/env python3
"""Retry ArXiv search with delays + Process results and add to DB - 2026-04-24"""
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
    if p.get('arxiv_id'):
        existing_arxiv.add(p['arxiv_id'].strip())

print(f"📊 DB: {len(db['papers'])} papers, {len(existing_arxiv)} arxiv IDs")

# ── ArXiv XML parser ──
def search_arxiv(query, max_results=50):
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
]

def is_llm_serving(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference',
                                       'language model serving', 'transformer inference',
                                       'foundation model inference', 'autoregressive inference',
                                       'gpt inference', 'llm system'])
    if not has_llm:
        return False
    ss = sum(1 for k in SERVING_KEYWORDS if k in text)
    ns = sum(1 for k in NOT_SERVING_KEYWORDS if k in text)
    return (ss >= 2 and ss > ns) or (ss >= 1 and ns == 0 and has_llm)

def guess_topic(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    if 'speculative' in text and ('decoding' in text or 'sampling' in text or 'execution' in text):
        return 'Speculative Decoding'
    if 'kv cache' in text or 'kv-cache' in text or 'kv compression' in text or 'prefix caching' in text:
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

# ── Retry ArXiv search with longer delays ──
print("\n🔍 Retrying ArXiv searches with longer delays...")
arxiv_queries = [
    "llm serving",
    "speculative decoding",
    "llm inference optimization",
    "kv cache llm",
    "llm inference latency",
    "llm inference throughput",
]

arxiv_results = []
for q in arxiv_queries:
    print(f"  Searching: {q}")
    results = search_arxiv(q, max_results=50)
    print(f"  Got {len(results)} entries")
    for p in results:
        if is_llm_serving(p['title'], p['abstract_en']):
            arxiv_results.append(p)
    time.sleep(5)  # 5s delay to avoid 429

# Deduplicate arxiv results
seen_arxiv = set()
unique_arxiv = []
for p in arxiv_results:
    aid = p.get('arxiv_id', '')
    if aid not in seen_arxiv:
        seen_arxiv.add(aid)
        unique_arxiv.append(p)

print(f"\n  ArXiv serving papers found: {len(unique_arxiv)}")

# ── Also load S2 results from previous search ──
prev_data = json.load(open('/tmp/search_results_20260424.json'))
s2_papers = prev_data['new_papers']
gh_repos = prev_data['new_repos']

# Combine all paper sources
all_candidates = unique_arxiv + s2_papers

# Deduplicate globally (by arxiv ID or title)
seen = set()
all_unique = []
for p in all_candidates:
    aid = p.get('arxiv_id', '').strip()
    key = aid if aid else p['title'].lower().strip()[:80]
    if key and key not in seen:
        seen.add(key)
        all_unique.append(p)

# Filter: remove suspicious/future arxiv IDs (month > current month 04)
# arxiv ID format: YYMM.NNNNN - for 2026, valid months are 2601-2604
# Also allow 2501-2504 (2025 recent)
def is_valid_arxiv_id(aid):
    if not aid:
        return True  # No arxiv ID - might be S2-only, keep for review
    aid = aid.strip()
    # Remove version suffix
    base = aid.split('v')[0]
    # Check format
    if len(base) != 9 and len(base) != 10:
        return False
    # Parse YYMM
    try:
        yymm = base[:4]
        mm = int(yymm[2:4])
        yy = int(yymm[:2])
        # Valid: 2025 (01-12) or 2026 (01-04)
        if yy == 25 and 1 <= mm <= 12:
            return True
        if yy == 26 and 1 <= mm <= 4:
            return True
        return False
    except:
        return False

valid_candidates = []
for p in all_unique:
    aid = p.get('arxiv_id', '').strip()
    # Keep papers with valid arxiv IDs or no arxiv ID but with good abstract
    if is_valid_arxiv_id(aid):
        valid_candidates.append(p)
    elif not aid and p.get('abstract_en') and len(p.get('abstract_en', '')) > 50:
        valid_candidates.append(p)
    else:
        print(f"  ⚠️ Skipping suspicious: {aid} - {p['title'][:50]}")

# Filter out existing papers
new_papers = []
for p in valid_candidates:
    aid = p.get('arxiv_id', '').strip()
    tkey = p['title'].lower().strip().replace('\n', ' ').replace('{', '').replace('}', '').replace('$', '').replace('\\', '')[:80]
    if aid and aid in existing_arxiv:
        continue
    if tkey in existing_titles:
        continue
    new_papers.append(p)

print(f"\n📈 Final results:")
print(f"  Total candidates (after dedup): {len(all_unique)}")
print(f"  Valid candidates (after ID check): {len(valid_candidates)}")
print(f"  New papers (not in DB): {len(new_papers)}")

# ── Fetch full abstract + intro for each new paper ──
def fetch_arxiv_page(arxiv_id):
    """Fetch abstract and first section from arXiv HTML page."""
    if not arxiv_id:
        return None
    url = f"https://arxiv.org/abs/{arxiv_id}"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')
        # Extract abstract
        abs_match = re.search(r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)</blockquote>', html, re.DOTALL)
        abstract = abs_match.group(1).strip() if abs_match else ''
        abstract = re.sub(r'<[^>]+>', '', abstract).strip()
        return {'abstract_en': abstract, 'url': url}
    except:
        return None

# ── Chinese translation stub ──
def translate_abstract_to_cn(abstract_en):
    """Simple keyword-based Chinese translation for common terms."""
    cn = abstract_en
    # This is a stub - actual translation would need an API
    # For now, we'll just note "[中文翻译待补充]"
    return "[中文翻译待补充] " + abstract_en[:200] + "..."

# ── Write markdown notes for each paper ──
def get_conference_dir(conference, year):
    """Determine directory path based on conference."""
    conf_lower = conference.lower().strip()
    if conf_lower == 'arxiv' or conf_lower == 'arXiv':
        return os.path.join(CLAW_DIR, 'arXiv', str(year))
    # Map conference names to directory names
    dir_map = {
        'osdi': 'osdi', 'sosp': 'sosp', 'nsdi': 'nsdi',
        'sigcomm': 'sigcomm', 'sigmod': 'sigmod',
        'atc': 'atc', 'eurosys': 'eurosys', 'dac': 'dac',
        'asplos': 'asplos', 'sc': 'sc',
        'neurips': 'nips', 'nips': 'nips',
        'iclr': 'iclr', 'icml': 'icml',
        'acl': 'acl', 'emnlp': 'emnlp',
    }
    dirname = dir_map.get(conf_lower, conf_lower)
    return os.path.join(CLAW_DIR, dirname, str(year))

def sanitize_filename(title, arxiv_id=''):
    """Create a safe filename from paper title."""
    # Remove special chars, limit length
    name = title.replace('\n', ' ').strip()
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name)
    if arxiv_id:
        # Add arxiv ID prefix
        base = arxiv_id.split('v')[0]
        name = f"{base}_{name[:60]}"
    else:
        name = name[:70]
    return name + '.md'

def write_paper_note(paper, year, conference, topic, github_repo=''):
    """Write a markdown note file for a paper."""
    dir_path = get_conference_dir(conference, year)
    os.makedirs(dir_path, exist_ok=True)
    
    fname = sanitize_filename(paper['title'], paper.get('arxiv_id', ''))
    filepath = os.path.join(dir_path, fname)
    
    # Avoid overwriting existing notes
    if os.path.exists(filepath):
        return filepath
    
    title = paper.get('title', '').strip()
    abstract_en = paper.get('abstract_en', '').strip()
    abstract_cn = translate_abstract_to_cn(abstract_en)
    arxiv_id = paper.get('arxiv_id', '').strip()
    authors = paper.get('authors', '')
    published = paper.get('published', '')
    
    md_content = f"""# {title}

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
*Auto-collected on 2026-04-24*
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    return filepath

# ── Process and add papers to DB ──
def safe_int_id(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0

max_id = max(safe_int_id(pp.get('id', 0)) for pp in db['papers']) if db['papers'] else 0

added_count = 0
added_files = []

for p in new_papers:
    # Determine year and conference
    published = p.get('published', '')
    if published:
        try:
            year = int(published[:4])
        except:
            year = 2026
    else:
        year = 2026
    
    conference = p.get('conference', '') or p.get('venue', '') or 'arXiv'
    if not conference or conference.lower() in ['arxiv', 'arxiv']:
        conference = 'arXiv'
    
    topic = p.get('topic', '') or guess_topic(p['title'], p.get('abstract_en', ''))
    
    # Try to find matching GitHub repo
    github_repo = ''
    title_lower = p['title'].lower()
    for r in gh_repos:
        repo_desc = r.get('description', '').lower() if r.get('description') else ''
        repo_name = r.get('name', '').lower()
        # Match by name similarity
        if any(word in repo_desc for word in title_lower.split()[:3]) and r.get('stars', 0) >= 50:
            github_repo = r.get('full_name', '')
            break
    
    # Write markdown note
    filepath = write_paper_note(p, year, conference, topic, github_repo)
    if filepath:
        added_files.append(filepath)
    
    # Add to database
    paper_entry = {
        'id': max_id + 1 + added_count,
        'title': p['title'].strip(),
        'authors': p.get('authors', '') or '',
        'arxiv_id': p.get('arxiv_id', '').strip(),
        'github_repo': github_repo,
        'conference': conference,
        'year': str(year),
        'topic': topic,
        'abstract_en': p.get('abstract_en', '').strip(),
        'abstract_cn': translate_abstract_to_cn(p.get('abstract_en', '').strip()),
    }
    db['papers'].append(paper_entry)
    added_count += 1
    print(f"  ✅ [{added_count}] {p['title'][:60]} | {conference} {year} | {topic}")

# ── Also add top GitHub repos as entries ──
gh_added = 0
for r in gh_repos[:20]:  # Top 20 repos by stars
    if r.get('stars', 0) < 100:
        continue
    # Check if already in DB
    full_name = r.get('full_name', '')
    already = any(full_name in (p.get('github_repo', '') or '') for p in db['papers'])
    if already:
        continue
    
    desc = r.get('description', '') or ''
    topic = guess_topic(r.get('name', ''), desc)
    
    # Create a DB entry for the repo
    paper_entry = {
        'id': max_id + 1 + added_count + gh_added,
        'title': f"[GitHub] {r.get('name', '')}: {desc[:100]}",
        'authors': r.get('full_name', '').split('/')[0] if '/' in r.get('full_name', '') else '',
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
    
    # Write note in github directory
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
*Auto-collected on 2026-04-24*
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
print(f"\n📁 Files created:")
for f in added_files[:20]:
    print(f"  {f}")
if len(added_files) > 20:
    print(f"  ... and {len(added_files) - 20} more")