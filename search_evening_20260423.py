#!/usr/bin/env python3
"""
晚间论文搜集任务 - 2026-04-23
搜索关键词: llm serving, speculative decoding, llm inference
搜索来源: arxiv, semantic scholar
过滤: 只保留LLM serving相关论文
"""
import json, time, os, re, sys, urllib.request, urllib.parse, xml.etree.ElementTree as ET

DB_PATH = '/home/admin/claw_notes/database.json'
PAPERS_DIR = '/home/admin/claw_notes/papers'

# Load existing database
with open(DB_PATH) as f:
    db = json.load(f)

existing_titles = set()
for p in db['papers']:
    t = p.get('title', '').lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '').replace('{', '').replace('}', '')
    existing_titles.add(t[:80])
    existing_titles.add(t[:60])

print(f"📊 现有数据库: {len(db['papers'])} 篇论文, {len(existing_titles)} 个唯一标题")

# ==================== Serving keyword filters ====================
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
    'kernel optimization', 'cuda kernel', 'metal kernel',
    'lora adapter serving', 'adapter routing', 'adapter caching',
    'agentic inference', 'inference scaling', 'decode phase',
    'kv compression', 'kv share', 'cache compression',
    'prefix caching', 'token eviction',
    'fp4 inference', 'fp8 inference', 'low-bit inference',
    'tpu inference', 'pim inference', 'processing-in-memory',
    'speculative offloading', 'draft tree',
    'long-context inference', 'long-context serving',
    'moe inference', 'moe serving', 'moe routing',
    'power-efficient inference', 'energy-efficient inference',
    'multi-gpu inference', 'multi-die',
    'attention sink', 'context-aware scheduling',
    'request scheduling', 'batch scheduling',
    'memory footprint', 'gpu memory',
    'cost-efficient inference', 'inference cost',
    'token generation', 'generation latency', 'generation speedup',
    'weight loading', 'weight quantization inference',
    'flash attention', 'flash decoding',
    'speculative sampling', 'draft model',
    'context window', 'sequence length',
    'tensor parallel inference', 'pipeline parallel inference',
    'speculative decoding', 'medusa', 'eagle',
]

TRAINING_EXCLUSIONS = [
    'training system', 'distributed training', 'fine-tuning system',
    'pre-training', 'gradient accumulation', 'optimizer',
    'learning rate schedule', 'backpropagation',
    'data parallelism training', 'model parallel training',
    'pipeline parallel training', 'tensor parallel training',
    'checkpoint saving', 'training efficiency',
    'training acceleration', 'training framework',
    'training infrastructure', 'distributed training system',
]

def is_llm_serving_paper(title, abstract):
    text = (title + ' ' + (abstract or '')).lower()
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 
                                       'language model serving', 'transformer inference', 
                                       'transformer serving', 'generative model inference',
                                       'foundation model inference', 'foundation model serving',
                                       'autoregressive inference', 'autoregressive serving'])
    if not has_llm:
        return False
    serving_score = sum(1 for k in SERVING_KEYWORDS if k in text)
    training_score = sum(1 for k in TRAINING_EXCLUSIONS if k in text)
    if serving_score >= 1 and serving_score > training_score:
        return True
    if serving_score >= 2:
        return True
    return False

# ==================== Method 1: arXiv API search ====================
def search_arxiv(query, max_results=50):
    """Search arXiv via API"""
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (compatible; PaperCollector/1.0)')
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        root = ET.fromstring(data)
        papers = []
        for entry in root.findall('atom:entry', ns):
            try:
                title = entry.find('atom:title', ns).text.strip().replace('\n', ' ').replace('$', '').replace('{', '').replace('}', '')
                summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ').replace('$', '').replace('{', '').replace('}', '')
                arxiv_id_full = entry.find('atom:id', ns).text.strip()
                aid = arxiv_id_full.replace('http://arxiv.org/abs/', '').split('v')[0]
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
        print(f"  ❌ arXiv search error: {e}")
        return []

# ==================== Method 2: Semantic Scholar search ====================
def search_semantic_scholar(query, limit=50, year_from=2025):
    """Search Semantic Scholar API"""
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={encoded_query}&limit={limit}&year={year_from}-2026&fields=title,abstract,url,externalIds,authors,publicationDate,fieldsOfStudy,venue&sort=relevance"
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (compatible; PaperCollector/1.0)')
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        results = []
        for p in data.get('data', []):
            title = p.get('title', '').strip().replace('$', '').replace('{', '').replace('}', '')
            abstract = p.get('abstract', '') or ''
            ext_ids = p.get('externalIds', {})
            arxiv_id = ext_ids.get('ArXiv', '')
            pub_date = p.get('publicationDate', '')
            venue = p.get('venue', '') or ''
            authors_list = p.get('authors', [])
            author_names = [a.get('name', '') for a in authors_list] if isinstance(authors_list, list) else []
            fields = p.get('fieldsOfStudy', []) or []
            results.append({
                'title': title,
                'abstract_en': abstract,
                'arxiv_id': arxiv_id,
                'published': pub_date,
                'authors': author_names,
                'venue': venue,
                'categories': fields,
                'source': 'semantic_scholar',
            })
        return results
    except Exception as e:
        print(f"  ❌ Semantic Scholar error: {e}")
        return []

# ==================== Search queries ====================
arxiv_queries = [
    "llm serving",
    "speculative decoding",
    "llm inference optimization",
    "kv cache llm",
    "continuous batching llm",
    "prefill decode disaggregation",
    "efficient llm inference",
    "llm inference latency throughput",
]

s2_queries = [
    "LLM serving system",
    "speculative decoding LLM",
    "KV cache LLM inference",
    "LLM inference optimization",
    "LLM inference efficiency acceleration",
    "distributed LLM inference serving",
    "LLM inference latency throughput",
    "paged attention continuous batching",
    "prefill decode disaggregation",
    "LLM inference system",
    "speculative execution language model",
    "efficient transformer inference",
    "GPU memory LLM serving",
    "MoE inference serving",
    "long context LLM inference",
]

# ==================== Conference mapping ====================
CONFERENCE_MAP = {
    'osdi': 'OSDI', 'sosp': 'SOSP', 'nsdi': 'NSDI', 'sigcomm': 'SIGCOMM',
    'sigmod': 'SIGMOD', 'atc': 'ATC', 'eurosys': 'EuroSys', 'dac': 'DAC',
    'asplos': 'ASPLOS', 'sc': 'SC', 'nips': 'NeurIPS', 'neurips': 'NeurIPS',
    'iclr': 'ICLR', 'icml': 'ICML', 'acl': 'ACL', 'emnlp': 'EMNLP',
}

def detect_conference(venue, categories, title, abstract):
    """Try to detect which conference this paper belongs to"""
    text = (venue + ' ' + ' '.join(categories) + ' ' + title + ' ' + (abstract or '')).lower()
    for key, conf_name in CONFERENCE_MAP.items():
        if key in text:
            return conf_name
    # Default based on source
    return 'arxiv'

# ==================== Run searches ====================
print("\n🔍 ====== 搜索阶段 ======\n")
all_found = []
seen_titles = set()

# arXiv searches
print("📚 搜索 arXiv...")
for q in arxiv_queries:
    print(f"  搜索: {q}")
    results = search_arxiv(q, max_results=30)
    count_before = len(all_found)
    for p in results:
        key = p['title'].lower().strip().replace('\n', ' ')[:80]
        if key not in seen_titles and is_llm_serving_paper(p['title'], p['abstract_en']):
            seen_titles.add(key)
            all_found.append(p)
    count_after = len(all_found)
    print(f"    发现 {count_after - count_before} 篇相关论文")
    time.sleep(3)  # Rate limit

# Semantic Scholar searches
print("\n📚 搜索 Semantic Scholar...")
for q in s2_queries:
    print(f"  搜索: {q}")
    results = search_semantic_scholar(q, limit=40, year_from=2025)
    count_before = len(all_found)
    for p in results:
        key = p['title'].lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '')[:80]
        if key not in seen_titles and is_llm_serving_paper(p['title'], p['abstract_en']):
            seen_titles.add(key)
            all_found.append(p)
    count_after = len(all_found)
    print(f"    发现 {count_after - count_before} 篇相关论文")
    time.sleep(1.5)

print(f"\n📊 搜索总计: {len(all_found)} 篇LLM serving相关论文")

# ==================== Deduplicate against existing DB ====================
new_papers = []
for p in all_found:
    key = p['title'].lower().strip().replace('\n', ' ').replace('$', '').replace('\\', '').replace('{', '').replace('}', '')[:80]
    short_key = key[:60]
    if key not in existing_titles and short_key not in existing_titles:
        new_papers.append(p)

duplicates = len(all_found) - len(new_papers)
print(f"  已在数据库中: {duplicates} 篇")
print(f"  新论文: {len(new_papers)} 篇")

print("\n📋 新论文列表:")
for i, p in enumerate(new_papers[:50], 1):
    aid = p.get('arxiv_id', '')
    src = p.get('source', '')
    venue = p.get('venue', '')
    print(f"  {i}. [{aid or src}] {p['title'][:70]} | {p.get('published', '')} | {venue}")

# ==================== Add papers to database ====================
print("\n📝 ====== 写入阶段 ======\n")
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
    
    # Determine year
    year = 2026
    if published:
        try:
            year = int(published[:4])
        except:
            year = 2026
    
    # Determine conference
    conf = detect_conference(venue, categories, title, abstract)
    if not conf or conf == 'arxiv':
        if arxiv_id:
            conf = 'arxiv'
        else:
            conf = 'semantic_scholar'
    
    # Find GitHub in abstract
    github = ""
    gh_matches = re.findall(r'github\.com/([^\s\)\.\,]+)', abstract)
    if gh_matches:
        github = gh_matches[0]
    
    # Create directory structure: papers/conf/year/
    dir_path = os.path.join(PAPERS_DIR, conf, str(year))
    os.makedirs(dir_path, exist_ok=True)
    
    safe_title = re.sub(r'[^\w\s-]', '', title[:60]).strip().replace(' ', '_')
    if not safe_title:
        safe_title = f"paper_{arxiv_id or int(time.time())}"
    filename = f"{safe_title}.md"
    filepath = os.path.join(dir_path, filename)
    
    if os.path.exists(filepath):
        print(f"  ⏭️ 已存在: {title[:60]}")
        continue
    
    # URLs
    url_link = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ''
    pdf_link = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ''
    
    # Write markdown file
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

{f'https://github.com/{github}' if github else '暂无 GitHub 仓库'}

---
*注: 此文件由晚间自动化论文搜集系统生成于 2026-04-23，部分内容待完善。*
"""
    
    with open(filepath, 'w') as f:
        f.write(md_content)
    
    # Add to database JSON
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
    print(f"  ✅ [{arxiv_id or conf}] {title[:70]}")

# Save database
with open(DB_PATH, 'w') as f:
    json.dump(db, f, indent=2)

print(f"\n✅ ====== 结果汇总 ======")
print(f"  新增论文: {added} 篇")
print(f"  数据库总计: {len(db['papers'])} 篇")
print(f"  搜索发现总计: {len(all_found)} 篇")
print(f"  重复已存在: {duplicates} 篇")