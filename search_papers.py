#!/usr/bin/env python3
"""Search and collect LLM serving papers from arXiv API."""
import json, urllib.request, urllib.parse, time, sys, re, os

# Existing papers
db = json.load(open('database.json'))
existing_titles = set()
for p in db['papers']:
    existing_titles.add(p['title'].lower().strip()[:60])

# Keywords for LLM serving
keywords = [
    "llm serving",
    "speculative decoding",
    "llm inference",
    "kv cache",
    "llm inference optimization",
    "llm inference efficiency",
]

# Fetch arXiv API
def search_arxiv(query, max_results=50):
    url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        return parse_arxiv_xml(data)
    except Exception as e:
        print(f"Error fetching {query}: {e}")
        return []

def parse_arxiv_xml(xml_data):
    import xml.etree.ElementTree as ET
    ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
    root = ET.fromstring(xml_data)
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
        arxiv_id = entry.find('atom:id', ns).text.strip()
        published = entry.find('atom:published', ns).text.strip()[:10]
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
        link = arxiv_id
        papers.append({
            'title': title,
            'abstract_en': summary,
            'arxiv_id': arxiv_id.replace('http://arxiv.org/abs/', ''),
            'url': arxiv_id,
            'published': published,
            'authors': authors,
            'categories': categories,
        })
    return papers

# LLM serving relevance filter
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
]

NOT_SERVING_KEYWORDS = [
    'training', 'fine-tuning', 'pre-training', 'alignment', 'rlhf',
    'safety', 'guard model', 'phishing', 'question answering',
    'retrieval augmented', 'rag', 'prompt tuning', 'prompt engineering',
    'sentiment analysis', 'text classification', 'translation',
    'summarization', 'code generation', 'math reasoning',
    'medical', 'clinical', 'health', 'drug', 'bioinformatics',
    'drug discovery', 'protein', 'molecule', 'chemistry',
    'education', 'survey', 'benchmark evaluation only',
    'social media', 'election', 'politics', 'law',
    'music', 'art', 'creative', 'story', 'game',
    'robotics', 'autonomous driving', 'vision-language',
    'speech recognition', 'asr', 'tts',
    'image generation', 'video generation', 'diffusion model',
    'graph neural', 'gnn', 'knowledge graph',
    'embedding', 'representation learning',
    'data augmentation', 'dataset',
]

def is_llm_serving_paper(title, abstract):
    text = (title + ' ' + abstract).lower()
    # Must have LLM-related term
    has_llm = any(k in text for k in ['llm', 'large language model', 'language model inference', 'language model serving'])
    if not has_llm:
        return False
    # Must have serving-related term
    serving_score = sum(1 for k in SERVING_KEYWORDS if k in text)
    not_serving_score = sum(1 for k in NOT_SERVING_KEYWORDS if k in text)
    if serving_score >= 2 and serving_score > not_serving_score:
        return True
    if serving_score >= 1 and not_serving_score == 0 and has_llm:
        # Check more carefully - is it truly about inference/serving?
        strong_terms = ['serving system', 'inference system', 'inference framework', 'inference engine',
                       'speculative decoding', 'kv cache', 'prefill', 'decode phase',
                       'inference latency', 'inference throughput', 'inference speedup',
                       'inference acceleration', 'inference optimization', 'efficient inference',
                       'parallel decoding', 'batched inference']
        return any(k in text for k in strong_terms)
    return False

# Search
all_found = []
for kw in keywords:
    print(f"Searching: {kw}")
    results = search_arxiv(kw, max_results=30)
    for p in results:
        if is_llm_serving_paper(p['title'], p['abstract_en']):
            all_found.append(p)
    time.sleep(1)

# Deduplicate
seen = set()
unique = []
for p in all_found:
    key = p['title'].lower().strip()[:60]
    if key not in seen:
        seen.add(key)
        unique.append(p)

# Filter out existing
new_papers = []
for p in unique:
    key = p['title'].lower().strip()[:60]
    if key not in existing_titles:
        new_papers.append(p)

print(f"\nTotal found (after relevance filter): {len(unique)}")
print(f"New papers (not in DB): {len(new_papers)}")
print("\nNew paper titles:")
for p in new_papers:
    print(f"  [{p['arxiv_id']}] {p['title'][:80]} | {p['published']}")

# Save for processing
with open('/tmp/new_papers_search.json', 'w') as f:
    json.dump(new_papers, f, indent=2)