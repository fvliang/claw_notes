#!/usr/bin/env python3
"""Fetch missing abstracts via Semantic Scholar API."""
import json
import urllib.request
import time

db = json.load(open('database.json'))
papers = db['papers']

# Revert the bad match for ID 548 first
for p in papers:
    if p['id'] == 548 and p.get('arxiv_id') == '2407.11310v2':
        print(f"Reverting bad match for ID 548")
        p['arxiv_id'] = None
        p['abstract_en'] = None

missing = [p for p in papers if not p.get('abstract_en')]
print(f"Fetching {len(missing)} missing abstracts from Semantic Scholar...")

def fetch_ss(title):
    """Search Semantic Scholar by title."""
    q = urllib.parse.quote(title)
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={q}&fields=title,abstract,externalIds&limit=3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clawbot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as e:
        print(f"  SS error: {e}")
        return None

    for paper in data.get('data', []):
        ptitle = paper.get('title', '')
        abstract = paper.get('abstract', '')
        if not abstract:
            continue
        # Check title similarity
        t_norm = title.lower().replace('-', ' ')
        p_norm = ptitle.lower().replace('-', ' ')
        if t_norm in p_norm or p_norm in t_norm or len(set(t_norm.split()) & set(p_norm.split())) >= 3:
            return abstract
    return None

for p in missing:
    print(f"\n[{p['id']}] {p['title'][:60]}...")
    abstract = fetch_ss(p['title'])
    if abstract:
        print(f"  Found abstract ({len(abstract)} chars)")
        p['abstract_en'] = abstract
        p['_needs_translate'] = True
        p['_needs_summary'] = True
    else:
        print("  Not found")
    time.sleep(1)

with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print("\nDone!")
