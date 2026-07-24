#!/usr/bin/env python3
"""Fetch missing abstracts - broader search."""
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

db = json.load(open('database.json'))
papers = db['papers']

missing = [p for p in papers if not p.get('abstract_en')]

def search_arxiv_broad(query):
    """Search arXiv with all: query, return list of (arxiv_id, title, abstract)."""
    q = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results=5&sortBy=relevance&sortOrder=descending"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clawbot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  Search failed: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    results = []
    for entry in root.findall("atom:entry", ns):
        t_el = entry.find("atom:title", ns)
        s_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        if t_el is None or id_el is None:
            continue
        atitle = (t_el.text or "").strip().replace("\n", " ")
        aid = id_el.text.strip().replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
        abstr = (s_el.text or "").strip().replace("\n", " ") if s_el is not None else ""
        results.append((aid, atitle, abstr))
    return results

for p in missing:
    title = p['title']
    print(f"\n[{p['id']}] {title[:70]}...")

    # Try searching key phrases
    keywords = title.replace(":", "").replace("-", " ").split()
    # Use first 4-5 significant words
    query = " ".join([w for w in keywords if len(w) > 2][:6])
    print(f"  Query: {query}")

    results = search_arxiv_broad(query)
    if not results:
        print("  No results")
        continue

    for aid, atitle, abstr in results:
        print(f"  -> {aid}: {atitle[:60]}...")
        # Simple similarity check
        t_norm = title.lower()
        a_norm = atitle.lower()
        # Check if key words match
        key_words = set(w for w in query.lower().split() if len(w) > 3)
        a_words = set(w for w in a_norm.split() if len(w) > 3)
        match = len(key_words & a_words) >= max(2, len(key_words) // 2)
        if match:
            print(f"  MATCH! Using {aid}")
            p['arxiv_id'] = aid
            p['abstract_en'] = abstr
            p['_needs_translate'] = True
            p['_needs_summary'] = True
            break
    time.sleep(3)

with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print("\nDone!")
