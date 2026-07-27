#!/usr/bin/env python3
import json
import time
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse

DB_PATH = "database.json"

def search_arxiv(title):
    """Search arXiv by title and return the first matching ID."""
    query = urllib.parse.quote(title[:200])
    url = f"http://export.arxiv.org/api/query?search_query=ti:{query}&start=0&max_results=3"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'claw-notes/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode('utf-8')
        root = ET.fromstring(data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        for entry in entries:
            entry_id = entry.find('atom:id', ns)
            entry_title = entry.find('atom:title', ns)
            if entry_id is not None and entry_title is not None:
                arxiv_id = entry_id.text.split('/abs/')[-1]
                t = entry_title.text.strip().lower().replace('\n', ' ')
                orig = title.strip().lower().replace('\n', ' ')
                if orig in t or t in orig or len(set(orig.split()) & set(t.split())) >= 3:
                    return arxiv_id
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

with open(DB_PATH, 'r') as f:
    db = json.load(f)

papers = db['papers']
need_fill = [p for p in papers if p.get('conference') == 'arXiv' and not p.get('arxiv_id')]
print(f"Need to fill arxiv_id for {len(need_fill)} papers")

filled = 0
for p in need_fill:
    print(f"[{p['id']}] Searching: {p['title'][:60]}...")
    arxiv_id = search_arxiv(p['title'])
    if arxiv_id:
        p['arxiv_id'] = arxiv_id
        print(f"  -> {arxiv_id}")
        filled += 1
    else:
        print(f"  -> Not found")
    time.sleep(3)

with open(DB_PATH, 'w') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print(f"\nDone! Filled {filled}/{len(need_fill)}")
