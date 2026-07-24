#!/usr/bin/env python3
"""Try to find abstracts via web search for remaining 6 papers."""
import json
import urllib.request
import re
import time

db = json.load(open('database.json'))
papers = db['papers']

missing = [p for p in papers if not p.get('abstract_en')]
print(f"Trying web search for {len(missing)} papers...")

def web_search(title):
    """Simple Bing web search to find abstract."""
    q = urllib.parse.quote(title + " abstract")
    url = f"https://www.bing.com/search?q={q}"
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return None
    
    # Try to extract abstract from search result snippets
    # Look for text between certain patterns
    snippets = re.findall(r'<p[^>]*>([^<]{100,800})</p>', html)
    for s in snippets:
        s_clean = re.sub(r'<[^>]+>', '', s)
        if len(s_clean) > 100 and 'abstract' not in s_clean.lower()[:20]:
            return s_clean
    return None

for p in missing:
    print(f"\n[{p['id']}] {p['title'][:50]}...")
    abstract = web_search(p['title'])
    if abstract:
        print(f"  Found ({len(abstract)} chars)")
        print(f"  {abstract[:100]}...")
        p['abstract_en'] = abstract
    else:
        print("  Not found")
    time.sleep(2)

with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print("\nDone!")
