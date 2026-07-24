#!/usr/bin/env python3
"""Fetch missing abstracts for papers without arxiv_id by searching arXiv."""
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

db = json.load(open('database.json'))
papers = db['papers']

missing = [p for p in papers if not p.get('abstract_en')]
print(f"Found {len(missing)} papers missing abstract_en")

def search_arxiv_by_title(title):
    """Search arXiv by title, return (arxiv_id, abstract) or (None, None)."""
    q = urllib.parse.quote(title)
    url = f"http://export.arxiv.org/api/query?search_query=ti:{q}&max_results=3"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clawbot/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  Search failed: {e}")
        return None, None

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None, None

    for entry in root.findall("atom:entry", ns):
        t_el = entry.find("atom:title", ns)
        s_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        if t_el is None or id_el is None:
            continue
        atitle = (t_el.text or "").strip().replace("\n", " ")
        aid = id_el.text.strip().replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
        abstr = (s_el.text or "").strip().replace("\n", " ") if s_el is not None else ""
        # Check similarity
        t_norm = title.lower().replace("-", " ").replace(":", "")
        a_norm = atitle.lower().replace("-", " ").replace(":", "")
        if t_norm in a_norm or a_norm in t_norm:
            return aid, abstr
    return None, None

for p in missing:
    print(f"\n[{p['id']}] {p['title'][:70]}...")
    arxiv_id, abstract = search_arxiv_by_title(p['title'])
    if arxiv_id:
        print(f"  Found arxiv_id: {arxiv_id}")
        print(f"  Abstract: {abstract[:120]}...")
        p['arxiv_id'] = arxiv_id
        p['abstract_en'] = abstract
        # Mark for translation + summary
        p['_needs_translate'] = True
        p['_needs_summary'] = True
    else:
        print(f"  Not found on arXiv")
    time.sleep(3)

# Save
with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print(f"\nDone! Updated database.json")
