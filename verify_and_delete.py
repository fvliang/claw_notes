#!/usr/bin/env python3
"""Verify no-link papers by searching arXiv. Delete if not found."""

import json
import re
import time
import urllib.request
import urllib.parse

db = json.load(open('database.json'))
papers = db['papers']

no_link = [p for p in papers if not any(p.get(k) for k in ['arxiv_id','github_repo','url','arxiv_url','pdf_url'])]
print(f'Papers with no links: {len(no_link)}')

found_count = 0
not_found = []

def search_arxiv(title):
    """Search arXiv by title, return arxiv_id if found."""
    # Truncate long titles
    short_title = title[:80]
    query = urllib.parse.quote(short_title)
    url = f'https://arxiv.org/search/?query={query}&searchtype=all&abstracts=hide&order=-announced_date_first&size=5'
    try:
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8')

        # Look for arxiv abs links in search results
        # Pattern: <a href="/abs/XXXX.XXXXX">Title</a>
        pattern = r'<a href="/abs/([0-9]{4}\.[0-9]{4,5}[v0-9]*)"[^>]*>(.*?)</a>'
        matches = re.findall(pattern, html, re.IGNORECASE)

        for arxiv_id, result_title in matches[:3]:
            result_title_clean = re.sub(r'<[^>]+>', '', result_title).strip()
            # Check title similarity
            t1 = title.lower().replace(':', '').replace('-', ' ')
            t2 = result_title_clean.lower().replace(':', '').replace('-', ' ')
            if t1 == t2 or t1 in t2 or t2 in t1:
                return arxiv_id
        return None
    except Exception as e:
        print(f'  ERROR: {e}')
        return None

for i, p in enumerate(no_link):
    title = p['title']
    print(f'[{i+1}/{len(no_link)}] {title[:70]}...')

    arxiv_id = search_arxiv(title)
    if arxiv_id:
        print(f'  -> FOUND: {arxiv_id}')
        p['arxiv_id'] = arxiv_id
        p['url'] = f'https://arxiv.org/abs/{arxiv_id}'
        p['pdf_url'] = f'https://arxiv.org/pdf/{arxiv_id}.pdf'
        found_count += 1
    else:
        print(f'  -> NOT FOUND')
        not_found.append(p)

    time.sleep(1)  # Be nice to arXiv

print(f'\n=== Results ===')
print(f'Found on arXiv: {found_count}/{len(no_link)}')
print(f'Not found: {len(not_found)}')

if not_found:
    print(f'\nDeleting {len(not_found)} papers not found on arXiv...')
    ids_to_delete = {p['id'] for p in not_found}
    db['papers'] = [p for p in papers if p['id'] not in ids_to_delete]
    print(f'Remaining papers: {len(db["papers"])}')

# Save
with open('database.json', 'w', encoding='utf-8') as f:
    json.dump(db, f, ensure_ascii=False, indent=2)
print('Saved database.json')
