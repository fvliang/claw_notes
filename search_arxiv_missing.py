#!/usr/bin/env python3
"""Search arXiv for papers missing abstracts."""

import json
import time
import requests
from pathlib import Path

DB_PATH = Path("database.json")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def search_arxiv(title):
    """Search arXiv by title."""
    try:
        query = title.replace(' ', '+').replace(':', '')[:100]
        url = f"http://export.arxiv.org/api/query?search_query=ti:{query}&start=0&max_results=3"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        # Parse simple XML
        text = resp.text
        entries = text.split('<entry>')[1:]
        
        results = []
        for entry in entries:
            # Get title
            title_start = entry.find('<title>') + 7
            title_end = entry.find('</title>')
            entry_title = entry[title_start:title_end].strip()
            
            # Get abstract
            abs_start = entry.find('<summary>') + 9
            abs_end = entry.find('</summary>')
            abstract = entry[abs_start:abs_end].strip()
            
            # Get arxiv id
            id_start = entry.find('<id>http://arxiv.org/abs/') + 25
            id_end = entry.find('</id>', id_start)
            arxiv_id = entry[id_start:id_end].strip()
            
            results.append({
                'title': entry_title,
                'abstract': abstract,
                'arxiv_id': arxiv_id
            })
        
        return results
    except Exception as e:
        print(f"  Error: {e}")
        return []

def main():
    db = load_db()
    
    missing_papers = []
    for p in db["papers"]:
        no_en = not p.get("abstract_en") or len(p.get("abstract_en", "")) <= 50
        no_arxiv = not p.get("arxiv_id")
        if no_en and no_arxiv:
            missing_papers.append(p)
    
    print(f"Found {len(missing_papers)} papers to search")
    
    for p in missing_papers:
        title = p.get("title", "")
        print(f"\nSearching: {title[:60]}...")
        
        results = search_arxiv(title)
        if results:
            best = results[0]
            print(f"  Found: {best['title'][:60]}...")
            print(f"  arxiv_id: {best['arxiv_id']}")
            
            p["arxiv_id"] = best['arxiv_id']
            p["abstract_en"] = best['abstract']
            print(f"  ✓ Added abstract ({len(best['abstract'])} chars)")
        else:
            print(f"  ✗ Not found")
        
        time.sleep(5)
    
    save_db(db)
    print("\nDone!")

if __name__ == "__main__":
    main()
