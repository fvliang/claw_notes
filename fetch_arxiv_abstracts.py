#!/usr/bin/env python3
"""Fetch missing abstracts from arXiv API."""

import json
import time
from pathlib import Path

import requests

DB_PATH = Path("database.json")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def fetch_arxiv_abstract(arxiv_id):
    """Fetch abstract from arXiv API."""
    if not arxiv_id:
        return None
    
    try:
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        # Parse XML response
        content = resp.text
        
        # Extract abstract
        start = content.find('<summary>')
        end = content.find('</summary>')
        if start > 0 and end > start:
            abstract = content[start+9:end].strip()
            # Remove XML entities
            abstract = abstract.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            return abstract
        
        return None
    except Exception as e:
        print(f"  Error fetching {arxiv_id}: {e}")
        return None


def fetch_missing_abstracts():
    """Fetch abstracts for papers missing them."""
    db = load_db()
    papers = db["papers"]
    
    missing = [p for p in papers if not p.get("abstract_en") or len(p.get("abstract_en", "")) < 50]
    has_arxiv = [p for p in missing if p.get("arxiv_id")]
    
    print(f"Papers missing abstract: {len(missing)}")
    print(f"With arxiv_id: {len(has_arxiv)}")
    
    for i, p in enumerate(has_arxiv):
        arxiv_id = p["arxiv_id"]
        print(f"\n[{i+1}/{len(has_arxiv)}] Fetching {arxiv_id}...")
        
        abstract = fetch_arxiv_abstract(arxiv_id)
        if abstract:
            p["abstract_en"] = abstract
            print(f"  ✓ Got abstract ({len(abstract)} chars)")
            save_db(db)
        else:
            print(f"  ✗ Failed")
        
        time.sleep(3)  # Be nice to arXiv API
    
    print(f"\nDone!")


if __name__ == "__main__":
    fetch_missing_abstracts()
