#!/usr/bin/env python3
"""Fetch missing abstracts from arXiv HTML pages."""

import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DB_PATH = Path("database.json")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def fetch_arxiv_html(arxiv_id):
    """Fetch abstract from arXiv HTML page."""
    if not arxiv_id:
        return None
    
    try:
        url = f"https://arxiv.org/abs/{arxiv_id}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find abstract
        abstract_div = soup.find('blockquote', class_='abstract')
        if abstract_div:
            # Remove the "Abstract:" heading
            for span in abstract_div.find_all('span', class_='descriptor'):
                span.decompose()
            abstract = abstract_div.get_text(strip=True)
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
        
        abstract = fetch_arxiv_html(arxiv_id)
        if abstract:
            p["abstract_en"] = abstract
            print(f"  ✓ Got abstract ({len(abstract)} chars)")
            save_db(db)
        else:
            print(f"  ✗ Failed")
        
        time.sleep(2)  # Be nice to arXiv
    
    print(f"\nDone!")


if __name__ == "__main__":
    fetch_missing_abstracts()
