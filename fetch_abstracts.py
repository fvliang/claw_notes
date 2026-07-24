#!/usr/bin/env python3
"""Fetch missing arxiv abstracts, translate to Chinese, and generate AI summaries."""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

DB_PATH = Path("database.json")
ARXIV_API = "http://export.arxiv.org/api/query"

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def fetch_arxiv_abstract(arxiv_id):
    """Fetch abstract from arxiv API."""
    try:
        url = f"{ARXIV_API}?id_list={arxiv_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        
        # Parse Atom XML
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        entry = root.find("atom:entry", ns)
        if entry is None:
            return None
        
        summary = entry.find("atom:summary", ns)
        if summary is not None:
            return summary.text.strip()
        return None
    except Exception as e:
        print(f"Error fetching {arxiv_id}: {e}")
        return None


def batch_fetch_missing_abstracts():
    """Fetch abstracts for papers missing abstract_en but having arxiv_id."""
    db = load_db()
    papers = db["papers"]
    
    missing = [p for p in papers if not p.get("abstract_en") and p.get("arxiv_id")]
    print(f"Found {len(missing)} papers missing abstracts with arxiv_id")
    
    fetched = 0
    for i, p in enumerate(missing):
        arxiv_id = p["arxiv_id"]
        print(f"[{i+1}/{len(missing)}] Fetching {arxiv_id}...")
        
        abstract = fetch_arxiv_abstract(arxiv_id)
        if abstract:
            p["abstract_en"] = abstract
            fetched += 1
            print(f"  ✓ Got {len(abstract)} chars")
        else:
            print(f"  ✗ Failed")
        
        # Rate limit: 3 seconds between requests
        time.sleep(3)
        
        # Save every 5 papers
        if (i + 1) % 5 == 0:
            save_db(db)
            print(f"  Saved progress")
    
    save_db(db)
    print(f"\nDone! Fetched {fetched}/{len(missing)} abstracts")
    return fetched


if __name__ == "__main__":
    batch_fetch_missing_abstracts()
