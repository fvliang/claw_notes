#!/usr/bin/env python3
"""Batch translate abstracts using free MyMemory API and generate simple summaries."""

import json
import time
import urllib.parse
from pathlib import Path

import requests

DB_PATH = Path("database.json")
TRANSLATE_API = "https://api.mymemory.translated.net/get"

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def translate_text(text):
    """Translate English text to Chinese using MyMemory API."""
    if not text or len(text) < 50:
        return None
    
    # Limit text length to avoid API issues
    text = text[:1000]
    
    try:
        params = {
            "q": text,
            "langpair": "en|zh"
        }
        resp = requests.get(TRANSLATE_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("responseStatus") == 200:
            return data["responseData"]["translatedText"]
        else:
            print(f"  Translation API error: {data.get('responseDetails')}")
            return None
    except Exception as e:
        print(f"  Translation error: {e}")
        return None


def generate_simple_summary(abstract_en, title, topic):
    """Generate a simple extractive summary."""
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    # Extract first 2-3 sentences as summary
    sentences = abstract_en.split(". ")
    if len(sentences) >= 2:
        summary = ". ".join(sentences[:2]) + "."
    else:
        summary = abstract_en[:300] + "..."
    
    # Add topic context
    return f"【{topic}】{summary}"


def process_papers(batch_size=5, max_papers=None):
    """Process papers in batches."""
    db = load_db()
    papers = db["papers"]
    
    # Find papers needing processing
    need_processing = []
    for p in papers:
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))
        needs_summary = not p.get("ai_summary") or "[AI总结生成中...]" in str(p.get("ai_summary", ""))
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        
        if (needs_cn or needs_summary) and has_en:
            need_processing.append((p, needs_cn, needs_summary))
    
    if max_papers:
        need_processing = need_processing[:max_papers]
    
    print(f"Found {len(need_processing)} papers to process")
    
    total = len(need_processing)
    processed = 0
    
    for i in range(0, total, batch_size):
        batch = need_processing[i:i+batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ---")
        
        for p, needs_cn, needs_summary in batch:
            print(f"Processing [{p['id']}]: {p['title'][:50]}...")
            
            # Translate to Chinese
            if needs_cn:
                print("  Translating...")
                cn = translate_text(p.get("abstract_en", ""))
                if cn:
                    p["abstract_cn"] = cn
                    print(f"  ✓ Translated ({len(cn)} chars)")
                else:
                    print("  ✗ Translation failed")
                time.sleep(1)  # Rate limit
            
            # Generate summary
            if needs_summary:
                print("  Generating summary...")
                summary = generate_simple_summary(
                    p.get("abstract_en", ""), 
                    p.get("title", ""), 
                    p.get("topic", "")
                )
                if summary:
                    p["ai_summary"] = summary
                    print(f"  ✓ Summary generated ({len(summary)} chars)")
                time.sleep(0.5)
            
            processed += 1
        
        # Save progress every batch
        save_db(db)
        print(f"Saved progress: {processed}/{total}")
        
        # Rate limit between batches
        time.sleep(2)
    
    print(f"\nDone! Processed {processed} papers")


if __name__ == "__main__":
    # Process first 100 papers as a test
    process_papers(batch_size=3, max_papers=100)
