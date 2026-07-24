#!/usr/bin/env python3
"""Batch translate abstracts with chunking support for long texts."""

import json
import time
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


def translate_chunk(text):
    """Translate a single chunk (max ~450 chars to be safe)."""
    if not text or len(text.strip()) < 10:
        return ""
    
    try:
        params = {
            "q": text[:450],  # Stay well under 500 limit
            "langpair": "en|zh"
        }
        resp = requests.get(TRANSLATE_API, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("responseStatus") == 200:
            return data["responseData"]["translatedText"]
        else:
            print(f"  API error: {data.get('responseDetails')}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def translate_text(text):
    """Translate text, splitting into chunks if needed."""
    if not text or len(text) < 50:
        return None
    
    # If text is short enough, translate directly
    if len(text) <= 450:
        return translate_chunk(text)
    
    # Split into sentences and group into chunks
    sentences = text.replace('. ', '.|').replace('? ', '?|').replace('! ', '!|').split('|')
    chunks = []
    current_chunk = ""
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Add period back if removed
        if sent and sent[-1] not in '.?!':
            sent += '.'
        
        if len(current_chunk) + len(sent) + 1 <= 450:
            current_chunk += " " + sent if current_chunk else sent
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sent
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Translate each chunk
    translations = []
    for i, chunk in enumerate(chunks):
        print(f"  Translating chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
        result = translate_chunk(chunk)
        if result:
            translations.append(result)
        time.sleep(0.5)  # Rate limit between chunks
    
    if translations:
        return " ".join(translations)
    return None


def generate_summary(abstract_en, title, topic):
    """Generate a simple extractive summary."""
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    sentences = abstract_en.split(". ")
    if len(sentences) >= 2:
        summary = ". ".join(sentences[:2]) + "."
    else:
        summary = abstract_en[:300] + "..."
    
    return f"【{topic}】{summary}"


def process_all_papers(batch_size=5):
    """Process all papers needing translation/summary."""
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
    
    print(f"Found {len(need_processing)} papers to process")
    
    total = len(need_processing)
    processed = 0
    
    for i in range(0, total, batch_size):
        batch = need_processing[i:i+batch_size]
        print(f"\n--- Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ---")
        
        for p, needs_cn, needs_summary in batch:
            print(f"[{p['id']}] {p['title'][:50]}...")
            
            if needs_cn:
                print("  Translating...")
                cn = translate_text(p.get("abstract_en", ""))
                if cn:
                    p["abstract_cn"] = cn
                    print(f"  ✓ CN ({len(cn)} chars)")
                else:
                    print("  ✗ Failed")
                time.sleep(1)
            
            if needs_summary:
                print("  Summarizing...")
                summary = generate_summary(p.get("abstract_en", ""), p.get("title", ""), p.get("topic", ""))
                if summary:
                    p["ai_summary"] = summary
                    print(f"  ✓ Summary ({len(summary)} chars)")
                time.sleep(0.3)
            
            processed += 1
        
        save_db(db)
        print(f"Progress: {processed}/{total}")
        time.sleep(2)
    
    print(f"\nDone! Processed {processed} papers")


if __name__ == "__main__":
    process_all_papers(batch_size=3)
