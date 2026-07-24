#!/usr/bin/env python3
"""Robust background translation using MyMemory API."""

import json
import time
import sys
from pathlib import Path

import requests

DB_PATH = Path("database.json")
PROGRESS_FILE = Path(".translation_progress_v2")
TRANSLATE_API = "https://api.mymemory.translated.net/get"

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return set(int(x) for x in f.read().strip().split("\n") if x)
    return set()

def save_progress(done_ids):
    with open(PROGRESS_FILE, "w") as f:
        for pid in sorted(done_ids):
            f.write(f"{pid}\n")

def translate_chunk(text):
    """Translate a single chunk with long retry."""
    if not text or len(text.strip()) < 10:
        return ""
    
    for attempt in range(5):
        try:
            params = {
                "q": text[:450],
                "langpair": "en|zh"
            }
            resp = requests.get(TRANSLATE_API, params=params, timeout=30)
            
            if resp.status_code == 429:
                wait = 20 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("responseStatus") == 200:
                return data["responseData"]["translatedText"]
            elif data.get("responseStatus") == 429:
                wait = 20 * (attempt + 1)
                print(f"    Rate limited (API), waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            else:
                return None
        except Exception as e:
            print(f"    Error: {e}", file=sys.stderr)
            time.sleep(10)
    
    return None

def translate_text(text):
    """Translate text, splitting into chunks if needed."""
    if not text or len(text) < 50:
        return None
    
    if len(text) <= 450:
        return translate_chunk(text)
    
    # Split into sentences
    sentences = text.replace('. ', '.|').replace('? ', '?|').replace('! ', '!|').split('|')
    chunks = []
    current = ""
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if sent and sent[-1] not in '.?!':
            sent += '.'
        
        if len(current) + len(sent) + 1 <= 450:
            current += " " + sent if current else sent
        else:
            if current:
                chunks.append(current)
            current = sent
    
    if current:
        chunks.append(current)
    
    translations = []
    for chunk in chunks:
        result = translate_chunk(chunk)
        if result:
            translations.append(result)
        time.sleep(5)  # Delay between chunks
    
    return " ".join(translations) if translations else None

def main():
    db = load_db()
    papers = db["papers"]
    done_ids = load_progress()
    
    need_processing = []
    for p in papers:
        if p["id"] in done_ids:
            continue
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        
        if needs_cn and has_en:
            need_processing.append(p)
    
    total = len(need_processing)
    print(f"Found {total} papers to translate ({len(done_ids)} already done)")
    
    for i, p in enumerate(need_processing):
        print(f"[{i+1}/{total}] ID:{p['id']} {p['title'][:40]}...")
        
        cn = translate_text(p.get("abstract_en", ""))
        if cn:
            p["abstract_cn"] = cn
            print(f"  ✓ CN ({len(cn)} chars)")
        else:
            print(f"  ✗ Failed")
        
        done_ids.add(p["id"])
        save_progress(done_ids)
        
        if (i + 1) % 5 == 0:
            save_db(db)
            print(f"  Saved DB ({i+1}/{total})")
        
        time.sleep(8)  # Conservative delay
    
    save_db(db)
    print("Done!")

if __name__ == "__main__":
    main()
