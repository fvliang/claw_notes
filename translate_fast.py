#!/usr/bin/env python3
"""Batch translate using translators library."""

import json
import time
import sys
from pathlib import Path

import translators as ts

DB_PATH = Path("database.json")
PROGRESS_FILE = Path(".translation_progress_v3")

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

def translate_text(text, translator='bing'):
    """Translate text using translators library."""
    if not text or len(text) < 50:
        return None
    
    # Truncate very long texts (Bing has limits)
    if len(text) > 3000:
        text = text[:3000] + "..."
    
    try:
        result = ts.translate_text(
            text,
            translator=translator,
            from_language='en',
            to_language='zh'
        )
        return result
    except Exception as e:
        print(f"    {translator} error: {e}", file=sys.stderr)
        return None

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
    
    translators_to_try = ['bing', 'google', 'yandex']
    current_translator_idx = 0
    
    for i, p in enumerate(need_processing):
        print(f"[{i+1}/{total}] ID:{p['id']} {p['title'][:40]}...")
        
        abstract_en = p.get("abstract_en", "")
        
        # Try each translator
        cn = None
        for t_idx in range(len(translators_to_try)):
            translator = translators_to_try[(current_translator_idx + t_idx) % len(translators_to_try)]
            print(f"  Trying {translator}...")
            cn = translate_text(abstract_en, translator)
            if cn:
                print(f"  ✓ CN ({len(cn)} chars) via {translator}")
                current_translator_idx = (current_translator_idx + t_idx + 1) % len(translators_to_try)
                break
            time.sleep(2)
        
        if cn:
            p["abstract_cn"] = cn
        else:
            print(f"  ✗ All translators failed")
        
        done_ids.add(p["id"])
        save_progress(done_ids)
        
        if (i + 1) % 10 == 0:
            save_db(db)
            print(f"  Saved DB ({i+1}/{total})")
        
        time.sleep(3)  # Be nice to APIs
    
    save_db(db)
    print("Done!")

if __name__ == "__main__":
    main()
