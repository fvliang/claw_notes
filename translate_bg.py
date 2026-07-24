#!/usr/bin/env python3
"""Background translation processor with resume support."""

import json
import time
import sys
from pathlib import Path

import requests

DB_PATH = Path("database.json")
PROGRESS_FILE = Path(".translation_progress")
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


def translate_chunk(text, retries=3):
    if not text or len(text.strip()) < 10:
        return ""
    
    for attempt in range(retries):
        try:
            params = {
                "q": text[:450],
                "langpair": "en|zh"
            }
            resp = requests.get(TRANSLATE_API, params=params, timeout=30)
            
            if resp.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("responseStatus") == 200:
                return data["responseData"]["translatedText"]
            elif data.get("responseStatus") == 429:
                wait = 10 * (attempt + 1)
                print(f"    Rate limited (API), waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            else:
                print(f"    API error: {data.get('responseDetails')}", file=sys.stderr)
                return None
        except Exception as e:
            print(f"    Error: {e}", file=sys.stderr)
            time.sleep(5)
    
    return None


def translate_text(text):
    if not text or len(text) < 50:
        return None
    
    if len(text) <= 450:
        return translate_chunk(text)
    
    # Split into chunks
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
        time.sleep(3)
    
    return " ".join(translations) if translations else None


def generate_summary(abstract_en, title, topic):
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    sentences = abstract_en.split(". ")
    if len(sentences) >= 2:
        summary = ". ".join(sentences[:2]) + "."
    else:
        summary = abstract_en[:300] + "..."
    
    return f"【{topic}】{summary}"


def main():
    db = load_db()
    papers = db["papers"]
    done_ids = load_progress()
    
    need_processing = []
    for p in papers:
        if p["id"] in done_ids:
            continue
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))
        needs_summary = not p.get("ai_summary") or "[AI总结生成中...]" in str(p.get("ai_summary", ""))
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        
        if (needs_cn or needs_summary) and has_en:
            need_processing.append((p, needs_cn, needs_summary))
    
    total = len(need_processing)
    print(f"Found {total} papers to process ({len(done_ids)} already done)")
    
    for i, (p, needs_cn, needs_summary) in enumerate(need_processing):
        print(f"[{i+1}/{total}] ID:{p['id']} {p['title'][:40]}...")
        
        if needs_cn:
            cn = translate_text(p.get("abstract_en", ""))
            if cn:
                p["abstract_cn"] = cn
                print(f"  ✓ CN ({len(cn)} chars)")
            else:
                print(f"  ✗ CN failed")
        
        if needs_summary:
            summary = generate_summary(p.get("abstract_en", ""), p.get("title", ""), p.get("topic", ""))
            if summary:
                p["ai_summary"] = summary
                print(f"  ✓ Summary ({len(summary)} chars)")
        
        done_ids.add(p["id"])
        save_progress(done_ids)
        
        if (i + 1) % 5 == 0:
            save_db(db)
            print(f"  Saved DB")
        
        time.sleep(4)  # Conservative delay
    
    save_db(db)
    print("Done!")


if __name__ == "__main__":
    main()
