#!/tmp/venv/bin/python3
"""Retry translation for remaining papers."""

import json
import time
import sys
from pathlib import Path

import translators as ts

DB_PATH = Path("database.json")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def translate_with(text, translator):
    if not text or len(text) < 20:
        return None
    
    if len(text) > 3000:
        text = text[:3000]
    
    try:
        result = ts.translate_text(
            text,
            translator=translator,
            from_language='en',
            to_language='zh'
        )
        return result
    except Exception as e:
        return None

def main():
    db = load_db()
    papers = db["papers"]
    
    # Find papers missing CN but having some EN
    need_processing = []
    for p in papers:
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))
        has_en = bool(p.get("abstract_en") and len(p.get("abstract_en", "")) > 20)
        
        if needs_cn and has_en:
            need_processing.append(p)
    
    total = len(need_processing)
    print(f"Found {total} papers to retry")
    
    translators = ['bing', 'caiyun', 'sogou', 'youdao']
    
    for i, p in enumerate(need_processing):
        print(f"[{i+1}/{total}] ID:{p['id']} {p['title'][:40]}...")
        
        abstract_en = p.get("abstract_en", "")
        
        cn = None
        for translator in translators:
            cn = translate_with(abstract_en, translator)
            if cn and len(cn) > 20:
                print(f"  ✓ CN ({len(cn)} chars) via {translator}")
                break
            time.sleep(1)
        
        if cn and len(cn) > 20:
            p["abstract_cn"] = cn
        else:
            print(f"  ✗ Failed")
        
        if (i + 1) % 10 == 0:
            save_db(db)
            print(f"  Saved DB ({i+1}/{total})")
        
        time.sleep(3)
    
    save_db(db)
    print("Done!")

if __name__ == "__main__":
    main()
