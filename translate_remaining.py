#!/tmp/venv/bin/python3
"""Translate remaining papers."""

import json
import time
from pathlib import Path

import translators as ts

DB_PATH = Path("database.json")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def translate_text(text):
    if not text or len(text) < 50:
        return None
    
    if len(text) > 3000:
        text = text[:3000]
    
    for translator in ['bing', 'caiyun', 'sogou', 'youdao']:
        try:
            result = ts.translate_text(text, translator=translator, from_language='en', to_language='zh')
            if result and len(result) > 20:
                return result
        except:
            pass
        time.sleep(2)
    
    return None

def main():
    db = load_db()
    
    target_ids = list(range(242, 267)) + [404]
    
    for pid in target_ids:
        paper = next((p for p in db["papers"] if p["id"] == pid), None)
        if not paper:
            continue
        
        needs_cn = not paper.get("abstract_cn") or "[中文翻译待补充]" in str(paper.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(paper.get("abstract_cn", ""))
        has_en = bool(paper.get("abstract_en") and len(paper.get("abstract_en", "")) > 50)
        
        if needs_cn and has_en:
            print(f"Translating ID:{pid} {paper['title'][:40]}...")
            cn = translate_text(paper["abstract_en"])
            if cn:
                paper["abstract_cn"] = cn
                print(f"  ✓ ({len(cn)} chars)")
            else:
                print(f"  ✗ Failed")
            time.sleep(3)
    
    save_db(db)
    print("Done!")

if __name__ == "__main__":
    main()
