#!/tmp/venv/bin/python3
"""Final attempt for remaining papers."""

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

def main():
    db = load_db()
    
    for p in db["papers"]:
        needs_cn = not p.get("abstract_cn") or "[中文翻译待补充]" in str(p.get("abstract_cn", "")) or "[自动翻译生成中...]" in str(p.get("abstract_cn", ""))
        has_en = bool(p.get("abstract_en") and len(p.get("abstract_en", "")) > 20)
        
        if needs_cn and has_en:
            print(f"Trying ID:{p['id']}...")
            for translator in ['bing', 'caiyun', 'sogou', 'youdao']:
                try:
                    result = ts.translate_text(
                        p["abstract_en"],
                        translator=translator,
                        from_language='en',
                        to_language='zh'
                    )
                    if result and len(result) > 10:
                        p["abstract_cn"] = result
                        print(f"  ✓ via {translator}")
                        break
                except:
                    pass
                time.sleep(2)
    
    save_db(db)
    print("Done!")

if __name__ == "__main__":
    main()
