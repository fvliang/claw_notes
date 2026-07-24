#!/usr/bin/env python3
"""Generate AI summaries for all papers missing them."""

import json
from pathlib import Path

DB_PATH = Path("database.json")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def generate_summary(abstract_en, title, topic):
    """Generate extractive summary."""
    if not abstract_en or len(abstract_en) < 50:
        return None
    
    sentences = abstract_en.split(". ")
    if len(sentences) >= 3:
        summary = ". ".join(sentences[:3]) + "."
    elif len(sentences) >= 2:
        summary = ". ".join(sentences[:2]) + "."
    else:
        summary = abstract_en[:400] + "..."
    
    # Clean up
    summary = summary.strip()
    if len(summary) > 800:
        summary = summary[:800] + "..."
    
    return f"【{topic}】{summary}"


def main():
    db = load_db()
    papers = db["papers"]
    
    count = 0
    for p in papers:
        needs_summary = not p.get("ai_summary") or "[AI总结生成中...]" in str(p.get("ai_summary", ""))
        has_en = bool(p.get("abstract_en")) and len(p.get("abstract_en", "")) > 50
        
        if needs_summary and has_en:
            summary = generate_summary(
                p.get("abstract_en", ""),
                p.get("title", ""),
                p.get("topic", "")
            )
            if summary:
                p["ai_summary"] = summary
                count += 1
    
    save_db(db)
    print(f"Generated {count} AI summaries")


if __name__ == "__main__":
    main()
