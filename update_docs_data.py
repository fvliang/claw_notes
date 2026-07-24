#!/usr/bin/env python3
"""Update docs/index.html with latest database content."""

import json
from pathlib import Path

DB_PATH = Path("database.json")
HTML_PATH = Path("docs/index.html")

def load_db():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def update_html():
    db = load_db()
    papers = db["papers"]
    
    # Read current HTML
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Find the data line (starts with "const P=")
    lines = content.split('\n')
    data_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('const P='):
            data_line_idx = i
            break
    
    if data_line_idx is None:
        print("Could not find data line!")
        return
    
    # Generate new data
    # Only include fields needed by the UI
    papers_data = []
    for p in papers:
        paper_data = {
            "id": p["id"],
            "title": p.get("title", ""),
            "authors": p.get("authors", ""),
            "topic": p.get("topic", ""),
            "conference": p.get("conference", "arXiv"),
            "added_date": p.get("added_date", ""),
            "arxiv_id": p.get("arxiv_id", ""),
            "github_repo": p.get("github_repo", ""),
        }
        
        # Only include abstract fields if they exist and are meaningful
        abstract_en = p.get("abstract_en", "")
        if abstract_en and len(abstract_en) > 50:
            paper_data["abstract_en"] = abstract_en
        
        abstract_cn = p.get("abstract_cn", "")
        if abstract_cn and "[中文翻译" not in str(abstract_cn) and "[自动翻译" not in str(abstract_cn) and len(str(abstract_cn)) > 10:
            paper_data["abstract_cn"] = abstract_cn
        
        ai_summary = p.get("ai_summary", "")
        if ai_summary and "[AI总结" not in str(ai_summary) and len(str(ai_summary)) > 50:
            paper_data["ai_summary"] = ai_summary
        
        papers_data.append(paper_data)
    
    # Generate compact JSON
    json_str = json.dumps(papers_data, ensure_ascii=False, separators=(',', ':'))
    new_data_line = f"const P={json_str};"
    
    # Replace data line
    lines[data_line_idx] = new_data_line
    
    # Write back
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write('\n'.join(lines))
    
    print(f"Updated docs/index.html with {len(papers_data)} papers")
    
    # Count stats
    has_en = sum(1 for p in papers_data if p.get("abstract_en"))
    has_cn = sum(1 for p in papers_data if p.get("abstract_cn"))
    has_summary = sum(1 for p in papers_data if p.get("ai_summary"))
    print(f"With EN abstract: {has_en}")
    print(f"With CN translation: {has_cn}")
    print(f"With AI summary: {has_summary}")


if __name__ == "__main__":
    update_html()
