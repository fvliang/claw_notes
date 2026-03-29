#!/usr/bin/env python3
"""
论文数据库管理 - 增删改查
"""
import json
import os

DB_PATH = os.path.expanduser("~/claw_notes/database.json")

def load_db():
    with open(DB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_db(db):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def add_paper(title, authors="", arxiv="", github="", conference="", year=2026, topic="LLM Serving", abstract_en="", abstract_cn=""):
    db = load_db()
    paper = {
        "id": len(db['papers']) + 1,
        "title": title,
        "authors": authors,
        "arxiv_id": arxiv,
        "github_repo": github,
        "conference": conference,
        "year": year,
        "topic": topic,
        "abstract_en": abstract_en,
        "abstract_cn": abstract_cn,
    }
    db['papers'].append(paper)
    save_db(db)
    print(f"✅ Added: {title}")
    return paper

def list_papers(topic=None, conference=None):
    db = load_db()
    papers = db['papers']
    
    if topic:
        papers = [p for p in papers if p['topic'] == topic]
    if conference:
        papers = [p for p in papers if p['conference'] == conference]
    
    papers.sort(key=lambda x: x.get('id', 0), reverse=True)
    
    print(f"\n📚 {len(papers)} papers\n")
    for p in papers:
        print(f"[{p['id']}] {p['title']}")
        print(f"    📍 {p['conference']} | 🏷️ {p['topic']}")
        if p.get('arxiv_id'):
            print(f"    🔗 arxiv.org/abs/{p['arxiv_id']}")
        print()
    return papers

def search_papers(keyword):
    db = load_db()
    keyword = keyword.lower()
    results = [p for p in db['papers'] if keyword in p['title'].lower() or keyword in p['topic'].lower()]
    
    print(f"\n🔍 Found {len(results)} papers\n")
    for p in results:
        print(f"[{p['id']}] {p['title']}")
    return results

def export_notion():
    """导出为Notion CSV格式"""
    db = load_db()
    lines = ["Title,Authors,Conference,Year,Topic,arXiv,GitHub"]
    
    for p in db['papers']:
        title = p['title'].replace('"', '""')
        authors = p['authors'].replace('"', '""')
        line = f'"{title}","{authors}",{p["conference"]},{p["year"]},{p["topic"]},{p.get("arxiv_id","")},{p.get("github_repo","")}'
        lines.append(line)
    
    with open(os.path.expanduser("~/claw_notes/notion_import.csv"), 'w') as f:
        f.write('\n'.join(lines))
    print(f"✅ Exported {len(db['papers'])} papers to notion_import.csv")

def generate_web():
    """生成静态网页"""
    db = load_db()
    topics = db.get('topics', [])
    
    html = '''<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Serving Papers</title>
    <style>
        body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
        h1 { text-align: center; color: #333; }
        .filter { margin-bottom: 20px; }
        .filter select, .filter input { padding: 10px; border: 1px solid #ddd; border-radius: 8px; margin-right: 10px; }
        .paper { background: white; padding: 20px; margin-bottom: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .paper h3 { color: #007AFF; margin-bottom: 8px; }
        .paper .meta { color: #666; font-size: 14px; }
        .paper .tags span { background: #e8f4ff; color: #007AFF; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin-right: 8px; }
        .paper a { color: #007AFF; margin-right: 15px; }
    </style>
</head>
<body>
    <h1>📚 LLM Serving Papers</h1>
    <div class="filter">
        <select id="topicFilter" onchange="filterPapers()">
            <option value="">All Topics</option>
'''
    for t in topics:
        html += f'            <option value="{t}">{t}</option>\n'
    
    html += '''        </select>
        <input type="text" id="search" placeholder="Search..." oninput="filterPapers()">
    </div>
    <div id="papers">
'''
    
    for p in db['papers']:
        links = []
        if p.get('arxiv_id'):
            links.append(f'<a href="https://arxiv.org/abs/{p["arxiv_id"]}">arXiv</a>')
        if p.get('github_repo'):
            links.append(f'<a href="https://github.com/{p["github_repo"]}">GitHub</a>')
        
        html += f'''        <div class="paper" data-topic="{p['topic']}">
            <h3>{p['title']}</h3>
            <div class="meta">{p['authors']} | {p['conference']} {p['year']}</div>
            <div class="tags"><span>{p['topic']}</span></div>
            <div>{" ".join(links)}</div>
        </div>
'''
    
    html += '''    </div>
    <script>
        function filterPapers() {
            const topic = document.getElementById('topicFilter').value;
            const search = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.paper').forEach(p => {
                const match = (!topic || p.dataset.topic === topic) && 
                              (!search || p.textContent.toLowerCase().includes(search));
                p.style.display = match ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>'''
    
    with open(os.path.expanduser("~/claw_notes/index.html"), 'w') as f:
        f.write(html)
    print("✅ Generated index.html")

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    
    if cmd == "list":
        list_papers()
    elif cmd == "export-notion":
        export_notion()
    elif cmd == "web":
        generate_web()
    elif cmd == "add":
        # 简单添加
        title = input("Title: ")
        authors = input("Authors: ")
        arxiv = input("arXiv ID (optional): ")
        topic = input("Topic: ")
        add_paper(title, authors, arxiv=arxiv, topic=topic)
    else:
        print("Usage:")
        print("  python3 db_manager.py list")
        print("  python3 db_manager.py add")
        print("  python3 db_manager.py export-notion")
        print("  python3 db_manager.py web")