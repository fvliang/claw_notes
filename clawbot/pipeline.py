"""End-to-end pipeline: search -> filter -> dedup -> save -> generate web."""
import logging
import time
from datetime import datetime
from typing import List

from . import config
from .models import Paper
from .arxiv_client import search_arxiv, fetch_recent_by_category
from .filter import filter_papers
from .dedup import DedupEngine
from .database import PaperDatabase
from .summarizer import generate_summaries_for_papers

logger = logging.getLogger("clawbot.pipeline")


def run_search_pipeline(
    db: PaperDatabase,
    queries: List[str] = None,
    max_results_per_query: int = 50,
    categories: List[str] = None,
) -> List[Paper]:
    """Run full search pipeline and return new papers found."""

    queries = queries or config.SEARCH_QUERIES
    dedup = DedupEngine(db.all_papers())
    all_new: List[Paper] = []

    # Phase 1: Keyword searches
    logger.info("=" * 50)
    logger.info("Phase 1: Keyword search (%d queries)", len(queries))
    logger.info("=" * 50)

    for q in queries:
        try:
            papers = search_arxiv(q, max_results=max_results_per_query)
            filtered = filter_papers(papers)
            new = dedup.dedup(filtered)
            logger.info("Query '%s': %d raw -> %d filtered -> %d new", q, len(papers), len(filtered), len(new))
            all_new.extend(new)
        except Exception as e:
            logger.error("Query '%s' failed: %s", q, e)
            continue

    # Phase 2: Category feed (cs.DC, cs.CL, cs.AR, cs.LG)
    if categories:
        logger.info("=" * 50)
        logger.info("Phase 2: Category feed (%s)", ", ".join(categories))
        logger.info("=" * 50)
        try:
            recent = fetch_recent_by_category(categories, max_results=200)
            filtered = filter_papers(recent)
            new = dedup.dedup(filtered)
            logger.info("Category feed: %d raw -> %d filtered -> %d new", len(recent), len(filtered), len(new))
            all_new.extend(new)
        except Exception as e:
            logger.error("Category feed failed: %s", e)

    # Add to database
    if all_new:
        logger.info("=" * 50)
        logger.info("Adding %d new papers to database", len(all_new))
        logger.info("=" * 50)
        for p in all_new:
            p.collected_date = datetime.now().strftime("%Y-%m-%d")
            db.add(p)
        db.save()
    else:
        logger.info("No new papers found.")

    return all_new


def generate_markdown_index(db: PaperDatabase, output_path: str = None):
    """Generate INDEX.md from database."""
    output_path = output_path or str(config.REPO_ROOT / "INDEX.md")
    papers = db.all_papers()

    # Group by year and topic
    by_year = {}
    for p in papers:
        year = p.year or 2026
        by_year.setdefault(year, []).append(p)

    lines = ["# LLM Serving Papers Index\n", f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n"]

    for year in sorted(by_year.keys(), reverse=True):
        lines.append(f"\n## {year}\n")
        year_papers = by_year[year]
        # Sort by topic
        by_topic = {}
        for p in year_papers:
            by_topic.setdefault(p.topic or "Uncategorized", []).append(p)

        for topic in sorted(by_topic.keys()):
            lines.append(f"\n### {topic}\n")
            for p in sorted(by_topic[topic], key=lambda x: x.title):
                arxiv_link = f"[arXiv](https://arxiv.org/abs/{p.arxiv_id})" if p.arxiv_id else ""
                github_link = f"[GitHub](https://github.com/{p.github_repo})" if p.github_repo else ""
                links = " | ".join(filter(None, [arxiv_link, github_link]))
                lines.append(f"- **{p.title}** — {p.authors or 'N/A'}")
                if links:
                    lines.append(f"  {links}")
                if p.abstract_en:
                    summary = p.abstract_en[:200].replace("\n", " ")
                    lines.append(f"  > {summary}...")
                lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Generated INDEX.md with %d papers", len(papers))


def generate_web_html(db: PaperDatabase, output_path: str = None):
    """Generate index.html from database."""
    output_path = output_path or str(config.INDEX_HTML)
    papers = db.all_papers()
    topics = sorted(set(p.topic or "Uncategorized" for p in papers))

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Serving Papers</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;max-width:1000px;margin:0 auto;padding:20px;line-height:1.6}
h1{color:#64ffda;text-align:center;margin-bottom:10px;font-size:2rem}
.stats{text-align:center;color:#888;margin-bottom:30px;font-size:0.9rem}
.filter{display:flex;gap:10px;margin-bottom:25px;flex-wrap:wrap;justify-content:center}
.filter input,.filter select{padding:10px 15px;border:1px solid #333;border-radius:8px;background:#1a1a2e;color:#e0e0e0;font-size:0.95rem;min-width:200px}
.paper{background:#1a1a2e;border:1px solid #222;border-radius:12px;padding:20px;margin-bottom:15px;transition:border-color 0.2s}
.paper:hover{border-color:#64ffda}
.paper h3{color:#64ffda;font-size:1.1rem;margin-bottom:8px}
.paper .meta{color:#888;font-size:0.85rem;margin-bottom:10px}
.paper .abstract{color:#bbb;font-size:0.9rem;margin-bottom:12px;line-height:1.5}
.paper .tags{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.paper .tags span{background:#16213e;color:#64ffda;padding:4px 12px;border-radius:20px;font-size:0.75rem}
.paper .links a{color:#64ffda;text-decoration:none;margin-right:15px;font-size:0.85rem}
.paper .links a:hover{text-decoration:underline}
.empty{text-align:center;color:#888;padding:40px}
</style>
</head>
<body>
<h1>📚 LLM Serving Papers</h1>
<div class="stats">"""

    stats = db.stats()
    html += f"{stats['total']} papers | Updated {datetime.now().strftime('%Y-%m-%d')}"
    html += """</div>
<div class="filter">
<select id="topicFilter" onchange="filterPapers()">
<option value="">All Topics</option>"""

    for t in topics:
        html += f'<option value="{t}">{t}</option>\n'

    html += """</select>
<input type="text" id="searchInput" placeholder="Search papers..." oninput="filterPapers()">
</div>
<div id="paperList">"""

    for p in papers:
        links = []
        if p.arxiv_id:
            links.append(f'<a href="https://arxiv.org/abs/{p.arxiv_id}">arXiv</a>')
        if p.github_repo:
            links.append(f'<a href="https://github.com/{p.github_repo}">GitHub</a>')
        if p.pdf_url:
            links.append(f'<a href="{p.pdf_url}">PDF</a>')

        abstract = (p.abstract_en or "")[:280].replace("\n", " ").replace("<", "&lt;").replace(">", "&gt;")
        if len(p.abstract_en or "") > 280:
            abstract += "..."

        html += f"""<div class="paper" data-topic="{p.topic or ''}">
<h3>{p.title}</h3>
<div class="meta">{p.authors or 'N/A'} | {p.conference or p.source} {p.year}</div>
<div class="abstract">{abstract}</div>
<div class="tags"><span>{p.topic or 'Uncategorized'}</span></div>
<div class="links">{" ".join(links)}</div>
</div>"""

    html += """</div>
<div class="empty" id="emptyState" style="display:none">No papers match your filters.</div>
<script>
function filterPapers(){
const topic=document.getElementById('topicFilter').value.toLowerCase();
const search=document.getElementById('searchInput').value.toLowerCase();
const papers=document.querySelectorAll('.paper');
let visible=0;
papers.forEach(p=>{
const matchTopic=!topic||p.dataset.topic.toLowerCase()===topic;
const matchSearch=!search||p.textContent.toLowerCase().includes(search);
const show=matchTopic&&matchSearch;
p.style.display=show?'block':'none';
if(show)visible++;
});
document.getElementById('emptyState').style.display=visible?'none':'block';
}
</script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Generated index.html with %d papers", len(papers))



def generate_docs_html(db: PaperDatabase, output_path: str = None):
    """Generate docs/index.html (mobile UI with read/unread/favorites)."""
    import json as _json
    output_path = output_path or str(config.REPO_ROOT / "docs" / "index.html")
    papers = db.all_papers()

    # Read template from existing docs/index.html
    docs_html_path = config.REPO_ROOT / "docs" / "index.html"
    if not docs_html_path.exists():
        logger.warning("docs/index.html not found, skipping docs generation")
        return

    with open(docs_html_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find the data line (starts with 'const P=')
    data_line_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("const P="):
            data_line_idx = i
            break

    if data_line_idx is None:
        logger.warning("Could not find data line in docs/index.html, skipping")
        return

    # Build paper data
    paper_data = []
    for p in papers:
        paper_data.append({
            "id": p.id,
            "title": p.title or "",
            "authors": p.authors or "",
            "topic": p.topic or "",
            "conference": p.conference or "",
            "added_date": getattr(p, "added_date", None) or getattr(p, "collected_date", None) or "2026-03-15",
            "abstract": p.abstract_en or "",
            "abstract_cn": getattr(p, "abstract_cn", None) or "",
            "ai_summary_en": getattr(p, "ai_summary_en", None) or "",
            "ai_summary_cn": getattr(p, "ai_summary_cn", None) or "",
            "arxiv_id": getattr(p, "arxiv_id", None) or "",
            "github_repo": getattr(p, "github_repo", None) or "",
        })

    # Replace data line
    new_data_line = "const P=" + _json.dumps(paper_data, ensure_ascii=False) + ";\n"
    lines[data_line_idx] = new_data_line

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info("Generated docs/index.html with %d papers", len(papers))


def run_full_pipeline(
    db_path: str = None,
    queries: List[str] = None,
    categories: List[str] = None,
    generate_web: bool = True,
    generate_index: bool = True,
    generate_summaries: bool = True,
) -> dict:
    """Run the complete daily pipeline."""
    db = PaperDatabase(Path(db_path) if db_path else None)
    new_papers = run_search_pipeline(db, queries=queries, categories=categories)

    # Generate AI summaries for papers missing them (including newly added)
    if generate_summaries:
        summary_count = generate_summaries_for_papers(db.all_papers())
        if summary_count > 0:
            db.save()

    if generate_web:
        generate_web_html(db)
        generate_docs_html(db)
    if generate_index:
        generate_markdown_index(db)

    return {
        "new_papers": len(new_papers),
        "total_papers": len(db.all_papers()),
        "new_titles": [p.title for p in new_papers],
    }
