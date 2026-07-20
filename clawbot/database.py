"""Database operations for the paper collection."""
import json
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .models import Paper
from .config import DB_PATH

logger = logging.getLogger("clawbot.db")


class PaperDatabase:
    """Manages the database.json file."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DB_PATH
        self._data = None
        self._papers: List[Paper] = []
        self._load()

    def _load(self):
        """Load database from disk."""
        if not self.db_path.exists():
            logger.warning("Database not found at %s, creating new", self.db_path)
            self._data = {"papers": [], "topics": ["LLM Serving", "Speculative Decoding", "KV Cache", "Inference Kernel", "Hardware Acceleration", "Memory Optimization"]}
            self._papers = []
            return

        with open(self.db_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        papers_raw = self._data.get("papers", [])
        self._papers = []
        for raw in papers_raw:
            try:
                p = Paper.from_legacy(raw)
                self._papers.append(p)
            except Exception as e:
                logger.warning("Failed to parse paper entry: %s", e)

        logger.info("Loaded %d papers from database", len(self._papers))

    def save(self):
        """Save database to disk."""
        # Reassign IDs to be sequential
        for i, p in enumerate(self._papers, start=1):
            p.id = i

        legacy_papers = [p.to_legacy() for p in self._papers]
        self._data["papers"] = legacy_papers

        # Ensure topics list exists
        if "topics" not in self._data:
            self._data["topics"] = ["LLM Serving", "Speculative Decoding", "KV Cache", "Inference Kernel", "Hardware Acceleration", "Memory Optimization"]

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

        logger.info("Saved %d papers to database", len(self._papers))

    def all_papers(self) -> List[Paper]:
        return list(self._papers)

    def add(self, paper: Paper) -> Paper:
        """Add a new paper, assigning ID."""
        paper.id = len(self._papers) + 1
        if not paper.added_date:
            paper.added_date = datetime.now().strftime("%Y-%m-%d")
        if not paper.collected_date:
            paper.collected_date = datetime.now().strftime("%Y-%m-%d")
        self._papers.append(paper)
        logger.info("Added paper [%d]: %s", paper.id, paper.title[:60])
        return paper

    def add_batch(self, papers: List[Paper]) -> List[Paper]:
        """Add multiple papers."""
        added = []
        for p in papers:
            added.append(self.add(p))
        return added

    def find_by_arxiv(self, arxiv_id: str) -> Optional[Paper]:
        for p in self._papers:
            if p.arxiv_id and p.arxiv_id.strip() == arxiv_id.strip():
                return p
        return None

    def find_by_title(self, title: str) -> Optional[Paper]:
        nt = Paper(title=title).normalized_title()
        for p in self._papers:
            if p.normalized_title() == nt:
                return p
        return None

    def search(self, keyword: str) -> List[Paper]:
        kw = keyword.lower()
        return [
            p for p in self._papers
            if kw in p.title.lower() or kw in p.abstract_en.lower() or kw in p.topic.lower()
        ]

    def stats(self) -> dict:
        topics = {}
        years = {}
        sources = {}
        for p in self._papers:
            topics[p.topic] = topics.get(p.topic, 0) + 1
            years[p.year] = years.get(p.year, 0) + 1
            sources[p.source] = sources.get(p.source, 0) + 1
        return {
            "total": len(self._papers),
            "by_topic": topics,
            "by_year": years,
            "by_source": sources,
        }
