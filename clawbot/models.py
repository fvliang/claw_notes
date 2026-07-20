"""Unified data model for papers."""
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json
from datetime import datetime


@dataclass
class Paper:
    """Normalized paper representation."""

    title: str
    authors: str = ""
    arxiv_id: str = ""
    github_repo: str = ""
    conference: str = ""
    year: int = 2026
    topic: str = "LLM Serving"
    abstract_en: str = ""
    abstract_cn: str = ""
    intro_en: str = ""
    intro_cn: str = ""
    published: str = ""           # arXiv publish date: YYYY-MM-DD
    added_date: str = ""          # when added to DB
    collected_date: str = ""      # when collected by bot
    source: str = "arXiv"         # arXiv, conference, GitHub
    has_content: bool = False
    is_github_project: bool = False
    is_placeholder_arxiv: bool = False
    file: str = ""
    id: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    url: str = ""
    pdf_url: str = ""
    comment: str = ""

    # Legacy field mappings for backward compatibility
    @classmethod
    def from_legacy(cls, data: dict) -> "Paper":
        """Create Paper from legacy database.json entry."""
        d = dict(data)  # copy

        # Normalize field names
        field_map = {
            "summary_en": "abstract_en",
            "summary_cn": "abstract_cn",
            "introduction_en": "intro_en",
            "introduction_cn": "intro_cn",
            "markdown_path": "file",
            "filepath": "file",
            "md_file": "file",
            "venue": "conference",
            "full_conference": "conference",
            "github": "github_repo",
            "link": "url",
            "pdf_link": "pdf_url",
            "blog_url": "url",
        }
        for old, new in field_map.items():
            if old in d and new not in d:
                d[new] = d.pop(old)
            elif old in d:
                # both exist, prefer non-empty
                if not d.get(new):
                    d[new] = d.pop(old)
                else:
                    d.pop(old)

        # Handle date fields
        if "date" in d and not d.get("published"):
            d["published"] = d.pop("date")
        if "created_at" in d and not d.get("added_date"):
            d["added_date"] = d.pop("created_at")
        if "updated_at" in d and not d.get("collected_date"):
            d["collected_date"] = d.pop("updated_at")

        # Clean up fields not in Paper
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        cleaned = {k: v for k, v in d.items() if k in valid_fields}

        # Handle tags (list or string)
        if "tags" in cleaned and isinstance(cleaned["tags"], str):
            cleaned["tags"] = [t.strip() for t in cleaned["tags"].split(",") if t.strip()]
        elif "tags" not in cleaned:
            cleaned["tags"] = []

        # Ensure year is int
        if "year" in cleaned and cleaned["year"]:
            try:
                cleaned["year"] = int(cleaned["year"])
            except (ValueError, TypeError):
                cleaned["year"] = 2026

        # Ensure id is int or None
        if "id" in cleaned and cleaned["id"] is not None:
            try:
                cleaned["id"] = int(cleaned["id"])
            except (ValueError, TypeError):
                cleaned["id"] = None

        return cls(**cleaned)

    def to_legacy(self) -> dict:
        """Export to legacy dict format for database.json compatibility."""
        d = asdict(self)
        # Remove None values and empty strings for cleanliness
        return {k: v for k, v in d.items() if v or v == 0 or v is False}

    def fingerprint(self) -> str:
        """Return a stable fingerprint for deduplication."""
        # Primary: arxiv_id
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.strip()}"
        # Secondary: normalized title
        t = self.title.lower().strip()
        t = t.replace("\n", " ").replace("  ", " ")
        t = t.replace("{", "").replace("}", "").replace("$", "")
        t = t.replace("\\", "")
        return f"title:{t}"

    def normalized_title(self) -> str:
        """Return normalized title for comparison."""
        t = self.title.lower().strip()
        t = t.replace("\n", " ").replace("  ", " ")
        t = t.replace("{", "").replace("}", "").replace("$", "")
        t = t.replace("\\", "")
        return t

    def __hash__(self):
        return hash(self.fingerprint())

    def __eq__(self, other):
        if not isinstance(other, Paper):
            return False
        return self.fingerprint() == other.fingerprint()
