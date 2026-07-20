"""Deduplication engine."""
import logging
from typing import List, Set
from .models import Paper

logger = logging.getLogger("clawbot.dedup")


class DedupEngine:
    """Track seen papers and deduplicate new ones."""

    def __init__(self, existing_papers: List[Paper] = None):
        self._seen: Set[str] = set()
        self._title_map: dict[str, str] = {}  # normalized title -> fingerprint
        if existing_papers:
            for p in existing_papers:
                self._add_fingerprint(p)

    def _add_fingerprint(self, paper: Paper):
        """Add a paper's fingerprints to the seen set."""
        fp = paper.fingerprint()
        self._seen.add(fp)
        # Also add title-based fingerprint for robustness
        nt = paper.normalized_title()
        self._title_map[nt] = fp
        # Add truncated variants
        for trunc in [80, 60, 50, 40]:
            if len(nt) > trunc:
                self._title_map[nt[:trunc]] = fp

    def is_new(self, paper: Paper) -> bool:
        """Check if paper is not yet seen."""
        fp = paper.fingerprint()
        if fp in self._seen:
            return False
        # Title-based check
        nt = paper.normalized_title()
        if nt in self._title_map:
            return False
        for trunc in [80, 60, 50, 40]:
            if len(nt) > trunc and nt[:trunc] in self._title_map:
                return False
        return True

    def add(self, paper: Paper):
        """Mark a paper as seen."""
        self._add_fingerprint(paper)

    def dedup(self, papers: List[Paper]) -> List[Paper]:
        """Filter out duplicates from a list of papers."""
        unique = []
        for p in papers:
            if self.is_new(p):
                unique.append(p)
                self.add(p)
            else:
                logger.debug("Duplicate skipped: %s", p.title[:60])
        logger.info("Dedup: %d/%d unique", len(unique), len(papers))
        return unique

    def stats(self) -> dict:
        return {"seen_fingerprints": len(self._seen), "seen_titles": len(self._title_map)}
