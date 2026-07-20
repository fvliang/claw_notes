"""arXiv API client with retry, backoff, and batch support."""
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from typing import List, Optional
import logging

from .config import (
    ARXIV_API_BASE, ARXIV_MAX_RESULTS, ARXIV_DELAY_BETWEEN_QUERIES,
    ARXIV_BATCH_SIZE, ARXIV_BATCH_DELAY, REQUEST_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
)
from .models import Paper

logger = logging.getLogger("clawbot.arxiv")


def _fetch(url: str, timeout: int = REQUEST_TIMEOUT, retries: int = MAX_RETRIES) -> str:
    """Fetch URL with retry and exponential backoff."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "clawbot/2.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                wait = RETRY_DELAY * (2 ** attempt) + 5
                logger.warning("arXiv rate limit (429), waiting %ds (attempt %d/%d)", wait, attempt + 1, retries)
                time.sleep(wait)
            elif e.code >= 500:
                wait = RETRY_DELAY * (2 ** attempt)
                logger.warning("arXiv server error %d, waiting %ds (attempt %d/%d)", e.code, wait, attempt + 1, retries)
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            last_err = e
            wait = RETRY_DELAY * (2 ** attempt)
            logger.warning("arXiv fetch error: %s, waiting %ds (attempt %d/%d)", e, wait, attempt + 1, retries)
            time.sleep(wait)
    raise last_err


def _parse_arxiv_xml(xml_data: str) -> List[Paper]:
    """Parse arXiv Atom XML into Paper objects."""
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError as e:
        logger.error("Failed to parse arXiv XML: %s", e)
        return []

    papers = []
    for entry in root.findall("atom:entry", ns):
        try:
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            id_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)

            if title_el is None or id_el is None:
                continue

            title = (title_el.text or "").strip().replace("\n", " ")
            summary = (summary_el.text or "").strip().replace("\n", " ") if summary_el is not None else ""
            arxiv_url = id_el.text.strip()
            arxiv_id = arxiv_url.replace("http://arxiv.org/abs/", "").replace("https://arxiv.org/abs/", "")
            published = (published_el.text or "")[:10] if published_el is not None else ""

            authors = [
                (a.find("atom:name", ns).text or "")
                for a in entry.findall("atom:author", ns)
                if a.find("atom:name", ns) is not None
            ]
            categories = [
                c.attrib.get("term", "")
                for c in entry.findall("atom:category", ns)
            ]
            comment_el = entry.find("arxiv:comment", ns)
            comment = (comment_el.text or "") if comment_el is not None else ""

            # Try to extract PDF link
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            paper = Paper(
                title=title,
                authors=", ".join(authors),
                arxiv_id=arxiv_id,
                abstract_en=summary,
                published=published,
                source="arXiv",
                url=arxiv_url,
                pdf_url=pdf_url,
                comment=comment,
                tags=categories,
            )
            papers.append(paper)
        except Exception as e:
            logger.warning("Failed to parse arXiv entry: %s", e)
            continue

    return papers


def search_arxiv(query: str, max_results: int = ARXIV_MAX_RESULTS, sort_by: str = "submittedDate") -> List[Paper]:
    """Search arXiv by query string."""
    encoded = urllib.parse.quote(query)
    url = (
        f"{ARXIV_API_BASE}?search_query=all:{encoded}"
        f"&sortBy={sort_by}&sortOrder=descending&max_results={max_results}"
    )
    logger.info("Searching arXiv: %s (max=%d)", query, max_results)
    xml_data = _fetch(url)
    papers = _parse_arxiv_xml(xml_data)
    logger.info("Found %d papers for query: %s", len(papers), query)
    time.sleep(ARXIV_DELAY_BETWEEN_QUERIES)
    return papers


def fetch_papers_by_ids(arxiv_ids: List[str]) -> List[Paper]:
    """Fetch paper metadata by arXiv ID list (batched)."""
    all_papers = []
    for i in range(0, len(arxiv_ids), ARXIV_BATCH_SIZE):
        batch = arxiv_ids[i : i + ARXIV_BATCH_SIZE]
        id_list = ",".join(batch)
        url = f"{ARXIV_API_BASE}?id_list={id_list}"
        logger.info("Fetching batch %d/%d (%s)", i // ARXIV_BATCH_SIZE + 1, (len(arxiv_ids) - 1) // ARXIV_BATCH_SIZE + 1, ",".join(batch))
        xml_data = _fetch(url)
        papers = _parse_arxiv_xml(xml_data)
        all_papers.extend(papers)
        if i + ARXIV_BATCH_SIZE < len(arxiv_ids):
            time.sleep(ARXIV_BATCH_DELAY)
    return all_papers


def fetch_recent_by_category(categories: List[str], days: int = 7, max_results: int = 200) -> List[Paper]:
    """Fetch recent papers from specific arXiv categories."""
    cat_query = " OR ".join(f"cat:{c}" for c in categories)
    url = (
        f"{ARXIV_API_BASE}?search_query={urllib.parse.quote(cat_query)}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    logger.info("Fetching recent papers from categories: %s", ", ".join(categories))
    xml_data = _fetch(url)
    papers = _parse_arxiv_xml(xml_data)
    logger.info("Found %d recent papers", len(papers))
    return papers
