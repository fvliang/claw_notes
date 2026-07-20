"""LLM serving relevance filter."""
import logging
from .models import Paper
from .config import SERVING_KEYWORDS, TRAINING_EXCLUSIONS, LLM_TERMS, STRONG_SIGNALS

logger = logging.getLogger("clawbot.filter")


def is_llm_serving(paper: Paper) -> bool:
    """Determine if a paper is about LLM serving/inference."""
    text = (paper.title + " " + paper.abstract_en).lower()

    # Must have LLM-related term
    has_llm = any(term in text for term in LLM_TERMS)
    if not has_llm:
        return False

    # Strong signal: direct match on high-confidence terms
    has_strong = any(sig in text for sig in STRONG_SIGNALS)
    if has_strong:
        return True

    # Score serving vs non-serving keywords
    serving_score = sum(1 for k in SERVING_KEYWORDS if k in text)
    exclusion_score = sum(1 for k in TRAINING_EXCLUSIONS if k in text)

    # High serving score with low exclusion -> accept
    if serving_score >= 3 and serving_score > exclusion_score * 2:
        return True

    # Moderate serving score, no exclusions -> accept
    if serving_score >= 2 and exclusion_score == 0:
        return True

    # Borderline: need strong signal or high serving score
    if serving_score >= 1 and exclusion_score == 0:
        # One more check: does it mention inference/serving prominently?
        inference_terms = [
            "inference system", "inference framework", "inference engine",
            "serving system", "serving framework", "deployment",
            "latency", "throughput", "gpu utilization", "memory optimization",
        ]
        return any(t in text for t in inference_terms)

    return False


def filter_papers(papers: list[Paper]) -> list[Paper]:
    """Filter a list of papers for LLM serving relevance."""
    accepted = []
    for p in papers:
        if is_llm_serving(p):
            accepted.append(p)
        else:
            logger.debug("Rejected: %s", p.title[:60])
    logger.info("Filter: %d/%d accepted", len(accepted), len(papers))
    return accepted
