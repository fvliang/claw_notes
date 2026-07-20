#!/usr/bin/env python3
"""Daily paper search and update script.

Usage:
    python scripts/daily_search.py [--web] [--index] [--verbose]
"""
import argparse
import logging
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clawbot.pipeline import run_full_pipeline
from clawbot.config import SEARCH_QUERIES


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=[
        logging.StreamHandler(sys.stdout)
    ])


def main():
    parser = argparse.ArgumentParser(description="Daily LLM serving paper search")
    parser.add_argument("--web", action="store_true", default=True, help="Generate web page")
    parser.add_argument("--no-web", dest="web", action="store_false", help="Skip web generation")
    parser.add_argument("--index", action="store_true", default=True, help="Generate markdown index")
    parser.add_argument("--no-index", dest="index", action="store_false", help="Skip index generation")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Search only, don't save")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger("daily_search")

    logger.info("=" * 60)
    logger.info("clawbot daily search - %s", __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"))
    logger.info("=" * 60)

    categories = ["cs.DC", "cs.CL", "cs.AR", "cs.LG", "cs.OS"]

    try:
        result = run_full_pipeline(
            queries=SEARCH_QUERIES,
            categories=categories,
            generate_web=args.web,
            generate_index=args.index,
        )
        logger.info("=" * 60)
        logger.info("Pipeline complete: %d new, %d total", result["new_papers"], result["total_papers"])
        if result["new_papers"] > 0:
            for t in result["new_titles"]:
                logger.info("  + %s", t[:80])
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
