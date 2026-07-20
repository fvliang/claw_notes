#!/usr/bin/env python3
"""Database CLI for manual operations.

Usage:
    python scripts/db_cli.py list [--topic TOPIC]
    python scripts/db_cli.py search KEYWORD
    python scripts/db_cli.py stats
    python scripts/db_cli.py dedup
    python scripts/db_cli.py migrate
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from clawbot.database import PaperDatabase
from clawbot.models import Paper


def cmd_list(args):
    db = PaperDatabase()
    papers = db.all_papers()
    if args.topic:
        papers = [p for p in papers if p.topic == args.topic]
    print(f"\n📚 {len(papers)} papers\n")
    for p in papers[-args.limit:]:
        print(f"[{p.id}] {p.title}")
        print(f"    📍 {p.conference or p.source} {p.year} | 🏷️ {p.topic}")
        if p.arxiv_id:
            print(f"    🔗 arxiv.org/abs/{p.arxiv_id}")
        print()


def cmd_search(args):
    db = PaperDatabase()
    results = db.search(args.keyword)
    print(f"\n🔍 Found {len(results)} papers matching '{args.keyword}'\n")
    for p in results:
        print(f"[{p.id}] {p.title}")


def cmd_stats(args):
    db = PaperDatabase()
    stats = db.stats()
    print(f"\n📊 Database Statistics\n")
    print(f"Total papers: {stats['total']}")
    print(f"\nBy topic:")
    for topic, count in sorted(stats["by_topic"].items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")
    print(f"\nBy year:")
    for year, count in sorted(stats["by_year"].items(), reverse=True):
        print(f"  {year}: {count}")
    print(f"\nBy source:")
    for source, count in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")


def cmd_dedup(args):
    """Remove duplicate papers from database."""
    db = PaperDatabase()
    from clawbot.dedup import DedupEngine

    papers = db.all_papers()
    dedup = DedupEngine()
    unique = []
    removed = 0

    for p in papers:
        if dedup.is_new(p):
            dedup.add(p)
            unique.append(p)
        else:
            removed += 1
            print(f"  Removed duplicate: {p.title[:60]}")

    if removed > 0:
        db._papers = unique
        db.save()
        print(f"\n✅ Removed {removed} duplicates, {len(unique)} papers remaining")
    else:
        print("\n✅ No duplicates found")


def cmd_migrate(args):
    """Migrate legacy database to normalized format."""
    db = PaperDatabase()
    # Just saving will normalize all entries
    db.save()
    print(f"✅ Migrated {len(db.all_papers())} papers to normalized format")


def main():
    parser = argparse.ArgumentParser(description="Database CLI")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List papers")
    p_list.add_argument("--topic", help="Filter by topic")
    p_list.add_argument("--limit", type=int, default=20, help="Limit output")

    p_search = sub.add_parser("search", help="Search papers")
    p_search.add_argument("keyword", help="Search keyword")

    sub.add_parser("stats", help="Show statistics")
    sub.add_parser("dedup", help="Remove duplicates")
    sub.add_parser("migrate", help="Migrate to normalized format")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "dedup":
        cmd_dedup(args)
    elif args.command == "migrate":
        cmd_migrate(args)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
