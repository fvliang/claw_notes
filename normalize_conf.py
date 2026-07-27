#!/usr/bin/env python3
"""Normalize conference names: extract base name and year."""

import json
import re
from pathlib import Path

DB_PATH = Path("database.json")

# Normalization rules: (pattern, base_name)
# Order matters - more specific first
CONF_PATTERNS = [
    # arXiv variants
    (r'^ar[xX]iv\s*预印本$', 'arXiv'),
    (r'^ar[xX]iv$', 'arXiv'),
    (r'^ar[xX]iv\s+\(?(\d{4})\)?$', 'arXiv'),  # arXiv 2024, arXiv (2024)
    # ACL variants
    (r'^ACL\s+Findings\s+(\d{4})$', 'ACL Findings'),
    (r'^ACL\s+(\d{4})$', 'ACL'),
    (r'^ACL$', 'ACL'),
    # Other conferences with years
    (r'^ASPLOS\s*\'?(\d{2,4})', 'ASPLOS'),  # ASPLOS 2026, ASPLOS '26
    (r'^ASPLOS$', 'ASPLOS'),
    (r'^ICML\s+(\d{4})$', 'ICML'),
    (r'^ICML$', 'ICML'),
    (r'^ICLR\s+(\d{4})$', 'ICLR'),
    (r'^ICLR$', 'ICLR'),
    (r'^NeurIPS\s+(\d{4})$', 'NeurIPS'),
    (r'^NeurIPS$', 'NeurIPS'),
    (r'^EMNLP\s+(\d{4})$', 'EMNLP'),
    (r'^EMNLP$', 'EMNLP'),
    (r'^NAACL\s+(\d{4})$', 'NAACL'),
    (r'^NAACL$', 'NAACL'),
    (r'^COLM\s+(\d{4})$', 'COLM'),
    (r'^COLM$', 'COLM'),
    (r'^MLSys\s+(\d{4})$', 'MLSys'),
    (r'^MLSys$', 'MLSys'),
    (r'^OSDI\s+(\d{4})$', 'OSDI'),
    (r'^OSDI$', 'OSDI'),
    (r'^SOSP\s+(\d{4})$', 'SOSP'),
    (r'^SOSP$', 'SOSP'),
    (r'^EuroSys\s+(\d{4})$', 'EuroSys'),
    (r'^EuroSys$', 'EuroSys'),
    (r'^AISTATS\s+(\d{4})$', 'AISTATS'),
    (r'^AISTATS$', 'AISTATS'),
    (r'^ATC$', 'ATC'),
    (r'^SC$', 'SC'),
    (r'^SIGMOD$', 'SIGMOD'),
    (r'^DAC$', 'DAC'),
    (r'^dac$', 'DAC'),
    (r'^DAC\s+(\d{4})$', 'DAC'),
    (r'^CVPR$', 'CVPR'),
    (r'^cvpr$', 'CVPR'),
    (r'^ISCA$', 'ISCA'),
    (r'^isca$', 'ISCA'),
    (r'^ISPASS$', 'ISPASS'),
    (r'^ispass$', 'ISPASS'),
    (r'^FPGA', 'FPGA'),
    # Lowercase variants
    (r'^acl$', 'ACL'),
    (r'^icml$', 'ICML'),
    (r'^euromlsys$', 'EuroMLSys'),
    # GitHub / semantic_scholar as-is
    (r'^GitHub$', 'GitHub'),
    (r'^semantic_scholar$', 'Semantic Scholar'),
    # Edge cases
    (r'^ar[xX]iv\s+\(EuroMLSys\s+(\d{4})\)$', 'EuroMLSys'),
    (r'^ACM International Conference o.*', 'ACM IC'),
    (r'^International Conference on ASIC.*', 'ASIC'),
    (r'^International Congress of Mathematicans.*', 'ICM'),
]

# Multi-conference papers that need special handling
MULTI_CONF_PATTERN = re.compile(r'^([A-Z]+)\s+(\d{4}),\s+([A-Z]+)\s+(\d{4})')


def normalize_conference(raw_conf):
    """Extract base conference name and year from raw conference string."""
    if not raw_conf:
        return 'arXiv', None

    raw = raw_conf.strip()

    # Check multi-conference first
    m = MULTI_CONF_PATTERN.match(raw)
    if m:
        # Use first conference, extract year from second if first has none
        return m.group(1), int(m.group(2))

    for pattern, base in CONF_PATTERNS:
        m = re.match(pattern, raw, re.IGNORECASE)
        if m:
            year = None
            if len(m.groups()) > 0 and m.group(1) and m.group(1).isdigit():
                y = int(m.group(1))
                if y < 100:
                    y = 2000 + y  # 26 -> 2026
                year = y
            return base, year

    # Fallback: try to extract 4-digit year from end
    m = re.search(r'(\d{4})$', raw)
    if m:
        year = int(m.group(1))
        base = re.sub(r'\s+\d{4}$', '', raw).strip()
        return base, year

    return raw, None


def normalize_database():
    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    changes = []
    for p in db["papers"]:
        raw = p.get("conference", "")
        base, year = normalize_conference(raw)

        old_conf = p.get("conference", "")
        old_year = p.get("year")

        if old_conf != base or (year and old_year != year):
            changes.append({
                "id": p["id"],
                "title": p["title"][:50],
                "old_conf": old_conf,
                "new_conf": base,
                "old_year": old_year,
                "new_year": year or old_year,
            })
            p["conference"] = base
            if year:
                p["year"] = year

    # Save
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"Normalized {len(changes)} papers")
    for c in changes[:20]:
        print(f"  [{c['id']}] {c['title']}... | {c['old_conf']} ({c['old_year']}) -> {c['new_conf']} ({c['new_year']})")
    if len(changes) > 20:
        print(f"  ... and {len(changes)-20} more")

    # Show final conference distribution
    confs = {}
    for p in db["papers"]:
        c = p.get("conference", "arXiv")
        confs[c] = confs.get(c, 0) + 1
    print("\nFinal conference distribution:")
    for c, n in sorted(confs.items(), key=lambda x: -x[1]):
        print(f"  {n:4d}  {c}")

    # Show year distribution per conference (top ones)
    print("\nYear distribution for multi-year conferences:")
    for c in ["arXiv", "ACL", "ASPLOS", "ICML", "ICLR", "NeurIPS"]:
        if c not in confs:
            continue
        years = {}
        for p in db["papers"]:
            if p.get("conference") == c:
                y = p.get("year", "?")
                years[y] = years.get(y, 0) + 1
        print(f"  {c}: {dict(sorted(years.items()))}")


if __name__ == "__main__":
    normalize_database()
