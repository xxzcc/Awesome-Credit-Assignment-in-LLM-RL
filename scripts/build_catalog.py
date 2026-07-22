#!/usr/bin/env python3
"""Validate the paper catalog and export its CSV representation."""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
JSON_PATH = ROOT / "data" / "papers.json"
CSV_PATH = ROOT / "data" / "papers.csv"

CSV_FIELDS = [
    "id",
    "title",
    "year",
    "collection",
    "update_batch",
    "scope",
    "setting",
    "category",
    "granularity",
    "methodology",
    "paper_url",
    "source",
    "code_url",
    "code_status",
    "venue",
    "publication_status",
    "description",
]

REQUIRED_TEXT_FIELDS = {
    "id",
    "title",
    "collection",
    "update_batch",
    "scope",
    "setting",
    "category",
    "granularity",
    "methodology",
    "paper_url",
    "source",
    "code_status",
    "publication_status",
    "description",
}

README_PAPER_LINK_PATTERN = re.compile(
    r"\[\[(?:Paper|OpenReview)\]\]\((?P<url>https://[^)]+)\)"
)
ALLOWED_PAPER_HOSTS = {
    "aclanthology.org",
    "arxiv.org",
    "openreview.net",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
}


def load_catalog() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    papers = document.get("papers")
    if not isinstance(papers, list):
        raise ValueError("data/papers.json must contain a papers array")
    return document, papers


def readme_catalog_urls() -> set[str]:
    readme = README_PATH.read_text(encoding="utf-8")
    recent = readme[
        readme.index("## Recent Additions") : readme.index("## Foundational & Background")
    ]
    methods = readme[
        readme.index("## Credit Assignment in Reasoning RL") : readme.index(
            "## Benchmarks"
        )
    ]
    return set(README_PAPER_LINK_PATTERN.findall(recent + methods))


def valid_paper_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in ALLOWED_PAPER_HOSTS


def validate_catalog(document: dict[str, Any], papers: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    ids: list[str] = []
    urls: list[str] = []

    if document.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if document.get("entry_count") != len(papers):
        errors.append(
            f"entry_count is {document.get('entry_count')}, but papers has {len(papers)} entries"
        )

    for index, paper in enumerate(papers, start=1):
        label = paper.get("id", f"row-{index}")
        missing = sorted(
            field for field in REQUIRED_TEXT_FIELDS if not str(paper.get(field, "")).strip()
        )
        if missing:
            errors.append(f"{label}: missing required fields: {', '.join(missing)}")

        if not isinstance(paper.get("year"), int):
            errors.append(f"{label}: year must be an integer")

        paper_url = str(paper.get("paper_url", ""))
        if not valid_paper_url(paper_url):
            errors.append(f"{label}: unsupported paper_url: {paper_url}")

        ids.append(str(paper.get("id", "")))
        urls.append(paper_url)

    duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
    duplicate_urls = sorted({value for value in urls if urls.count(value) > 1})
    if duplicate_ids:
        errors.append(f"duplicate IDs: {', '.join(duplicate_ids)}")
    if duplicate_urls:
        errors.append(f"duplicate paper URLs: {', '.join(duplicate_urls)}")

    catalog_urls = set(urls)
    readme_urls = readme_catalog_urls()
    missing_from_readme = sorted(catalog_urls - readme_urls)
    missing_from_catalog = sorted(readme_urls - catalog_urls)
    if missing_from_readme:
        errors.append("catalog URLs missing from README: " + ", ".join(missing_from_readme))
    if missing_from_catalog:
        errors.append("README URLs missing from catalog: " + ", ".join(missing_from_catalog))

    if errors:
        raise ValueError("Catalog validation failed:\n- " + "\n- ".join(errors))


def render_csv(papers: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=CSV_FIELDS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(papers)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the catalog and verify that data/papers.csv is current.",
    )
    args = parser.parse_args()

    try:
        document, papers = load_catalog()
        validate_catalog(document, papers)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    csv_output = render_csv(papers)
    if args.check:
        if not CSV_PATH.exists() or CSV_PATH.read_text(encoding="utf-8") != csv_output:
            print("data/papers.csv is stale; run python3 scripts/build_catalog.py", file=sys.stderr)
            return 1
        print(f"Catalog is valid and current: {len(papers)} entries")
        return 0

    CSV_PATH.write_text(csv_output, encoding="utf-8")
    print(f"Validated {len(papers)} entries and wrote data/papers.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
