#!/usr/bin/env python3
"""Convert a human-curated cluster DOCX into validated clusters.json.

Authoritative document format:

    Heading 1: MSB 201: Microbiology, Pharmacology and Immunology
    Heading 2: Microbiology
    Heading 3: [MSB201-MICRO-001] Bacterial structure and composition
    Heading 3: [MSB201-MICRO-002] Principles of bacterial genetics

Only Heading 3 entries become examinable clusters. Heading 1 courses and
Heading 2 domains are organizational metadata and are never emitted as
clusters.

Example:
    python3 manual_cluster_import.py \
        --input MSB201_clusters.docx \
        --output clusters.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any


COURSE_RE = re.compile(
    r"^\s*(?P<code>[A-Z]{2,6}\s*[-;]?\s*\d{3})\s*:\s*(?P<name>.+?)\s*$",
    re.I,
)
CLUSTER_RE = re.compile(r"^\s*\[(?P<id>[A-Za-z0-9][A-Za-z0-9_.-]*)\]\s+(?P<name>\S.+?)\s*$")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def slug_terms(text: str) -> list[str]:
    stopwords = {
        "a", "about", "after", "also", "an", "and", "are", "as", "at", "be",
        "being", "by", "for", "from", "have", "into", "is", "more", "most",
        "of", "on", "or", "other", "such", "than", "that", "the", "their",
        "these", "this", "those", "through", "to", "was", "with", "within",
        "without",
    }
    terms = re.findall(r"[a-z0-9][a-z0-9/-]{2,}", text.lower())
    return list(dict.fromkeys(term for term in terms if term not in stopwords))[:30]


def style_name(paragraph: Any) -> str:
    return str(getattr(getattr(paragraph, "style", None), "name", "") or "")


def parse_course(text: str, paragraph_number: int) -> tuple[str, str]:
    match = COURSE_RE.match(text)
    if not match:
        raise ValueError(
            f"paragraph {paragraph_number}: course heading must look like "
            f"'CODE 123: Course name'; got {text!r}"
        )
    code = re.sub(r"[\s;]+", " ", match.group("code").upper())
    return code, normalize(match.group("name"))


def import_docx(path: Path) -> list[dict[str, Any]]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "DOCX input requires python-docx. Install it with: pip install python-docx"
        ) from exc

    document = Document(str(path))
    records: list[dict[str, Any]] = []
    current_course_code = ""
    current_course_name = ""
    current_domain = ""
    seen_ids: set[str] = set()

    for paragraph_number, paragraph in enumerate(document.paragraphs, 1):
        text = normalize(paragraph.text)
        if not text:
            continue
        style = style_name(paragraph).lower()

        if style == "heading 1":
            current_course_code, current_course_name = parse_course(text, paragraph_number)
            current_domain = ""
            continue

        if style == "heading 2":
            if not current_course_code:
                raise ValueError(
                    f"paragraph {paragraph_number}: domain appears before a course heading"
                )
            current_domain = text.rstrip(":").strip()
            if not current_domain:
                raise ValueError(f"paragraph {paragraph_number}: domain cannot be empty")
            continue

        if style == "heading 3":
            if not current_course_code:
                raise ValueError(
                    f"paragraph {paragraph_number}: cluster appears before a course heading"
                )
            if not current_domain:
                raise ValueError(
                    f"paragraph {paragraph_number}: cluster appears before a domain heading"
                )
            match = CLUSTER_RE.match(text)
            if not match:
                raise ValueError(
                    f"paragraph {paragraph_number}: mature cluster must start with "
                    f"'[STABLE-ID]'; got {text!r}"
                )
            cluster_id = match.group("id")
            name = normalize(match.group("name"))
            if cluster_id in seen_ids:
                raise ValueError(
                    f"paragraph {paragraph_number}: duplicate cluster ID {cluster_id!r}"
                )
            if len(name) < 3:
                raise ValueError(f"paragraph {paragraph_number}: cluster name is too short")
            seen_ids.add(cluster_id)
            records.append(
                {
                    "cluster_id": cluster_id,
                    "name": name,
                    "parent_id": current_course_code,
                    "level": "mature_cluster",
                    "keywords": slug_terms(name),
                    "course_code": current_course_code,
                    "course_name": current_course_name,
                    "domain": current_domain,
                    "source_document": str(path),
                    "source_paragraph": paragraph_number,
                    "source": "manual_docx",
                }
            )
            continue

        raise ValueError(
            f"paragraph {paragraph_number}: unexpected non-empty paragraph style "
            f"{style_name(paragraph)!r}; use Heading 1/2/3 only"
        )

    if not records:
        raise ValueError("No mature clusters found")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Human-curated cluster DOCX")
    parser.add_argument("--output", type=Path, required=True, help="Validated JSON output")
    args = parser.parse_args()
    try:
        records = import_docx(args.input)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    courses = len({record["course_code"] for record in records})
    domains = len({(record["course_code"], record["domain"]) for record in records})
    print(f"Validated {len(records)} mature clusters from {courses} course(s) and {domains} domain(s)")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
