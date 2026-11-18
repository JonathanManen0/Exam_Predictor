#!/usr/bin/env python3
"""Build an auditable reference-material analysis for an exam unit.

The script accepts a curriculum cluster file and a reference source, extracts
page/section evidence, optionally compares it with slide evidence, and writes:

  extracted_evidence.jsonl
  cluster_comparison.csv
  reading_map.md
  review_queue.csv

Docling is used when available for structured PDF blocks; ``pypdf`` remains a
fallback. JSON, JSONL, Markdown, and plain text work without Docling.

Example:
    python reference_analysis.py \
        --clusters clusters.json \
        --reference physiology.pdf \
        --slides slide_evidence.jsonl \
        --output-dir analysis
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_WEIGHTS = {
    "summary_or_key_point": 1.5,
    "review_question": 1.5,
    "table_or_algorithm": 1.2,
    "repeated_heading": 0.8,
    "bold_or_emphasized_term": 0.5,
    "body_text": 0.2,
}

SUMMARY_TERMS = re.compile(
    r"\b(summary|key points?|take[- ]?home|learning objectives?|objectives?)\b",
    re.IGNORECASE,
)
REVIEW_TERMS = re.compile(
    r"\b(review questions?|self[- ]?assessment|questions for review|test yourself)\b",
    re.IGNORECASE,
)
EMPHASIS_TERMS = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__|<strong>.*?</strong>", re.I)


@dataclass
class Cluster:
    cluster_id: str
    name: str
    parent_id: str = ""
    keywords: tuple[str, ...] = ()


@dataclass
class Evidence:
    source_id: str
    source_path: str
    page: int
    section: str
    block_type: str
    cluster_id: str
    evidence_type: str
    content: str
    importance_signal: str
    confidence: float
    matched_terms: str = ""


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    value = json.loads(line)
                    if isinstance(value, list):
                        records.extend(value)
                    else:
                        records.append(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get("clusters", value.get("records", [value]))
    raise ValueError(f"{path} must contain a JSON object or array")


def load_clusters(path: Path) -> list[Cluster]:
    clusters = []
    for item in load_json_records(path):
        cluster_id = str(item.get("cluster_id") or item.get("id") or "").strip()
        name = str(item.get("name") or item.get("title") or cluster_id).strip()
        if not cluster_id:
            raise ValueError(f"{path}: every cluster needs an id or cluster_id")
        keywords = item.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                name=name,
                parent_id=str(item.get("parent_id", "")),
                keywords=tuple(str(word).lower() for word in keywords),
            )
        )
    if not clusters:
        raise ValueError(f"{path} contains no clusters")
    return clusters


def extract_pages(path: Path) -> list[tuple[int, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF input requires pypdf. Install it with: pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    text = path.read_text(encoding="utf-8")
    if suffix in {".json", ".jsonl"}:
        records = load_json_records(path)
        pages = []
        for index, record in enumerate(records, 1):
            pages.append((int(record.get("page", index)), str(record.get("text", record.get("content", "")))))
        return pages
    return [(number, chunk) for number, chunk in enumerate(re.split(r"\f", text), 1)]


def extract_docling_blocks(path: Path) -> list[dict[str, Any]]:
    """Extract page-aware blocks through Docling without requiring its schema."""
    from docling.document_converter import DocumentConverter

    document = DocumentConverter().convert(str(path)).document
    blocks: list[dict[str, Any]] = []
    for item, _level in document.iterate_items():
        text = clean_text(str(getattr(item, "text", "") or ""))
        if not text:
            continue
        page = 1
        provenance = getattr(item, "prov", None) or []
        if provenance:
            page = int(getattr(provenance[0], "page_no", 1) or 1)
        label = str(getattr(item, "label", "paragraph") or "paragraph").lower()
        block_type = {
            "title": "heading",
            "section_header": "heading",
            "list_item": "list",
            "table": "table",
            "caption": "caption",
        }.get(label, "paragraph")
        blocks.append({"page": page, "section": "", "block_type": block_type, "text": text})
    return blocks


def extract_blocks(path: Path, extractor: str) -> tuple[list[dict[str, Any]], str]:
    if extractor in {"auto", "docling"} and path.suffix.lower() == ".pdf":
        try:
            return extract_docling_blocks(path), "docling"
        except ImportError:
            if extractor == "docling":
                raise RuntimeError("Docling is not installed; use --extractor auto or install docling")
        except Exception as exc:
            if extractor == "docling":
                raise RuntimeError(f"Docling extraction failed: {exc}") from exc

    blocks: list[dict[str, Any]] = []
    current_section = ""
    for page, raw_text in extract_pages(path):
        if not raw_text.strip():
            continue
        current_section = section_for(raw_text, current_section)
        chunks = [chunk for chunk in re.split(r"\n\s*\n+", raw_text) if chunk.strip()]
        if not chunks:
            chunks = [raw_text]
        for chunk in chunks:
            blocks.append(
                {
                    "page": page,
                    "section": current_section,
                    "block_type": "paragraph",
                    "text": clean_text(chunk),
                }
            )
    return blocks, "pypdf"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def section_for(text: str, current: str) -> str:
    headings = re.findall(r"(?m)^\s{0,3}(?:#{1,6}\s+|[A-Z][A-Z0-9][A-Z0-9 :,&/-]{4,}$)(.+)?$", text)
    if headings:
        candidate = clean_text(headings[-1] or "")
        if candidate:
            return candidate[:200]
    return current or "Unsectioned"


def classify_evidence(text: str) -> list[tuple[str, str, float]]:
    clean = clean_text(text)
    found: list[tuple[str, str, float]] = []
    if SUMMARY_TERMS.search(clean):
        found.append(("summary_or_key_point", "high", 0.90))
    if REVIEW_TERMS.search(clean) or re.search(r"(?m)^\s*(?:\d+[.)]|[-*])\s+.+\?$", text):
        found.append(("review_question", "high", 0.88))
    if re.search(r"(?m)^\s*\|.+\|", text) or re.search(
        r"\b(algorithm|flowchart|table|pathway|sequence)\b", clean, re.I
    ):
        found.append(("table_or_algorithm", "medium", 0.84))
    if EMPHASIS_TERMS.search(text):
        found.append(("bold_or_emphasized_term", "medium", 0.80))
    if not found:
        found.append(("body_text", "low", 0.65))
    return found


ALIASES = {
    "bayes theorem": ("bayes theorem", "bayes rule", "inverse probability", "posterior probability"),
    "decision trees": ("decision tree", "tree diagram", "branch probabilities", "sequential outcomes"),
    "multiplication rule": ("multiplication rule", "product rule", "successive selections"),
    "joint probability": ("joint probability", "intersection probability"),
    "marginal probability": ("marginal probability", "row total", "column total"),
    "conditional probability": ("conditional probability", "given that", "conditioned on"),
    "addition rule": ("addition rule", "union probability"),
    "independence": ("independence", "independent events", "statistically independent"),
    "simple probability": ("simple probability", "sample space", "simple event", "compound event"),
}
GENERIC_TERMS = {"probability", "rule", "theorem", "event", "events"}


def assign_clusters(text: str, clusters: Iterable[Cluster], limit: int = 3) -> list[tuple[str, float, str]]:
    normalized = clean_text(text).lower()
    words = set(re.findall(r"[a-z0-9][a-z0-9/-]+", normalized))
    scored: list[tuple[float, Cluster, list[str]]] = []
    for cluster in clusters:
        phrases = {
            phrase
            for phrase in (set(cluster.keywords) | {cluster.name.lower()})
            if phrase not in GENERIC_TERMS
        }
        phrases |= set(ALIASES.get(cluster.name.lower(), ()))
        matched = [
            phrase
            for phrase in phrases
            if (
                phrase in normalized
                if " " in phrase
                else phrase in words
            )
        ]
        if not matched:
            continue
        score = min(1.0, 0.40 + 0.15 * len(matched) + 0.05 * max(map(len, matched)))
        scored.append((score, cluster, sorted(matched)))
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return [
        (cluster.cluster_id, round(score, 2), ", ".join(matched))
        for score, cluster, matched in scored[:limit]
        if score >= 0.45
    ]


def make_evidence(
    source: Path, clusters: list[Cluster], source_id: str, extractor: str
) -> tuple[list[Evidence], list[dict[str, Any]], str]:
    evidence: list[Evidence] = []
    review: list[dict[str, Any]] = []
    blocks, backend = extract_blocks(source, extractor)
    for block_number, block in enumerate(blocks, 1):
        text = block["text"]
        matches = assign_clusters(text, clusters)
        if not matches:
            review.append({"block": block_number, "page": block["page"], "reason": "no cluster match", "text": text[:500]})
            continue
        for cluster_id, assignment_confidence, matched_terms in matches:
            for evidence_type, signal, confidence in classify_evidence(text):
                evidence.append(
                    Evidence(
                        source_id=source_id,
                        source_path=str(source),
                        page=block["page"],
                        section=block.get("section", ""),
                        block_type=block.get("block_type", "paragraph"),
                        cluster_id=cluster_id,
                        evidence_type=evidence_type,
                        content=text[:2000],
                        importance_signal=signal,
                        confidence=round(min(confidence, assignment_confidence), 2),
                        matched_terms=matched_terms,
                    )
                )
    return evidence, review, backend


def load_slide_clusters(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for record in load_json_records(path):
        cluster_id = str(record.get("cluster_id", "")).strip()
        if not cluster_id:
            continue
        totals[cluster_id]["slide_count"] += float(record.get("slide_count", 1))
        totals[cluster_id]["objective_mentions"] += float(record.get("objective_mentions", 0))
        totals[cluster_id]["summary_slides"] += float(record.get("summary_slides", 0))
    return {key: dict(value) for key, value in totals.items()}


def build_comparison(
    clusters: list[Cluster], evidence: list[Evidence], slide_data: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidence:
        if item.cluster_id:
            grouped[item.cluster_id].append(item)

    rows = []
    for cluster in clusters:
        items = grouped[cluster.cluster_id]
        unique_blocks: dict[tuple[int, str, str], Evidence] = {}
        for item in items:
            key = (item.page, item.content, item.block_type)
            current = unique_blocks.get(key)
            if current is None or EVIDENCE_WEIGHTS.get(item.evidence_type, 0.0) > EVIDENCE_WEIGHTS.get(
                current.evidence_type, 0.0
            ):
                unique_blocks[key] = item
        score = sum(
            EVIDENCE_WEIGHTS.get(item.evidence_type, 0.0)
            for item in unique_blocks.values()
        )
        pages = sorted({item.page for item in unique_blocks.values()})
        types = sorted({item.evidence_type for item in unique_blocks.values()})
        slides = slide_data.get(cluster.cluster_id, {})
        compared_with_slides = bool(slide_data)
        if slides.get("slide_count", 0) > 0 and items:
            relation = "selected"
        elif slides.get("slide_count", 0) > 0:
            relation = "mentioned"
        elif items:
            relation = "omitted_from_slides" if compared_with_slides else "reference_only"
        else:
            relation = "not_found" if compared_with_slides else "not_found_in_reference"
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.name,
                "parent_id": cluster.parent_id,
                "reference_score": round(score, 2),
                "reference_pages": ",".join(map(str, pages)),
                "evidence_types": ",".join(types),
                "reference_evidence_count": len(unique_blocks),
                "reference_record_count": len(items),
                "slide_count": int(slides.get("slide_count", 0)),
                "objective_mentions": int(slides.get("objective_mentions", 0)),
                "relation_to_slides": relation,
            }
        )
    return sorted(rows, key=lambda row: (-row["reference_score"], row["cluster_name"]))


def write_outputs(
    output_dir: Path,
    evidence: list[Evidence],
    rows: list[dict[str, Any]],
    review: list[dict[str, Any]],
    backend: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "extracted_evidence.jsonl").open("w", encoding="utf-8") as handle:
        for item in evidence:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    fields = list(rows[0]) if rows else [
        "cluster_id", "cluster_name", "parent_id", "reference_score",
        "reference_pages", "evidence_types", "reference_evidence_count",
        "slide_count", "objective_mentions", "relation_to_slides",
    ]
    with (output_dir / "cluster_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["cluster_id", "cluster_name", "reason", "reference_pages"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["relation_to_slides"] in {"omitted_from_slides", "not_found"}:
                writer.writerow(
                    {
                        "cluster_id": row["cluster_id"],
                        "cluster_name": row["cluster_name"],
                        "reason": "Verify cluster assignment and lecturer treatment manually",
                        "reference_pages": row["reference_pages"],
                    }
                )
        for item in review:
            writer.writerow(
                {
                    "cluster_id": "",
                    "cluster_name": "",
                    "reason": f"Block {item['block']}: {item['reason']}",
                    "reference_pages": item["page"],
                }
            )

    lines = [
        "# Reading map",
        "",
        f"Extraction backend: **{backend}**",
        "Evidence is assigned at block level and may map to multiple clusters.",
        "",
    ]
    for rank, row in enumerate(rows, 1):
        pages = row["reference_pages"] or "No reference pages found"
        lines.extend(
            [
                f"## {rank}. {row['cluster_name']} (`{row['cluster_id']}`)",
                f"- **Priority score:** {row['reference_score']}",
                f"- **Pages:** {pages}",
                f"- **Evidence:** {row['evidence_types'] or 'none'}",
                f"- **Slide relationship:** {row['relation_to_slides']}",
                "",
            ]
        )
    (output_dir / "reading_map.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "extraction_summary.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "evidence_records": len(evidence),
                "unmatched_blocks": len(review),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, required=True, help="JSON file containing cluster records")
    parser.add_argument("--reference", type=Path, required=True, help="PDF, text, Markdown, JSON, or JSONL source")
    parser.add_argument("--slides", type=Path, help="Optional JSON/JSONL slide-cluster evidence")
    parser.add_argument("--output-dir", type=Path, default=Path("reference_analysis"))
    parser.add_argument("--source-id", help="Stable source identifier; defaults to the filename and hash")
    parser.add_argument(
        "--extractor",
        choices=("auto", "docling", "pypdf"),
        default="auto",
        help="PDF extraction backend; auto uses Docling when installed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        clusters = load_clusters(args.clusters)
        source_hash = hashlib.sha256(args.reference.read_bytes()).hexdigest()[:12]
        source_id = args.source_id or f"{args.reference.stem}-{source_hash}"
        evidence, review, backend = make_evidence(
            args.reference, clusters, source_id, args.extractor
        )
        slide_data = load_slide_clusters(args.slides)
        rows = build_comparison(clusters, evidence, slide_data)
        write_outputs(args.output_dir, evidence, rows, review, backend)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote analysis to {args.output_dir}")
    print(f"Evidence records: {len(evidence)}")
    print(f"Clusters: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
