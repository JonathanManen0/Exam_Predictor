#!/usr/bin/env python3
"""Map extracted lecture slides and transcript segments to mature clusters.

Inputs are intentionally flexible:

* clusters.json: the output of ``curriculum_clusters.py``
* slides.json: a list or {"slides": [...]} containing slide text and timings
* transcript.jsonl: timestamped records from ``lecture_transcription.py``

Each slide may provide ``start``/``end`` seconds, ``start_time``/``end_time``,
or a ``start``/``end`` timestamp in HH:MM:SS format. If no slide timings are
available, slides are still mapped independently and transcript evidence is
reported separately.

Outputs:

    slide_cluster_matches.jsonl
    cluster_evidence.csv
    mapping_review_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


STOPWORDS = {
    "about", "after", "also", "and", "are", "being", "from", "have", "into",
    "more", "most", "other", "such", "than", "that", "their", "these",
    "this", "those", "through", "with", "within", "without", "which",
}

CONCEPT_ALIASES = {
    "simple probability": (
        "simple probability", "sample space", "simple event", "compound event",
        "equally likely", "experiment", "outcome",
    ),
    "joint probability": (
        "joint probability", "intersection probability", "and probability",
        "two-way table",
    ),
    "marginal probability": (
        "marginal probability", "row total", "column total", "single event",
    ),
    "addition rule": (
        "addition rule", "union probability", "mutually exclusive",
    ),
    "conditional probability": (
        "conditional probability", "given that", "conditioned on",
    ),
    "decision trees": (
        "decision tree", "tree diagram", "branch probabilities",
        "sequential outcomes",
    ),
    "independence": (
        "independence", "independent events", "statistically independent",
    ),
    "multiplication rule": (
        "multiplication rule", "product rule", "successive selections",
        "multiplicative rule",
    ),
    "bayes theorem": (
        "bayes theorem", "bayes' theorem", "bayes rule", "inverse probability",
        "posterior probability",
    ),
}
GENERIC_TERMS = {"probability", "rule", "theorem", "event", "events"}


@dataclass
class Cluster:
    cluster_id: str
    name: str
    keywords: tuple[str, ...]
    course_code: str
    parent_id: str


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if isinstance(value, dict):
                    records.append(value)
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("clusters", "slides", "records", "segments"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
        return [value]
    raise ValueError(f"{path}: expected an object, array, or JSONL records")


def parse_time(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?", text)
    if not match:
        return None
    hours, minutes, seconds, fraction = match.groups()
    return (
        (int(hours or 0) * 3600)
        + int(minutes) * 60
        + int(seconds)
        + float(f"0.{fraction}" if fraction else 0)
    )


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9/-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def record_text(record: dict[str, Any]) -> str:
    values = [
        record.get("title", ""),
        record.get("text", ""),
        record.get("notes", ""),
        record.get("flowchart_desc", ""),
        record.get("table_desc", ""),
        record.get("table_text", ""),
    ]
    for key in ("bullets", "key_terms", "terms"):
        value = record.get(key, [])
        if isinstance(value, list):
            values.extend(str(item) for item in value)
    return " ".join(str(value) for value in values if value)


def slide_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_tokens = set(re.findall(r"[a-z]{3,}", record_text(left).lower()))
    right_tokens = set(re.findall(r"[a-z]{3,}", record_text(right).lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    jaccard = overlap / max(1, len(left_tokens | right_tokens))
    containment = overlap / max(1, min(len(left_tokens), len(right_tokens)))
    return max(jaccard, containment)


def coalesce_slides(slides: list[dict[str, Any]], threshold: float = 0.88) -> list[dict[str, Any]]:
    """Collapse repeated time samples of the same visible slide."""
    coalesced: list[dict[str, Any]] = []
    for slide in slides:
        if coalesced and slide_similarity(coalesced[-1], slide) >= threshold:
            previous = coalesced[-1]
            _, end = timing(slide)
            if end is not None:
                previous["end"] = end
            if len(record_text(slide)) > len(record_text(previous)):
                previous["text"] = slide.get("text", previous.get("text", ""))
            continue
        coalesced.append(dict(slide))
    return coalesced


def timing(record: dict[str, Any]) -> tuple[float | None, float | None]:
    start = next(
        (parse_time(record.get(key)) for key in ("start", "start_time", "from", "timestamp_start")
         if parse_time(record.get(key)) is not None),
        None,
    )
    end = next(
        (parse_time(record.get(key)) for key in ("end", "end_time", "to", "timestamp_end")
         if parse_time(record.get(key)) is not None),
        None,
    )
    return start, end


def load_clusters(path: Path) -> list[Cluster]:
    clusters = []
    for record in load_records(path):
        if record.get("level") not in (None, "mature_cluster", "topic"):
            continue
        cluster_id = str(record.get("cluster_id", "")).strip()
        name = str(record.get("name", "")).strip()
        if not cluster_id or not name:
            continue
        keyword_values = record.get("keywords", [])
        if isinstance(keyword_values, str):
            keyword_values = [keyword_values]
        clusters.append(
            Cluster(
                cluster_id=cluster_id,
                name=name,
                keywords=tuple(str(value).lower() for value in keyword_values),
                course_code=str(record.get("course_code", "")),
                parent_id=str(record.get("parent_id", "")),
            )
        )
    if not clusters:
        raise ValueError(f"{path} contains no usable mature clusters")
    return clusters


def score_match(text: str, cluster: Cluster) -> tuple[float, str]:
    normalized = re.sub(r"\s+", " ", text.lower())
    evidence_tokens = tokens(normalized)
    cluster_name = cluster.name.lower()
    phrases = set(CONCEPT_ALIASES.get(cluster_name, (cluster_name,)))
    phrases.update(
        keyword.lower() for keyword in cluster.keywords
        if keyword.lower() not in GENERIC_TERMS
    )
    matched_phrases = [
        phrase for phrase in phrases
        if (phrase in normalized if " " in phrase else phrase in evidence_tokens)
    ]
    cluster_tokens = {
        token for phrase in phrases for token in tokens(phrase)
    }
    if not evidence_tokens or not cluster_tokens or not matched_phrases:
        return 0.0, "no tokens"
    overlap = evidence_tokens & cluster_tokens
    overlap_score = len(overlap) / max(1, len(cluster_tokens))
    phrase_score = max(
        SequenceMatcher(None, phrase, normalized).ratio()
        for phrase in matched_phrases
    )
    exact_name = cluster_name in normalized
    alias_bonus = min(0.35, 0.12 * len(matched_phrases))
    score = min(
        1.0,
        overlap_score * 0.35
        + phrase_score * 0.15
        + alias_bonus
        + (0.30 if exact_name else 0.0),
    )
    evidence = f"matched: {', '.join(sorted(matched_phrases))}"
    if exact_name:
        evidence += "; exact cluster name"
    return round(score, 4), evidence


def top_matches(text: str, clusters: list[Cluster], limit: int, threshold: float) -> list[dict[str, Any]]:
    scored = []
    for cluster in clusters:
        score, evidence = score_match(text, cluster)
        if score >= threshold:
            scored.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.name,
                    "score": score,
                    "evidence": evidence,
                }
            )
    return sorted(scored, key=lambda item: (-item["score"], item["cluster_name"]))[:limit]


def transcript_for_slide(
    slide_start: float | None,
    slide_end: float | None,
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if slide_start is None:
        return []
    end = slide_end if slide_end is not None else float("inf")
    selected = []
    for segment in transcript:
        start, finish = timing(segment)
        if start is None:
            continue
        finish = finish if finish is not None else start
        # Some whisper.cpp runs emit a startup artifact spanning the entire
        # recording. It is not a real speech segment and would contaminate
        # every slide's transcript window.
        if finish - start > 120:
            continue
        if finish >= slide_start and start <= end:
            selected.append(segment)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("lecture_mapping"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.16)
    args = parser.parse_args()

    try:
        clusters = load_clusters(args.clusters)
        slides = load_records(args.slides)
        slides = coalesce_slides(slides)
        transcript = load_records(args.transcript) if args.transcript else []
        if not slides:
            raise ValueError("No slides found")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    matches: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "slide_count": 0,
            "weighted_slide_evidence": 0.0,
            "transcript_seconds": 0.0,
            "weighted_transcript_seconds": 0.0,
            "match_count": 0,
            "max_score": 0.0,
            "scores": [],
            "sources": set(),
        }
    )
    review: list[dict[str, Any]] = []

    for index, slide in enumerate(slides, 1):
        start, end = timing(slide)
        associated = transcript_for_slide(start, end, transcript)
        speech = " ".join(str(item.get("text", "")) for item in associated)
        combined = f"{record_text(slide)} {speech}".strip()
        ranked = top_matches(combined, clusters, args.top_k, args.threshold)
        slide_id = str(slide.get("key") or slide.get("slide_id") or f"slide-{index:04d}")
        record = {
            "slide_id": slide_id,
            "slide_number": slide.get("slide", index),
            "source": slide.get("file", slide.get("source_path", "")),
            "start": start,
            "end": end,
            "slide_text": record_text(slide),
            "transcript_segment_ids": [
                item.get("segment_id", position) for position, item in enumerate(associated, 1)
            ],
            "transcript_text": speech,
            "matches": ranked,
            "needs_review": not ranked or ranked[0]["score"] < 0.25,
        }
        matches.append(record)
        if record["needs_review"]:
            review.append(
                {
                    "slide_id": slide_id,
                    "reason": "no confident cluster match" if not ranked else "low-confidence top match",
                    "top_score": ranked[0]["score"] if ranked else 0,
                    "text": combined[:500],
                }
            )
        for item in ranked:
            stats = aggregate[item["cluster_id"]]
            stats["slide_count"] += 1
            stats["weighted_slide_evidence"] += item["score"]
            stats["match_count"] += 1
            stats["max_score"] = max(stats["max_score"], item["score"])
            stats["scores"].append(item["score"])
            stats["sources"].add("slide")
        if associated and start is not None:
            speech_end = end if end is not None else start
            for item in ranked:
                aggregate[item["cluster_id"]]["transcript_seconds"] += max(0, speech_end - start)
                aggregate[item["cluster_id"]]["weighted_transcript_seconds"] += (
                    max(0, speech_end - start) * item["score"]
                )
                aggregate[item["cluster_id"]]["sources"].add("transcript")

    with (args.output_dir / "slide_cluster_matches.jsonl").open("w", encoding="utf-8") as handle:
        for record in matches:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    cluster_by_id = {cluster.cluster_id: cluster for cluster in clusters}
    rows = []
    for cluster in clusters:
        stats = aggregate[cluster.cluster_id]
        scores = stats["scores"]
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "cluster_name": cluster.name,
                "course_code": cluster.course_code,
                "slide_count": stats["slide_count"],
                "weighted_slide_evidence": round(stats["weighted_slide_evidence"], 4),
                "transcript_seconds": round(stats["transcript_seconds"], 3),
                "weighted_transcript_seconds": round(
                    stats["weighted_transcript_seconds"], 3
                ),
                "match_count": stats["match_count"],
                "max_match_score": round(stats["max_score"], 4),
                "mean_match_score": round(sum(scores) / len(scores), 4) if scores else 0,
                "evidence_sources": ",".join(sorted(stats["sources"])),
            }
        )
    rows.sort(key=lambda row: (-row["max_match_score"], row["cluster_name"]))
    with (args.output_dir / "cluster_evidence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (args.output_dir / "mapping_review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["slide_id", "reason", "top_score", "text"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review)

    print(f"Mapped {len(matches)} slides to {len(clusters)} clusters")
    print(f"Manual review queue: {len(review)} slides")
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
