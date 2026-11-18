#!/usr/bin/env python3
"""Generate transparent pilot cluster weights and three predictor papers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASPECTS = {
    "Simple probability": "definitions, sample spaces, events, and direct probability",
    "Joint probability": "intersection probabilities from a two-way table or experiment",
    "Marginal probability": "row/column totals and probabilities without conditioning",
    "Addition rule": "union probabilities and mutually exclusive events",
    "Conditional probability": "conditional probabilities from wording or a two-way table",
    "Decision trees": "constructing and reading a tree diagram",
    "Independence": "testing or applying independence",
    "Multiplication rule": "successive selections and product-rule calculations",
    "Bayes theorem": "reverse conditional probability and diagnostic-style reasoning",
}

QUESTION_TEMPLATES = {
    "Simple probability": "Define the core probability terminology and construct the sample space for a stated experiment.",
    "Joint probability": "Given a two-way table or paired experiment, calculate and interpret a joint probability.",
    "Marginal probability": "Use row and column totals to calculate marginal probabilities and explain what they represent.",
    "Addition rule": "Calculate the probability of a union, distinguishing mutually exclusive from overlapping events.",
    "Conditional probability": "Calculate and interpret a conditional probability from a clinical or two-way-table scenario.",
    "Decision trees": "List the outcomes of a multi-stage experiment and represent them on a tree diagram.",
    "Independence": "Determine whether two events are independent and justify the conclusion numerically.",
    "Multiplication rule": "Calculate the probability of successive selections using the multiplication rule.",
    "Bayes theorem": "Use Bayes theorem to reverse a conditional probability in a diagnostic-style problem.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_clusters(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("clusters", [])
    return [item for item in value if isinstance(item, dict)]


def rank_clusters(clusters: list[dict[str, Any]], lecture: list[dict[str, str]],
                  reference: list[dict[str, str]]) -> list[dict[str, Any]]:
    lecture_by_id = {row["cluster_id"]: row for row in lecture}
    reference_by_id = {row["cluster_id"]: row for row in reference}
    slide_counts = [
        number(row.get("weighted_slide_evidence", row.get("slide_count", "")))
        for row in lecture
    ]
    transcript = [
        number(row.get("weighted_transcript_seconds", row.get("transcript_seconds", "")))
        for row in lecture
    ]
    ref_scores = [number(row.get("reference_score", "")) for row in reference]
    max_slide = max(slide_counts) if slide_counts and max(slide_counts) > 0 else 0.0
    max_transcript = max(transcript) if transcript and max(transcript) > 0 else 0.0
    max_reference = max(ref_scores) if ref_scores and max(ref_scores) > 0 else 0.0
    ranked = []
    for cluster in clusters:
        cluster_id = str(cluster["cluster_id"])
        lecture_row = lecture_by_id.get(cluster_id, {})
        reference_row = reference_by_id.get(cluster_id, {})
        raw_slides = number(lecture_row.get("slide_count", ""))
        raw_speech = number(lecture_row.get("transcript_seconds", ""))
        slides = number(
            lecture_row.get("weighted_slide_evidence", raw_slides)
        )
        speech = number(
            lecture_row.get("weighted_transcript_seconds", raw_speech)
        )
        reference_score = number(reference_row.get("reference_score", ""))
        # Curriculum membership is deliberately a small floor. The remaining
        # terms are normalized evidence, not probabilities.
        curriculum = 0.10
        lecture_score = (
            0.45 * (slides / max_slide if max_slide > 0 else 0.0)
            + 0.35 * (speech / max_transcript if max_transcript > 0 else 0.0)
        )
        reference_score_norm = (
            0.10 * math.log1p(reference_score) / math.log1p(max_reference)
            if max_reference > 0 else 0.0
        )
        total = curriculum + lecture_score + reference_score_norm
        has_lecture = slides > 0 or speech > 0
        has_reference = reference_score > 0
        if has_lecture and has_reference:
            evidence_status = "lecture_and_reference"
            uncertainty = "lower"
        elif has_lecture or has_reference:
            evidence_status = "single_source"
            uncertainty = "moderate"
        else:
            evidence_status = "curriculum_only"
            uncertainty = "high"
        ranked.append(
            {
                **cluster,
                "curriculum_score": round(curriculum, 4),
                "lecture_score": round(lecture_score, 4),
                "reference_score": round(reference_score_norm, 4),
                "weight": round(total, 4),
                "slide_count": int(raw_slides),
                "weighted_slide_evidence": round(slides, 4),
                "transcript_seconds": round(raw_speech, 1),
                "weighted_transcript_seconds": round(speech, 1),
                "reference_raw_score": round(reference_score, 2),
                "aspect": ASPECTS.get(
                    cluster["name"],
                    "core definitions, mechanisms, applications, and differences relevant to this cluster",
                ),
                "evidence_status": evidence_status,
                "uncertainty": uncertainty,
            }
        )
    return sorted(ranked, key=lambda row: (-row["weight"], row["cluster_id"]))


def choose_questions(ranked: list[dict[str, Any]], exam_index: int, count: int) -> list[dict[str, Any]]:
    if exam_index == 1:
        selected = ranked[:count]
    elif exam_index == 2:
        # Diversify by moving lower-ranked curriculum areas into the paper.
        selected = sorted(ranked, key=lambda row: (-row["weight"], row["cluster_id"]))
        selected = selected[::2] + selected[1::2]
        selected = selected[:count]
    else:
        # A coverage-oriented paper starts with underrepresented clusters, then
        # fills remaining places by weight.
        selected = sorted(ranked, key=lambda row: (row["slide_count"] > 0,
                                                    row["reference_raw_score"],
                                                    -row["weight"],
                                                    row["cluster_id"]))
        selected = selected[:count]
    return selected


def write_exam(path: Path, title: str, selected: list[dict[str, Any]]) -> None:
    lines = [
        f"# {title}",
        "",
        "Pilot predictor generated from the ranked cluster evidence. These are "
        "question archetypes, not claims about exact wording.",
        "",
    ]
    for index, row in enumerate(selected, 1):
        template = QUESTION_TEMPLATES.get(
            row["name"],
            f"Develop a cluster-specific archetype covering {row['aspect']}."
        )
        lines.extend(
            [
                f"{index}. **{row['name']}** — {template}",
                f"   - Cluster: `{row['cluster_id']}`",
                f"   - Evidence weight: `{row['weight']}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--lecture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("prediction"))
    parser.add_argument("--questions", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    if args.questions < 1:
        parser.error("--questions must be positive")

    ranked = rank_clusters(
        load_clusters(args.clusters),
        read_csv(args.lecture),
        read_csv(args.reference),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.allow_overwrite:
        parser.error(
            "refusing to overwrite an existing sealed manifest; use --allow-overwrite explicitly"
        )
    if args.output_dir.exists():
        stale = sorted(path for path in args.output_dir.iterdir() if path.is_file())
        if stale and not args.allow_overwrite:
            parser.error(
                "output directory is not empty; use --allow-overwrite explicitly"
            )
    with (args.output_dir / "ranked_clusters.json").open("w", encoding="utf-8") as handle:
        json.dump(ranked, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "ranked_clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["cluster_id", "name", "weight", "curriculum_score", "lecture_score",
                  "reference_score", "slide_count", "weighted_slide_evidence",
                  "transcript_seconds", "weighted_transcript_seconds",
                  "reference_raw_score", "evidence_status", "uncertainty"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in ranked)
    for index in (1, 2, 3):
        selected = choose_questions(ranked, index, min(args.questions, len(ranked)))
        write_exam(args.output_dir / f"predictor_{index}.md",
                   f"Exam Predictor {index}", selected)
    (args.output_dir / "assumptions.md").write_text(
        "# Pilot prediction assumptions\n\n"
        "- The supplied past-paper PDF was not used for topic weighting because it is not a validated exam paper.\n"
        "- The curriculum floor is 0.10; lecture evidence contributes 0.80 and reference evidence contributes 0.10.\n"
        "- The three papers contain one archetype per selected cluster and are not yet a real-exam-sized format.\n"
        "- Clusters without lecture or reference evidence remain in the ranking with high uncertainty; they are not treated as impossible.\n"
        "- Exact question wording must be authored after the archetypes are reviewed.\n",
        encoding="utf-8",
    )
    sealed_at = datetime.now(timezone.utc).isoformat()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    input_hashes = {}
    for path in (args.clusters, args.lecture, args.reference):
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "sealed_at": sealed_at,
        "seed": args.seed,
        "code_sha256": code_hash,
        "input_sha256": input_hashes,
        "weights": {
            "curriculum": 0.10,
            "lecture_slides": 0.45,
            "lecture_transcript": 0.35,
            "reference": 0.10,
        },
        "files": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "manifest.json" or not path.is_file():
            continue
        manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Ranked {len(ranked)} clusters and wrote three predictors to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
