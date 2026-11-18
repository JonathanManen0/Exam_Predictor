#!/usr/bin/env python3
"""Analyze past papers for question formats and proportions.

Past papers are used only to learn assessment structure, not to predict topics.
The analyzer extracts questions from PDF/text/Markdown/JSON/JSONL files,
classifies them as MCQ, SAQ, LEQ, or OTHER, and writes auditable outputs:

    questions.jsonl
    format_summary.csv
    report.md

Example:
    python3 past_paper_analysis.py \
        --input past_papers/ \
        --output-dir past_paper_analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".json", ".jsonl"}
QUESTION_START_RE = re.compile(
    r"(?m)^\s*(?P<number>\d{1,3})\s*[\).:-]\s+(?P<body>.+?)(?=^\s*\d{1,3}\s*[\).:-]\s+|\Z)",
    re.S,
)
LETTER_OPTION_RE = re.compile(r"(?m)^\s*[A-D][\).]\s+\S+")
MARK_RE = re.compile(r"\(?\s*(\d{1,3})\s*marks?\s*\)?", re.I)
MCQ_TERMS = re.compile(
    r"\b(select the (?:one|best|correct)|choose the|which of the following|"
    r"all of the following except|most likely|single best answer)\b",
    re.I,
)
SAQ_TERMS = re.compile(
    r"\b(short answer|briefly|list|state|mention|define|outline|give two|"
    r"give three|enumerate)\b",
    re.I,
)
LEQ_TERMS = re.compile(
    r"\b(essay|long essay|discuss|describe in detail|critically discuss|"
    r"explain fully|write an essay)\b",
    re.I,
)


@dataclass
class Question:
    paper_id: str
    source_path: str
    number: str
    text: str
    question_type: str
    marks: int | None
    confidence: float
    evidence: str


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(value, list):
                records.extend(value)
            elif isinstance(value, dict):
                records.append(value)
            else:
                raise ValueError(f"{path}:{line_number}: expected an object or array")
        return records
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        records = value.get("questions", value.get("records", [value]))
        if not isinstance(records, list):
            raise ValueError(f"{path}: questions/records must be an array")
        return [item for item in records if isinstance(item, dict)]
    raise ValueError(f"{path}: expected a JSON object or array")


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires pypdf. Install it with: pip install pypdf"
        ) from exc
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def read_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix in {".json", ".jsonl"}:
        records = load_json_records(path)
        return "\n\n".join(
            f"{item.get('number', item.get('question_number', ''))}. "
            f"{item.get('text', item.get('question', item.get('content', '')))}"
            for item in records
        )
    return path.read_text(encoding="utf-8")


def discover_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
    raise FileNotFoundError(input_path)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def classify_question(text: str) -> tuple[str, float, str]:
    normalized = clean_text(text)
    has_options = len(LETTER_OPTION_RE.findall(text)) >= 2
    if has_options or MCQ_TERMS.search(normalized):
        evidence = "options" if has_options else "MCQ wording"
        return "MCQ", 0.95 if has_options else 0.80, evidence
    if LEQ_TERMS.search(normalized):
        return "LEQ", 0.82, "essay/discussion wording"
    if SAQ_TERMS.search(normalized):
        return "SAQ", 0.78, "short-answer wording"
    return "OTHER", 0.40, "no decisive format signal"


def extract_questions(path: Path) -> list[Question]:
    paper_id = path.stem
    text = read_source(path)
    questions: list[Question] = []
    matches = list(QUESTION_START_RE.finditer(text))
    if not matches and text.strip():
        matches = [re.match(r"(?s)(.*)", text)]
    for index, match in enumerate(matches, 1):
        if match is None:
            continue
        if "body" in match.groupdict():
            number = match.group("number")
            body = match.group("body")
        else:
            number = "1"
            body = match.group(0)
        body = clean_text(body)
        if not body:
            continue
        question_type, confidence, evidence = classify_question(body)
        mark_match = MARK_RE.search(body)
        questions.append(
            Question(
                paper_id=paper_id,
                source_path=str(path),
                number=number or str(index),
                text=body,
                question_type=question_type,
                marks=int(mark_match.group(1)) if mark_match else None,
                confidence=confidence,
                evidence=evidence,
            )
        )
    return questions


def write_outputs(output_dir: Path, questions: list[Question]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(asdict(question), ensure_ascii=False) + "\n")

    by_paper: dict[str, Counter[str]] = defaultdict(Counter)
    marks_by_paper: dict[str, Counter[str]] = defaultdict(Counter)
    for question in questions:
        by_paper[question.paper_id][question.question_type] += 1
        if question.marks is not None:
            marks_by_paper[question.paper_id][question.question_type] += question.marks

    rows = []
    for paper_id in sorted(by_paper):
        total_questions = sum(by_paper[paper_id].values())
        total_marks = sum(marks_by_paper[paper_id].values())
        for question_type in ("MCQ", "SAQ", "LEQ", "OTHER"):
            count = by_paper[paper_id][question_type]
            marks = marks_by_paper[paper_id][question_type]
            rows.append(
                {
                    "paper_id": paper_id,
                    "question_type": question_type,
                    "question_count": count,
                    "question_proportion": round(count / total_questions, 4)
                    if total_questions else 0,
                    "marks": marks,
                    "mark_proportion": round(marks / total_marks, 4)
                    if total_marks else "",
                    "total_questions_in_paper": total_questions,
                    "total_explicit_marks": total_marks,
                }
            )
    fields = list(rows[0]) if rows else [
        "paper_id", "question_type", "question_count", "question_proportion",
        "marks", "mark_proportion", "total_questions_in_paper",
        "total_explicit_marks",
    ]
    with (output_dir / "format_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    aggregate = Counter(question.question_type for question in questions)
    total = sum(aggregate.values())
    lines = [
        "# Past-paper format analysis",
        "",
        "This analysis records question formats and proportions only.",
        "",
        "## Aggregate format distribution",
        "",
        "| Format | Questions | Proportion |",
        "|---|---:|---:|",
    ]
    for question_type in ("MCQ", "SAQ", "LEQ", "OTHER"):
        count = aggregate[question_type]
        proportion = count / total if total else 0
        lines.append(f"| {question_type} | {count} | {proportion:.1%} |")
    lines.extend(["", "## Review queue", ""])
    uncertain = [q for q in questions if q.confidence < 0.7]
    if uncertain:
        lines.append("Manually verify these low-confidence classifications:")
        for question in uncertain:
            lines.append(
                f"- `{question.paper_id}` question {question.number}: "
                f"{question.text[:180]}"
            )
    else:
        lines.append("No low-confidence classifications were generated.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Past-paper file or directory")
    parser.add_argument("--output-dir", type=Path, default=Path("past_paper_analysis"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = discover_inputs(args.input)
        if not paths:
            raise ValueError(f"No supported input files found under {args.input}")
        questions = [question for path in paths for question in extract_questions(path)]
        if not questions:
            raise ValueError("No questions could be extracted")
        write_outputs(args.output_dir, questions)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Analyzed {len(questions)} questions from {len(paths)} paper(s)")
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
