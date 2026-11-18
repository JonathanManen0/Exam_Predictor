#!/usr/bin/env python3
"""Extract mature examinable clusters from an MBChB curriculum DOCX.

The curriculum is read as:

    year -> course -> examinable content item

The output contains only mature examinable clusters. Year and course remain
metadata and parent references; they are not emitted as clusters themselves.
Splitting is deliberately conservative: clearly named, distinct subject areas
are separated, while ambiguous material is retained for later review.

Example:
    python3 curriculum_clusters.py \
        --input MBChB_Curriculum_clean_corrected.docx \
        --supplement msb201_supplement.json \
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


YEAR_RE = re.compile(r"^\s*YEAR\s+(ONE|TWO|THREE|FOUR|FIVE|SIX)\s*$", re.I)
COURSE_RE = re.compile(
    r"^\s*(?P<code>[A-Z]{2,5}\s*[-;]?\s*\d{3})\s*:\s*(?P<title>.+?)\s*$",
    re.I,
)
CONTENT_RE = re.compile(r"^\s*(course\s+)?contents?\s*:?\s*$", re.I)
IGNORE_SECTIONS = {
    "purpose",
    "objective",
    "objectives",
    "outcome",
    "outcomes",
    "learning outcome",
    "learning outcomes",
}
AREA_MARKERS = (
    "Principles of bacterial genetics:",
    "Pathogenic bacteria:",
    "Viruses:",
    "Classification, identification, and nomenclature of fungi,",
    "Classification, identification, and nomenclature of parasites and arthropods",
    "Medical entomology:",
    "Public health bacteriology:",
    "Innate and adaptive immune response",
    "Immune effector mechanisms.",
    "Immune evasion mechanisms.",
    "Principles of vaccination,",
    "General concepts:",
    "Mechanisms of drug absorption,",
    "General properties of autacoids:",
    "Biogenic amines:",
    "Smooth muscle/endothelial autacoids:",
    "Cytokines:",
    "Adrenergics:",
    "Chemotherapy of microbial diseases:",
    "Principles of diagnostic procedures",
    "Laboratory diagnostic techniques:",
    "Quality control.",
    "Viral cultivation,",
    "Culture of microorganisms;",
    "Immune diagnosis of microbial and parasitic infections.",
)
STOP_HEADINGS = {
    "purpose",
    "objective",
    "objectives",
    "outcome",
    "outcomes",
    "learning outcome",
    "learning outcomes",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def slug(text: str) -> str:
    value = normalize(text).lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "unnamed"


def course_code(value: str) -> str:
    return re.sub(r"[\s;]+", " ", normalize(value).upper())


def keywords(text: str) -> list[str]:
    stopwords = {
        "about", "after", "also", "and", "are", "being", "from", "including",
        "into", "may", "more", "such", "than", "that", "their", "these",
        "this", "through", "with", "within",
    }
    terms = re.findall(r"[A-Za-z][A-Za-z0-9/-]{2,}", text.lower())
    return list(dict.fromkeys(term for term in terms if term not in stopwords))[:20]


def split_mature_topics(text: str) -> list[str]:
    """Split only at clearly identifiable subject-area boundaries.

    The curriculum often compresses several examinable areas into one
    paragraph. This first pass uses explicit labels and a small number of
    unambiguous paired concepts; it does not split every comma or semicolon.
    """
    text = normalize(text)
    if not text:
        return []

    # Make the two distinct bacterial concepts explicit.
    text = re.sub(
        r"^Structure,\s*composition,\s*physiology\s+and\s+regulation\s+of\s+bacteria\.",
        "Bacterial structure and composition. Bacterial physiology and regulation.",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(Bacterial physiology and regulation\.)\s+(Principles of bacterial\s+g?\s*genetics:?)",
        r"\1\n\2",
        text,
        flags=re.I,
    )

    markers = sorted(AREA_MARKERS, key=len, reverse=True)
    marker_pattern = "|".join(re.escape(marker) for marker in markers)
    parts = []
    for line in text.splitlines():
        line_parts = [line]
        for marker in markers:
            next_parts: list[str] = []
            for part in line_parts:
                next_parts.extend(
                    re.split(
                        rf"(?<=[.;])\s+(?={re.escape(marker)})",
                        part,
                        flags=re.I,
                    )
                )
            line_parts = next_parts
        parts.extend(part.strip(" ;") for part in line_parts if part.strip(" ;"))

    # A leading labelled area followed by a full stop is a safe boundary.
    result: list[str] = []
    for part in parts:
        if part.lower().startswith("bacterial structure and composition."):
            result.extend(
                [
                    "Bacterial structure and composition.",
                    part[len("Bacterial structure and composition.") :].strip(),
                ]
            )
        else:
            result.append(part)
    return [normalize(part) for part in result if normalize(part)]


def style_name(paragraph: Any) -> str:
    return str(getattr(getattr(paragraph, "style", None), "name", "") or "")


def is_heading(paragraph: Any, level: int | None = None) -> bool:
    style = style_name(paragraph).lower()
    return style == f"heading {level}" if level else style.startswith("heading ")


def add_cluster(
    records: list[dict[str, Any]],
    *,
    cluster_id: str,
    name: str,
    parent_id: str,
    level: str,
    source_paragraph: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "cluster_id": cluster_id,
        "name": normalize(name),
        "parent_id": parent_id,
        "level": level,
        "keywords": keywords(name),
        "source_paragraph": source_paragraph,
    }
    if metadata:
        record.update(metadata)
    records.append(record)


def load_supplement(path: Path) -> dict[str, list[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("courses"), dict):
        raise ValueError(f"{path} must contain an object with a 'courses' mapping")
    result: dict[str, list[str]] = {}
    for course, topics in value["courses"].items():
        if not isinstance(topics, list) or not all(isinstance(topic, str) for topic in topics):
            raise ValueError(f"{path}: topics for {course} must be a list of strings")
        result[course_code(course)] = [normalize(topic) for topic in topics if normalize(topic)]
    return result


def merge_supplement(
    records: list[dict[str, Any]], supplement: dict[str, list[str]]
) -> list[dict[str, Any]]:
    """Replace condensed course-description topics for supplemented courses."""
    if not supplement:
        return records

    output = [record for record in records if record.get("course_code") not in supplement]
    course_info = {
        record["course_code"]: record
        for record in records
        if record.get("course_code") in supplement
    }
    for code, topics in supplement.items():
        info = course_info.get(code)
        if not info:
            raise ValueError(
                f"Supplement course {code!r} was not found in the DOCX course descriptions"
            )
        current_index = 0
        parent_id = info["parent_id"]
        year = info.get("year", "")
        course_name = info.get("course_name", code)
        for topic in topics:
            for mature_topic in split_mature_topics(topic):
                current_index += 1
                output.append(
                    {
                        "cluster_id": (
                            f"{parent_id}.cluster-supplement-{current_index:03d}-"
                            f"{slug(mature_topic)[:60]}"
                        ),
                        "name": mature_topic,
                        "parent_id": parent_id,
                        "level": "mature_cluster",
                        "keywords": keywords(mature_topic),
                        "source_paragraph": info.get("source_paragraph"),
                        "course_code": code,
                        "course_name": course_name,
                        "year": year,
                        "content_index": current_index,
                        "source": "supplement",
                    }
                )
    return output


def extract_clusters(path: Path, supplement_path: Path | None = None) -> list[dict[str, Any]]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "DOCX input requires python-docx. Install it with: pip install python-docx"
        ) from exc

    document = Document(str(path))
    records: list[dict[str, Any]] = []
    current_year = ""
    current_year_id = ""
    current_course = ""
    course_code_for_current_course = ""
    current_course_id = ""
    in_content = False
    in_course_descriptions = False
    content_index = 0
    current_section = ""
    course_has_content_heading = False
    pending_direct_content: list[tuple[int, str]] = []

    def add_topic(paragraph_index: int, text: str) -> None:
        nonlocal content_index
        for mature_topic in split_mature_topics(text):
            content_index += 1
            content_id = (
                f"{current_course_id}.cluster-{content_index:03d}-{slug(mature_topic)[:60]}"
            )
            add_cluster(
                records,
                cluster_id=content_id,
                name=mature_topic,
                parent_id=current_course_id,
                level="mature_cluster",
                source_paragraph=paragraph_index,
                metadata={
                    "course_code": course_code_for_current_course,
                    "course_name": current_course,
                    "year": current_year,
                    "content_index": content_index,
                },
            )

    def flush_pending_content() -> None:
        if not current_course_id or course_has_content_heading:
            pending_direct_content.clear()
            return
        for paragraph_index, text in pending_direct_content:
            add_topic(paragraph_index, text)
        pending_direct_content.clear()

    for paragraph_index, paragraph in enumerate(document.paragraphs):
        text = normalize(paragraph.text)
        if not text:
            continue

        if text.upper() == "3.4 COURSE DESCRIPTIONS":
            in_course_descriptions = True
            continue
        if is_heading(paragraph, 1) and text.upper().startswith("4.0 APPENDICES"):
            break
        if not in_course_descriptions:
            continue

        year_match = YEAR_RE.match(text)
        if year_match:
            flush_pending_content()
            current_year = f"Year {year_match.group(1).title()}"
            current_year_id = f"year-{slug(year_match.group(1))}"
            current_course = ""
            current_course_id = ""
            course_code_for_current_course = ""
            in_content = False
            current_section = ""
            continue

        course_match = COURSE_RE.match(text)
        if course_match and is_heading(paragraph):
            flush_pending_content()
            code = course_code(course_match.group("code"))
            course_code_for_current_course = code
            title = normalize(course_match.group("title"))
            title = re.sub(r"\s*\([^)]*units?\)\s*$", "", title, flags=re.I).strip()
            current_course = f"{code}: {title}"
            current_course_id = f"{current_year_id}.{slug(code)}" if current_year_id else slug(code)
            current_course = current_course.rstrip(" ;:")
            current_course_id = current_course_id.rstrip("-")
            current_content_parent = current_course_id
            content_index = 0
            in_content = False
            current_section = ""
            course_has_content_heading = False
            pending_direct_content.clear()
            continue

        if is_heading(paragraph, 4):
            heading = text.rstrip(":").strip().lower()
            in_content = bool(CONTENT_RE.match(text))
            current_section = heading
            course_has_content_heading = course_has_content_heading or in_content
            if heading in STOP_HEADINGS:
                in_content = False
            continue

        if not current_course_id:
            continue

        if re.match(r"^course\s+(purpose|objectives?|outcomes?)\b", text, re.I):
            current_section = re.match(
                r"^course\s+(purpose|objectives?|outcomes?)", text, re.I
            ).group(1).lower()
            continue

        # A few curriculum entries use "Course Content ..." in a normal
        # paragraph rather than a Heading 4; retain the text after the label.
        if text.lower().startswith("course content"):
            course_has_content_heading = True
            in_content = True
            text = re.sub(r"^course\s+content\s*:?\s*", "", text, flags=re.I).strip()
            if not text:
                continue

        if in_content:
            add_topic(paragraph_index, text)
        elif not current_section or current_section == "general guidelines":
            # Some short courses omit the Content heading and place their
            # examinable topics directly below the course heading.
            if text.rstrip(":").lower() == "general guidelines":
                current_section = "general guidelines"
            else:
                pending_direct_content.append((paragraph_index, text))

    flush_pending_content()

    if not records:
        raise ValueError("No year/course/content clusters were found in the document")
    if supplement_path:
        records = merge_supplement(records, load_supplement(supplement_path))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Curriculum DOCX file")
    parser.add_argument(
        "--supplement",
        type=Path,
        action="append",
        help="Optional JSON supplement; may be supplied more than once",
    )
    parser.add_argument("--output", type=Path, default=Path("clusters.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        records = extract_clusters(args.input)
        for supplement in args.supplement or []:
            records = merge_supplement(records, load_supplement(supplement))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    counts: dict[str, int] = {}
    for record in records:
        counts[record["level"]] = counts.get(record["level"], 0) + 1
    print(f"Wrote {len(records)} clusters to {args.output}")
    print(", ".join(f"{level}={count}" for level, count in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
