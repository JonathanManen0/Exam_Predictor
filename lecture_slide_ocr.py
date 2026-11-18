#!/usr/bin/env python3
"""Extract timestamped OCR from sampled lecture-video frames.

OCR is treated as noisy evidence. The output keeps raw and normalized text so
downstream mapping can combine it with the more reliable Whisper transcript.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = re.sub(r"[|]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ocr_quality(text: str) -> dict[str, Any]:
    words = re.findall(r"[A-Za-z]{2,}", text)
    tokens = re.findall(r"[A-Za-z0-9]+", text)
    suspicious = re.findall(r"\b(?:iA|Ch|wz|yi|ve|zc|Db)\b", text)
    return {
        "word_count": len(words),
        "token_count": len(tokens),
        "suspicious_token_count": len(suspicious),
        "quality": (
            "empty" if not tokens else
            "low" if len(suspicious) >= 3 else
            "medium" if suspicious else
            "higher"
        ),
    }


def content_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def similar_text(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return max(
        overlap / max(1, len(left_tokens | right_tokens)),
        overlap / max(1, min(len(left_tokens), len(right_tokens))),
    )


def coalesce_records(records: list[dict[str, Any]], threshold: float = 0.88) -> list[dict[str, Any]]:
    coalesced: list[dict[str, Any]] = []
    for record in records:
        if coalesced and similar_text(coalesced[-1]["text"], record["text"]) >= threshold:
            previous = coalesced[-1]
            previous["end"] = record["end"]
            previous["sample_count"] += 1
            if len(record["text"]) > len(previous["text"]):
                previous["text"] = record["text"]
            if len(record["raw_text"]) > len(previous["raw_text"]):
                previous["raw_text"] = record["raw_text"]
            continue
        record["sample_count"] = 1
        coalesced.append(record)
    for index, record in enumerate(coalesced, 1):
        record["slide_id"] = f"slide-{index:04d}"
        record["slide"] = index
    return coalesced


def ocr_frame(path: Path) -> tuple[str, str]:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import pytesseract

    image = Image.open(path).convert("L")
    width, height = image.size
    # Remove small player/viewer margins while retaining the whole slide.
    # The recordings tested here show a PDF/whiteboard viewer. Its toolbar and
    # side controls create most of the false OCR tokens, so keep the page area.
    image = image.crop((
        int(width * 0.045),
        int(height * 0.08),
        int(width * 0.95),
        int(height * 0.90),
    ))
    image = image.resize((image.width * 2, image.height * 2))
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Sharpness(image).enhance(1.5)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    processed = pytesseract.image_to_string(image, config="--psm 6")
    raw = pytesseract.image_to_string(Image.open(path), config="--psm 6")
    return raw, processed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=3)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"input does not exist: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="lecture-ocr-") as directory:
        frames = Path(directory)
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(args.input),
                "-vf", f"fps=1/{args.interval}", "-q:v", str(args.jpeg_quality),
                str(frames / "frame-%06d.jpg"), "-y",
            ],
            check=True,
        )
        records: list[dict[str, Any]] = []
        for index, frame in enumerate(sorted(frames.glob("frame-*.jpg"))):
            raw, processed = ocr_frame(frame)
            records.append(
                {
                    "slide_id": f"frame-{index + 1:06d}",
                    "slide": index + 1,
                    "start": round(index * args.interval, 3),
                    "end": round((index + 1) * args.interval, 3),
                    "text": normalize_text(processed),
                    "raw_text": normalize_text(raw),
                    "ocr_method": "cropped-upscaled-autocontrast",
                    "quality": ocr_quality(normalize_text(processed)),
                }
            )

    records = coalesce_records(records)
    quality_counts = {}
    for record in records:
        label = record["quality"]["quality"]
        quality_counts[label] = quality_counts.get(label, 0) + 1
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {len(records)} coalesced OCR records to {args.output}; "
        f"quality={quality_counts}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
