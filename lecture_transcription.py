#!/usr/bin/env python3
"""Transcribe lecture recordings locally with whisper.cpp.

Uses the existing native whisper.cpp installation and model locally. Video
inputs are converted to temporary mono WAV audio with ffmpeg. No recording is
uploaded.

Example:
    python3 lecture_transcription.py \
        --input lecture03.mp4 \
        --output-dir units/MSB201/transcripts \
        --language en

Install:
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any


def timestamp(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def srt_timestamp(seconds: float) -> str:
    return timestamp(seconds).replace(".", ",")


def parse_native_timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().replace(",", ".")
    match = __import__("re").match(r"^(\d+):(\d{2}):(\d{2})(?:\.(\d+))?$", text)
    if not match:
        return 0.0
    hours, minutes, seconds, fraction = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + float(
        f"0.{fraction}" if fraction else 0
    )


def extract_audio(input_path: Path, output_path: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for video input but was not found")
    command = [
        "ffmpeg", "-y", "-i", str(input_path), "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"ffmpeg audio extraction failed: {result.stderr[-1000:]}")


def audio_duration(input_path: Path) -> float:
    if input_path.suffix.lower() == ".wav":
        with wave.open(str(input_path), "rb") as handle:
            return handle.getnframes() / handle.getframerate()
    if not shutil.which("ffprobe"):
        return 0.0
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)],
        capture_output=True, text=True,
    )
    try:
        return max(0.0, float(result.stdout.strip()))
    except ValueError:
        return 0.0


def native_segments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    entries = raw.get("transcription", raw.get("segments", []))
    if not isinstance(entries, list):
        raise ValueError("whisper.cpp JSON did not contain transcription segments")
    records = []
    for index, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            continue
        offsets = entry.get("offsets", {})
        timestamps = entry.get("timestamps", {})
        if isinstance(offsets, dict) and offsets:
            start_raw = offsets.get("from", 0)
            end_raw = offsets.get("to", start_raw)
            # Native offsets are integer milliseconds.
            start = float(start_raw) / 1000.0 if isinstance(start_raw, (int, float)) else parse_native_timestamp(start_raw)
            end = float(end_raw) / 1000.0 if isinstance(end_raw, (int, float)) else parse_native_timestamp(end_raw)
        else:
            start = parse_native_timestamp(
                timestamps.get("from", entry.get("start", 0))
            )
            end = parse_native_timestamp(
                timestamps.get("to", entry.get("end", start))
            )
        text = str(entry.get("text", "")).strip()
        if text:
            records.append(
                {
                    "segment_id": index,
                    "start": round(start, 3),
                    "end": round(max(end, start), 3),
                    "start_timestamp": timestamp(start),
                    "end_timestamp": timestamp(max(end, start)),
                    "text": text,
                    "confidence": None,
                    "words": [],
                }
            )
    return records


def transcribe(
    input_path: Path,
    output_dir: Path,
    model_path: Path,
    language: str | None,
    threads: int,
    beam_size: int,
) -> dict[str, Any]:
    whisper_cli = Path.home() / "whisper.cpp/build/bin/whisper-cli"
    if not whisper_cli.is_file():
        raise RuntimeError(f"whisper.cpp executable not found: {whisper_cli}")
    if not model_path.is_file():
        raise RuntimeError(f"Whisper model not found: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lecture-transcription-") as temp_dir:
        audio_path = Path(temp_dir) / "audio.wav"
        if input_path.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}:
            audio_path = input_path
        else:
            extract_audio(input_path, audio_path)
        input_duration = audio_duration(audio_path)

        raw_base = Path(temp_dir) / "whisper"
        command = [
            str(whisper_cli), "--model", str(model_path), "--file", str(audio_path),
            "--output-json", "--output-json-full", "--output-file", str(raw_base),
            "--threads", str(threads), "--beam-size", str(beam_size),
            "--language", language or "auto", "--no-prints",
            "--no-speech-thold", "0.8", "--suppress-nst",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"whisper.cpp failed: {result.stderr[-1500:]}")
        raw_path = raw_base.with_suffix(".json")
        if not raw_path.is_file():
            raise RuntimeError("whisper.cpp completed without producing JSON output")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        records = native_segments(raw)
        if input_duration:
            for record in records:
                record["end"] = round(min(record["end"], input_duration), 3)
                record["end_timestamp"] = timestamp(record["end"])
                if record["start"] > input_duration:
                    record["start"] = round(input_duration, 3)
                    record["start_timestamp"] = timestamp(record["start"])

    metadata = {
        "input": str(input_path),
        "backend": "whisper.cpp",
        "whisper_cli": str(whisper_cli),
        "model": str(model_path),
        "language": language or raw.get("result", {}).get("language", "unknown"),
        "duration_seconds": round(input_duration, 3),
        "segment_count": len(records),
    }

    stem = input_path.stem
    (output_dir / f"{stem}.metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    (output_dir / f"{stem}.txt").write_text(
        "\n\n".join(
            f"[{record['start_timestamp']} - {record['end_timestamp']}] {record['text']}"
            for record in records
        )
        + "\n",
        encoding="utf-8",
    )

    with (output_dir / f"{stem}.srt").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                f"{record['segment_id']}\n"
                f"{srt_timestamp(record['start'])} --> {srt_timestamp(record['end'])}\n"
                f"{record['text']}\n\n"
            )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local audio or video recording")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path.home() / "whisper.cpp/models/ggml-large-v3-turbo.bin",
        help="Local whisper.cpp model path",
    )
    parser.add_argument("--language", default=None, help="Language code, e.g. en; omit for detection")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--beam-size", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        print(f"error: input file does not exist: {args.input}", file=sys.stderr)
        return 2
    try:
        metadata = transcribe(
            args.input,
            args.output_dir,
            args.model.expanduser(),
            args.language,
            args.threads,
            args.beam_size,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
