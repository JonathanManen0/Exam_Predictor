# Exam Predictor: Third-Party Review Summary

## Purpose

This project predicts which curriculum clusters and question archetypes are most likely to appear on an upcoming exam. It converts a manually curated curriculum into stable cluster IDs, extracts evidence from lecture recordings and reference materials, combines that evidence into transparent rankings, and produces three sealed predictor exams.

## Workflow

1. **Curriculum definition**
   - A manually authored DOCX is the authoritative source.
   - `manual_cluster_import.py` validates stable mature-cluster IDs and exports `clusters.json`.
   - `curriculum_clusters.py` remains a drafting/helper extractor, not the production authority.

2. **Lecture processing**
   - `lecture_transcription.py` extracts audio and runs local whisper.cpp transcription.
   - `lecture_slide_ocr.py` samples lecture frames, enhances them, runs OCR, and coalesces repeated frames.
   - `lecture_cluster_mapping.py` maps slide and transcript evidence to curriculum clusters using aliases, timing, and weighted matches.

3. **Reference analysis**
   - `reference_analysis.py` prefers Docling for structured PDF extraction and falls back to pypdf.
   - It records page/block provenance, detects evidence types, assigns blocks to one or more clusters, normalizes duplicate evidence, and writes comparison and review outputs.

4. **Past-paper analysis**
   - `past_paper_analysis.py` estimates question-format proportions such as MCQ, SAQ, and LEQ.
   - This input must be manually validated because arbitrary examination PDFs may not be clean past papers.

5. **Prediction**
   - `predictor.py` combines curriculum, lecture, and reference evidence.
   - It ranks clusters, reports evidence status and uncertainty, and creates three diversified predictor papers.
   - Outputs are sealed with a timestamp and SHA-256 manifest.

## Current safeguards against overfitting

- Stable manual cluster boundaries prevent the model from inventing topics from source wording.
- Evidence is fused across curriculum, lecture, and reference sources.
- Repeated OCR frames and duplicate reference blocks are coalesced or normalized.
- Every curriculum cluster receives a small baseline weight.
- Predictions use topic/aspect archetypes rather than copied questions.
- Three diversified predictor papers are generated.
- The design includes comparison with slide-count-only and uniform baselines.

## Known limitations

- The current pilot has very few validated exams, so generalization is not yet demonstrated statistically.
- Evidence weights are hand-designed rather than learned from a held-out exam set.
- Broad terminology slides can still contribute too much lecture evidence.
- The supplied examination PDF produced too few reliable questions to establish a trustworthy format distribution.
- OCR is supplementary and may contain noisy text.
- Docling can leave structural blocks unmatched and may emit non-fatal OCR/table warnings.

## Recommended independent review

Reviewers should inspect:

- evidence leakage between pre-exam inputs and evaluation outputs;
- duplicate suppression and evidence normalization;
- cluster assignment precision and alias false positives;
- whether lecture mentions are distinguished from substantive teaching;
- whether weight tuning is performed only on prior exams;
- predictor size and format constraints;
- sealed-manifest integrity;
- performance against slide-count and uniform baselines.

## Source files

- `manual_cluster_import.py` — authoritative cluster importer and validator
- `reference_analysis.py` — reference-material extraction and comparison
- `lecture_transcription.py` — local Whisper transcription
- `lecture_slide_ocr.py` — slide OCR and repeated-frame coalescing
- `lecture_cluster_mapping.py` — lecture-to-cluster evidence mapping
- `past_paper_analysis.py` — question-format analysis
- `predictor.py` — ranking, predictor generation, and sealing
- `curriculum_clusters.py` — experimental curriculum drafting helper
- `msb201_supplement.json` — supplemental cluster data
- `sample_clusters.json` — example cluster input
