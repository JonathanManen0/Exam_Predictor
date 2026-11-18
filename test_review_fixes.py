import csv
import json
import tempfile
import unittest
from pathlib import Path

from lecture_transcription import native_segments
from lecture_slide_ocr import ocr_quality
from lecture_cluster_mapping import Cluster, score_match
from reference_analysis import Cluster as ReferenceCluster, load_slide_clusters, assign_clusters
from predictor import rank_clusters
from past_paper_analysis import extract_questions


class ReviewFixTests(unittest.TestCase):
    def test_predictor_all_zero_evidence(self):
        clusters = [{"cluster_id": "A", "name": "Topic A"}]
        lecture = [{"cluster_id": "A", "slide_count": "0", "weighted_slide_evidence": "0",
                    "transcript_seconds": "0", "weighted_transcript_seconds": "0"}]
        reference = [{"cluster_id": "A", "reference_score": "0"}]
        ranked = rank_clusters(clusters, lecture, reference)
        self.assertEqual(ranked[0]["weight"], 0.1)
        self.assertEqual(ranked[0]["evidence_status"], "curriculum_only")

    def test_whisper_millisecond_offsets(self):
        raw = {"transcription": [{
            "offsets": {"from": 5000, "to": 7000},
            "text": " test ",
        }]}
        records = native_segments(raw)
        self.assertEqual(records[0]["start"], 5.0)
        self.assertEqual(records[0]["end"], 7.0)

    def test_ocr_by_is_not_suspicious(self):
        quality = ocr_quality("shown by the diagram")
        self.assertEqual(quality["suspicious_token_count"], 0)
        self.assertEqual(quality["quality"], "higher")

    def test_generic_bacteria_keyword_is_rejected(self):
        gram_negative = Cluster("GN", "Gram-negative bacteria", ("bacteria", "gram-negative"), "", "")
        gram_positive = Cluster("GP", "Gram-positive bacteria", ("bacteria", "gram-positive"), "", "")
        negative_score, _ = score_match("gram-negative bacteria", gram_negative)
        positive_score, _ = score_match("gram-negative bacteria", gram_positive)
        self.assertGreater(negative_score, positive_score)

    def test_reference_accepts_mapper_jsonl_nested_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "slides.jsonl"
            path.write_text(json.dumps({
                "slide_id": "slide-1",
                "matches": [{"cluster_id": "A", "score": 0.8}],
            }) + "\n", encoding="utf-8")
            data = load_slide_clusters(path)
        self.assertEqual(data["A"]["slide_count"], 1)

    def test_reference_rejects_common_stopword(self):
        cluster = ReferenceCluster("A", "The immune system", keywords=("the", "immune"))
        matches = assign_clusters("the immune system", [cluster])
        self.assertTrue(matches)

    def test_past_paper_sections_and_marks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "paper.txt"
            path.write_text(
                "1. Which of the following is correct?\n"
                "A. One\nB. Two\nC. Three\nD. Four\n(1 mark)\n"
                "2. Briefly outline two principles.\n"
                "SECTION B\n"
                "3. Describe the pathogenesis in detail. (20 marks)\n",
                encoding="utf-8",
            )
            questions = extract_questions(path)
        self.assertEqual(questions[0].question_type, "MCQ")
        self.assertEqual(questions[1].question_type, "SAQ")
        self.assertEqual(questions[2].question_type, "LEQ")
        self.assertEqual(questions[2].marks, 20)


if __name__ == "__main__":
    unittest.main()
