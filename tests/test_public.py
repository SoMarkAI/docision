import tempfile
import unittest
from pathlib import Path
from docision.io import normalize_document
from docision.metrics import probability_metrics, permutation_metrics
from docision.permutations import orders
from docision.evaluate import score_benchmark


class PublicTests(unittest.TestCase):
    def test_original_jev_schema_and_reference_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "page.png").touch()
            task = {
                "image": "page.png",
                "question": {"instructions": "Paid?", "criteria": {"A": "Yes", "B": "No"}},
                "label": "A",
            }
            result = normalize_document(task, folder)
            self.assertEqual(result["questions"][0]["candidates"], ["Yes", "No"])
            task["label"] = "C"
            with self.assertRaises(ValueError):
                normalize_document(task, folder)

    def test_missing_media_and_nonimage_rejected(self):
        for task in [
            {"image": "missing.png", "question": "q", "candidates": ["x", "y"]},
            {"modality": "audio", "image": "x", "question": "q", "candidates": ["x", "y"]},
        ]:
            with self.assertRaises(ValueError):
                normalize_document(task)

    def test_accuracy_and_extreme_logits(self):
        rows = [
            {"probabilities": [0.75, 0.25], "target_index": 0},
            {"probabilities": [0.25, 0.75], "target_index": 0},
        ]
        self.assertEqual(probability_metrics(rows), {"count": 2, "accuracy": 0.5})
        self.assertEqual(
            probability_metrics(
                [{"probabilities": [1.0, 0.0], "raw_logits": [0.0, -1000.0], "target_index": 1}]
            ),
            {"count": 1, "accuracy": 0.0},
        )

    def test_metrics_reject_malformed_and_inconsistent_scores(self):
        for row in [
            {"probabilities": [0.6, 0.6], "target_index": 0},
            {"probabilities": [0.5, 0.5], "target_index": True},
            {"probabilities": [0.5, 0.5], "target_index": 0, "raw_logits": [1.0, 0.0]},
            {"probabilities": [float("nan"), 0], "target_index": 0},
        ]:
            with self.assertRaises(ValueError):
                probability_metrics([row])

    def test_circular_rotations_cover_every_position(self):
        for count in [2, 3, 5, 40]:
            rotations = orders(count, "circular")
            self.assertEqual(len(rotations), count)
            for semantic_index in range(count):
                self.assertEqual(
                    sorted(order.index(semantic_index) for order in rotations), list(range(count))
                )
        self.assertEqual(orders(5, "random", 8, 7), orders(5, "random", 8, 7))
        self.assertEqual(len(orders(2, "random", 100, 7)), 2)

    def test_evaluator_hides_answers_and_remaps_semantics(self):
        class Engine:
            def score_many(self, image, requests, **kwargs):
                result = []
                for question in requests:
                    assert "label" not in question
                    index = question["candidates"].index("correct")
                    p = [float(i == index) for i in range(len(question["candidates"]))]
                    result.append(
                        {
                            "candidates": [{"raw_logit": v * 10, "probability": v} for v in p],
                            "decision": {"index": index},
                            "num_cached_tokens": 0,
                        }
                    )
                return result

        questions = [
            {
                "id": "q",
                "image": "page",
                "context": "",
                "question": "q",
                "candidates": ["wrong", "correct", "other"],
                "label": "B",
            }
        ]
        rows = score_benchmark(Engine(), questions, "circular")
        self.assertTrue(all(row["semantic_prediction"] == 1 for row in rows))
        self.assertEqual(permutation_metrics(rows)["permutation_consistency"], 1)
        self.assertEqual(permutation_metrics(rows)["all_variants_accuracy"], 1)
        self.assertEqual([r["target_index"] for r in rows], [1, 0, 2])

    def test_permutation_consistency_does_not_imply_correctness(self):
        rows = [
            {"id": "a", "variant": i, "semantic_prediction": 1, "semantic_target": 0}
            for i in range(3)
        ]
        metrics = permutation_metrics(rows)
        self.assertEqual(metrics["permutation_consistency"], 1)
        self.assertEqual(metrics["all_variants_accuracy"], 0)


class SavedPredictionTests(unittest.TestCase):
    def test_tampered_semantic_mapping_is_rejected(self):
        from docision.evaluate import summarize

        row = {
            "id": "q",
            "variant": 0,
            "order": [0, 1],
            "target_index": 0,
            "semantic_target": 0,
            "semantic_prediction": 1,
            "probabilities": [0.9, 0.1],
        }
        with self.assertRaises(ValueError):
            summarize([row])

    def test_identity_only_agreement_is_not_robustness_measurement(self):
        from docision.evaluate import summarize

        row = {
            "id": "q",
            "variant": 0,
            "order": [0, 1],
            "target_index": 0,
            "semantic_target": 0,
            "semantic_prediction": 0,
            "probabilities": [0.9, 0.1],
        }
        result = summarize([row])
        self.assertIsNone(result["permutation"]["identity_agreement"])
        self.assertEqual(result["original_order"]["accuracy"], 1)


class CommandTests(unittest.TestCase):
    def test_documented_module_commands_are_live(self):
        import subprocess
        import sys

        for module in ["docision.cli", "docision.evaluate", "docision.compare", "docision.service"]:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertIn("usage:", result.stdout)
