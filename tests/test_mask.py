import unittest
import torch
from docision._runtime.hf import isolated_mask


class MaskTests(unittest.TestCase):
    def test_candidates_are_isolated_and_answer_sees_all(self):
        spans = [[3, 5], [5, 8]]
        visible = isolated_mask(10, spans, "cpu", torch.float32)[0, 0].isfinite()
        self.assertTrue(visible[6, :3].all())
        self.assertFalse(visible[6, 3:5].any())
        self.assertTrue(visible[6, 5:7].all())
        self.assertFalse(visible[6, 7:].any())
        self.assertTrue(visible[9].all())
        self.assertFalse(visible.triu(1).any())
        cached = isolated_mask(10, spans, "cpu", torch.float32, query_start=3)[0, 0].isfinite()
        self.assertTrue(torch.equal(cached, visible[3:]))
