"""Focused contract checks for the standalone dense/compact student models."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import student_models


@unittest.skipIf(torch is None, "PyTorch is not installed")
class StudentModelsTest(unittest.TestCase):
    def test_output_contract_and_all_head_gradients(self) -> None:
        for variant in student_models.MODEL_SPECS:
            with self.subTest(variant=variant):
                model = student_models.make_model(variant)
                spatial = torch.randn(2, 22, 19, 19)
                global_input = torch.randn(2, 19)
                output = model(spatial, global_input)
                self.assertEqual(tuple(output.policy_logits.shape), (2, 362))
                self.assertEqual(tuple(output.value_logits.shape), (2, 3))
                self.assertEqual(tuple(output.score.shape), (2,))
                self.assertEqual(tuple(output.ownership_logits.shape), (2, 361))
                loss = sum(tensor.square().mean() for tensor in output)
                self.assertTrue(bool(torch.isfinite(loss)))
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_size_and_checkpoint_roundtrip(self) -> None:
        dense = student_models.make_model("dense")
        compact = student_models.make_model("compact")
        self.assertGreater(student_models.count_parameters(dense),
                           2 * student_models.count_parameters(compact))
        self.assertLess(student_models.count_parameters(dense), 1_000_000)
        compact.eval()
        restored = student_models.make_model("compact")
        restored.load_state_dict(compact.state_dict(), strict=True)
        restored.eval()
        spatial = torch.randn(1, 22, 19, 19)
        global_input = torch.randn(1, 19)
        with torch.inference_mode():
            actual = compact(spatial, global_input)
            replay = restored(spatial, global_input)
        for a, b in zip(actual, replay):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_bad_variant_or_shapes_fail_loud(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown model variant"):
            student_models.make_model("tiny")
        model = student_models.make_model("compact")
        with self.assertRaisesRegex(ValueError, "spatial must have shape"):
            model(torch.zeros(1, 22, 13, 13), torch.zeros(1, 19))
        with self.assertRaisesRegex(ValueError, "global_input must have shape"):
            model(torch.zeros(1, 22, 19, 19), torch.zeros(2, 19))

    @unittest.skipIf(torch is None or not torch.backends.mps.is_available(),
                     "Apple MPS is unavailable")
    def test_mps_forward_backward(self) -> None:
        model = student_models.make_model("compact").to("mps")
        spatial = torch.randn(1, 22, 19, 19, device="mps")
        global_input = torch.randn(1, 19, device="mps")
        output = model(spatial, global_input)
        sum(tensor.square().mean() for tensor in output).backward()
        torch.mps.synchronize()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
