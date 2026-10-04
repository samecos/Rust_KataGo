"""Exporter ABI, corruption rejection, and FP32 golden input checks."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

import export_student as export
import student_models
import student_reference
import verify_student_cuda


class ExportStudentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(4)

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.output = self.folder / "compact.rgmodel"
        torch.manual_seed(29)
        self.model = student_models.make_model("compact")

    def write_export(self) -> dict:
        return export.export_model(self.model.state_dict(), "compact", self.output)

    def rewrite(self, change, *, payload_change=None) -> None:
        raw = self.output.read_bytes()
        header_length = struct.unpack("<I", raw[16:20])[0]
        header = json.loads(raw[20:20 + header_length])
        payload = raw[20 + header_length:]
        change(header)
        header_bytes = json.dumps(header).encode("utf-8")
        if payload_change:
            payload = payload_change(payload)
        self.output.write_bytes(export.MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + payload)

    def test_both_variants_exact_roundtrip_and_byte_layout(self) -> None:
        self.assertEqual(len(export.MAGIC), 16)
        for variant in ("compact", "dense"):
            with self.subTest(variant=variant):
                model = student_models.make_model(variant)
                output = self.folder / f"{variant}.anything"
                receipt = export.export_model(model.state_dict(), variant, output)
                header, restored = export.read_export(output)
                self.assertEqual({name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}, export.expected_tensor_shapes(variant))
                self.assertEqual(header["input_version"], 7)
                self.assertEqual(receipt["bytes"], 20 + receipt["header_bytes"] + receipt["payload_bytes"])
                self.assertEqual(receipt["parameters"], student_models.count_parameters(model))
                self.assertEqual(receipt["sha256"], export.sha256_file(output))
                self.assertEqual(json.loads(Path(str(output) + ".receipt.json").read_text())["sha256"], receipt["sha256"])
                raw = output.read_bytes()
                first = model.stem.weight.detach().numpy().reshape(-1)
                offset = 20 + receipt["header_bytes"]
                np.testing.assert_array_equal(np.frombuffer(raw[offset:offset + first.nbytes], dtype="<f4"), first)
                for key, tensor in model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, restored[key]), key)

    def test_original_checkpoint_export_uses_metadata(self) -> None:
        checkpoint = self.folder / "trained.pt"
        torch.save({"variant": "compact", "score_scale_points": 20.0, "model_state": self.model.state_dict(), "epoch": 3}, checkpoint)
        receipt = export.export_checkpoint(checkpoint, self.output)
        self.assertEqual(receipt["source"]["sha256"], export.sha256_file(checkpoint))
        restored, _ = export.load_exported_model(self.output)
        spatial = torch.zeros(1, 22, 19, 19)
        global_input = torch.zeros(1, 19)
        with torch.no_grad():
            for source, replay in zip(self.model(spatial, global_input), restored(spatial, global_input), strict=True):
                torch.testing.assert_close(source, replay, atol=0, rtol=0)

    def test_checkpoint_missing_scale_unknown_variant_and_architecture_rejected(self) -> None:
        base = {"variant": "compact", "score_scale_points": 20.0, "model_state": self.model.state_dict()}
        for patch in ({"score_scale_points": None}, {"score_scale_points": True}, {"variant": "tiny"}, {"width": 64}, {"blocks": True}):
            with self.subTest(patch=patch):
                checkpoint = self.folder / "invalid.pt"
                torch.save({**base, **patch}, checkpoint)
                with self.assertRaises(ValueError):
                    export.export_checkpoint(checkpoint, self.output)
                self.assertFalse(self.output.exists())

    def test_invalid_parameters_do_not_replace_existing_output(self) -> None:
        self.output.write_bytes(b"existing artifact")
        original = self.model.state_dict()
        invalid = []
        missing = dict(original)
        missing.pop("stem.bias")
        invalid.append(missing)
        invalid.append({**original, "unexpected": torch.ones(1)})
        invalid.append({**original, "stem.bias": torch.ones(1)})
        invalid.append({**original, "stem.bias": torch.ones(48, dtype=torch.int64)})
        invalid.append({**original, "stem.bias": torch.full((48,), float("nan"))})
        invalid.append({**original, "stem.bias": torch.full((48,), 1e300, dtype=torch.float64)})
        for state in invalid:
            with self.subTest(keys=len(state)):
                with self.assertRaises(ValueError):
                    export.export_model(state, "compact", self.output)
                self.assertEqual(self.output.read_bytes(), b"existing artifact")

    def test_fp16_checkpoint_weights_have_exact_fp32_conversion(self) -> None:
        half_state = {name: tensor.half() for name, tensor in self.model.state_dict().items()}
        export.export_model(half_state, "compact", self.output)
        _, restored = export.read_export(self.output)
        for name in restored:
            self.assertTrue(torch.equal(restored[name], half_state[name].float()), name)

    def test_corrupt_header_and_tensor_descriptor_rejected(self) -> None:
        changes = [
            lambda h: h.update(format_version=True),
            lambda h: h.update(input_version=8),
            lambda h: h.update(width=64),
            lambda h: h.update(score_scale=1),
            lambda h: h.update(extra=1),
            lambda h: h["tensors"][0].update(offset=4),
            lambda h: h["tensors"][0].update(length=1),
            lambda h: h["tensors"][0].update(name="stem.bias"),
            lambda h: h["tensors"][0].update(shape=[48, 22, 3, True]),
            lambda h: h["tensors"].pop(),
        ]
        for change in changes:
            with self.subTest(change=change):
                self.write_export()
                self.rewrite(change)
                with self.assertRaises(ValueError):
                    export.read_export(self.output)

    def test_payload_truncation_trailing_bytes_and_nonfinite_rejected(self) -> None:
        for mutation in (lambda p: p[:-1], lambda p: p + b"\x00", lambda p: struct.pack("<f", float("inf")) + p[4:]):
            with self.subTest(mutation=mutation):
                self.write_export()
                self.rewrite(lambda h: None, payload_change=mutation)
                with self.assertRaises(ValueError):
                    export.read_export(self.output)

    def test_bad_magic_header_length_and_duplicate_fields_rejected(self) -> None:
        for content in (b"wrong", b"x" * 20, export.MAGIC + struct.pack("<I", 2 ** 32 - 1),
                        export.MAGIC + struct.pack("<I", 25) + b'{"variant":1,"variant":2}'):
            self.output.write_bytes(content)
            with self.assertRaises(ValueError):
                export.read_export(self.output)

    def test_contiguous_tensor_permutations_match_rust_reader_contract(self) -> None:
        self.write_export()
        raw = self.output.read_bytes()
        header_length = struct.unpack("<I", raw[16:20])[0]
        header = json.loads(raw[20:20 + header_length])
        payload = raw[20 + header_length:]
        header["tensors"].reverse()
        chunks = []
        offset = 0
        for tensor in header["tensors"]:
            length = tensor["length"] * 4
            chunks.append(payload[tensor["offset"]:tensor["offset"] + length])
            tensor["offset"] = offset
            offset += length
        encoded = json.dumps(header).encode("utf-8")
        self.output.write_bytes(export.MAGIC + struct.pack("<I", len(encoded)) + encoded + b"".join(chunks))
        _, state = export.read_export(self.output)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(state[name], value))

    def test_synthetic_reference_binary_layout_and_output_identity(self) -> None:
        self.write_export()
        manifest = student_reference.write_reference(self.output, self.folder / "reference", [1, 3])
        self.assertEqual(manifest["tf32"], False)
        for case in manifest["cases"]:
            batch = case["batch"]
            spatial = np.fromfile(case["spatial"], dtype="<f4").reshape(batch, 22, 19, 19)
            global_input = np.fromfile(case["global_input"], dtype="<f4").reshape(batch, 19)
            reference = json.loads(Path(case["reference"]).read_text())
            self.assertEqual(reference["shapes"]["policy_logits"], [batch, 362])
            with torch.no_grad():
                expected = self.model(torch.from_numpy(spatial), torch.from_numpy(global_input))
            for name in student_reference.HEADS:
                np.testing.assert_array_equal(reference[name], getattr(expected, name).numpy().reshape(-1))

    def test_packed_input_reference_decodes_msb_bits_and_checks_contract(self) -> None:
        bits = np.zeros((3, 22, 361), dtype=np.uint8)
        bits[0, 0, 0] = 1
        bits[1, 21, 360] = 1
        path = self.folder / "inputs.npz"
        np.savez(path, packed=np.packbits(bits, axis=2, bitorder="big"), global_input=np.ones((3, 19), np.float32))
        spatial, global_input, _ = student_reference.input_pool(3, 29, path)
        self.assertEqual(spatial[0, 0, 0, 0], 1)
        self.assertEqual(spatial[1, 21, 18, 18], 1)
        self.assertEqual(spatial.sum(), 2)
        self.assertEqual(global_input.shape, (3, 19))
        with self.assertRaisesRegex(ValueError, "insufficient"):
            student_reference.input_pool(4, 29, path)


class CompareOutputsTest(unittest.TestCase):
    def reference(self) -> dict:
        shapes = {"policy_logits": [1, 362], "value_logits": [1, 3], "score": [1], "ownership_logits": [1, 361]}
        result = {"batch": 1, "shapes": shapes}
        for name, shape in shapes.items():
            result[name] = np.full(int(np.prod(shape)), 0.25).tolist()
        return result

    def test_exact_and_tolerated_outputs_pass_both_limits(self) -> None:
        reference = self.reference()
        actual = {name: np.asarray(reference[name]) + 1e-5 for name in student_reference.HEADS}
        result = verify_student_cuda.compare_outputs(reference, actual, atol=1e-4, rtol=1e-4, max_abs=2e-4)
        self.assertTrue(result["passed"])
        self.assertEqual(result["heads"]["policy_logits"]["values"], 362)

    def test_single_bad_value_and_global_absolute_cap_fail(self) -> None:
        reference = self.reference()
        actual = {name: list(reference[name]) for name in student_reference.HEADS}
        actual["ownership_logits"][200] += 0.001
        result = verify_student_cuda.compare_outputs(reference, actual, atol=1e-4, rtol=1e-4, max_abs=2e-4)
        self.assertFalse(result["passed"])
        self.assertEqual(result["heads"]["ownership_logits"]["element_tolerance_failures"], 1)
        actual = {name: np.asarray(reference[name]) + 1e-4 for name in student_reference.HEADS}
        result = verify_student_cuda.compare_outputs(reference, actual, atol=1, rtol=1, max_abs=1e-5)
        self.assertFalse(result["passed"])

    def test_missing_truncated_and_nonfinite_outputs_rejected(self) -> None:
        reference = self.reference()
        mutations = (lambda a: a.pop("value_logits"), lambda a: a["policy_logits"].pop(),
                     lambda a: a["score"].__setitem__(0, float("nan")))
        for mutation in mutations:
            actual = {name: list(reference[name]) for name in student_reference.HEADS}
            mutation(actual)
            with self.assertRaises(ValueError):
                verify_student_cuda.compare_outputs(reference, actual, atol=1e-4, rtol=1e-4, max_abs=2e-4)


if __name__ == "__main__":
    unittest.main()
