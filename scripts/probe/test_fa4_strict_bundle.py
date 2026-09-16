"""CPU integrity checks for the measured bundle; no compiler/CUDA imports."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import fa4_strict_bundle as bundle_tool


class BundleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name) / "relocated-bundle"
        shutil.copytree(bundle_tool.DEFAULT_BUNDLE, self.bundle,
                        ignore=shutil.ignore_patterns("__pycache__"))

    def test_relocated_bundle_matches_registered_identity_without_subprocesses(self):
        with patch.object(bundle_tool.subprocess, "run", side_effect=AssertionError("CPU only")):
            result = bundle_tool.verify_bundle(self.bundle)
        self.assertEqual(result["status"], "CPU_BUNDLE_IDENTITY_PASS")
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(result["runtime_fingerprint"],
                         "sha256:e142b91fe865445357aa754015d4ed9693ff8f9aa4a65eb631dbb9183c17fbc0")

    def test_runtime_byte_corruption_is_rejected(self):
        artifact = self.bundle / "attention.sm120.cubin"
        data = bytearray(artifact.read_bytes())
        data[-1] ^= 1
        artifact.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            bundle_tool.verify_bundle(self.bundle)

    def test_source_corruption_cannot_hide_behind_runtime_identity(self):
        original = bundle_tool.artifact_fingerprint(self.bundle)
        source = self.bundle / "source/helpers.cu"
        source.write_bytes(source.read_bytes() + b"\n// changed\n")
        self.assertEqual(original, bundle_tool.artifact_fingerprint(self.bundle))
        with self.assertRaisesRegex(ValueError, "file identity mismatch: source/helpers.cu"):
            bundle_tool.verify_bundle(self.bundle)

    def test_manifest_cannot_read_outside_bundle(self):
        path = self.bundle / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["files"][0]["path"] = "../outside-file"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "escaping manifest path"):
            bundle_tool.verify_bundle(self.bundle)

    def test_unregistered_extra_file_is_rejected(self):
        (self.bundle / "unregistered.cubin").write_bytes(b"candidate")
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            bundle_tool.verify_bundle(self.bundle)

    def test_missing_recorded_evidence_is_rejected(self):
        (self.bundle / "evidence/no-vcopy-run-r1.json").unlink()
        with self.assertRaises(FileNotFoundError):
            bundle_tool.verify_bundle(self.bundle)


if __name__ == "__main__":
    unittest.main()
