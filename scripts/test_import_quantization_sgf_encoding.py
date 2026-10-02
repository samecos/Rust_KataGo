"""CPU-only tests for the explicit FF4 Latin1 import boundary."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name("import_quantization_sgf_encoding.py")
SPEC = importlib.util.spec_from_file_location("encoding_import_under_test", SCRIPT)
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)


def sgf(extra=b"", moves=b";B[pd];W[dp]", ca=b""):
    return b"(;FF[4]GM[1]SZ[19]" + ca + b"RU[Chinese]KM[375]AP[foxwq]" + extra + moves + b")"


class ConversionTests(unittest.TestCase):
    def assert_rejected(self, raw, code):
        with self.assertRaises(IMPORTER.ImportFailure) as caught:
            IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(caught.exception.code, code)

    def test_ascii_default_and_exact_reverse(self):
        raw = sgf(b"PB[A]PW[B]")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(result.data, raw[:2] + b"CA[UTF-8]" + raw[2:])
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        self.assertEqual(result.evidence["operation"]["kind"], "insert-ca")
        self.assertTrue(result.evidence["whole_tree_equivalent_except_encoding_declaration"])

    def test_high_bytes_strict_latin1_roundtrip(self):
        raw = sgf(b"PB[\x80\xb5\xff]PW[\xe4\xbd?]")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        self.assertIn("\u0080\u00b5\u00ff", result.data.decode("utf-8"))

    def test_declared_latin1_replaced_in_original_position(self):
        raw = sgf(b"PB[\xff]CA[ISO-8859-1]C[end]")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(result.evidence["operation"]["kind"], "replace-ca-value")
        self.assertEqual(result.data.count(b"CA[UTF-8]"), 1)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)

    def test_escape_crlf_and_empty_values_preserved(self):
        raw = sgf(b"PB[]C[a\\\r\nb\r\nc\td\\]e\\\\f]", b";B[];W[]") + b"\r\n"
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        self.assertTrue(result.data.endswith(b"\r\n"))
        self.assertIn(b"\\\r\n", result.data)
        self.assertTrue(result.evidence["original_escape_spelling_equal"])

    def test_odd_even_backslashes_high_byte_escape_and_value_delimiters(self):
        for count in [1, 2, 3, 4]:
            comment = b"C[x" + b"\\" * count + b"]" + (b"]" if count % 2 else b"")
            raw = sgf(comment + b"CANNOT[ASCII]")
            result = IMPORTER.convert_sgf_bytes(raw)
            self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        raw = sgf(b"C[(;\x81\\\xff);)\r\n\r\n]")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)

    def test_setup_and_native_us_komi_dependency_are_not_rewritten(self):
        raw = sgf(b"US[GoGoD]AB[aa][bb]AW[cc]HA[2]PL[W]", b";W[];B[dd]")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        self.assertIn(b"KM[375]", result.data)
        self.assert_rejected(sgf(b"US[GoGoD\xff]"), "non_ascii_semantics")

    def test_5c_high_byte_combination_preserves_byte_first_escape(self):
        raw = sgf(b"PB[\x81\\X]")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        self.assertIn(b"\xc2\x81\\X]", result.data)

    def test_5d_is_a_delimiter_not_guessed_gbk_tail(self):
        raw = sgf(b"PB[\xb5]PW[test]")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertIn(b"PB[\xc2\xb5]PW[test]", result.data)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        self.assert_rejected(sgf(b"PB[\x81]X]"), "sgf_syntax")

    def test_nested_branches_and_duplicate_ap_preserved(self):
        raw = sgf(b"AP[GNU Go:3.8]C[\xff]", b";B[aa](;W[bb](;B[cc])(;B[dd]))(;W[ee])")
        result = IMPORTER.convert_sgf_bytes(raw)
        self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
        self.assertEqual(result.evidence["duplicate_ap_occurrences"], 1)
        self.assertEqual(result.evidence["tree_count"], 1)
        self.assertIn(b"(;W[bb](;B[cc])(;B[dd]))", result.data)

    def test_deep_real_style_tree_without_recursive_parser_or_json(self):
        # Deeper than both the first implementation's 128 limit and typical
        # Python recursion limits; only structural encoding behavior is tested.
        for depth in [343, 1100]:
            raw = sgf(moves=b"")[:-1] + b"(;B[aa]" * depth + b")" * (depth + 1)
            result = IMPORTER.convert_sgf_bytes(raw)
            self.assertEqual(IMPORTER.recover_source_bytes(result.data, result.evidence), raw)
            self.assertEqual(result.evidence["node_count_all_branches"], depth + 1)
            self.assertEqual(result.evidence["semantic_evidence_encoding"], "ordered-flat-sgf-events-json-v1")

    def test_deep_tree_resource_and_syntax_failures_remain_explicit(self):
        raw = sgf(moves=b"")[:-1] + b"(;B[aa]" * (IMPORTER.MAX_DEPTH + 1) + b")" * (IMPORTER.MAX_DEPTH + 2)
        self.assert_rejected(raw, "resource_limit")
        with mock.patch.object(IMPORTER, "MAX_NODES", 2):
            self.assert_rejected(sgf(), "resource_limit")
        deep = sgf(moves=b"")[:-1] + b"(;B[aa]" * 343 + b")" * 343
        self.assert_rejected(deep, "sgf_syntax")
        self.assert_rejected(sgf(moves=b"(;B[aa]);W[bb]"), "sgf_syntax")

    def test_encoding_declarations_rejected_conservatively(self):
        for ca in [b"CA[UTF-8]", b"CA[GBK]", b"CA[]", b"CA[latin1]"]:
            with self.subTest(ca=ca):
                self.assert_rejected(sgf(ca=ca), "unsupported_encoding")
        self.assert_rejected(sgf(ca=b"CA[ISO-8859-1][ISO-8859-1]"), "encoding_declaration")
        self.assert_rejected(sgf(ca=b"CA[ISO-8859-1]CA[ISO-8859-1]"), "encoding_declaration")
        self.assert_rejected(sgf(moves=b";B[aa]CA[ISO-8859-1]"), "encoding_declaration")

    def test_ff_is_unique_single_root_property(self):
        for raw in [sgf(b"FF[4]"), sgf().replace(b"FF[4]", b"FF[4][4]"),
                    sgf(moves=b";B[aa]FF[4]"), sgf().replace(b"FF[4]", b""),
                    sgf().replace(b"FF[4]", b"FF[3]")]:
            with self.subTest(raw=raw):
                self.assert_rejected(raw, "format_version")

    def test_duplicate_non_ap_property_and_semantic_arity_rejected(self):
        self.assert_rejected(sgf(b"C[a]C[b]"), "duplicate_property")
        self.assert_rejected(sgf(b"KM[375]"), "duplicate_property")
        self.assert_rejected(sgf(moves=b";B[aa][bb]"), "semantic_arity")

    def test_nn_fields_and_unknown_high_byte_metadata_rejected(self):
        self.assert_rejected(sgf().replace(b"Chinese", b"Chin\xffse"), "non_ascii_semantics")
        self.assert_rejected(sgf(b"AP[\xff]"), "non_ascii_semantics")
        self.assert_rejected(sgf(b"XX[\xff]"), "unsupported_non_ascii_property")

    def test_malformed_and_multiple_games_not_repaired(self):
        for raw in [b"", sgf()[1:], sgf()[:-1], sgf() + b"junk", sgf() + sgf(),
                    b"(;FF[4]C[unterminated)", b"()", b"(;FF)",
                    sgf(moves=b";B[aa]()"), b"\xef\xbb\xbf" + sgf()]:
            with self.subTest(raw=raw):
                self.assert_rejected(raw, "sgf_syntax")

    def test_reversal_rejects_derivative_or_evidence_tampering(self):
        result = IMPORTER.convert_sgf_bytes(sgf())
        with self.assertRaises(IMPORTER.ImportFailure):
            IMPORTER.recover_source_bytes(result.data.replace(b"375", b"325"), result.evidence)
        altered = json.loads(json.dumps(result.evidence))
        altered["operation"]["text_start"] += 1
        with self.assertRaises(IMPORTER.ImportFailure):
            IMPORTER.recover_source_bytes(result.data, altered)

    def test_negative_gates_survive_python_optimized_mode(self):
        code = (f"import sys; sys.path.insert(0, {str(SCRIPT.parent)!r}); "
                "import import_quantization_sgf_encoding as m\n"
                f"base={sgf()!r}\n"
                "bad=[base+base,base.replace(b'FF[4]',b'FF[3]FF[4]'),"
                "base.replace(b'RU[Chinese]',b'RU[Chin\\xffse]'),"
                "base.replace(b';B[pd]',b';B[pd]CA[ISO-8859-1]')]\n"
                "for value in bad:\n"
                " try: m.convert_sgf_bytes(value)\n"
                " except m.ImportFailure: continue\n"
                " raise SystemExit('optimized mode admitted invalid input')\n")
        p = subprocess.run([sys.executable, "-O", "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)


class DirectoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="sgf-encoding-import-test-")
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def test_manifest_mixed_results_preserves_all_input_bytes(self):
        (self.source / "nested").mkdir()
        files = {"nested/one.sgf": sgf(b"PB[\xff]"), "two.sgf": sgf()[1:]}
        for name, data in files.items():
            (self.source / name).write_bytes(data)
        output = self.base / "new-output"
        manifest = IMPORTER.import_directory(self.source, output)
        self.assertEqual(manifest["counts"], {"sources": 2, "accepted": 1, "rejected": 1, "ignored_regular_files": 0})
        self.assertEqual(manifest, json.loads((output / "manifest.json").read_text(encoding="utf-8")))
        self.assertTrue((output / "sgf/nested/one.sgf").is_file())
        for name, data in files.items():
            self.assertEqual((self.source / name).read_bytes(), data)
        accepted = next(r for r in manifest["records"] if r["status"] == "accepted")
        self.assertEqual(accepted["source"]["sha256"], hashlib.sha256(files["nested/one.sgf"]).hexdigest())

    def test_existing_or_nested_output_rejected_before_writing(self):
        (self.source / "a.sgf").write_bytes(sgf())
        existing = self.base / "existing"
        existing.mkdir()
        (existing / "keep").write_bytes(b"unchanged")
        for output in [existing, self.source, self.source / "child", self.source / "a" / "b"]:
            with self.subTest(output=output), self.assertRaises(IMPORTER.ImportFailure):
                IMPORTER.import_directory(self.source, output)
        self.assertEqual((existing / "keep").read_bytes(), b"unchanged")
        self.assertFalse((self.source / "child").exists())

    def test_symlink_file_and_directory_rejected(self):
        original = self.base / "real.sgf"
        original.write_bytes(sgf())
        link = self.source / "link.sgf"
        try:
            link.symlink_to(original)
        except (OSError, NotImplementedError):
            self.skipTest("OS does not permit creating test symlinks")
        output = self.base / "out"
        with self.assertRaises(IMPORTER.ImportFailure):
            IMPORTER.import_directory(self.source, output)
        self.assertFalse(output.exists())
        link.unlink()
        link = self.source / "linked-dir"
        link.symlink_to(self.base, target_is_directory=True)
        with self.assertRaises(IMPORTER.ImportFailure):
            IMPORTER.import_directory(self.source, output)
        link.unlink()

    def test_windows_junction_rejected(self):
        if os.name != "nt":
            self.skipTest("Windows reparse-point check")
        target = self.base / "target"
        target.mkdir()
        (target / "keep").write_bytes(b"keep")
        link = self.source / "junction"
        p = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        if p.returncode:
            self.skipTest("Cannot create test junction")
        try:
            with self.assertRaises(IMPORTER.ImportFailure):
                IMPORTER.import_directory(self.source, self.base / "out")
        finally:
            os.rmdir(link)
        self.assertEqual((target / "keep").read_bytes(), b"keep")

    def test_unsafe_relative_paths_rejected(self):
        for name in ["../a.sgf", "/a.sgf", "C:/a.sgf", "a\\b.sgf", "a:stream.sgf", "CON.sgf", "a./b.sgf",
                     "COM¹.sgf", "NUL .sgf", "a?.sgf", 'a".sgf']:
            with self.subTest(name=name), self.assertRaises(IMPORTER.ImportFailure):
                IMPORTER.validate_relative_path(name)

    def test_cli_help_is_cpu_only(self):
        p = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0)
        self.assertIn("--input", p.stdout)
        self.assertIn("--output", p.stdout)


if __name__ == "__main__":
    unittest.main()
