"""Safety and reproducibility checks using tiny synthetic tar archives."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import prepare_tar_sample


def _write_tar(path: Path, files: dict[str, bytes], *, link: bool = False,
               mode: str = "w") -> None:
    with tarfile.open(path, mode) as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if link:
            info = tarfile.TarInfo("fixture/tdata/link.npz")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../outside"
            archive.addfile(info)


class PrepareTarSampleTests(unittest.TestCase):
    def test_selects_by_member_path_and_records_exact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.tar"
            files = {f"fixture/tdata/{name}.npz": name.encode()
                     for name in ("AAA", "BBB", "CCC", "DDD")}
            _write_tar(source, files)
            data_root = root / "data"
            with patch.object(prepare_tar_sample, "DATA_ROOT", data_root):
                manifest = prepare_tar_sample.prepare_archive(source, 2, data_root / "fixture")
            ranked = sorted(files, key=lambda name: (hashlib.sha256(name.encode()).hexdigest(), name))[:2]
            selected = {item["archive_path"] for item in manifest["selected_files"]}
            self.assertEqual(selected, set(ranked))
            self.assertEqual(manifest["source_tar"]["npz_members_total"], 4)
            self.assertEqual(manifest["source_tar"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            saved = json.loads((data_root / "fixture" / "manifest.json").read_text())
            self.assertEqual(saved, manifest)
            for item in manifest["selected_files"]:
                payload = (data_root / "fixture" / item["extracted_path"]).read_bytes()
                self.assertEqual(payload, files[item["archive_path"]])
                self.assertEqual(item["sha256"], hashlib.sha256(payload).hexdigest())

    def test_rejects_traversal_before_writing_any_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.tar"
            _write_tar(source, {
                "fixture/tdata/okay.npz": b"okay",
                "fixture/../outside.npz": b"bad",
            })
            data_root = root / "data"
            with patch.object(prepare_tar_sample, "DATA_ROOT", data_root):
                with self.assertRaisesRegex(ValueError, "unsafe tar member path"):
                    prepare_tar_sample.prepare_archive(source, 1, data_root / "fixture")
            self.assertFalse(data_root.exists())

    def test_rejects_symlink_even_if_it_would_not_be_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.tar"
            _write_tar(source, {"fixture/tdata/okay.npz": b"okay"}, link=True)
            data_root = root / "data"
            with patch.object(prepare_tar_sample, "DATA_ROOT", data_root):
                with self.assertRaisesRegex(ValueError, "not a regular file"):
                    prepare_tar_sample.prepare_archive(source, 1, data_root / "fixture")
            self.assertFalse(data_root.exists())

    def test_gzip_tar_with_model_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.tgz"
            files = {f"fixture/kata1-model/{name}.npz": name.encode()
                     for name in ("AAA", "BBB", "CCC")}
            _write_tar(source, files, mode="w:gz")
            data_root = root / "data"
            with patch.object(prepare_tar_sample, "DATA_ROOT", data_root):
                manifest = prepare_tar_sample.prepare_archive(source, 2, data_root / "fixture")
            ranked = sorted(files, key=lambda name: (hashlib.sha256(name.encode()).hexdigest(), name))[:2]
            selected = {item["archive_path"] for item in manifest["selected_files"]}
            self.assertEqual(selected, set(ranked))
            for item in manifest["selected_files"]:
                self.assertTrue(item["extracted_path"].startswith("kata1-model/"))
                payload = (data_root / "fixture" / item["extracted_path"]).read_bytes()
                self.assertEqual(payload, files[item["archive_path"]])


if __name__ == "__main__":
    unittest.main()
