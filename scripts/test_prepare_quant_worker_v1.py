"""Five CPU-only checks. No executable, CUDA DLL, model or socket is started."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import unittest

import prepare_quant_worker_v1 as p


UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def fixture_export():
    envelope = {"schema": "rustgo-precision-recipe", "version": 1,
                "quantization_semantics_version": 1, "model_sha256": "a" * 64,
                "graph_sha256": p.GRAPH, "projections": [
                    {"id": "a", "expected_n": 3, "expected_k": 8, "precision": "fp16"},
                    {"id": "b", "expected_n": 8, "expected_k": 3, "precision": "int8"}]}
    # This is the ordered serde struct layout, independent of compact().
    raw = ('{"schema":"rustgo-precision-recipe","version":1,'
           '"quantization_semantics_version":1,"model_sha256":"' + "a" * 64 +
           '","graph_sha256":"' + p.GRAPH + '","projections":['
           '{"id":"a","expected_n":3,"expected_k":8,"precision":"fp16"},'
           '{"id":"b","expected_n":8,"expected_k":3,"precision":"int8"}]}').encode()
    digest = hashlib.sha256(raw).hexdigest()
    return {"schema": "rustgo-recipe-identity-export", "version": 1,
            "model_sha256": "a" * 64, "graph_sha256": p.GRAPH,
            "source_recipe_file_sha256": "b" * 64, "recipe_sha256": digest,
            "canonical_encoding": "serde-json-compact-ordered-v1", "canonical_envelope": envelope}, raw


class PrepareQuantWorkerV1Tests(unittest.TestCase):
    def test_recipe_struct_order_and_raw_source_are_distinct(self):
        export, golden = fixture_export()
        self.assertEqual(p.compact(export["canonical_envelope"], ordered=True), golden)
        self.assertEqual(p.recipe_identity(export, "a" * 64, "b" * 64), hashlib.sha256(golden).hexdigest())
        self.assertNotEqual(hashlib.sha256(p.compact(export["canonical_envelope"])).hexdigest(), export["recipe_sha256"])
        for kind in ("reorder", "raw-source", "mxfp8", "unsorted"):
            bad = copy.deepcopy(export)
            if kind == "reorder":
                envelope = bad["canonical_envelope"]
                bad["canonical_envelope"] = {key: envelope[key] for key in sorted(envelope)}
            elif kind == "raw-source":
                bad["recipe_sha256"] = "b" * 64
            elif kind == "mxfp8":
                bad["canonical_envelope"]["projections"][0]["precision"] = "mxfp8"
            else:
                bad["canonical_envelope"]["projections"].reverse()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                p.recipe_identity(bad, "a" * 64, "b" * 64)

    def test_quant_json_sort_null_and_new_binary_identity(self):
        # Golden JSON covers nested key ordering, UTF-8, exact null, bool and
        # integer encoding (the types present in the actual Rust contract).
        self.assertEqual(p.compact({"z": None, "a": {"z": True, "a": "é\n"}, "n": 13030}),
                         b'{"a":{"a":"\xc3\xa9\\n","z":true},"n":13030,"z":null}')
        with self.assertRaisesRegex(ValueError, "unsupported JSON scalar"):
            p.compact({"not_supported": 1.5})
        fingerprint = {"gpu_name": "NVIDIA GeForce RTX 5070 Ti", "compute_capability": "12.0",
                       "sm_count": 70, "l2_cache_bytes": 50331648,
                       "backend_build": {"kernel_build_id": "fixture", "compiled_sm": ["sm_120"],
                                         "capabilities": {"dual_ffn": False}, "cublaslt_version": 130100}}
        env = p.selected_environment({}, UUID)
        execution = {"nnMaxBatchSize": "3", "numNNServerThreadsPerModel": "1",
                     "nnUseFP16": None, "nnUseNHWC": None}
        contract, identity = p.profile_contract("a" * 64, "b" * 64, "c" * 64,
                                                fingerprint, 13030, env, execution)
        self.assertEqual(set(contract), {"schema", "model_sha256", "recipe_sha256", "executable_sha256",
                                        "backend_build", "device", "driver_version", "tactics",
                                        "actual_execution", "execution_config"})
        self.assertEqual(len(contract["tactics"]), 25)
        self.assertIsNone(contract["tactics"]["KATAGO_CUDA_MXFP8_SCALE_CLEAR_FUSION"])
        self.assertNotIn("KATAGO_CUDA_BATCH_TRACE", contract["tactics"])
        self.assertEqual(contract["actual_execution"], {"mxfp8_scale_clear_fusion": False})
        self.assertEqual(contract["execution_config"], execution)
        self.assertEqual(identity, "rustgo-quant-v1:" + hashlib.sha256(
            json.dumps(contract, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest())
        _, changed = p.profile_contract("a" * 64, "b" * 64, "d" * 64,
                                         fingerprint, 13030, env, execution)
        self.assertNotEqual(identity, changed)
        _, changed_driver = p.profile_contract("a" * 64, "b" * 64, "c" * 64,
                                                fingerprint, 13031, env, execution)
        self.assertNotEqual(identity, changed_driver)
        env["KATAGO_CUDA_STEM_MAP3D"] = "0"
        with self.assertRaisesRegex(ValueError, "unexpected optional tactic"):
            p.profile_contract("a" * 64, "b" * 64, "c" * 64, fingerprint, 13030, env, execution)

    def test_environment_does_not_inherit_optional_tactics(self):
        values = {"Path": "fixture", "katago_cuda_stem_map3d": "1", "KATAGO_NN_BATCH_WINDOW_US": "9",
                  "Cuda_Visible_Devices": "old", "KATAGO_CUDA_INT8_GEMM_TUNE": "1"}
        env = p.selected_environment(values, UUID)
        self.assertEqual(env["Path"], "fixture")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], UUID)
        self.assertEqual(env["KATAGO_CUDA_INT8_GEMM_TUNE"], "0")
        self.assertFalse(any(k.upper() in ("KATAGO_CUDA_STEM_MAP3D", "KATAGO_NN_BATCH_WINDOW_US") for k in env))
        self.assertEqual(values["katago_cuda_stem_map3d"], "1")
        for bad in ({"PATH": "a", "Path": "b"}, {"bad=name": "v"}, {"x": "nul\0"}):
            with self.assertRaises(ValueError):
                p.selected_environment(bad, UUID)
        with self.assertRaises(ValueError):
            p.selected_environment({}, "0,1")
        # The generated launcher embeds a bound Source as data, including odd
        # path characters. This inspects generation only, never runs PowerShell.
        command_source = {"path": "D:\\fixture\\quote'\n表达.json", "bytes": 17, "sha256": "f" * 64}
        script = p.launcher_script(command_source).decode("utf-8")
        literal = re.search(r"FromBase64String\('([A-Za-z0-9+/=]+)'\)", script).group(1)
        self.assertEqual(p.decode(base64.b64decode(literal)), command_source)
        self.assertNotIn(command_source["path"], script)
        self.assertIn("param([switch]$Once)", script)
        self.assertIn("& $workerExecutable @workerArguments", script)

    def test_fixed_worker_config_hash_fields_and_rejection(self):
        recipe = str(Path("fixture-recipe.json").resolve())
        for batch in (3, 8):
            raw = (f"nnBackend = cudaquantbackend\nnnMaxBatchSize = {batch}\n"
                   "numNNServerThreadsPerModel = 1\nuseFP16 = auto\ninputsUseNHWC = false\n"
                   f"cudaQuantPlan = {recipe}\n").encode()
            self.assertEqual(p.config_values(raw, batch, recipe),
                             {"nnMaxBatchSize": str(batch), "numNNServerThreadsPerModel": "1",
                              "nnUseFP16": None, "nnUseNHWC": None})
            for suffix in ("cudaQuantExpectedProfile = old\n", "cudaTacticPlan = other\n",
                           "nnUseFP16 = true\n", "nnMaxBatchSize = 99\n"):
                with self.subTest(batch=batch, suffix=suffix), self.assertRaises(ValueError):
                    p.config_values(raw + suffix.encode(), batch, recipe)
            with self.assertRaises(ValueError):
                p.config_values(raw, batch, recipe + ".changed")

    def test_source_change_duplicate_json_and_no_overwrite(self):
        # Deliberately retain tiny fixtures; no recursive Windows cleanup.
        directory = Path(tempfile.mkdtemp(prefix="rustgo-prepare-worker-cpu-"))
        target = directory / "fixture.bin"
        p.publish(target, b"first")
        deadline = time.monotonic_ns() + 10_000_000_000
        inventory = p.Inventory(deadline)
        item = inventory.add(target)
        self.assertEqual(inventory.raw(item), b"first")
        inventory.recheck()
        with self.assertRaises(FileExistsError):
            p.publish(target, b"replacement")
        self.assertEqual(target.read_bytes(), b"first")
        with self.assertRaisesRegex(ValueError, "Source SHA mismatch"):
            p.Inventory(deadline).add(target, "0" * 64)
        target.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            inventory.recheck()
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            p.decode(b'{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, "original preparation deadline"):
            p.remaining(time.monotonic_ns() - 1)
        build = {"schema": "rustgo-typed-worker-native-build-result-v1", "actual_exit_code": 0,
                 "tool_actual_exit_code": 0, "sources_unchanged": True,
                 "root_terminal_before_deadline": True, "binary": item,
                 "status": "BUILT_NOT_GPU_QUALIFIED", "original_wire_protocol": True,
                 "new_binary_speed_qualified": False}
        p.validate_build(build, item)
        for key, value in (("binary", {**item, "sha256": "0" * 64}), ("actual_exit_code", False),
                           ("original_wire_protocol", False), ("new_binary_speed_qualified", True)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                p.validate_build({**build, key: value}, item)


if __name__ == "__main__":
    unittest.main()
