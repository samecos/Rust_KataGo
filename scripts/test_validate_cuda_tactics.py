"""CPU checks for tactic marker matching and capability-dependent cases."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from validate_cuda_tactics import Case, has_marker, numeric_environment, padbatch_case, q64_serial_cases, run_case, run_numeric_gate


class TacticValidationTests(unittest.TestCase):
    def test_q64_token_does_not_match_serial_variant(self):
        log = "[cuda-tactic] name=attention requested=fa2 launch=fa2 effective=fa2 tile=q64-serial\n"
        self.assertFalse(has_marker(log, "tile=q64"))
        self.assertTrue(has_marker(log, "tile=q64-serial"))
        self.assertFalse(has_marker(log, "tile=q128"))
        self.assertTrue(has_marker(log.replace("q64-serial", "q64"), "tile=q64"))

    def test_error_fragments_still_match_punctuation(self):
        self.assertTrue(has_marker("Error: kernel_build_id mismatch: plan abc", "kernel_build_id mismatch"))
        self.assertTrue(has_marker("invalid value 'bad' for tactic", "invalid value"))
        self.assertFalse(has_marker("unrelated_launch=fused", "launch=fused"))

    def test_numeric_dump_uses_plan_over_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = Path(directory) / "plan.json"
            plan.write_text(json.dumps({"apply": {"tactic_overrides": {
                "KATAGO_CUDA_ATTN_TILE": "q64-serial", "KATAGO_CUDA_DUALFFN": "1",
            }}}), encoding="utf-8")
            case = Case("priority", env={"KATAGO_CUDA_ATTN_TILE": "q128", "KATAGO_CUDA_DUALFFN": "0"}, plan=plan)
            with patch.dict("os.environ", {"KATAGO_CUDA_NOGRAPH": "1"}, clear=True):
                env = numeric_environment(case)
            self.assertEqual(env["KATAGO_CUDA_ATTN_TILE"], "q64-serial")
            self.assertEqual(env["KATAGO_CUDA_DUALFFN"], "1")
            self.assertNotIn("KATAGO_CUDA_NOGRAPH", env)

    def test_full_capability_exercises_env_and_plan_priority(self):
        fp = {"backend_build": {"capabilities": {"attention_q64": True, "attention_q64_serial": True}}}
        cases = q64_serial_cases(fp, Path("serial.json"))
        self.assertEqual(len(cases), 2)
        self.assertTrue(all(case.exit_code == 0 and case.numeric for case in cases))
        self.assertEqual(cases[1].env["KATAGO_CUDA_ATTN_TILE"], "q128")
        self.assertEqual(cases[1].plan, Path("serial.json"))

    def test_ordinary_q64_does_not_qualify_for_serial_capability(self):
        fp = {"backend_build": {"capabilities": {"attention_q64": True, "attention_q64_serial": False}}}
        cases = q64_serial_cases(fp, Path("serial.json"))
        self.assertEqual(cases[0].exit_code, 0)
        self.assertIn("fallback=q128", cases[0].expect[0])
        self.assertEqual(cases[1].exit_code, 1)
        self.assertIn("backend capability attention_q64_serial=false", cases[1].expect)
        self.assertFalse(any(case.numeric for case in cases))

    def test_numeric_gate_rejects_a_different_attention_path(self):
        marker = "name=attention requested=fa2 launch=fa2 effective=fa2 tile=q64-serial"
        case = Case("serial", env={"KATAGO_CUDA_ATTN_TILE": "q64-serial"}, expect=(marker,), numeric=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "serial-numeric").mkdir()
            (output / "serial-numeric/meta.json").write_text("{}", encoding="utf-8")
            completed = SimpleNamespace(returncode=0, stdout=marker.replace("q64-serial", "q128"), stderr="")
            with patch("validate_cuda_tactics.subprocess.run", return_value=completed) as run:
                errors = run_numeric_gate(Path("unused.exe"), Path("model.onnx"), output,
                                          case, Path("compare.py"), Path("python"))
            self.assertEqual(run.call_count, 1, "wrong-path dump must fail before numeric comparison")
            self.assertIn("numeric dump missing attention marker", errors[0])

    def test_padbatch_fixture_declares_capacity_above_active_clients(self):
        case = padbatch_case()
        self.assertEqual(case.batch, (3, 4))
        self.assertGreater(case.workers, 1)
        self.assertLess(case.workers, max(case.batch))
        with tempfile.TemporaryDirectory() as directory:
            # This fake subprocess tests only command construction and evidence
            # checking. The actual GPU marker still has to pass in run_case.
            completed = SimpleNamespace(returncode=0,
                stdout="(workers 3, physicalMax 4)\n[cuda-tactic] name=padbatch launch=on phys=4 rows=3\n",
                stderr="")
            with patch("validate_cuda_tactics.subprocess.run", return_value=completed) as run:
                errors = run_case(Path("unused.exe"), Path("model.onnx"), Path(directory), case)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--batch") + 1], "3,4")
            self.assertEqual(command[command.index("--workers") + 1], "3")
            self.assertEqual(errors, [])

    def test_numeric_gate_rejects_inactive_large_batch_layout(self):
        marker = "name=gemm_layout launch=nn_k384 effective=nn_k384 k=384 requested=nn_k384_b8"
        case = Case("layout", expect=(marker,), numeric=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "layout-numeric").mkdir()
            (output / "layout-numeric/meta.json").write_text("{}", encoding="utf-8")
            completed = SimpleNamespace(returncode=0,
                stdout="name=gemm_layout launch=tn effective=tn k=384 requested=nn_k384_b8", stderr="")
            with patch("validate_cuda_tactics.subprocess.run", return_value=completed) as run:
                errors = run_numeric_gate(Path("unused.exe"), Path("model.onnx"), output,
                                          case, Path("compare.py"), Path("python"))
            self.assertEqual(run.call_count, 1, "an inactive candidate must fail before comparison")
            self.assertIn("numeric dump missing gemm_layout marker", errors[0])

    def test_padbatch_fixture_does_not_pass_without_actual_padding_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = SimpleNamespace(returncode=0, stdout="(workers 3, physicalMax 4)\n", stderr="")
            with patch("validate_cuda_tactics.subprocess.run", return_value=completed):
                errors = run_case(Path("unused.exe"), Path("model.onnx"), Path(directory), padbatch_case())
            self.assertEqual(errors, ["missing marker: name=padbatch launch=on phys=4"])


if __name__ == "__main__":
    unittest.main()
