"""Hermetic CPU receipts; synthetic protobuf outputs are never GPU evidence."""
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_quantization_corpus as benchmark
from benchmark_workers import distribution
import collect_quantization_corpus as collection
import compare_quantization_corpus as comparison
from compare_worker_outputs import message_dict
import plan_quantization_search as planner
import quantization_selection_ledger as ledger
from tune_runtime import decide
from worker_protocol_tools import Protocol


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(planner.pretty(value))


def lines(values):
    return "".join(collection.json_line(v) for v in values).encode()


def artifact(path):
    raw = path.read_bytes()
    return dict(file=path.name, bytes=len(raw), sha256=planner.digest(raw))


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = Protocol(ledger.ROOT / "crates/kata_worker/proto/worker.proto")
        cls.target = ledger.ROOT / "target" / "quantization-ledger-cpu-tests"
        cls.target.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.protocol.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-", dir=self.target)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        sgfs = self.root / "sgfs"
        sgfs.mkdir()
        (sgfs / "synthetic.sgf").write_bytes(b"(;GM[1]SZ[19]RU[Chinese]KM[7.5];B[aa])")
        game_id = planner.digest(b"synthetic-game")
        moves = [[1 + i % 2, i] for i in range(8)]
        self.game = dict(game_id=game_id, split="calibration", retained_plies=[8],
                         sources=[dict(path="synthetic.sgf", sha256=planner.file_digest(sgfs / "synthetic.sgf"))],
                         semantics=dict(board_size=19, komi_half_points=15, initial_player=1, initial_stones=[], moves=moves))
        params = {key: False for key in collection.PARAMETER_BOOLS}
        params.update(include_ownership=True, skip_cache=True, policy_temperature=1.0, policy_optimism=0.0,
                      draw_equivalent_wins_for_white=0.5, playout_doubling_advantage=0.0, symmetry=0, max_history=1000)
        self.record = dict(name="synthetic-position", game_id=game_id, split="calibration", ply=8, phase="early",
                           semantic_position_sha256=planner.digest(b"semantic"), board_state_sha256=planner.digest(b"board"),
                           position=dict(board_size=19, rules="chinese", komi=7.5, initial_player=1, next_player=1,
                                         initial_stones=[], moves=[dict(color=c, vertex=v) for c, v in moves]), parameters=params)
        blobs = {"games.jsonl": lines([self.game]), "rejections.jsonl": b"", "duplicates.jsonl": b""}
        for split in planner.SPLITS:
            blobs[split + ".requests.jsonl"] = lines([self.record]) if split == "calibration" else b""
            blobs[split + ".worker-fixture.json"] = b"{}"
        for name, raw in blobs.items():
            (self.corpus / name).write_bytes(raw)
        counts = [dict(split=s, assigned_games=int(s == "calibration"), games_with_retained_positions=int(s == "calibration"),
                       distinct_semantic_positions=int(s == "calibration"), requested_games=1) for s in planner.SPLITS]
        manifest = {key: None for key in planner.CORPUS_KEYS}
        manifest.update(schema=collection.SCHEMA, status="NOT_READY", production_certified=False,
                        options=dict(input=str(sgfs), output=str(self.corpus), seed="synthetic", games_per_split=[1, 1, 1],
                                     positions_per_game=1, assume_rules=None, assume_komi=None),
                        artifacts=[artifact(self.corpus / name) for name in blobs], unique_games=1, rejected_sources=0, duplicate_records=0,
                        request_set_sha256=planner.digest(planner.compact([[self.record], [], []])),
                        readiness=dict(status="NOT_READY", symmetry_multiplicity_counted=False, holdout_minimum_games=128,
                                       holdout_minimum_distinct_positions=4096, splits=counts))
        put(self.corpus / "manifest.json", manifest)
        self.model = self.root / "synthetic-model.bin"
        self.model.write_bytes(b"SYNTHETIC CPU TEST ONLY: not an executable model")
        self.binary = self.root / "synthetic-worker.exe"
        self.binary.write_bytes(b"SYNTHETIC CPU TEST ONLY: never execute")
        self.model_sha = planner.file_digest(self.model)
        self.graph_sha = planner.digest(b"synthetic-graph")
        projections = [dict(id="trunk.block00.pair00." + suffix, layer_index=index, n=n, k=k)
                       for suffix, index, n, k in [("attention.qkv", 3, 1152, 384), ("attention.out", 3, 384, 384),
                                                   ("ffn.dual", 5, 1152, 384), ("ffn.down", 5, 384, 1152)]]
        put(self.root / "model-manifest.json", dict(schema="rustgo-layer-graph-v1", model_sha256=self.model_sha,
                                                    graph_sha256=self.graph_sha, projections=projections))
        template = dict(schema="rustgo-precision-recipe", version=1, quantization_semantics_version=1,
                        model_sha256=self.model_sha, graph_sha256=self.graph_sha,
                        projections=[dict(id=p["id"], expected_n=p["n"], expected_k=p["k"], precision="fp16") for p in projections])
        put(self.root / "template.json", template)
        self.plan = planner.run(SimpleNamespace(model=self.model, model_manifest=self.root / "model-manifest.json",
                                                fp16_template=self.root / "template.json", corpus_manifest=self.corpus / "manifest.json",
                                                sgf_root=None, output=self.root / "search", seed="synthetic", max_candidates=8, max_finalists=2))
        self.fp16, self.int8 = self.plan["identity"]["candidate_recipe_sha256"]
        self.collections = {}
        for entry in self.plan["candidates"]:
            self.collections[entry["recipe_sha256"]] = self.make_collection(entry)
        self.reference = self.collections[self.fp16]
        self.workload = dict(schema="rustgo-quantization-ledger-workload-v1", split="calibration", metric="rpc_throughput",
                             concurrency=1, capacity=1, warmup=1, cycles=1, task_timeout=180,
                             max_numeric_evaluations=2, max_performance_evaluations=1,
                             minimum_improvement=0.01, maximum_relative_spread=0.05, smoke=True)
        put(self.root / "workload.json", self.workload)
        put(self.root / "baselines.json", dict(schema="rustgo-quantization-ledger-baselines-v1",
                                               fp16_reference_report_sha256=planner.file_digest(self.reference / "report.json"),
                                               baselines=[dict(recipe_sha256=self.fp16, collection_report_sha256=planner.file_digest(self.reference / "report.json"))]))
        self.ledger = self.root / "ledger"

    def make_collection(self, entry):
        directory = self.root / entry["name"]
        directory.mkdir()
        (directory / "source").mkdir()
        for name in ("manifest.json", "games.jsonl", "calibration.requests.jsonl"):
            shutil.copyfile(self.corpus / name, directory / "source" / name)
        shutil.copyfile(self.root / "search" / entry["recipe_file"], directory / "recipe.json")
        profile = dict(model_sha256=self.model_sha, graph_sha256=self.graph_sha, recipe_sha256=entry["recipe_sha256"],
                       inference_profile_id="rustgo-quant-v1:" + planner.digest(entry["name"].encode()))
        cfg = f"nnBackend=cudaquantbackend\nnnMaxBatchSize=1\nnumNNServerThreadsPerModel=1\ncudaQuantPlan={directory / 'recipe.json'}\ncudaQuantExpectedProfile={profile['inference_profile_id']}\n"
        (directory / "worker.cfg").write_text(cfg, encoding="utf-8")
        log = f"[cuda-quant] model_sha256={self.model_sha} graph_sha256={self.graph_sha} recipe_sha256={entry['recipe_sha256']} inference_profile={profile['inference_profile_id']} validation=unverified\n"
        (directory / "worker.log").write_text(log, encoding="utf-8")
        pb = self.protocol.pb
        hello = pb.WorkerHello(protocol_version=1, worker_id="synthetic", model_sha256=self.model_sha, model_version=17,
                               input_profile="katago-eval-v1", max_in_flight=1, max_board_size=19, supports_ownership=True,
                               supports_friendly_pass_search=True, backend_info=f"backend=cudaquantbackend; inference-profile={profile['inference_profile_id']}")
        source_sha = planner.file_digest(self.corpus / "calibration.requests.jsonl")
        requests, requests_sha = collection.make_requests(self.protocol, [self.record], self.model_sha, "quant-corpus-" + source_sha[:24], 180000)
        request = requests[0]
        result = pb.EvalResult(task_id=1, generation=1, session_id=request.session_id, input_hash=request.input_hash,
                               model_sha256=self.model_sha, elapsed_us=10, queue_us=0, context_us=0, evaluator_us=10,
                               output=pb.NNOutput(policy=[1 / 362] * 362, ownership=[0] * 361,
                                                  white_win_prob=0.5, white_loss_prob=0.5))
        request_raw, result_raw = request.SerializeToString(deterministic=True), result.SerializeToString(deterministic=True)
        for name, raw in (("requests/000001.pb", request_raw), ("results/000001.pb", result_raw)):
            (directory / name).parent.mkdir(exist_ok=True)
            (directory / name).write_bytes(raw)
        meta = {key: self.record[key] for key in comparison.META_KEYS}
        index = dict(**meta, task_id=1, request_sha256=planner.digest(request_raw), input_hash=request.input_hash.hex(),
                     request=message_dict(request), source_sgfs=self.game["sources"])
        output_sha = planner.digest(result.output.SerializeToString(deterministic=True))
        output = dict(**meta, result=message_dict(result), result_sha256=planner.digest(result_raw), output_sha256=output_sha)
        (directory / "requests-index.jsonl").write_bytes(lines([index]))
        (directory / "outputs.jsonl").write_bytes(lines([output]))
        report = dict(schema="rustgo-quantization-collection-v1", status="COLLECTED_UNVERIFIED", drained=True,
                      backend="cudaquantbackend", expected_requests=1, completed_requests=1, model=str(self.model), model_sha256=self.model_sha,
                      binary=str(self.binary), binary_sha256=planner.file_digest(self.binary), schema_sha256=planner.file_digest(ledger.ROOT / "crates/kata_worker/proto/worker.proto"),
                      config_text=cfg, config_sha256=planner.file_digest(directory / "worker.cfg"), recipe_file_sha256=planner.file_digest(directory / "recipe.json"),
                      actual_profile=profile, requests_sha256=requests_sha, corpus_manifest_sha256=planner.file_digest(self.corpus / "manifest.json"),
                      corpus_request_set_sha256=ledger.read_json(self.corpus / "manifest.json")["request_set_sha256"],
                      source_request_file_sha256=source_sha, ordered_output_hashes_sha256=planner.digest((1).to_bytes(8, "little") + bytes.fromhex(output_sha)),
                      split="calibration", corpus_status="NOT_READY", unique_games=1, capacity=1, request_window=1, batch=1,
                      final_heartbeat=dict(completed_requests=1, nn_rows=1, failed_requests=0, in_flight=0, nn_batches=1),
                      hello=message_dict(hello), controlled_environment={}, runtime_artifacts=[], batching="synthetic CPU fixture",
                      source_artifacts=[artifact(directory / "source" / name) for name in ("manifest.json", "games.jsonl", "calibration.requests.jsonl")],
                      artifacts=[artifact(directory / name) for name in ("outputs.jsonl", "requests-index.jsonl", "worker.cfg", "recipe.json", "worker.log")])
        put(directory / "report.json", report)
        return directory

    def init(self):
        return ledger.initialize(self.root / "search/search-plan.json", self.root / "workload.json", self.root / "baselines.json", self.ledger)

    def numeric(self, recipe):
        return ledger.ingest(self.ledger, "NUMERIC", recipe, reference=self.reference, candidate=self.collections[recipe])

    def make_abba(self, *, candidate_ns=9000000, candidate_latency_us=None):
        contract, events = ledger.replay(self.ledger)
        results = ledger.derive_state(contract, events)[1]
        reference = comparison.load_collection(self.reference, self.protocol)
        arm_data = {name: comparison.load_collection(self.collections[recipe], self.protocol, full_reference=reference["recipe"])
                    for name, recipe in (("baseline", self.fp16), ("candidate", self.int8))}
        _, measured, request_sha = benchmark.prepare_requests(self.protocol, reference, 1, 1, 180)
        directory = self.root / "abba"
        directory.mkdir()
        report = dict(schema="rustgo-quantization-corpus-abba-v1", status="SMOKE_ONLY", order=list(benchmark.ORDER),
                      holdout_outputs_read=0, publish_allowed=False, production_certified=False, deployment_adopted=False,
                      primary_metric=self.workload["metric"], cycles=1, warmup_requests=1, smoke=True, minimum_improvement=0.01,
                      maximum_relative_spread=0.05, reference_report_sha256=reference["report_sha256"],
                      prepared_request_sha256=request_sha, measured_requests=1, semantic_positions=1, runs=[], numerical_checks={},
                      protocol_sha256=planner.file_digest(ledger.ROOT / "crates/kata_worker/proto/worker.proto"),
                      source_scripts={name: planner.file_digest(ledger.ROOT / "scripts" / name) for name in ledger.SOURCE_NAMES
                                      if name not in ("quantization_selection_ledger.py", "plan_quantization_search.py")})
        for name, data in arm_data.items():
            metrics = comparison.compare(reference, data)
            metrics.pop("cases")
            report["numerical_checks"][name] = metrics
        scores = []
        for index, name in enumerate(benchmark.ORDER):
            data = arm_data[name]
            source = self.collections[self.fp16 if name == "baseline" else self.int8]
            arm_dir = directory / f"{index + 1:02d}-{name}"
            arm_dir.mkdir()
            shutil.copyfile(source / "worker.log", arm_dir / "worker.log")
            ns = 10000000 if name == "baseline" else (candidate_ns[index - 1] if isinstance(candidate_ns, tuple) else candidate_ns)
            start = 1000000000 * (index + 1)
            elapsed = ns / 1e9
            rows = [[ns / 1000, 10, 0, 0, 10, None]]
            if name == "candidate" and candidate_latency_us is not None:
                rows[0][0] = candidate_latency_us
            columns = ["round_trip_us", "worker_elapsed_us", "queue_us", "context_us", "evaluator_us", "refill_delay_us"]
            measurement = dict(started_ns=start, ended_ns=start + ns, elapsed_seconds=elapsed, completed=1,
                               rpc_requests_per_second=1 / elapsed, timings={key: distribution([rows[0][i]]) for i, key in enumerate(columns)})
            before = dict(in_flight=0, completed_requests=1, failed_requests=0, nn_rows=1, nn_batches=1)
            after = dict(in_flight=0, completed_requests=2, failed_requests=0, nn_rows=2, nn_batches=2)
            arm = dict(status="PASS", drained=True, arm=name, order_index=index, collection_report_sha256=data["report_sha256"],
                       concurrency=1, protocol_capacity=1, warmup_requests=1, measured_requests=1, batch_capacity=1,
                       files=[dict(path=str(p), sha256=h) for p, h in benchmark.runtime_files((source, data), None)[2]],
                       environment={k: v for k, v in benchmark.arm_environment(data["report"]).items() if k.startswith("KATAGO_")},
                       hello=data["report"]["hello"], before_heartbeat=before, after_heartbeat=after,
                       counter_deltas={k: after[k] - before[k] for k in after if k != "in_flight"}, measurement=measurement)
            arm["command"] = [str(self.binary), "nnworker", "--server", f"127.0.0.1:{50000 + index}", "--worker-id",
                              f"quant-corpus-abba-{index}", "--capacity", "1", "--once", "--model", str(self.model),
                              "--model-sha256", self.model_sha, "--config", str(source / "worker.cfg")]
            put(arm_dir / "report.json", arm)
            put(arm_dir / "timings.json", dict(columns=columns, rows=rows))
            report["runs"].append(arm)
            scores.append(1 / elapsed if self.workload["metric"] == "rpc_throughput" else 1 / rows[0][0])
        decision = decide([scores[0], scores[3]], scores[1:3], 0.01, 0.05)
        decision["meets_pairwise_speed_and_stability_limits"] = decision.pop("accepted")
        report["pairwise"] = decision
        put(directory / "report.json", report)
        return directory

    def test_numeric_and_abba_are_registered_without_publication(self):
        self.init()
        self.numeric(self.fp16)
        self.numeric(self.int8)
        state = ledger.ingest(self.ledger, "PERFORMANCE", self.int8, performance=self.make_abba())
        self.assertEqual(state["results"]["PERFORMANCE"][self.int8]["gate"], "PASS")
        self.assertEqual((state["numeric_attempts"], state["performance_attempts"]), (2, 1))
        self.assertFalse(state["publish_allowed"])
        self.assertEqual(state["status"], "SMOKE_ONLY")
        self.assertEqual(state["holdout_outputs_read"], 0)

    def test_event_tampering(self):
        self.init()
        event = self.ledger / "events/000000.json"
        data = ledger.read_json(event)
        data["payload"]["contract_sha256"] = "f" * 64
        put(event, data)
        with self.assertRaisesRegex(ValueError, "event hash"):
            ledger.inspect(self.ledger)

    def test_source_and_manifest_tampering(self):
        self.init()
        self.model.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "immutable source"):
            ledger.inspect(self.ledger)

    def test_manifest_lies_fail_initialization(self):
        plan_path = self.root / "search/search-plan.json"
        data = ledger.read_json(plan_path)
        data["status"] = "PLANNED_UNVERIFIED"
        put(plan_path, data)
        with self.assertRaisesRegex(ValueError, "regenerated"):
            self.init()
        self.assertFalse(self.ledger.exists())

    def test_cross_model_evidence_consumes_attempt(self):
        self.init()
        path = self.collections[self.int8] / "report.json"
        data = ledger.read_json(path)
        data["model_sha256"] = "a" * 64
        put(path, data)
        with self.assertRaisesRegex(ValueError, "cross-model"):
            self.numeric(self.int8)
        state = ledger.inspect(self.ledger)
        self.assertEqual(state["numeric_attempts"], 1)
        self.assertEqual(state["results"]["NUMERIC"][self.int8]["gate"], "INVALID_EVIDENCE")

    def test_reused_candidate_and_replayed_evidence(self):
        self.init()
        self.numeric(self.int8)
        duplicate = self.root / "renamed-collection"
        shutil.copytree(self.collections[self.int8], duplicate)
        with self.assertRaisesRegex(ValueError, "already reserved"):
            ledger.ingest(self.ledger, "NUMERIC", self.int8, reference=self.reference, candidate=duplicate)
        self.assertEqual(ledger.inspect(self.ledger)["numeric_attempts"], 1)

    def test_budget_exhaustion(self):
        self.workload["max_numeric_evaluations"] = 1
        put(self.root / "workload.json", self.workload)
        self.init()
        self.numeric(self.fp16)
        with self.assertRaisesRegex(ValueError, "budget exhausted"):
            self.numeric(self.int8)
        self.assertEqual(ledger.inspect(self.ledger)["numeric_attempts"], 1)

    def test_holdout_rejected_before_output_read(self):
        self.init()
        path = self.collections[self.int8] / "report.json"
        data = ledger.read_json(path)
        data["split"] = "holdout"
        put(path, data)
        original = Path.read_bytes
        def guarded(path):
            if "results" in path.parts or path.name == "outputs.jsonl":
                self.fail("holdout output was read")
            return original(path)
        with patch.object(Path, "read_bytes", guarded):
            with self.assertRaisesRegex(ValueError, "holdout"):
                self.numeric(self.int8)

    def test_raw_output_tamper_not_top_level_pass(self):
        self.init()
        path = self.collections[self.int8] / "results/000001.pb"
        path.write_bytes(path.read_bytes() + b"tampered")
        with self.assertRaises(ValueError):
            self.numeric(self.int8)
        self.assertEqual(ledger.inspect(self.ledger)["results"]["NUMERIC"][self.int8]["gate"], "INVALID_EVIDENCE")

    def test_frozen_snapshot_tamper(self):
        self.init()
        self.numeric(self.int8)
        files = list((self.ledger / "evidence").rglob("outputs.jsonl"))
        files[0].write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "frozen evidence"):
            ledger.inspect(self.ledger)

    def test_raw_timing_tamper_and_no_retry(self):
        self.init()
        self.numeric(self.fp16)
        self.numeric(self.int8)
        directory = self.make_abba()
        path = directory / "02-candidate/timings.json"
        data = ledger.read_json(path)
        data["rows"][0][0] = 1
        put(path, data)
        with self.assertRaisesRegex(ValueError, "raw rows"):
            ledger.ingest(self.ledger, "PERFORMANCE", self.int8, performance=directory)
        with self.assertRaisesRegex(ValueError, "already reserved"):
            ledger.ingest(self.ledger, "PERFORMANCE", self.int8, performance=directory)

    def test_replayed_arm_even_after_relabeling(self):
        self.init()
        self.numeric(self.fp16)
        self.numeric(self.int8)
        directory = self.make_abba()
        top = ledger.read_json(directory / "report.json")
        arm = top["runs"][2]
        arm["measurement"] = dict(top["runs"][1]["measurement"])
        put(directory / "03-candidate/report.json", arm)
        put(directory / "report.json", top)
        with self.assertRaisesRegex(ValueError, "replayed/overlapping"):
            ledger.ingest(self.ledger, "PERFORMANCE", self.int8, performance=directory)

    def test_numeric_gate_failure_consumes_budget_and_blocks_performance(self):
        directory = self.collections[self.int8]
        result_path = directory / "results/000001.pb"
        result = self.protocol.pb.EvalResult.FromString(result_path.read_bytes())
        result.output.white_win_prob, result.output.white_loss_prob = 0.7, 0.3
        raw = result.SerializeToString(deterministic=True)
        result_path.write_bytes(raw)
        entry = next(collection.read_jsonl((directory / "outputs.jsonl").read_bytes(), "test"))
        entry.update(result=message_dict(result), result_sha256=planner.digest(raw),
                     output_sha256=planner.digest(result.output.SerializeToString(deterministic=True)))
        (directory / "outputs.jsonl").write_bytes(lines([entry]))
        report = ledger.read_json(directory / "report.json")
        report["ordered_output_hashes_sha256"] = planner.digest((1).to_bytes(8, "little") + bytes.fromhex(entry["output_sha256"]))
        report["artifacts"] = [artifact(directory / item["file"]) for item in report["artifacts"]]
        put(directory / "report.json", report)
        self.init()
        state = self.numeric(self.int8)
        self.assertEqual(state["results"]["NUMERIC"][self.int8]["gate"], "FAIL")
        with self.assertRaisesRegex(ValueError, "numeric PASS"):
            ledger.ingest(self.ledger, "PERFORMANCE", self.int8, performance=self.root / "missing")
        self.numeric(self.fp16)
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "BASELINE_RETAINED")
        self.assertEqual(summary["candidates"][0]["status"], "REJECTED_NUMERIC_GATE")
        self.assertIsNone(summary["performance_ranking"])

    def test_missing_snapshot_file_is_not_silently_ignored(self):
        self.init()
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            ledger.snapshot(self.ledger, 99, "missing", self.reference, ["missing.pb"])

    def test_continuous_label_requires_bound_raw_buffer_evidence(self):
        self.init()
        path = self.collections[self.int8] / "report.json"
        data = ledger.read_json(path)
        data["window_mode"] = "continuous"
        put(path, data)
        with self.assertRaisesRegex(ValueError, "buffer evidence"):
            self.numeric(self.int8)

    def test_continuous_buffer_is_snapshotted_and_mode_is_bound(self):
        directory = self.collections[self.int8]
        raw = (directory / "results/000001.pb").read_bytes()
        summary = dict(schema="rustgo-continuous-corpus-buffer-v1", submitted=1, received_basic_identity_checked=1,
                       heartbeat_count=0, protobuf_payload_bytes=len(raw), max_pending=1, pending_task_ids=[],
                       artifacts=[dict(task_id=1, file="results/000001.pb", bytes=len(raw), sha256=planner.digest(raw))],
                       validation="SYNTHETIC CPU TEST")
        put(directory / "continuous-buffer.json", summary)
        report = ledger.read_json(directory / "report.json")
        report.update(window_mode="continuous", continuous_numerically_validated_results=1,
                      collection_buffer_contract=dict(schema="rustgo-continuous-corpus-buffer-contract-v1", max_results=1,
                                                      max_protobuf_payload_bytes=256 * 1024 * 1024),
                      continuous_buffer={k: v for k, v in summary.items() if k != "artifacts"})
        report["artifacts"].append(artifact(directory / "continuous-buffer.json"))
        put(directory / "report.json", report)
        self.init()
        state = self.numeric(self.int8)
        recorded = state["results"]["NUMERIC"][self.int8]
        self.assertEqual(recorded["execution"]["window_mode"], "continuous")
        self.assertEqual(recorded["collection_buffer_contract"]["max_results"], 1)
        self.assertEqual(len(list((self.ledger / "evidence").rglob("continuous-buffer.json"))), 1)

    def test_anchor_rejects_rollback(self):
        original = self.init()["head_sha256"]
        self.numeric(self.int8)
        with self.assertRaisesRegex(ValueError, "expected anchor"):
            ledger.inspect(self.ledger, original)

    def test_exclusive_lock_is_not_stolen(self):
        self.init()
        (self.ledger / ".ledger.lock").write_text("other writer")
        with self.assertRaises(FileExistsError):
            self.numeric(self.int8)

    def test_performance_before_numeric_is_rejected(self):
        self.init()
        with self.assertRaisesRegex(ValueError, "numeric PASS"):
            ledger.ingest(self.ledger, "PERFORMANCE", self.int8, performance=self.root / "missing")
        self.assertEqual(ledger.inspect(self.ledger)["performance_attempts"], 0)

    def summary(self, **kwargs):
        return ledger.summarize_selection(self.ledger, self.root / "summary", **kwargs)["report"]

    def measured(self, **kwargs):
        self.init()
        self.numeric(self.fp16)
        self.numeric(self.int8)
        return ledger.ingest(self.ledger, "PERFORMANCE", self.int8, performance=self.make_abba(**kwargs))

    def test_summary_incomplete_does_not_rank_unmeasured_or_change_ledger(self):
        self.init()
        before = self.numeric(self.fp16)
        summary = self.summary(expect_head=before["head_sha256"])
        self.assertEqual(summary["conclusion"], "EVIDENCE_INCOMPLETE")
        self.assertEqual(summary["candidates"][0]["next_action"], "REGISTER_PREDECLARED_EVIDENCE_IF_AVAILABLE")
        self.assertIsNone(summary["performance_ranking"])
        self.assertEqual(summary["positive_pairwise_candidates_unranked"], [])
        self.assertEqual(summary["baseline_optimality"], "NOT_ESTABLISHED")
        self.assertEqual(before, ledger.inspect(self.ledger))

    def test_summary_no_gain_retains_registered_baseline(self):
        before = self.measured(candidate_ns=10100000)
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "BASELINE_RETAINED")
        self.assertEqual(summary["candidates"][0]["status"], "REJECTED_NO_GAIN")
        self.assertEqual(summary["retained_baseline_recipe_sha256"], [self.fp16])
        self.assertEqual(summary["next_steps"], [dict(action="RETAIN_REGISTERED_BASELINE")])
        self.assertEqual(before, ledger.inspect(self.ledger))

    def test_summary_unstable_is_rejected_without_retry_or_slow_claim(self):
        self.measured(candidate_ns=(8000000, 9000000))
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "BASELINE_RETAINED")
        self.assertEqual(summary["candidates"][0]["status"], "REJECTED_UNSTABLE")
        self.assertGreater(summary["candidates"][0]["pairwise"]["improvement"], 0)
        self.assertEqual(summary["next_steps"], [dict(action="RETAIN_REGISTERED_BASELINE")])

    def test_summary_no_challenger_has_no_comparative_claim(self):
        catalog = ledger.read_json(self.root / "baselines.json")
        catalog["baselines"].append(dict(recipe_sha256=self.int8,
                                         collection_report_sha256=planner.file_digest(self.collections[self.int8] / "report.json")))
        put(self.root / "baselines.json", catalog)
        self.init()
        self.numeric(self.fp16)
        self.numeric(self.int8)
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "EVIDENCE_INCOMPLETE")
        self.assertEqual(summary["candidates"], [])
        self.assertEqual(summary["next_steps"], [dict(action="CLOSE_WITHOUT_COMPARATIVE_CLAIM", reason="NO_REGISTERED_CHALLENGER")])

    def test_summary_requires_every_raw_file_to_be_bound_by_result_event(self):
        self.init()
        self.numeric(self.int8)
        path = sorted((self.ledger / "events").iterdir())[-1]
        event = ledger.read_json(path)
        event["evidence"] = [item for item in event["evidence"] if not item["relative_path"].endswith("candidate/results/000001.pb")]
        event.pop("event_sha256")
        event["event_sha256"] = ledger.digest(ledger.compact(event))
        put(path, event)
        with self.assertRaisesRegex(ValueError, "every required frozen artifact"):
            self.summary()

    def test_summary_positive_smoke_never_adopts_and_binds_runtime_artifacts(self):
        before = self.measured()
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "CANDIDATE_REQUIRES_CONFIRMATION")
        self.assertEqual(summary["scope"], "SMOKE_DIAGNOSTIC_ONLY")
        self.assertEqual(summary["positive_pairwise_candidates_unranked"], [self.int8])
        self.assertEqual(summary["head_sha256"], before["head_sha256"])
        identity = summary["candidates"][0]["identity"]
        self.assertEqual(identity["execution"]["actual_profile"]["recipe_sha256"], self.int8)
        self.assertEqual(identity["execution"]["binary_sha256"], planner.file_digest(self.binary))
        self.assertTrue(identity["performance_artifacts"])
        self.assertTrue(any(item["path"].endswith("worker.cfg") for item in identity["runtime_sources"]))
        self.assertEqual(summary["holdout_outputs_read"], 0)
        self.assertIn("QUALIFIED_SELECTION_CORPUS", summary["missing_qualification"])
        self.assertIn("BEST_VALIDATED_FP16_AND_INT8_BASELINE_QUALIFICATION", summary["missing_qualification"])
        self.assertIn("INDEPENDENT_ABBA_CONFIRMATION", summary["missing_qualification"])
        self.assertIn("FROZEN_INDEPENDENT_HOLDOUT", summary["missing_qualification"])
        for key in ("formal_adoption_allowed", "automatic_optimization_complete", "publish_allowed", "production_certified", "deployment_adopted"):
            self.assertFalse(summary[key])

    def test_summary_revalidates_frozen_outputs_not_original_output_directories(self):
        self.measured()
        for path in self.collections.values():
            (path / "outputs.jsonl").unlink()
            (path / "results/000001.pb").unlink()
            (path / "report.json").unlink()
        shutil.rmtree(self.root / "abba")
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "CANDIDATE_REQUIRES_CONFIRMATION")

    def test_summary_rejects_changed_actual_runtime_binary(self):
        self.measured()
        self.binary.write_bytes(b"another executable")
        with self.assertRaisesRegex(ValueError, "immutable source"):
            self.summary()
        self.assertFalse((self.root / "summary").exists())

    def test_summary_rejects_frozen_timing_tamper(self):
        self.measured()
        next((self.ledger / "evidence").rglob("timings.json")).write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "frozen evidence"):
            self.summary()

    def test_summary_budget_exhausted_does_not_request_repeat(self):
        self.workload["max_numeric_evaluations"] = 1
        put(self.root / "workload.json", self.workload)
        self.init()
        self.numeric(self.fp16)
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "EVIDENCE_INCOMPLETE")
        self.assertEqual(summary["candidates"][0]["next_action"], "CLOSE_WITHOUT_RETRY")
        self.assertEqual(summary["candidates"][0]["reason"], "BUDGET_EXHAUSTED")

    def test_summary_invalid_attempt_does_not_request_retry(self):
        self.init()
        self.numeric(self.fp16)
        path = self.collections[self.int8] / "results/000001.pb"
        path.write_bytes(b"invalid")
        with self.assertRaises(ValueError):
            self.numeric(self.int8)
        summary = self.summary()
        self.assertEqual(summary["candidates"][0]["next_action"], "CLOSE_WITHOUT_RETRY")
        self.assertEqual(summary["candidates"][0]["reason"], "ATTEMPT_ALREADY_CONSUMED")
        self.assertEqual(summary["remaining_attempts"]["NUMERIC"], 0)

    def test_summary_uses_frozen_p95_metric_even_when_throughput_improves(self):
        self.workload["metric"] = "p95_latency"
        put(self.root / "workload.json", self.workload)
        self.measured(candidate_ns=9000000, candidate_latency_us=12000)
        summary = self.summary()
        self.assertEqual(summary["conclusion"], "BASELINE_RETAINED")
        self.assertEqual(summary["candidates"][0]["status"], "REJECTED_NO_GAIN")

    def test_summary_rejects_stale_anchor_and_existing_destination(self):
        head = self.init()["head_sha256"]
        self.numeric(self.fp16)
        with self.assertRaisesRegex(ValueError, "expected anchor"):
            self.summary(expect_head=head)
        self.summary()
        with self.assertRaisesRegex(ValueError, "NEW directory"):
            self.summary()

    def test_summary_recomputes_decision_even_with_rehashed_payload(self):
        self.measured()
        path = sorted((self.ledger / "events").iterdir())[-1]
        event = ledger.read_json(path)
        event["payload"]["pairwise"]["improvement"] = 999
        event.pop("event_sha256")
        event["event_sha256"] = ledger.digest(ledger.compact(event))
        put(path, event)
        with self.assertRaisesRegex(ValueError, "frozen raw timings"):
            self.summary()


if __name__ == "__main__":
    unittest.main()
