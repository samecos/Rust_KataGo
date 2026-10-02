"""Hermetic catalog tests: synthetic models/binaries, no subprocesses or GPU."""
from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import quantization_execution_catalog as catalog

ledger = catalog.ledger
planner = ledger.planner
collection = planner.corpus_io


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(planner.pretty(value))


def artifact(path):
    return dict(file=path.name, bytes=path.stat().st_size, sha256=planner.file_digest(path))


class CatalogTests(unittest.TestCase):
    def setUp(self):
        target = ledger.ROOT / "target/quantization-execution-catalog-cpu-tests"
        target.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-", dir=target)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(patch.stopall)
        patch.object(subprocess, "Popen", side_effect=AssertionError("CPU catalog cannot start any process")).start()
        self.corpus, self.sgfs = self.root / "corpus", self.root / "sgfs"
        self.corpus.mkdir()
        self.sgfs.mkdir()
        sgf = self.sgfs / "synthetic.sgf"
        sgf.write_bytes(b"(;GM[1]SZ[19]RU[Chinese]KM[7.5];B[aa])")
        game_id = planner.digest(b"synthetic-game")
        moves = [[1 + i % 2, i] for i in range(8)]
        game = dict(game_id=game_id, split="calibration", retained_plies=[8],
                    sources=[dict(path=sgf.name, sha256=planner.file_digest(sgf))],
                    semantics=dict(board_size=19, komi_half_points=15, initial_player=1, initial_stones=[], moves=moves))
        params = {key: False for key in collection.PARAMETER_BOOLS}
        params.update(include_ownership=True, skip_cache=True, policy_temperature=1.0, policy_optimism=0.0,
                      draw_equivalent_wins_for_white=0.5, playout_doubling_advantage=0.0, symmetry=0, max_history=1000)
        record = dict(name="synthetic-position", game_id=game_id, split="calibration", ply=8, phase="early",
                      semantic_position_sha256=planner.digest(b"semantic"), board_state_sha256=planner.digest(b"board"),
                      position=dict(board_size=19, rules="chinese", komi=7.5, initial_player=1, next_player=1,
                                    initial_stones=[], moves=[dict(color=c, vertex=v) for c, v in moves]), parameters=params)
        lines = lambda rows: "".join(collection.json_line(row) for row in rows).encode()
        blobs = {"games.jsonl": lines([game]), "rejections.jsonl": b"", "duplicates.jsonl": b""}
        for split in planner.SPLITS:
            blobs[split + ".requests.jsonl"] = lines([record]) if split == "calibration" else b""
            blobs[split + ".worker-fixture.json"] = b"{}"
        for name, raw in blobs.items():
            (self.corpus / name).write_bytes(raw)
        counts = [dict(split=split, assigned_games=int(split == "calibration"), games_with_retained_positions=int(split == "calibration"),
                       distinct_semantic_positions=int(split == "calibration"), requested_games=1) for split in planner.SPLITS]
        manifest = {key: None for key in planner.CORPUS_KEYS}
        manifest.update(schema=collection.SCHEMA, status="NOT_READY", production_certified=False,
                        options=dict(input=str(self.sgfs), output=str(self.corpus), seed="synthetic", games_per_split=[1, 1, 1],
                                     positions_per_game=1, assume_rules=None, assume_komi=None),
                        artifacts=[artifact(self.corpus / name) for name in blobs], unique_games=1, rejected_sources=0, duplicate_records=0,
                        request_set_sha256=planner.digest(planner.compact([[record], [], []])),
                        readiness=dict(status="NOT_READY", symmetry_multiplicity_counted=False, holdout_minimum_games=128,
                                       holdout_minimum_distinct_positions=4096, splits=counts))
        put(self.corpus / "manifest.json", manifest)
        self.model, self.binary = self.root / "model.bin", self.root / "unified.exe"
        self.model.write_bytes(b"SYNTHETIC MODEL NEVER EXECUTE")
        self.binary.write_bytes(b"SYNTHETIC UNIFIED BINARY NEVER EXECUTE")
        self.model_sha = planner.file_digest(self.model)
        graph_sha = planner.digest(b"synthetic-graph")
        projections = []
        for pair, hidden in enumerate((1152, 48)):
            for suffix, offset, n, k in (("attention.qkv", 0, 1152, 384), ("attention.out", 0, 384, 384),
                                        ("ffn.dual", 2, hidden, 384), ("ffn.down", 2, 384, hidden)):
                projections.append(dict(id=f"trunk.block00.pair{pair:02}." + suffix, layer_index=3 + 4 * pair + offset, n=n, k=k))
        put(self.root / "model-manifest.json", dict(schema="rustgo-layer-graph-v1", model_sha256=self.model_sha,
                                                    graph_sha256=graph_sha, projections=projections))
        put(self.root / "template.json", dict(schema="rustgo-precision-recipe", version=1, quantization_semantics_version=1,
                    model_sha256=self.model_sha, graph_sha256=graph_sha,
                    projections=[dict(id=p["id"], expected_n=p["n"], expected_k=p["k"], precision="fp16") for p in projections]))
        self.search = self.root / "search/search-plan.json"
        planner.run(SimpleNamespace(model=self.model, model_manifest=self.root / "model-manifest.json", fp16_template=self.root / "template.json",
                    corpus_manifest=self.corpus / "manifest.json", sgf_root=self.sgfs, output=self.search.parent,
                    seed="synthetic", max_candidates=8, max_finalists=2))
        self.workload = dict(schema="rustgo-quantization-ledger-workload-v1", split="calibration", metric="rpc_throughput",
                             concurrency=1, capacity=1, warmup=1, cycles=1, task_timeout=180, max_numeric_evaluations=5,
                             max_performance_evaluations=8, minimum_improvement=0.01, maximum_relative_spread=0.05, smoke=True)
        self.workload_path = self.root / "workload.json"
        put(self.workload_path, self.workload)
        self.controls = [self.make_control("cudabackend"), self.make_control("cudaint8backend")]
        self.output = self.root / "catalog"

    def make_control(self, backend):
        root = self.root / backend
        root.mkdir()
        binary = root / "legacy.exe"
        binary.write_bytes(b"SYNTHETIC LEGACY BINARY NEVER EXECUTE")
        item = lambda path: dict(path=str(path), sha256=planner.file_digest(path))
        plan = root / "plan.json"
        put(plan, dict(schema=2, kind="cuda-tactic-plan", plan_id="synthetic-legacy-plan",
                       target=dict(model_sha256=self.model_sha, max_batch_size=8), backend_build=dict(kernel_build_id="synthetic"),
                       selection=dict(binary_sha256=planner.file_digest(binary)), apply=dict(tactic_overrides={"KATAGO_CUDA_ATTN_TILE": "q64"})))
        cfg = root / "legacy.cfg"
        text = (f"rules=chinese\nkomi=7.5\nnnBackend={backend}\nnnMaxBatchSize=8\nnumNNServerThreadsPerModel=1\n"
                "nnCacheSizePowerOfTwo=18\nnnMutexPoolSizePowerOfTwo=14\n")
        text += f"cudaTacticPlan={plan}\n" if backend == "cudabackend" else "cudaInt8Scope=ffn\ncudaInt8MinFfnWidth=0\n"
        cfg.write_text(text, encoding="utf-8")
        path = root / "spec.json"
        put(path, dict(schema=catalog.legacy.SCHEMA, label=backend, backend=backend, binary=item(binary), model=item(self.model), config=item(cfg),
                       plan=item(plan) if backend == "cudabackend" else None, batch=8,
                       environment={} if backend == "cudabackend" else catalog.legacy.INT8_ENVIRONMENT))
        return path

    def create(self, **overrides):
        args = dict(search_plan=self.search, workload=self.workload_path, unified_binary=self.binary, batches=[1],
                    legacy_specs=self.controls, output=self.output, reference_batch=1, sgf_root=self.sgfs)
        args.update(overrides)
        return catalog.create_catalog(**args)

    def read(self):
        return ledger.read_json(self.output / "catalog.json")

    def replan(self, name):
        self.search = self.root / name / "search-plan.json"
        planner.run(SimpleNamespace(model=self.model, model_manifest=self.root / "model-manifest.json", fp16_template=self.root / "template.json",
                    corpus_manifest=self.corpus / "manifest.json", sgf_root=self.sgfs, output=self.search.parent,
                    seed="synthetic", max_candidates=8, max_finalists=2))

    def refresh_corpus(self):
        manifest = ledger.read_json(self.corpus / "manifest.json")
        manifest["artifacts"] = [artifact(self.corpus / item["file"]) for item in manifest["artifacts"]]
        records = [list(collection.read_jsonl((self.corpus / (split + ".requests.jsonl")).read_bytes(), split)) for split in planner.SPLITS]
        manifest["request_set_sha256"] = planner.digest(planner.compact(records))
        put(self.corpus / "manifest.json", manifest)

    def assert_same_stage_keys(self, first, second, stages=("numeric_pairs", "performance_pairs")):
        for stage in stages:
            self.assertEqual([row["pair_key"] for row in first[stage]], [row["pair_key"] for row in second[stage]])

    def changed_workload_catalog(self, **changes):
        self.workload.update(changes)
        put(self.workload_path, self.workload)
        output = self.root / "changed-workload"
        self.create(output=output)
        return ledger.read_json(output / "catalog.json")

    def test_complete_catalog_covers_all_controls_and_preserves_unknown_qualification(self):
        result = self.create()
        contract = catalog.validate_catalog(self.output / "catalog.json", result["catalog"]["sha256"])
        self.assertEqual(result["unique_execution_specs"], 5)
        self.assertEqual((len(contract["numeric_pairs"]), len(contract["performance_pairs"])), (5, 8))
        reference = contract["identity"]["canonical_reference_execution_spec_id"]
        controls = set(contract["identity"]["required_control_execution_spec_ids"])
        for row in contract["execution_specs"]:
            if row["kind"] == "unified_v1_recipe":
                targets = {pair["baseline_execution_spec_id"] for pair in contract["performance_pairs"] if pair["candidate_execution_spec_id"] == row["execution_spec_id"]}
                self.assertEqual(targets, controls | ({reference} if row["execution_spec_id"] != reference else set()))
                self.assertEqual(row["semantic"]["execution"]["environment"], {})
            else:
                self.assertEqual(row["roles"], ["REQUIRED_LEGACY_CONTROL"])
        self.assertFalse(contract["gpu_execution_authorized"])
        self.assertFalse(contract["publish_allowed"])
        self.assertIsNone(contract["remaining_budget"])
        self.assertEqual(contract["independent_confirmation"], "NOT_REGISTERED_OR_AUTHORIZED")

    def test_semantic_aliases_and_source_plan_relocation_deduplicate(self):
        first = self.create()
        original = ledger.read_json(self.controls[0])
        alias = self.root / "alias"
        alias.mkdir()
        plan = alias / "renamed-plan.json"
        plan.write_bytes(Path(original["plan"]["path"]).read_bytes())
        cfg = alias / "renamed.cfg"
        cfg.write_bytes(Path(original["config"]["path"]).read_bytes().replace(original["plan"]["path"].encode(), str(plan).encode()))
        original["label"] = "source alias"
        original["plan"]["path"] = str(plan)
        original["config"] = dict(path=str(cfg), sha256=planner.file_digest(cfg))
        path = alias / "renamed-spec.json"
        put(path, original)
        result = self.create(legacy_specs=[*self.controls, path, path], batches=[1, 1], output=self.root / "alias-catalog")
        self.assertEqual(result["unique_execution_specs"], 5)
        self.assertEqual(result["catalog_id"], first["catalog_id"])

    def test_numeric_budget_is_checked_before_output_directory_creation(self):
        self.workload["max_numeric_evaluations"] = 4
        put(self.workload_path, self.workload)
        with self.assertRaisesRegex(ValueError, "numeric budget"):
            self.create()
        self.assertFalse(self.output.exists())

    def test_performance_budget_may_not_drop_required_controls(self):
        self.workload["max_performance_evaluations"] = 7
        put(self.workload_path, self.workload)
        with self.assertRaisesRegex(ValueError, "every required control"):
            self.create()

    def test_multi_batch_exceeds_initial_pair_budget_and_is_rejected(self):
        self.workload["max_numeric_evaluations"] = 8
        put(self.workload_path, self.workload)
        with self.assertRaisesRegex(ValueError, "performance budget"):
            self.create(batches=[1, 8])

    def test_missing_either_legacy_backend_is_not_a_complete_control_set(self):
        for control in self.controls:
            with self.subTest(control=control), self.assertRaisesRegex(ValueError, "incomplete control set"):
                self.create(legacy_specs=[control])

    def test_legacy_batch_cannot_be_selected_as_canonical_reference(self):
        with self.assertRaisesRegex(ValueError, "registered unified batch"):
            self.create(reference_batch=8)

    def test_cross_model_control_rejected_before_catalog_creation(self):
        value = ledger.read_json(self.controls[1])
        other = self.root / "other.bin"
        other.write_bytes(b"ANOTHER MODEL")
        value["model"] = dict(path=str(other), sha256=planner.file_digest(other))
        put(self.controls[1], value)
        with self.assertRaisesRegex(ValueError, "different model"):
            self.create()

    def test_holdout_workload_is_rejected_before_plan_or_any_output_access(self):
        self.workload["split"] = "holdout"
        put(self.workload_path, self.workload)
        with patch.object(ledger, "validate_search", side_effect=AssertionError("should reject before source planning")):
            with self.assertRaisesRegex(ValueError, "holdout"):
                self.create()

    def test_holdout_model_outputs_are_never_opened(self):
        forbidden = self.corpus / "holdout.outputs.jsonl"
        forbidden.write_bytes(b"not input evidence")
        original = Path.open
        def checked(path, *args, **kwargs):
            if path.suffix == ".pb" or (path.suffix == ".jsonl" and "outputs" in path.name):
                self.fail("unexpected model output access")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", checked):
            self.create()
            catalog.validate_catalog(self.output / "catalog.json")

    def test_changed_source_binary_fails_revalidation(self):
        self.create()
        self.binary.write_bytes(b"CHANGED")
        with self.assertRaisesRegex(ValueError, "immutable source changed"):
            catalog.validate_catalog(self.output / "catalog.json")

    def test_changed_source_sgf_fails_revalidation(self):
        self.create()
        (self.sgfs / "synthetic.sgf").write_bytes(b"CHANGED")
        with self.assertRaisesRegex(ValueError, "immutable source changed"):
            catalog.validate_catalog(self.output / "catalog.json")

    def test_removed_control_pair_cannot_be_accepted_by_verify(self):
        self.create()
        value = self.read()
        value["performance_pairs"].pop()
        put(self.output / "catalog.json", value)
        with self.assertRaisesRegex(ValueError, "independently rebuilt"):
            catalog.validate_catalog(self.output / "catalog.json")

    def test_catalog_anchor_rejects_tampering_before_rebuild(self):
        result = self.create()
        value = self.read()
        value["publish_allowed"] = True
        put(self.output / "catalog.json", value)
        with self.assertRaisesRegex(ValueError, "expected anchor"):
            catalog.validate_catalog(self.output / "catalog.json", result["catalog"]["sha256"])

    def test_incomplete_fp16_template_is_not_implicitly_filled(self):
        value = ledger.read_json(self.root / "template.json")
        value["projections"].pop()
        put(self.root / "template.json", value)
        with self.assertRaisesRegex(ValueError, "incomplete FP16 template"):
            self.create()

    def test_execution_semantics_model_shape_precision_batch_and_tactics_affect_id(self):
        self.create()
        semantic = next(row["semantic"] for row in self.read()["execution_specs"] if row["kind"] == "unified_v1_recipe")
        original = catalog.execution_spec_id(semantic)
        variants = []
        for key in ("model_sha256", "graph_sha256", "projection_shapes_sha256"):
            value = deepcopy(semantic)
            value[key] = "f" * 64
            variants.append(value)
        for key, replacement in (("batch", 8), ("binary_sha256", "f" * 64), ("environment", {"KATAGO_CUDA_ATTN_TILE": "q64"})):
            value = deepcopy(semantic)
            value["execution"][key] = replacement
            variants.append(value)
        value = deepcopy(semantic)
        value["execution"]["configuration"]["numNNServerThreadsPerModel"] = 2
        variants.append(value)
        value = deepcopy(semantic)
        value["execution"]["precision_recipe"]["projections"][0]["precision"] = "mxfp8"
        variants.append(value)
        self.assertTrue(all(catalog.execution_spec_id(value) != original for value in variants))
        self.assertEqual(len(set(map(catalog.execution_spec_id, variants))), len(variants))

    def test_pair_key_is_undirected_but_stage_and_workload_bound(self):
        key = catalog.pair_key("ABBA", "work", "ref", "a", "b")
        self.assertEqual(key, catalog.pair_key("ABBA", "work", "ref", "b", "a"))
        self.assertNotEqual(key, catalog.pair_key("NUMERIC_COMPARE", "work", "ref", "a", "b"))
        self.assertNotEqual(key, catalog.pair_key("ABBA", "other-work", "ref", "a", "b"))

    def test_abba_key_ignores_reference_that_does_not_execute_in_pair(self):
        self.assertEqual(catalog.pair_key("ABBA", "work", "reference-b1", "a", "b"),
                         catalog.pair_key("ABBA", "work", "reference-b8", "a", "b"))
        self.assertNotEqual(catalog.pair_key("NUMERIC_COMPARE", "work", "reference-b1", "a", "b"),
                            catalog.pair_key("NUMERIC_COMPARE", "work", "reference-b8", "a", "b"))

    def test_numeric_budget_key_binds_collection_not_reference_comparison(self):
        self.create()
        first = self.read()
        for row in first["numeric_pairs"]:
            self.assertEqual(row["pair_key"], row["collection_attempt_key"])
            self.assertEqual(row["budget_unit"], "UNIQUE_SPEC_COLLECTION")
            self.assertEqual(row["collection_attempt_key"], catalog.collection_attempt_key(
                    first["numeric_workload_id"], row["candidate_execution_spec_id"]))
            another_registration = catalog.numeric_registration(first["numeric_workload_id"],
                    "another-reference", row["candidate_execution_spec_id"])
            self.assertEqual(another_registration["collection_attempt_key"], row["collection_attempt_key"])
            self.assertEqual(another_registration["pair_key"], row["pair_key"])
            self.assertNotEqual(another_registration["comparison_key"], row["comparison_key"])

    def test_equivalent_timeout_number_representation_preserves_both_stage_keys(self):
        self.create()
        first = self.read()
        second = self.changed_workload_catalog(task_timeout=180.0)
        self.assert_same_stage_keys(first, second)
        self.assertEqual(first["numeric_workload_id"], second["numeric_workload_id"])
        self.assertEqual(first["abba_workload_id"], second["abba_workload_id"])
        first_source = next(row for row in first["source_records"] if row["path"] == str(self.workload_path))
        second_source = next(row for row in second["source_records"] if row["path"] == str(self.workload_path))
        self.assertNotEqual(first_source["sha256"], second_source["sha256"])

    def test_metric_is_bound_for_decision_but_never_resets_experiment_keys(self):
        self.create()
        first = self.read()
        second = self.changed_workload_catalog(metric="p95_latency")
        self.assert_same_stage_keys(first, second)
        self.assertNotEqual(first["catalog_id"], second["catalog_id"])
        self.assertNotEqual(first["identity"]["decision_settings"], second["identity"]["decision_settings"])

    def test_abba_warmup_cycles_do_not_reset_numeric_collection_keys(self):
        self.create()
        first = self.read()
        second = self.changed_workload_catalog(warmup=2, cycles=2)
        self.assert_same_stage_keys(first, second, ("numeric_pairs",))
        self.assertNotEqual(first["abba_workload_id"], second["abba_workload_id"])
        self.assertFalse(set(row["pair_key"] for row in first["performance_pairs"]) &
                         set(row["pair_key"] for row in second["performance_pairs"]))

    def test_real_timeout_change_changes_both_stage_keys(self):
        self.create()
        first = self.read()
        second = self.changed_workload_catalog(task_timeout=180.1)
        self.assertNotEqual(first["numeric_workload_id"], second["numeric_workload_id"])
        self.assertNotEqual(first["abba_workload_id"], second["abba_workload_id"])

    def test_actual_concurrency_and_capacity_changes_both_stage_keys(self):
        self.create()
        first = self.read()
        second = self.changed_workload_catalog(concurrency=2, capacity=2, warmup=2)
        self.assertNotEqual(first["numeric_workload_id"], second["numeric_workload_id"])
        self.assertNotEqual(first["abba_workload_id"], second["abba_workload_id"])

    def test_json_whitespace_and_record_labels_are_provenance_only(self):
        self.create()
        first = self.read()
        path = self.corpus / "calibration.requests.jsonl"
        row = next(collection.read_jsonl(path.read_bytes(), "calibration"))
        row.update(name="source-renamed", phase="different-metadata", semantic_position_sha256="a" * 64)
        row["parameters"]["policy_temperature"] = 1  # Same protobuf double as 1.0.
        path.write_bytes(b"  " + planner.compact(row, sorted_keys=True) + b"   \n")
        self.refresh_corpus()
        self.replan("reformatted-search")
        output = self.root / "reformatted-catalog"
        self.create(output=output)
        second = ledger.read_json(output / "catalog.json")
        self.assert_same_stage_keys(first, second)
        self.assertNotEqual(first["identity"]["search_id"], second["identity"]["search_id"])

    def test_unselected_split_does_not_change_selected_experiment_keys(self):
        self.create()
        first = self.read()
        row = next(collection.read_jsonl((self.corpus / "calibration.requests.jsonl").read_bytes(), "calibration"))
        game = next(collection.read_jsonl((self.corpus / "games.jsonl").read_bytes(), "games"))
        other_game, other_row = deepcopy(game), deepcopy(row)
        other_sgf = self.sgfs / "unselected.sgf"
        other_sgf.write_bytes(b"(;GM[1]SZ[19]RU[Chinese]KM[6.5];B[aa])")
        other_game.update(game_id="b" * 64, split="selection", sources=[dict(path=other_sgf.name, sha256=planner.file_digest(other_sgf))])
        other_game["semantics"]["komi_half_points"] = 13
        other_row.update(game_id=other_game["game_id"], split="selection", name="unselected", semantic_position_sha256="c" * 64,
                         board_state_sha256="d" * 64)
        other_row["position"]["komi"] = 6.5
        (self.corpus / "games.jsonl").write_text(collection.json_line(game) + collection.json_line(other_game), encoding="utf-8")
        (self.corpus / "selection.requests.jsonl").write_text(collection.json_line(other_row), encoding="utf-8")
        manifest = ledger.read_json(self.corpus / "manifest.json")
        manifest["unique_games"] = 2
        manifest["legacy_fixture_scope"] = {"label": "changed provenance"}
        manifest["readiness"]["splits"][1].update(assigned_games=1, games_with_retained_positions=1, distinct_semantic_positions=1)
        put(self.corpus / "manifest.json", manifest)
        self.refresh_corpus()
        self.replan("other-split-search")
        output = self.root / "other-split-catalog"
        self.create(output=output)
        second = ledger.read_json(output / "catalog.json")
        self.assert_same_stage_keys(first, second)
        self.assertNotEqual(first["identity"]["search_id"], second["identity"]["search_id"])

    def test_order_position_parameters_and_timeout_follow_actual_semantics(self):
        one = next(collection.read_jsonl((self.corpus / "calibration.requests.jsonl").read_bytes(), "calibration"))
        two = deepcopy(one)
        two["position"]["komi"] = 6.5
        three = deepcopy(one)
        three["parameters"]["policy_optimism"] = 0.2
        digest = lambda rows: catalog.ordered_request_semantics(rows)["sha256"]
        self.assertNotEqual(digest([one, two]), digest([two, one]))
        self.assertNotEqual(digest([one]), digest([two]))
        self.assertNotEqual(digest([one]), digest([three]))
        duplicate = deepcopy(one)
        duplicate["name"] = "another metadata label"
        with self.assertRaisesRegex(ValueError, "duplicate actual Position"):
            digest([one, duplicate])

    def test_budget_change_does_not_mint_fresh_attempt_pair_keys(self):
        self.create()
        first = self.read()
        self.workload["max_numeric_evaluations"] = 8
        put(self.workload_path, self.workload)
        other = self.root / "larger-planning-budget"
        self.create(output=other)
        second = ledger.read_json(other / "catalog.json")
        self.assertNotEqual(first["catalog_id"], second["catalog_id"])
        self.assertEqual(first["workload_id"], second["workload_id"])
        for stage in ("numeric_pairs", "performance_pairs"):
            self.assertEqual([row["pair_key"] for row in first[stage]], [row["pair_key"] for row in second[stage]])

    def test_distinct_valid_batch_changes_only_unified_execution_ids(self):
        self.create()
        first = self.read()
        other = self.root / "batch-eight"
        self.create(output=other, batches=[8], reference_batch=8)
        second = ledger.read_json(other / "catalog.json")
        by_kind = lambda value, kind: {row["execution_spec_id"] for row in value["execution_specs"] if row["kind"] == kind}
        self.assertEqual(by_kind(first, "legacy_explicit_spec"), by_kind(second, "legacy_explicit_spec"))
        self.assertFalse(by_kind(first, "unified_v1_recipe") & by_kind(second, "unified_v1_recipe"))

    def test_existing_catalog_cannot_be_overwritten_or_used_as_remaining_budget(self):
        self.create()
        with self.assertRaisesRegex(ValueError, "NEW directory"):
            self.create()
        second = self.create(output=self.root / "another-plan")
        self.assertIsNone(second["attempts_consumed"])
        self.assertIsNone(second["remaining_budget"])
        self.assertFalse(second["gpu_execution_authorized"])


if __name__ == "__main__":
    unittest.main()
