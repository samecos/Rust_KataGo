#!/usr/bin/env python3
"""Bounded v1 FP16/INT8 experiment driver; prepare is strictly CPU-only.

run is an explicit, single-use authorization to execute the frozen local tools.
It does not qualify the FP16 control as an optimal baseline, consume holdout,
confirm finalists, resume interrupted work, or adopt/publish a deployment.
Keep prepared.json's SHA separately: the local audit chain is not a signature.
"""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import quantization_selection_ledger as ledger

ROOT = ledger.ROOT
SCHEMA = "rustgo-bounded-quantization-driver-v1"
require = ledger.require
read_json = ledger.read_json
write_new = ledger.write_new
compact = ledger.compact
digest = ledger.digest
file_record = ledger.file_record
SAFE_ENVIRONMENT = {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "SYSTEMDRIVE", "TEMP", "TMP",
                    "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
                    "PROGRAMFILES(X86)", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "CUDA_PATH",
                    "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE"}


def environment():
    # An allowlist also removes inherited KATAGO_* and Python startup injection.
    return {key: value for key, value in os.environ.items() if key.upper() in SAFE_ENVIRONMENT}


def commands(plan, workload, binary, batch, sgf_root, root, python):
    commands_out, rows = [], []
    reference = root / "collections" / "00-fp16"
    corpus = Path(plan["source_paths"]["corpus_manifest"])
    search = Path(plan["source_paths_for_driver"]["search_plan"])
    count = plan["driver_request_count"]
    cpu_timeout = 900
    # Unified collection permits separate GTP discovery and Worker startup.
    collect_timeout = 690 + count * workload["task_timeout"]
    abba_timeout = 4 * (390 + (workload["warmup"] + count * workload["cycles"]) * workload["task_timeout"])

    def add(phase, tool, arguments, recipe=None, charge=None, output=None, timeout=cpu_timeout):
        entry = dict(id=f"{len(commands_out):03d}-{phase}", phase=phase, recipe_sha256=recipe,
                     charge=charge, argv=[str(python), "-E", "-s", str(ROOT / "scripts" / tool), *map(str, arguments)],
                     launcher_argv=[str(python), "-E", "-s", str(Path(__file__).resolve()), "_child"],
                     cwd=str(ROOT), output_directory=str(output) if output else None, timeout_seconds=timeout)
        commands_out.append(entry)

    for index, candidate in enumerate(plan["candidates"]):
        recipe = candidate["recipe_sha256"]
        name = f"{index:02d}-{candidate['name']}"
        directory = root / "collections" / name
        comparison = root / "comparisons" / name
        performance = root / "performance" / name
        rows.append(dict(recipe_sha256=recipe, collection_directory=str(directory), comparison_directory=str(comparison),
                         performance_directory=str(performance) if index else None))
        add("collect", "collect_quantization_corpus.py",
            ["--binary", binary, "--model", plan["source_paths"]["model"], "--requests", corpus.parent / (workload["split"] + ".requests.jsonl"),
             "--manifest", corpus, "--output", directory, "--backend", "cudaquantbackend", "--recipe", search.parent / candidate["recipe_file"],
             "--batch", batch, "--capacity", workload["capacity"], "--window", workload["concurrency"], "--window-mode", "continuous",
             "--max-buffered-result-mib", 256, "--max-requests", 100000, "--startup-timeout", 300, "--task-timeout", workload["task_timeout"]],
            recipe, "NUMERIC", directory, collect_timeout)
        add("compare", "compare_quantization_corpus.py", ["--reference", reference, "--candidate", directory, "--output", comparison],
            recipe, output=comparison)
        if index == 0:
            add("ledger-init", "quantization_selection_ledger.py",
                ["init", "--search-plan", search, "--workload", root / "workload.json", "--baselines", root / "baselines.json",
                 "--sgf-root", sgf_root, "--output", root / "ledger"], output=root / "ledger")
        add("ingest-numeric", "quantization_selection_ledger.py",
            ["ingest-numeric", "--ledger", root / "ledger", "--recipe-sha256", recipe, "--reference", reference, "--candidate", directory], recipe)
        if index:
            add("benchmark", "benchmark_quantization_corpus.py",
                ["--reference", reference, "--baseline", reference, "--candidate", directory, "--output", performance,
                 "--concurrency", workload["concurrency"], "--capacity", workload["capacity"], "--cycles", workload["cycles"],
                 "--warmup", workload["warmup"], "--metric", workload["metric"], "--startup-timeout", 300, "--task-timeout", workload["task_timeout"],
                 *(["--smoke"] if workload["smoke"] else [])], recipe, "PERFORMANCE", performance, abba_timeout)
            add("ingest-performance", "quantization_selection_ledger.py",
                ["ingest-performance", "--ledger", root / "ledger", "--recipe-sha256", recipe, "--performance", performance], recipe)
    add("summarize", "quantization_selection_ledger.py", ["summarize-selection", "--ledger", root / "ledger", "--output", root / "summary"],
        output=root / "summary")
    return commands_out, rows


def build_contract(search, workload_path, binary, batch, sgf_root, root, python):
    require(type(batch) is int and 1 <= batch <= 64, "batch must be explicitly 1..64")
    require(sgf_root.is_dir(), "explicit SGF root must exist")
    plan, records = ledger.validate_search(search, root.parent, sgf_root)
    workload = read_json(workload_path)
    ledger.validate_workload(workload, plan)  # rejects holdout before any output access
    candidates = plan["candidates"]
    require(1 <= len(candidates) <= 3 and candidates[0]["name"] == "fp16", "only initial v1 seeds with FP16 first are supported")
    require(workload["max_numeric_evaluations"] >= len(candidates)
            and workload["max_performance_evaluations"] >= len(candidates) - 1, "budget cannot cover the entire frozen seed set")
    corpus_path = Path(plan["source_paths"]["corpus_manifest"])
    _, _, _, requests, _, _ = ledger.planner.validate_corpus(corpus_path, sgf_root)
    count = len(requests[ledger.planner.SPLITS.index(workload["split"])])
    require(0 < count <= 100000, "selected full split must contain 1..100000 requests")
    require(count * workload["cycles"] + workload["warmup"] <= 1000000, "ABBA request budget exceeds one million")
    require(workload["smoke"] or count * workload["cycles"] >= 1024, "formal ABBA requires >=1024 complete-corpus requests")
    # These two temporary fields never modify the existing planner contract.
    command_plan = dict(plan, source_paths_for_driver=dict(search_plan=str(search)), driver_request_count=count)
    steps, rows = commands(command_plan, workload, binary, batch, sgf_root, root, python)
    records.extend([file_record(workload_path), file_record(binary), file_record(python), file_record(Path(__file__)), *ledger.source_versions()])
    records = list({record["path"]: record for record in records}.values())
    return dict(schema=SCHEMA, root=str(root), search_plan=str(search), workload_source=str(workload_path),
                binary=str(binary), batch=batch, sgf_root=str(sgf_root), python=str(python),
                search_id=plan["search_id"], workload_id=digest(compact(workload)), workload=workload,
                source_records=records, commands=steps, candidates=rows, reference_recipe_sha256=candidates[0]["recipe_sha256"],
                model_sha256=plan["identity"]["source_sha256"]["model"], binary_sha256=file_record(binary)["sha256"],
                request_count=count, expected_numeric_attempts=len(candidates), expected_performance_attempts=len(candidates)-1,
                baseline_scope="registered unified FP16 control only; best legacy FP16/INT8 qualification is absent",
                restart_policy="single-use; no retries/resume; failed and interrupted attempts remain consumed",
                holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)


def prepare(search, workload, binary, batch, sgf_root, output):
    search, workload, binary, sgf_root, output = (Path(p).resolve() for p in (search, workload, binary, sgf_root, output))
    require(not output.exists() and output.parent.is_dir(), "prepare output must be a NEW directory with an existing parent")
    contract = build_contract(search, workload, binary, batch, sgf_root, output, Path(sys.executable).resolve())
    contract["environment"] = environment()
    output.mkdir()
    write_new(output / "workload.json", contract["workload"])
    command_records = []
    for entry in contract["commands"]:
        path = output / "commands" / (entry["id"] + ".json")
        write_new(path, entry)
        command_records.append(file_record(path))
    contract["prepared_artifacts"] = [file_record(output / "workload.json"), *command_records]
    verify_sources(contract)
    write_new(output / "prepared.json", contract)
    return dict(status="PREPARED_NO_EXECUTION", prepared=file_record(output / "prepared.json"),
                search_id=contract["search_id"], candidates=len(contract["candidates"]),
                numeric_budget=contract["expected_numeric_attempts"], performance_budget=contract["expected_performance_attempts"],
                subprocesses_started=0, gpu_executions=0, publish_allowed=False)


def validate_prepared(path, expect_sha=None):
    path = Path(path).resolve()
    require(path.name == "prepared.json" and not path.is_symlink(), "run requires the original prepared.json")
    require(expect_sha is None or file_record(path)["sha256"] == expect_sha, "prepared SHA differs from expected anchor")
    contract = read_json(path)
    require(contract.get("schema") == SCHEMA and contract.get("root") == str(path.parent), "prepared location/schema differs")
    for record in contract["source_records"] + contract["prepared_artifacts"]:
        ledger.verify_record(record)
    rebuilt = build_contract(Path(contract["search_plan"]), Path(contract["workload_source"]), Path(contract["binary"]),
                             contract["batch"], Path(contract["sgf_root"]), path.parent, Path(contract["python"]))
    require(set(contract) == set(rebuilt) | {"environment", "prepared_artifacts"}, "unknown/incomplete prepared contract fields")
    require(all(ledger.same(value, contract[key]) for key, value in rebuilt.items()), "prepared contract/commands differ from rebuilt inputs")
    require(type(contract["environment"]) is dict and all(type(k) is str and k.upper() in SAFE_ENVIRONMENT and type(v) is str
                                                         for k, v in contract["environment"].items()), "unsafe prepared environment")
    expected_artifacts = [file_record(path.parent / "workload.json")]
    for entry in contract["commands"]:
        command_path = path.parent / "commands" / (entry["id"] + ".json")
        require(ledger.same(read_json(command_path), entry), "frozen command differs from contract")
        expected_artifacts.append(file_record(command_path))
    require(ledger.same(expected_artifacts, contract["prepared_artifacts"]), "prepared artifact inventory differs")
    return contract


class WindowsJob:
    """Private kill-on-close job, assigned before the helper's stdin gate opens."""
    def __init__(self):
        require(os.name == "nt", "actual execution currently requires Windows process containment")
        class Basic(ctypes.Structure):
            _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64), ("Flags", wintypes.DWORD),
                        ("MinWorkingSet", ctypes.c_size_t), ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("Priority", wintypes.DWORD), ("Scheduling", wintypes.DWORD)]
        class Io(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("Basic", Basic), ("Io", Io), ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
                        ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t)]
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateJobObjectW.argtypes, self.api.CreateJobObjectW.restype = [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway permitted.
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.get_last_error()
            self.close()
            raise ctypes.WinError(error)

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None


def execute(command, environment_values, stdout, stderr, timeout):
    """Only this function can spawn; the caller persists ATTEMPT_STARTED first."""
    process, job = None, WindowsJob()
    try:
        process = subprocess.Popen(command["launcher_argv"],
                                   stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, cwd=ROOT, env=environment_values,
                                   creationflags=subprocess.CREATE_NO_WINDOW, close_fds=True)
        job.assign(process)
        # Passing the already validated value avoids rereading a mutable file
        # between the parent's identity check and the child's execution.
        process.stdin.write(compact(command))
        process.stdin.close()
        return process.wait(timeout=timeout)
    finally:
        # This private job contains only our gated helper and its descendants.
        job.close()
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Assignment failure leaves only the unreleased helper.
                process.kill()
                process.wait(timeout=5)


def append_event(root, events, kind, payload):
    event = dict(schema=SCHEMA, sequence=len(events), previous_sha256=events[-1]["sha256"] if events else ledger.ZERO,
                 event_type=kind, payload=payload)
    event["sha256"] = digest(compact(event))
    write_new(root / "events" / f"{len(events):04d}.json", event)
    events.append(event)
    return event


def verify_sources(contract):
    for record in contract["source_records"] + contract["prepared_artifacts"]:
        ledger.verify_record(record)


def verify_history(root, events):
    for event in events:
        require(ledger.same(read_json(root / "events" / f"{event['sequence']:04d}.json"), event), "driver event changed")
        for record in event["payload"].get("artifacts", []) + event["payload"].get("logs", []):
            ledger.verify_record(record)


def collection_identity(contract, row):
    path = Path(row["collection_directory"]) / "report.json"
    report = read_json(path)
    require(report.get("status") == "COLLECTED_UNVERIFIED" and report.get("backend") == "cudaquantbackend"
            and report.get("split") == contract["workload"]["split"] and report.get("window_mode") == "continuous"
            and report.get("batch") == contract["batch"] and report.get("capacity") == contract["workload"]["capacity"]
            and report.get("request_window") == contract["workload"]["concurrency"]
            and report.get("model_sha256") == contract["model_sha256"] and report.get("binary_sha256") == contract["binary_sha256"]
            and report.get("actual_profile", {}).get("recipe_sha256") == row["recipe_sha256"], "collector actual execution differs from contract")
    return report, file_record(path)


def run(prepared, expect_sha=None, *, runner=None):
    prepared = Path(prepared).resolve()
    contract = validate_prepared(prepared, expect_sha)
    root = prepared.parent
    require(not (root / "events").exists() and not (root / "result.json").exists(), "prepared run is single-use; no retry/resume")
    # Retained on every outcome, including interruption: never steal this lock.
    with (root / ".run.lock").open("x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
    events, counts, numeric_gates = [], {"NUMERIC": 0, "PERFORMANCE": 0}, {}
    rows = {row["recipe_sha256"]: row for row in contract["candidates"]}
    reference = contract["reference_recipe_sha256"]
    (root / "events").mkdir()
    append_event(root, events, "RUN_STARTED", dict(prepared=file_record(prepared),
                 declared_numeric_budget=contract["expected_numeric_attempts"], declared_performance_budget=contract["expected_performance_attempts"],
                 declared_recipe_sha256=list(rows), environment=contract["environment"]))
    runner = runner or execute
    state = None
    try:
        for command in contract["commands"]:
            recipe, phase = command["recipe_sha256"], command["phase"]
            if phase in ("benchmark", "ingest-performance") and numeric_gates.get(recipe) == "FAIL":
                append_event(root, events, "STEP_SKIPPED", dict(command_id=command["id"], recipe_sha256=recipe, reason="NUMERIC_GATE_FAIL_NO_RETRY"))
                continue
            verify_sources(contract)
            verify_history(root, events)
            require(file_record(prepared)["sha256"] == events[0]["payload"]["prepared"]["sha256"], "prepared manifest changed during execution")
            if state is not None:
                ledger.inspect(root / "ledger", state["head_sha256"])
            charge = command["charge"]
            if charge:
                counts[charge] += 1
                require(counts[charge] <= contract["expected_" + charge.lower() + "_attempts"], "driver budget exceeded")
            append_event(root, events, "ATTEMPT_STARTED", dict(command_id=command["id"], recipe_sha256=recipe,
                         command=command, counts=dict(counts)))
            log = root / "logs" / command["id"]
            log.parent.mkdir(exist_ok=True)
            with log.with_suffix(".stdout.log").open("xb") as stdout, log.with_suffix(".stderr.log").open("xb") as stderr:
                code = runner(command, contract["environment"], stdout, stderr, command["timeout_seconds"])
            logs = [file_record(log.with_suffix(suffix)) for suffix in (".stdout.log", ".stderr.log")]
            append_event(root, events, "PROCESS_EXITED", dict(command_id=command["id"], returncode=code, logs=logs))
            verify_sources(contract)
            artifacts = []
            if phase == "compare":
                comparison_path = Path(rows[recipe]["comparison_directory"]) / "report.json"
                comparison = read_json(comparison_path)
                gate = comparison.get("experimental_gate", {}).get("result")
                require(gate in ("PASS", "FAIL") and code == (0 if gate == "PASS" else 1)
                        and comparison.get("collection_validation") != "FAIL" and not comparison.get("error"), "comparison execution failed")
                require(recipe != reference or gate == "PASS", "FP16 reference failed; stop before ledger initialization")
                numeric_gates[recipe] = gate
                artifacts.append(file_record(comparison_path))
            else:
                require(code == 0, f"{command['id']} subprocess failed with status {code}")
            if phase == "collect":
                _, artifact = collection_identity(contract, rows[recipe])
                artifacts.append(artifact)
            elif phase == "ledger-init":
                state = ledger.inspect(root / "ledger")
            elif phase in ("ingest-numeric", "ingest-performance"):
                state = ledger.inspect(root / "ledger")
                kind = "NUMERIC" if phase == "ingest-numeric" else "PERFORMANCE"
                result = state["results"][kind][recipe]
                require(result["gate"] != "INVALID_EVIDENCE", "ledger rejected raw evidence")
                if kind == "NUMERIC":
                    require(result["gate"] == numeric_gates[recipe], "comparator and raw ledger numeric gate differ")
                artifacts.append(file_record(root / "ledger" / "events" / f"{state['event_count']-1:06d}.json"))
            elif phase == "benchmark":
                artifacts.append(file_record(Path(rows[recipe]["performance_directory"]) / "report.json"))
            elif phase == "summarize":
                summary = read_json(root / "summary/selection-summary.json")
                require(summary["head_sha256"] == state["head_sha256"] and summary["publish_allowed"] is False
                        and summary["deployment_adopted"] is False and summary["production_certified"] is False
                        and summary["holdout_outputs_read"] == 0, "summary changed execution scope/head")
                artifacts.append(file_record(root / "summary/selection-summary.json"))
            if phase == "compare" and recipe == reference:
                _, artifact = collection_identity(contract, rows[reference])
                catalog = dict(schema="rustgo-quantization-ledger-baselines-v1", fp16_reference_report_sha256=artifact["sha256"],
                               baselines=[dict(recipe_sha256=reference, collection_report_sha256=artifact["sha256"])])
                write_new(root / "baselines.json", catalog)
                artifacts.append(file_record(root / "baselines.json"))
            append_event(root, events, "STEP_VALIDATED", dict(command_id=command["id"], artifacts=artifacts,
                         ledger_head_sha256=state["head_sha256"] if state else None))
        require(state["numeric_attempts"] == counts["NUMERIC"] and state["performance_attempts"] == counts["PERFORMANCE"],
                "completed ledger accounting differs from preregistered driver attempts")
        result = dict(schema=SCHEMA, status="FINITE_EXECUTION_COMPLETE_REQUIRES_REVIEW", counts=counts,
                      numeric_gates=numeric_gates, summary=file_record(root / "summary/selection-summary.json"),
                      ledger_head_sha256=state["head_sha256"], baseline_optimality="NOT_ESTABLISHED",
                      holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)
        append_event(root, events, "RUN_COMPLETED", result)
        result["driver_head_sha256"] = events[-1]["sha256"]
        write_new(root / "result.json", result)
        return result
    except BaseException as error:
        failed_logs = []
        if "log" in locals():
            failed_logs = [file_record(log.with_suffix(suffix)) for suffix in (".stdout.log", ".stderr.log") if log.with_suffix(suffix).is_file()]
        result = dict(schema=SCHEMA, status="FAILED_NO_RETRY", error=f"{type(error).__name__}: {error}", counts=counts,
                      numeric_gates=numeric_gates, failed_command_id=command["id"] if "command" in locals() else None,
                      logs=failed_logs, holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)
        append_event(root, events, "RUN_FAILED", result)
        result["driver_head_sha256"] = events[-1]["sha256"]
        write_new(root / "result.json", result)
        raise


EXECUTION_DRIVER_SCHEMA = "rustgo-bounded-catalog-execution-driver-v2"


def execution_contract(catalog_path, catalog_sha, registry, registry_sha, head, root, python):
    """Rebuild an immutable finite plan against one existing durable ledger."""
    import quantization_execution_catalog as catalogs
    import legacy_quantization_execution as legacy
    for name, value in (("catalog SHA", catalog_sha), ("ledger contract SHA", registry_sha), ("ledger head", head)):
        ledger.planner.corpus_io.check_sha(value, name)
    require(not root.is_relative_to(registry), "driver output must be outside the durable ledger")
    catalog = catalogs.validate_catalog(catalog_path, catalog_sha)
    registry_record = file_record(registry / "contract.json")
    require(registry_record["sha256"] == registry_sha, "external ledger contract differs from anchor")
    durable, events, state = ledger.execution_replay(registry, head)
    require(not any(key not in state["collections"] and key not in state.get("performance", {})
                    and key not in state["failures"] for key in state["reservations"]),
            "external ledger has pending attempts; no retry/resume")
    plan = read_json(catalog["inputs"]["search_plan"])
    work = catalog["workload"]
    require(work["split"] in ("calibration", "selection"), "holdout execution is forbidden")
    source_rows = [file_record(catalog_path), registry_record, file_record(python), file_record(__file__),
                   *ledger.execution_sources(), *catalog["source_records"],
                   file_record(registry / "events" / f"{len(events)-1:06d}.json")]
    specs = {row["execution_spec_id"]: row for row in catalog["execution_specs"]}
    recipes = {row["recipe_sha256"]: Path(catalog["inputs"]["search_plan"]).parent / row["recipe_file"] for row in plan["candidates"]}
    legacy_specs = [legacy.load_spec(path) for path in catalog["inputs"]["legacy_specs"]]
    reference = catalog["identity"]["canonical_reference_execution_spec_id"]
    require(catalog["numeric_pairs"][0]["candidate_execution_spec_id"] == reference,
            "canonical unified reference must be the first collection")
    entries, collections, performance, directories = [], [], [], {}

    def command(phase, tool, arguments, key, output, timeout):
        argv = [str(python), "-E", "-s", str(ROOT / "scripts" / tool), *map(str, arguments)]
        entry = dict(id=f"{len(entries):03d}-{phase}", phase=phase, attempt_key=key, argv=argv,
                     argv_sha256=digest(compact(argv)),
                     launcher_argv=[str(python), "-E", "-s", str(Path(__file__).resolve()), "_child"],
                     cwd=str(ROOT), output_directory=str(output), timeout_seconds=timeout)
        entries.append(entry)
        return entry["id"]

    count = catalog["selected_request_count"]
    corpus = Path(plan["source_paths"]["corpus_manifest"])
    for index, row in enumerate(catalog["numeric_pairs"]):
        key, spec_id = row["collection_attempt_key"], row["candidate_execution_spec_id"]
        require(key not in state["failures"], "collection already failed; no retry across prepares")
        spec, observed = specs[spec_id], state["collections"].get(key)
        if key in state["reservations"]:
            require(observed is not None, "collection already consumed without valid evidence; no retry")
        directory = Path(observed["original_directory"]) if observed else root / "collections" / f"{index:02d}-{digest(spec_id.encode())[:16]}"
        directories[spec_id] = directory
        entry = dict(execution_spec_id=spec_id, collection_attempt_key=key, comparison_key=row["comparison_key"],
                     directory=str(directory), action="REUSE_VALIDATED" if observed else "COLLECT", command_id=None,
                     reused_observation=observed)
        if observed:
            source_rows.extend(observed["original_files"] + observed["runtime_sources"])
        else:
            execution = spec["semantic"]["execution"]
            if spec["kind"] == "unified_v1_recipe":
                binary, model = catalog["inputs"]["unified_binary"], plan["source_paths"]["model"]
                specific = ["--recipe", recipes[execution["recipe_sha256"]]]
            else:
                matching = [loaded for loaded in legacy_specs if ledger.same(loaded["semantic"], execution)]
                require(matching, "planned legacy execution has no bound original spec")
                loaded = matching[0]
                binary, model = loaded["binary"], loaded["spec"]["model"]["path"]
                # Keep the original source config/plan and explicit q64/tune0 spec.
                source_spec = next(path for path in catalog["inputs"]["legacy_specs"]
                                   if ledger.same(legacy.load_spec(path)["semantic"], execution))
                specific = ["--config", loaded["config"], "--execution-spec", source_spec]
            args = ["--binary", binary, "--model", model, "--requests", corpus.parent / (work["split"] + ".requests.jsonl"),
                    "--manifest", corpus, "--output", directory, "--backend", execution["backend"], *specific,
                    "--batch", execution["batch"], "--capacity", work["capacity"], "--window", work["concurrency"],
                    "--window-mode", "continuous", "--max-buffered-result-mib", 256, "--max-requests", 100000,
                    "--startup-timeout", 300, "--task-timeout", work["task_timeout"]]
            # Includes discovery, both legacy probes, Worker startup and Drain.
            entry["command_id"] = command("collect", "collect_quantization_corpus.py", args, key, directory,
                                           1290 + count * work["task_timeout"])
        collections.append(entry)
    for index, pair in enumerate(catalog["performance_pairs"]):
        key = pair["pair_key"]
        require(key not in state["failures"], "ABBA already failed; no retry across prepares")
        observed = state.get("performance", {}).get(key)
        if key in state["reservations"]:
            require(observed is not None, "ABBA already consumed without valid evidence; no retry")
        directory = root / "performance" / f"{index:02d}-{digest(key.encode())[:16]}"
        entry = dict(pair_key=key, action="REUSE_VALIDATED" if observed else "BENCHMARK", command_id=None,
                     directory=str(directory), reused_observation=observed)
        if observed:
            source_rows.extend(observed.get("original_files", []) + observed.get("runtime_sources", []))
        if not observed:
            args = ["--reference", directories[reference], "--baseline", directories[pair["baseline_execution_spec_id"]],
                    "--candidate", directories[pair["candidate_execution_spec_id"]], "--output", directory,
                    "--concurrency", work["concurrency"], "--capacity", work["capacity"], "--cycles", work["cycles"],
                    "--warmup", work["warmup"], "--metric", work["metric"], "--startup-timeout", 300,
                    "--task-timeout", work["task_timeout"], *(["--smoke"] if work["smoke"] else [])]
            entry["command_id"] = command("benchmark", "benchmark_quantization_corpus.py", args, key, directory,
                4 * (990 + (work["warmup"] + count * work["cycles"]) * work["task_timeout"]))
        performance.append(entry)
    needed = dict(COLLECTION=sum(r["action"] == "COLLECT" for r in collections),
                  ABBA=sum(r["action"] == "BENCHMARK" for r in performance))
    consumed = {stage: sum(r["stage"] == stage for r in state["reservations"].values()) for stage in needed}
    require(all(needed[stage] <= durable["budgets"][stage] - consumed[stage] for stage in needed),
            "external durable budget cannot cover the complete remaining execution set")
    sources = catalogs.unique_records(source_rows)
    for item in sources:
        ledger.verify_record(item)
    return dict(schema=EXECUTION_DRIVER_SCHEMA, root=str(root), catalog_path=str(catalog_path), catalog_sha256=catalog_sha,
                registry=str(registry), registry_contract_sha256=registry_sha, initial_registry_head_sha256=head,
                registry_id=durable["ledger_id"], python=str(python), workload=work,
                reference_execution_spec_id=reference, collections=collections, performance=performance,
                commands=entries, source_records=sources, needed_attempts=needed, initial_consumed_attempts=consumed,
                baseline_scope="all registered unified candidates against canonical reference and every required legacy control",
                restart_policy="single-use prepare; reuse validated terminal evidence only; pending/failed keys never retried",
                holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)


def prepare_execution(catalog, catalog_sha, registry, registry_sha, head, output):
    catalog, registry, output = (Path(path).resolve() for path in (catalog, registry, output))
    require(not output.exists() and output.parent.is_dir(), "prepare output must be NEW with existing parent")
    require(not (registry / ".execution-driver.lock").exists(), "external ledger driver lock exists; never steal it")
    contract = execution_contract(catalog, catalog_sha, registry, registry_sha, head, output, Path(sys.executable).resolve())
    contract["environment"] = environment()
    output.mkdir()
    records = []
    for entry in contract["commands"]:
        path = output / "commands" / (entry["id"] + ".json")
        write_new(path, entry)
        records.append(file_record(path))
    contract["prepared_artifacts"] = records
    verify_sources(contract)
    write_new(output / "prepared-execution.json", contract)
    return dict(status="PREPARED_NO_EXECUTION", prepared=file_record(output / "prepared-execution.json"),
                registry_head_sha256=head, needed_attempts=contract["needed_attempts"],
                ledger_created=False, attempts_reserved=0, subprocesses_started=0, publish_allowed=False)


def validate_execution_prepared(path, expect_sha):
    path = Path(path).resolve()
    ledger.planner.corpus_io.check_sha(expect_sha, "prepared manifest SHA")
    require(path.name == "prepared-execution.json" and not path.is_symlink()
            and file_record(path)["sha256"] == expect_sha, "prepared execution path/SHA differs")
    contract = read_json(path)
    require(contract.get("schema") == EXECUTION_DRIVER_SCHEMA and contract.get("root") == str(path.parent),
            "prepared execution schema/location differs")
    verify_sources(contract)
    rebuilt = execution_contract(Path(contract["catalog_path"]), contract["catalog_sha256"], Path(contract["registry"]),
        contract["registry_contract_sha256"], contract["initial_registry_head_sha256"], path.parent, Path(contract["python"]))
    require(set(contract) == set(rebuilt) | {"environment", "prepared_artifacts"}
            and all(ledger.same(value, contract[key]) for key, value in rebuilt.items()),
            "prepared execution contract differs from rebuilt catalog/ledger")
    require(type(contract["environment"]) is dict and all(type(k) is str and k.upper() in SAFE_ENVIRONMENT
            and type(v) is str for k, v in contract["environment"].items()), "unsafe prepared environment")
    records = []
    for entry in contract["commands"]:
        command_path = path.parent / "commands" / (entry["id"] + ".json")
        require(ledger.same(read_json(command_path), entry), "frozen execution command differs")
        records.append(file_record(command_path))
    require(ledger.same(records, contract["prepared_artifacts"]), "execution artifact inventory differs")
    return contract


def append_execution_event(root, events, kind, payload):
    event = dict(schema=EXECUTION_DRIVER_SCHEMA, sequence=len(events), previous_sha256=events[-1]["sha256"] if events else ledger.ZERO,
                 event_type=kind, payload=payload)
    event["sha256"] = digest(compact(event))
    write_new(root / "events" / f"{len(events):04d}.json", event)
    events.append(event)
    return event


def run_execution(prepared, expect_sha, *, runner=None):
    prepared = Path(prepared).resolve()
    contract = validate_execution_prepared(prepared, expect_sha)
    root, registry = prepared.parent, Path(contract["registry"])
    require(not any((root / name).exists() for name in ("events", "result.json", ".run.lock")), "prepared execution is single-use; no retry/resume")
    require(all(not Path(entry["output_directory"]).exists() for entry in contract["commands"]),
            "unexecuted helper output already exists; no evidence overwrite/retry")
    guard = registry / ".execution-driver.lock"
    # This external guard serializes our complete launch window; ledger APIs
    # still enforce their own locks and exact expected head at every mutation.
    with guard.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(prepared=file_record(prepared), pid=os.getpid())) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    guard_record = file_record(guard)
    events, gates, outcomes = [], {}, {}
    counts = dict(COLLECTION=0, ABBA=0)
    active, log, head = None, None, contract["initial_registry_head_sha256"]
    runner = runner or execute
    try:
        with (root / ".run.lock").open("x", encoding="utf-8") as stream:
            stream.write(f"pid={os.getpid()}\n"); stream.flush(); os.fsync(stream.fileno())
        (root / "events").mkdir()
        append_execution_event(root, events, "RUN_STARTED", dict(prepared=file_record(prepared), registry=str(registry),
            registry_head_sha256=head, needed_attempts=contract["needed_attempts"], environment=contract["environment"]))
        state = ledger.execution_register_catalog(registry, Path(contract["catalog_path"]), contract["catalog_sha256"], head)
        head = state["head_sha256"]
        catalog = read_json(contract["catalog_path"])
        commands_by_id = {entry["id"]: entry for entry in contract["commands"]}

        def check():
            verify_sources(contract); verify_history(root, events)
            ledger.verify_record(guard_record)
            require(file_record(prepared)["sha256"] == expect_sha, "prepared execution changed during run")
            return ledger.execution_replay(registry, head)

        def invoke(stage, key, command_id, qualification=None):
            nonlocal active, log, head
            check()
            command = commands_by_id[command_id]
            receipt = ledger.execution_reserve(registry, contract["catalog_sha256"], stage, key, head)
            head = receipt["head_sha256"]
            active = receipt["reservation"]
            counts[stage] += 1
            if qualification is not None:
                require(ledger.same(active["numeric_qualification"], qualification), "reserved observed numerical identity changed")
            # Fsynced external receipt is bound before opening the helper gate.
            append_execution_event(root, events, "ATTEMPT_RESERVED", dict(receipt=receipt, command=command,
                artifacts=[file_record(registry / "events" / f"{len(ledger.execution_replay(registry, head)[1])-1:06d}.json")]))
            check()
            append_execution_event(root, events, "ATTEMPT_STARTED", dict(reservation_id=active["reservation_id"], command=command,
                registry_head_sha256=head))
            log = root / "logs" / command["id"]
            log.parent.mkdir(exist_ok=True)
            with log.with_suffix(".stdout.log").open("xb") as stdout, log.with_suffix(".stderr.log").open("xb") as stderr:
                code = runner(command, contract["environment"], stdout, stderr, command["timeout_seconds"])
            logs = [file_record(log.with_suffix(s)) for s in (".stdout.log", ".stderr.log")]
            append_execution_event(root, events, "PROCESS_EXITED", dict(reservation_id=active["reservation_id"], returncode=code, logs=logs))
            check()
            require(code == 0, f"{command_id} failed with exit status {code}; no retry")

        for row in contract["collections"]:
            key, spec_id = row["collection_attempt_key"], row["execution_spec_id"]
            check()
            if row["action"] == "COLLECT":
                invoke("COLLECTION", key, row["command_id"])
                state = ledger.execution_ingest_collection(registry, active["reservation_id"], Path(row["directory"]), head)
                head, active = state["head_sha256"], None
            else:
                append_execution_event(root, events, "COLLECTION_REUSED", dict(attempt_key=key, observation=row["reused_observation"]))
            state = ledger.execution_compare(registry, contract["catalog_sha256"], row["comparison_key"], head)
            head = state["head_sha256"]
            gate = state["numerical_comparisons"][row["comparison_key"]]["metrics"]["experimental_gate"]["result"]
            require(gate in ("PASS", "FAIL"), "invalid numeric gate")
            gates[spec_id] = gate
            append_execution_event(root, events, "NUMERIC_VALIDATED", dict(execution_spec_id=spec_id, gate=gate,
                comparison_key=row["comparison_key"], registry_head_sha256=head))
            require(spec_id != contract["reference_execution_spec_id"] or gate == "PASS", "canonical FP16 reference failed; stop")
        for row, pair in zip(contract["performance"], catalog["performance_pairs"], strict=True):
            _, _, replayed = check()
            key = row["pair_key"]
            if any(gates[spec_id] != "PASS" for spec_id in pair["requires_numeric_pass_for"]):
                outcomes[key] = "SKIPPED_NUMERIC_FAIL"
                append_execution_event(root, events, "PAIR_SKIPPED", dict(pair_key=key, reason="NUMERIC_FAIL_NO_ABBA_RESERVATION"))
                continue
            qualification = ledger.execution_qualification(catalog, pair, replayed)
            if row["action"] == "BENCHMARK":
                invoke("ABBA", key, row["command_id"], qualification)
                state = ledger.execution_ingest_abba(registry, active["reservation_id"], Path(row["directory"]), head)
                head, active = state["head_sha256"], None
            else:
                state = ledger.execution_inspect(registry, head)
            outcomes[key] = "VALIDATED_TERMINAL_EVIDENCE"
            append_execution_event(root, events, "PAIR_VALIDATED", dict(pair_key=key, reused=row["action"] == "REUSE_VALIDATED",
                numeric_qualification=qualification, registry_head_sha256=head))
        check()
        final_state = ledger.execution_inspect(registry, head)
        require(final_state["consumed_attempts"] == {stage: contract["initial_consumed_attempts"][stage] + counts[stage] for stage in counts},
                "durable accounting differs from actual driver reservations")
        summary = ledger.execution_summary(registry, contract["catalog_sha256"], root / "summary", head)
        require(summary["report"]["head_sha256"] == head and summary["report"]["publish_allowed"] is False
                and summary["report"]["deployment_adopted"] is False and summary["report"]["production_certified"] is False
                and summary["report"]["holdout_outputs_read"] == 0, "execution summary changes scope/head")
        check()
        result = dict(schema=EXECUTION_DRIVER_SCHEMA, status="FINITE_EXECUTION_COMPLETE_REQUIRES_REVIEW", registry=str(registry),
            registry_head_sha256=head, numeric_gates=gates, pair_outcomes=outcomes, summary=summary["summary_file"],
            actual_new_attempts=counts,
            holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)
        append_execution_event(root, events, "RUN_COMPLETED", result)
        result["driver_head_sha256"] = events[-1]["sha256"]
        write_new(root / "result.json", result)
        ledger.verify_record(guard_record)
        guard.unlink()  # Only our own successfully completed external guard.
        return result
    except BaseException as error:
        terminal_error = None
        try:
            state = ledger.execution_inspect(registry)
            head = state["head_sha256"]
            if active is not None:
                if any(r["reservation_id"] == active["reservation_id"] for r in state["pending_reservations"]):
                    state = ledger.execution_fail(registry, active["reservation_id"], f"driver stopped: {type(error).__name__}: {error}"[:4096], head)
                    head = state["head_sha256"]
        except BaseException as final_error:
            terminal_error = f"{type(final_error).__name__}: {final_error}"
        logs = [file_record(log.with_suffix(s)) for s in (".stdout.log", ".stderr.log")
                if log is not None and log.with_suffix(s).is_file()]
        result = dict(schema=EXECUTION_DRIVER_SCHEMA, status="FAILED_NO_RETRY", error=f"{type(error).__name__}: {error}",
            registry=str(registry), registry_head_sha256=head, active_reservation=active, terminal_record_error=terminal_error,
            logs=logs, numeric_gates=gates, pair_outcomes=outcomes, external_guard_retained=True,
            actual_new_attempts=counts,
            holdout_outputs_read=0, deployment_adopted=False, publish_allowed=False, production_certified=False)
        append_execution_event(root, events, "RUN_FAILED", result)
        result["driver_head_sha256"] = events[-1]["sha256"]
        write_new(root / "result.json", result)
        raise


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "_child":
        raw = sys.stdin.buffer.read(131073)
        require(0 < len(raw) <= 131072, "parent did not release a bounded contained command")
        entry = ledger.planner.strict_json(raw)
        sys.argv = entry["argv"][3:]
        sys.path.insert(0, str(Path(sys.argv[0]).parent))
        runpy.run_path(sys.argv[0], run_name="__main__")
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="CPU-only validation; never starts subprocesses")
    for name in ("search-plan", "workload", "binary", "sgf-root", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--batch", type=int, required=True)
    execute_parser = sub.add_parser("run", help="Execute the entire immutable prepared experiment once")
    execute_parser.add_argument("--prepared", type=Path, required=True)
    execute_parser.add_argument("--expect-manifest-sha256")
    prep_execution = sub.add_parser("prepare-execution", help="CPU-only catalog plan against an existing durable ledger")
    for name in ("catalog", "ledger", "output"):
        prep_execution.add_argument("--" + name, type=Path, required=True)
    for name in ("expect-catalog-sha256", "expect-ledger-sha256", "expect-head"):
        prep_execution.add_argument("--" + name, required=True)
    run_execution_parser = sub.add_parser("run-execution", help="Execute a frozen catalog once against its existing ledger")
    run_execution_parser.add_argument("--prepared", type=Path, required=True)
    run_execution_parser.add_argument("--expect-manifest-sha256", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.search_plan, args.workload, args.binary, args.batch, args.sgf_root, args.output)
        elif args.command == "run":
            result = run(args.prepared, args.expect_manifest_sha256)
        elif args.command == "prepare-execution":
            result = prepare_execution(args.catalog, args.expect_catalog_sha256, args.ledger, args.expect_ledger_sha256,
                                       args.expect_head, args.output)
        else:
            result = run_execution(args.prepared, args.expect_manifest_sha256)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        print(f"run_quantization_search: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
