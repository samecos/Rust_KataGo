"""Prepare finite speed qualification for a new executable using existing recipes.

This creates metadata only. Old measured profiles remain bound to their old
executable; the replacement must earn its own speed result and actual identity.
"""
import argparse
import copy
import json
from pathlib import Path
import time

from benchmark_quant_speed import require, save, source, validate_schedule, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--profiles-sha256", required=True)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--build-result", required=True, type=Path)
    parser.add_argument("--build-result-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--native-starts-so-far", required=True, type=int)
    parser.add_argument("--cumulative-native-limit", required=True, type=int)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=128)
    args = parser.parse_args()
    require(source(args.profiles)["sha256"] == args.profiles_sha256, "wrong handoff digest")
    require(source(args.build_result)["sha256"] == args.build_result_sha256, "wrong build receipt digest")
    binary = source(args.binary)
    require(binary["sha256"] == args.binary_sha256, "wrong replacement executable digest")
    build = json.loads(args.build_result.read_text(encoding="utf-8"))
    require(build["schema"] == "rustgo-typed-worker-native-build-result-v1", "wrong build receipt schema")
    require(build["status"] == "BUILT_NOT_GPU_QUALIFIED" and
            not build.get("deadline_exceeded") and not build.get("error"),
            "replacement build did not reach a successful terminal status")
    require(build.get("tool_actual_exit_code") == 0 and
            build.get("root_terminal_before_deadline") is True,
            "replacement build lacks an observed normal tool exit within its deadline")
    require(build["actual_exit_code"] == 0 and build["sources_unchanged"] is True,
            "replacement build did not complete with stable sources")
    require(build["binary"] == binary, "replacement executable differs from build receipt")
    handoff = json.loads(args.profiles.read_text(encoding="utf-8"))
    require(handoff["schema"] == "rustgo-speed-profile-handoff-v1", "wrong handoff schema")
    require(handoff["accuracy"] == "USER_VERIFICATION_NOT_EVALUATED", "wrong scope")
    require(handoff["workers"] == 32, "only the measured C32 load is supported")
    require(binary["sha256"] != handoff["binary"]["sha256"], "this entry is for a changed executable")
    require(1 <= args.iterations <= 20000 and 1 <= args.warmup <= 1000, "invalid lengths")
    variants, runs = {}, []
    for profile in handoff["profiles"]:
        require(profile["speed_confirmed"], "profile lacks independent speed confirmation")
        a, old_b = profile["baseline"], profile["candidate"]
        for variant in (a, old_b):
            for key in ("binary", "model", "config", "recipe", "tactic_plan"):
                if variant.get(key):
                    verify(variant[key])
        b = copy.deepcopy(old_b)
        b["id"] += "-rebuilt"
        b["binary"] = binary
        b["previous_measured_candidate"] = old_b
        # The original config has no old expected-profile pin. The first new
        # normal sample records the new actual profile; repeated arms must agree.
        config = Path(b["config"]["path"]).read_text(encoding="utf-8")
        require("cudaQuantExpectedProfile" not in config, "old profile pin must not be used for a new binary")
        for variant in (a, b):
            previous = variants.get(variant["id"])
            require(previous is None or previous == variant, "conflicting variant identity")
            variants[variant["id"]] = variant
        for stage, flavor, variant in (("A1", "baseline", a), ("B1", "candidate", b),
                                       ("B2", "candidate", b), ("A2", "baseline", a)):
            runs.append(dict(variant=variant["id"], comparison=profile["group"], flavor=flavor,
                             stage=stage, iterations=args.iterations, warmup=args.warmup))
    require(0 < len(runs) <= 8, "unexpected measured profile count")
    require(0 <= args.native_starts_so_far and
            args.native_starts_so_far + len(runs) <= args.cumulative_native_limit,
            "inherited cumulative native limit exceeded")
    schedule = dict(schema="rustgo-speed-abba-schedule-v1", handoff=source(args.profiles), runs=runs)
    validate_schedule(schedule, variants)
    for key in ("environment", "source_nnbench", "runner"):
        verify(handoff[key])
    require(Path(handoff["runner"]["path"]).resolve() == Path(__file__).with_name("benchmark_quant_speed.py").resolve(),
            "wrong runner path")
    now = time.monotonic_ns()
    plan = dict(schema="rustgo-quant-speed-plan-v1", scope="SPEED_ONLY_USER_OWNS_ACCURACY",
                runner=handoff["runner"], preparer=source(__file__), handoff=source(args.profiles),
                build_result=source(args.build_result), binary=binary,
                environment=handoff["environment"], source_nnbench=handoff["source_nnbench"],
                workers=32, maximum_model_starts=len(runs), per_run_timeout_seconds=300,
                inherited_native_starts=args.native_starts_so_far,
                cumulative_native_limit=args.cumulative_native_limit,
                old_speed_certificate_applies_to_new_binary=False,
                created_monotonic_ns=now, deadline_monotonic_ns=now + 3600 * 1_000_000_000,
                variants=list(variants.values()), accuracy="USER_VERIFICATION_NOT_EVALUATED")
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path, schedule_path = args.output / "plan.json", args.output / "schedule.json"
    save(plan_path, plan)
    save(schedule_path, schedule)
    print(json.dumps(dict(plan=source(plan_path), schedule=source(schedule_path)), indent=2))


if __name__ == "__main__":
    main()
