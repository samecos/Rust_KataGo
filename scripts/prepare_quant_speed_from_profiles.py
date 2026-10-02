"""Prepare a fresh speed-only ABBA plan from the measured profile handoff.

This creates metadata only. It does not run inference or evaluate accuracy.
Existing experiments, deadlines and output directories are never reused.
"""
import argparse
import json
from pathlib import Path
import time

from benchmark_quant_speed import require, save, source, validate_schedule, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--profiles-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=128)
    args = parser.parse_args()
    require(source(args.profiles)["sha256"] == args.profiles_sha256, "wrong handoff digest")
    handoff = json.loads(args.profiles.read_text(encoding="utf-8"))
    require(handoff["schema"] == "rustgo-speed-profile-handoff-v1", "wrong handoff schema")
    require(handoff["accuracy"] == "USER_VERIFICATION_NOT_EVALUATED", "wrong scope")
    require(1 <= args.iterations <= 20000 and 1 <= args.warmup <= 1000, "invalid lengths")
    require(handoff["workers"] == 32, "handoff was measured at C32")
    variants, runs = {}, []
    for profile in handoff["profiles"]:
        require(profile["speed_confirmed"], "profile lacks independent speed confirmation")
        a, b = profile["baseline"], profile["candidate"]
        for variant in (a, b):
            for key in ("binary", "model", "config", "recipe", "tactic_plan"):
                if variant.get(key):
                    verify(variant[key])
            previous = variants.get(variant["id"])
            require(previous is None or previous == variant, "conflicting variant identity")
            variants[variant["id"]] = variant
        for stage, flavor, variant in (("A1", "baseline", a), ("B1", "candidate", b),
                                       ("B2", "candidate", b), ("A2", "baseline", a)):
            runs.append(dict(variant=variant["id"], comparison=profile["group"], flavor=flavor,
                             stage=stage, iterations=args.iterations, warmup=args.warmup))
    require(0 < len(runs) <= 32, "unexpected handoff size")
    schedule = dict(schema="rustgo-speed-abba-schedule-v1", handoff=source(args.profiles), runs=runs)
    validate_schedule(schedule, variants)
    verify(handoff["environment"])
    verify(handoff["source_nnbench"])
    verify(handoff["runner"])
    require(Path(handoff["runner"]["path"]).resolve() == Path(__file__).with_name("benchmark_quant_speed.py").resolve(),
            "wrong runner path")
    start = time.monotonic_ns()
    plan = dict(schema="rustgo-quant-speed-plan-v1", scope="SPEED_ONLY_USER_OWNS_ACCURACY",
                runner=handoff["runner"], preparer=source(__file__), handoff=source(args.profiles),
                binary=handoff["binary"], environment=handoff["environment"],
                source_nnbench=handoff["source_nnbench"], workers=32,
                maximum_model_starts=len(runs), per_run_timeout_seconds=300,
                created_monotonic_ns=start, deadline_monotonic_ns=start + 7200 * 1_000_000_000,
                variants=list(variants.values()))
    args.output.mkdir(parents=True, exist_ok=False)
    plan_path, schedule_path = args.output / "plan.json", args.output / "schedule.json"
    save(plan_path, plan)
    save(schedule_path, schedule)
    print(json.dumps(dict(plan=source(plan_path), schedule=source(schedule_path)), indent=2))


if __name__ == "__main__":
    main()
