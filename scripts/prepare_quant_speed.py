"""Prepare a fresh finite speed-only plan from existing B11/B15 artifacts."""
import argparse
import json
from pathlib import Path
import time

import benchmark_quant_speed as speed


MODELS = {
    "b11": ("models/b11c768h12nbt3tflrs-fson-silu.bin.gz", "1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6"),
    "b15": ("models/b15-ffn-pruned-a8.bin.gz", "3f216ee88226ee49ca826eaa5f7e1e9982ff48a96625914bfc4e5ad20ee095a7"),
}
MIX = {
    "b11": {3: ["2679f0300d0eba79", "14361c8b286f9fd9", "ad4b263d00b37314"],
            8: ["1e2664bc2795d555", "7278e02de8c262de", "892b407c8867ef90"]},
    "b15": {3: ["3b0b6fa84c054154", "b7d9b6582120f25c", "fdb168f273ba6b7f"],
            8: ["783388b1b12823c2", "bf8469af95079221", "96055015016560eb"]},
}
TACTICS = {"b11": "plans/worker-tf3-sm120-c32.json",
           "b15": "autotune-output/pruned-worker-plan-20260920/result/plan.json"}
ALL_INT8 = {"b11": "target/unified-quant-g1/corpus-b11-ffn-r5/recipe.json",
            "b15": "target/unified-quant-g1/corpus-b15-ffn-w32-r5-r2/recipe.json"}
LEGACY_FP16_BINARIES = {
    "b11": ("target/release/katago-rs.exe", "d218b98dfdf2eb75820465cbf625dfd3f690b91c3bb112f3be1a56d2cc644dcc"),
    "b15": ("target/pruned-support/release/katago-rs.exe", "6ca478ab0065f66140a71fdc26a1b1e253475467ff5971d92ba57b41f9821d32"),
}


def prepare(output):
    binary = speed.source("target/unified-quant-restored-selection-cli-native-r1/run/build/release/katago-rs.exe")
    speed.require(binary["sha256"] == "dd665d03fe31739712a205eaa6f63852dbc27fee339dfe91f38067237e57bfdc", "wrong existing CLI")
    models = {name: speed.source(path) for name, (path, _) in MODELS.items()}
    speed.require(all(models[name]["sha256"] == sha for name, (_, sha) in MODELS.items()), "wrong model")
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    variants = []

    def add(model, cap, label, backend, recipe=None, env=None, tactic=None):
        identity = f"{model}-b{cap}-{label}"
        cfg = dict(rules="chinese", komi="7.5", numSearchThreads="1", nnBackend=backend,
                   nnMaxBatchSize=str(cap), numNNServerThreadsPerModel="1", nnRandomize="false",
                   nnForcedSymmetry="0", nnRandSeed="speed-only-r2", debugSkipNeuralNet="false",
                   nnCacheSizePowerOfTwo="10", nnMutexPoolSizePowerOfTwo="10", inputsUseNHWC="false", useFP16="auto")
        if recipe:
            cfg["cudaQuantPlan"] = str(Path(recipe).resolve(strict=True))
        if tactic:
            cfg["cudaTacticPlan"] = str(Path(tactic).resolve(strict=True))
        if backend == "cudaint8backend":
            cfg.update(cudaInt8Scope="ffn", cudaInt8MinFfnWidth="0")
        path = output / "configs" / (identity + ".cfg")
        path.write_text("".join(f"{k} = {v}\n" for k, v in cfg.items()), encoding="utf-8")
        record = dict(id=identity, group=f"{model}-b{cap}", model=models[model], backend=backend,
                      role="baseline" if label.startswith("legacy") else "candidate", batch_cap=cap,
                      config=speed.source(path), environment=env or {})
        if recipe:
            record["recipe"] = speed.source(recipe)
            metadata = json.loads(Path(recipe).read_text(encoding="utf-8"))
            speed.require(metadata["model_sha256"] == models[model]["sha256"], "recipe model differs")
            record["recipe_graph_sha256"] = metadata["graph_sha256"]
        if tactic:
            record["tactic_plan"] = speed.source(tactic)
            binary_path, expected_sha = LEGACY_FP16_BINARIES[model]
            record["binary"] = speed.source(binary_path)
            speed.require(record["binary"]["sha256"] == expected_sha, "wrong legacy FP16 binary")
        variants.append(record)

    for model in MODELS:
        for cap in (3, 8):
            registration = Path(f"target/unified-quant-restored-selection-mixed-prepared-r1/{model}-c{cap}-mix-000/registration.json")
            mapping = json.loads(registration.read_text(encoding="utf-8"))["environment"]
            env = {k: v for k, v in mapping.items() if v is not None}
            env.update(KATAGO_CUDA_NOGRAPH="0", KATAGO_CUDA_NOPIPELINE="0", KATAGO_CUDA_PADBATCH="0", KATAGO_CUDA_BATCH_TRACE="0")
            add(model, cap, "legacy-fp16", "cudabackend", tactic=TACTICS[model])
            add(model, cap, "legacy-int8-q64", "cudaint8backend",
                env={"KATAGO_CUDA_ATTN_TILE": "q64", "KATAGO_CUDA_INT8_GEMM_TUNE": "0", "KATAGO_CUDA_BATCH_TRACE": "0"})
            add(model, cap, "unified-fp16", "cudaquantbackend",
                recipe=f"target/unified-quant-calibration-full-r1/{model}-000-recipe.json", env=env)
            for index, digest in enumerate(MIX[model][cap], 1):
                add(model, cap, f"mix-{index}", "cudaquantbackend",
                    recipe=f"target/unified-quant-mixture-proposals-real-r1/{model}-{digest}.recipe.json", env=env)
            all_int8_env = {**env, "KATAGO_CUDA_DUALFFN": "0"}
            add(model, cap, "all-int8-q128", "cudaquantbackend", recipe=ALL_INT8[model], env=all_int8_env)
            add(model, cap, "all-int8-q64", "cudaquantbackend", recipe=ALL_INT8[model], env={**all_int8_env, "KATAGO_CUDA_ATTN_TILE": "q64"})
    # Also expose the B11 production FP16 cap16 reference. This prevents a
    # matched-cap gain from being mistaken for a gain over the production setup.
    add("b11", 16, "legacy-fp16", "cudabackend", tactic=TACTICS["b11"])
    env = next(v["environment"] for v in variants if v["id"] == "b11-b8-all-int8-q64")
    add("b11", 16, "all-int8-q64", "cudaquantbackend", recipe=ALL_INT8["b11"], env=env)
    now = time.monotonic_ns()
    plan = dict(schema="rustgo-quant-speed-plan-v1", scope="SPEED_ONLY_USER_OWNS_ACCURACY",
                runner=speed.source(speed.__file__), preparer=speed.source(__file__), binary=binary,
                source_nnbench=speed.source("crates/katago/src/cmd/nnbench.rs"),
                environment=speed.source("target/unified-quant-b15-compat-partial-tests-cpu-r1/environment.json"),
                workers=32, maximum_model_starts=96, per_run_timeout_seconds=300,
                created_monotonic_ns=now, deadline_monotonic_ns=now + 7200 * 10**9, variants=variants,
                historical_campaign="NOT_RESTARTED_OR_MUTATED", accuracy="USER_VERIFICATION_NOT_EVALUATED")
    speed.save(output / "plan.json", plan)
    smoke = dict(schema="rustgo-quant-speed-schedule-v1", stage="smoke",
                 runs=[dict(variant=f"{model}-b3-{label}", iterations=2, warmup=2)
                       for model in MODELS for label in ("legacy-fp16", "mix-3")])
    screen = dict(schema="rustgo-quant-speed-schedule-v1", stage="screen",
                  runs=[dict(variant=v["id"], iterations=256, warmup=64) for v in variants])
    speed.save(output / "smoke-schedule.json", smoke)
    speed.save(output / "screen-schedule.json", screen)
    print(json.dumps(dict(plan=speed.source(output / "plan.json"),
                         smoke_schedule=speed.source(output / "smoke-schedule.json"),
                         screen_schedule=speed.source(output / "screen-schedule.json"), variants=len(variants))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    prepare(parser.parse_args().output.resolve())
