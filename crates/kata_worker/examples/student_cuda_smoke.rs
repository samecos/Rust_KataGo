//! Bounded CUDA compatibility check through the original Worker evaluator.
//! No server is started or contacted. Uses real model inference, not a dummy.
use kata_core::config::ConfigParser;
use kata_worker::{Evaluator, evaluator::Engine, wire};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    anyhow::ensure!(
        args.len() == 3,
        "usage: student_cuda_smoke <model> <result.json>"
    );
    let cfg = ConfigParser::from_str(
        "nnBackend=cudabackend\ninputsUseNHWC=false\nrequireMaxBoardSize=true\nnumNNServerThreadsPerModel=1\nnnMaxBatchSize=8\nnnCacheSizePowerOfTwo=10\nnnMutexPoolSizePowerOfTwo=8\n",
        false,
        false,
    )?;
    let engine = Engine::load(&args[1], None, &cfg, 8, false)?;
    let meta = engine.metadata();
    anyhow::ensure!(meta.model_version == 8 && !meta.supports_shortterm_error);
    anyhow::ensure!(meta.execution_profile_id.starts_with("rustgo-student-v1:"));
    let mut reports = Vec::new();
    for next in [1, 2] {
        for sym in 0..8 {
            let moves = if next == 1 {
                vec![]
            } else {
                vec![wire::Move {
                    color: 1,
                    vertex: 0,
                }]
            };
            let request = wire::EvalRequest {
                task_id: (next * 8 + sym) as u64,
                generation: 1,
                session_id: "student-smoke".into(),
                model_sha256: meta.model_sha256.clone(),
                position: Some(wire::Position {
                    board_size: 19,
                    komi: 7.5,
                    rules: "chinese".into(),
                    initial_player: 1,
                    next_player: next,
                    moves,
                    ..Default::default()
                }),
                parameters: Some(wire::EvalParameters {
                    symmetry: sym,
                    policy_temperature: 1.0,
                    policy_optimism: 0.75,
                    draw_equivalent_wins_for_white: 0.5,
                    include_ownership: true,
                    max_history: 1000,
                    skip_cache: true,
                    ..Default::default()
                }),
                ..Default::default()
            };
            let out = engine
                .evaluate(&request)
                .result
                .map_err(|e| anyhow::anyhow!("{}: {}", e.code, e.message))?;
            anyhow::ensure!(out.policy.len() == 362 && out.ownership.len() == 361);
            anyhow::ensure!(!out.has_shortterm_error && out.var_time_left == -1.0);
            anyhow::ensure!(
                out.policy
                    .iter()
                    .chain(out.ownership.iter())
                    .all(|v| v.is_finite())
            );
            let sum: f32 = out.policy.iter().filter(|v| **v >= 0.0).sum();
            anyhow::ensure!((sum - 1.0).abs() < 1e-4);
            anyhow::ensure!(
                (out.white_win_prob + out.white_loss_prob + out.white_no_result_prob - 1.0).abs()
                    < 1e-6
            );
            anyhow::ensure!(out.ownership.iter().all(|v| (-1.0..=1.0).contains(v)));
            anyhow::ensure!((out.white_lead - out.white_score_mean).abs() < 1e-5);
            // The standard decoder assigns a zero score to no-result games.
            // Their mixture adds variance even with no learned stdev head.
            anyhow::ensure!(
                (out.white_score_mean_sq * (1.0 - out.white_no_result_prob)
                    - out.white_score_mean.powi(2))
                .abs()
                    < 1e-3 + 2e-5 * out.white_score_mean_sq.abs()
            );
            if next == 2 {
                anyhow::ensure!(out.policy[0] < 0.0);
            }
            reports.push(serde_json::json!({"next_player":next,"symmetry":sym,"policy_sum":sum,
                "white_score_mean":out.white_score_mean,"has_shortterm_error":out.has_shortterm_error}));
        }
    }
    let (rows, batches) = engine.stats();
    let result = serde_json::json!({"status":"PASS","model_sha256":meta.model_sha256,
        "execution_profile_id":meta.execution_profile_id,"model_version":meta.model_version,
        "requests":reports,"nn_rows":rows,"nn_batches":batches,"server_started":false});
    std::fs::write(&args[2], serde_json::to_vec_pretty(&result)?)?;
    println!("PASS: 16 real CUDA Worker evaluations, all symmetries, both players");
    Ok(())
}
