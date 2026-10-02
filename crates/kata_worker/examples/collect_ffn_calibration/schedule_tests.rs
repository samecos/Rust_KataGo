use super::*;
use std::path::PathBuf;

fn input(target: usize, physical: usize, n: usize) -> InputDescriptor {
    InputDescriptor {
        id: format!("input-{n}"),
        manifest_path: PathBuf::from("inputs.json"),
        manifest_sha256: "a".repeat(64),
        target_batch: target,
        physical_batch: physical,
        tensor_sha256: format!("{n:064x}"),
        rows: Vec::new(),
    }
}
fn plan() -> Plan {
    let source = Source {
        path: PathBuf::from("unused"),
        bytes: 0,
        sha256: "a".repeat(64),
    };
    Plan {
        schema: "rustgo-ffn-calibration-collection-plan-v1".into(),
        mode: "bounded-diagnostic".into(),
        model: source.clone(),
        binary: true,
        compressed: true,
        graph_sha256: "a".repeat(64),
        proposal: source.clone(),
        proposal_object_sha256: "a".repeat(64),
        recipe: source.clone(),
        resolved_recipe_sha256: "a".repeat(64),
        group_ids: vec![],
        provenance: source,
        numeric_input_indices: vec![0, 1],
        cost: vec![CostBatch {
            physical_batch: 3,
            input_indices: vec![2, 3],
            warmup: 3,
            measurements: 4,
        }],
        expected_forwards: 17,
        expected_numeric_rows: 2,
        chunk_inputs: 128,
    }
}
#[test]
fn one_warm_per_batch_and_durable_call_order_budget() {
    let inputs = vec![
        input(1, 1, 0),
        input(1, 1, 1),
        input(3, 3, 2),
        input(3, 3, 3),
    ];
    let p = plan();
    let calls = p.validate(&inputs, "complete-calibration", 2044).unwrap();
    assert_eq!(calls.len(), 17);
    assert_eq!(
        calls
            .iter()
            .filter(|c| c.phase == "cost-warmup")
            .map(|c| c.input_index)
            .collect::<Vec<_>>(),
        vec![2, 2, 2]
    );
    assert_eq!(calls[5].phase, "cost-before");
    assert_eq!(calls[10].phase, "cost-after");
    assert_eq!(calls[11].input_index, 3);
    assert_eq!(calls[11].phase, "cost-before");
    let mut wrong = plan();
    wrong.expected_forwards += 3;
    assert!(
        wrong
            .validate(&inputs, "complete-calibration", 2044)
            .is_err()
    );
}
#[test]
fn duplicate_numeric_or_cost_and_tail_batch_rejected() {
    let inputs = vec![
        input(1, 1, 0),
        input(1, 1, 1),
        input(3, 3, 2),
        input(3, 3, 3),
    ];
    let mut p = plan();
    p.numeric_input_indices[1] = 0;
    assert!(p.validate(&inputs, "complete-calibration", 2044).is_err());
    let mut p = plan();
    p.cost[0].input_indices[1] = 2;
    assert!(p.validate(&inputs, "complete-calibration", 2044).is_err());
    let mut p = plan();
    p.cost[0].physical_batch = 1;
    p.cost[0].input_indices = vec![2];
    let mut inputs = inputs;
    inputs[2] = input(3, 1, 2);
    assert!(p.validate(&inputs, "complete-calibration", 2044).is_err());
}
#[test]
fn complete_means_exact_ordered_b1_not_equal_row_count_or_prefix() {
    let inputs: Vec<_> = (0..2044).map(|n| input(1, 1, n)).collect();
    let mut p = plan();
    p.mode = "complete-calibration".into();
    p.cost.clear();
    p.numeric_input_indices = (0..2044).collect();
    p.expected_forwards = 2044;
    p.expected_numeric_rows = 2044;
    assert!(p.validate(&inputs, "complete-calibration", 2044).is_ok());
    assert!(p.validate(&inputs, "diagnostic-prefix", 2044).is_err());
    p.numeric_input_indices.swap(0, 1);
    assert!(p.validate(&inputs, "complete-calibration", 2044).is_err());
    p.numeric_input_indices = (0..2043).collect();
    p.expected_forwards = 2043;
    p.expected_numeric_rows = 2043;
    assert!(p.validate(&inputs, "complete-calibration", 2044).is_err());
}
#[test]
fn callback_cannot_change_tensor_phase_batch_or_iteration() {
    let c = call(0, &input(3, 3, 0), "cost-measure", 2);
    let good = serde_json::json!({"phase":"cost-measure","physical_batch":3,"iteration":2,"tensor_sha256":c.tensor_sha256});
    assert!(crate::check_callback(&c, &good).is_ok());
    for (key, bad) in [
        ("phase", serde_json::json!("numeric")),
        ("iteration", serde_json::json!(3)),
        ("physical_batch", serde_json::json!(1)),
        ("tensor_sha256", serde_json::json!("b".repeat(64))),
    ] {
        let mut changed = good.clone();
        changed[key] = bad;
        assert!(crate::check_callback(&c, &changed).is_err());
    }
}
