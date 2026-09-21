// Appended only to an isolated source copy by probe_local.py.
#[cfg(test)]
mod move_latency_probe {
    use super::*;
    use std::time::Instant;

    fn numbers(name: &str, fallback: &str) -> Vec<usize> {
        std::env::var(name).unwrap_or(fallback.into()).split(',')
            .map(|n| n.parse().unwrap()).collect()
    }

    fn node(id: usize) -> Box<SearchNode> {
        let mut n = Box::new(SearchNode::new(P_BLACK, false,
            (id % 65536) as u32, Hash128::new(id as u64, 123)));
        n.initialize_children();
        n.state.store(STATE_EXPANDED0, Ordering::Relaxed);
        n.stats.visits.store(1, Ordering::Relaxed);
        let mut output = NNOutput::default();
        output.white_owner_map = Some(vec![0.125; 361].into_boxed_slice());
        n.nn_output.store(Box::into_raw(Box::new(Arc::new(output))), Ordering::Relaxed);
        n
    }

    fn add(s: &Search, n: Box<SearchNode>) -> *mut SearchNode {
        let key = n.graph_hash;
        let ptr = Box::into_raw(n);
        let table = s.node_table.as_ref().unwrap();
        table.entries[table.get_index_for_hash(key) as usize].lock().insert(key, ptr);
        ptr
    }

    fn fixture(count: usize, keep: usize, threads: usize) -> Search<'static> {
        let mut s = super::tests::search_with_dummy();
        s.search_params.num_threads = threads as i32;
        let cut = count - count * keep / 100;
        let pointers: Vec<_> = (0..count).map(|i| add(&s, node(i))).collect();
        for i in 0..count {
            let start = if i < cut { 0 } else { cut };
            let end = if i < cut { cut } else { count };
            let n = unsafe { &*pointers[i] };
            let mut edge = 0;
            for k in 1..=4 {
                let c = start + (i - start) * 4 + k;
                if c < end {
                    n.get_children().get(edge).store(pointers[c]);
                    edge += 1;
                }
            }
            // Shared descendants (DAG). Avoid cycles here because the original
            // eval-cache traversal has no cycle guard.
            if i + 2 < end && edge == 0 && i % 3 == 0 {
                n.get_children().get(0).store(pointers[i + 2]);
            }
        }
        s.root_node = Some(Box::new(SearchNode::clone_for_tree(
            unsafe { &*pointers[cut] }, false, false)));
        s
    }

    fn mark_once(s: &mut Search) {
        s.search_node_age += 1;
        let mut pending = vec![s.root_node.as_deref().unwrap() as *const SearchNode];
        while let Some(ptr) = pending.pop() {
            let n = unsafe { &*ptr };
            if n.node_age.swap(s.search_node_age, Ordering::AcqRel) == s.search_node_age { continue; }
            for i in 0..n.get_children().get_capacity() {
                if let Some(child) = n.get_children().get(i).get_if_allocated() {
                    pending.push(child as *const SearchNode);
                } else { break; }
            }
        }
    }

    fn parallel_sweep(s: &Search) {
        // Scoped workers exclusively partition shards after traversal ends.
        // Node destructors do not recursively dereference child pointers.
        let table_address = &**s.node_table.as_ref().unwrap() as *const SearchNodeTable<SearchNode> as usize;
        let threads = s.search_params.num_threads.max(1) as usize;
        let age = s.search_node_age;
        let free_prop = s.search_params.subtree_value_bias_free_prop;
        s.perform_task_with_threads(&|thread| {
            let table = unsafe { &*(table_address as *const SearchNodeTable<SearchNode>) };
            let first = thread as usize * table.entries.len() / threads;
            let last = (thread as usize + 1) * table.entries.len() / threads;
            for shard in &table.entries[first..last] {
                shard.lock().retain(|_, ptr| {
                    if ptr.is_null() { return true; }
                    let node = unsafe { &**ptr };
                    if node.node_age.load(Ordering::Acquire) >= age { return true; }
                    if let Some(entry) = &node.subtree_value_bias_table_entry {
                        entry.update(-node.last_subtree_value_bias_delta_sum * free_prop,
                                     -node.last_subtree_value_bias_weight * free_prop);
                    }
                    unsafe { drop(Box::from_raw(*ptr)); }
                    false
                });
            }
        }, i32::MAX);
    }

    fn record_once(s: &Search) {
        let mut seen = HashSet::new();
        let mut pending = vec![s.root_node.as_deref().unwrap() as *const SearchNode];
        let root = pending[0];
        while let Some(ptr) = pending.pop() {
            if !seen.insert(ptr) { continue; }
            let n = unsafe { &*ptr };
            for i in 0..n.get_children().get_capacity() {
                if let Some(child) = n.get_children().get(i).get_if_allocated() {
                    pending.push(child as *const SearchNode);
                } else { break; }
            }
            let min = s.search_params.eval_cache_min_visits;
            if n.stats.visits.load(Ordering::Acquire) >= min && !n.force_non_terminal {
                s.eval_cache.as_ref().unwrap().update(s.get_eval_cache_key(n.graph_hash), n, min, ptr == root);
            }
        }
    }

    #[test]
    #[ignore = "explicit lifecycle benchmark"]
    fn local_abba() {
        let threads = numbers("MOVE_PROBE_THREADS", "8")[0];
        for count in numbers("MOVE_PROBE_NODES", "10000,100000,300000") {
            for keep in numbers("MOVE_PROBE_KEEP", "10,90") {
                let mut expected = None;
                for round in 0..numbers("MOVE_PROBE_ROUNDS", "1")[0] {
                    for (order, candidate) in [false, true, true, false].into_iter().enumerate() {
                        let mut s = fixture(count, keep, threads);
                        let root = s.root_node.as_deref().unwrap() as *const SearchNode;
                        let started = Instant::now();
                        if candidate { mark_once(&mut s); }
                        else { s.apply_recursively_any_order_multithreaded(&[unsafe { &*root }], &|_, _| {}); }
                        let mark = started.elapsed().as_secs_f64() * 1000.0;
                        let started = Instant::now();
                        if candidate { parallel_sweep(&s); }
                        else { s.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_multithreaded(true); }
                        let sweep = started.elapsed().as_secs_f64() * 1000.0;
                        let mut keys = Vec::new();
                        for shard in &s.node_table.as_ref().unwrap().entries {
                            for (key, ptr) in shard.lock().iter() {
                                keys.push(key.hash0);
                                let n = unsafe { &**ptr };
                                for i in 0..n.get_children().get_capacity() {
                                    if let Some(child) = n.get_children().get(i).get_if_allocated() {
                                        assert_eq!(child.node_age.load(Ordering::Acquire), s.search_node_age);
                                    } else { break; }
                                }
                            }
                        }
                        keys.sort_unstable();
                        if let Some(ref e) = expected { assert_eq!(&keys, e); }
                        else { expected = Some(keys.clone()); }
                        println!("MOVE_PROBE {}", serde_json::json!({"kind":"reuse", "nodes":count,
                            "keep_pct":keep,"threads":threads,"round":round,"order":order,
                            "variant":if candidate {"candidate"} else {"baseline"},
                            "mark_ms":mark,"sweep_ms":sweep,"total_ms":mark+sweep,"remaining":keys.len()}));
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "explicit graph-sharing stress test"]
    fn shared_cache() {
        for depth in [10, 14, 18, 20, 24, 28] {
            let mut s = super::tests::search_with_dummy();
            s.eval_cache = Some(Arc::new(EvalCacheTable::new(16)));
            s.search_params.eval_cache_min_visits = 100;
            s.root_node = Some(node(10000));
            let layers: Vec<_> = (0..depth).map(|i| [add(&s, node(i*2)), add(&s, node(i*2+1))]).collect();
            let root = s.root_node.as_deref_mut().unwrap() as *mut SearchNode;
            let mut parents = vec![root];
            for layer in &layers {
                for p in parents {
                    for (edge, child) in layer.iter().enumerate() {
                        unsafe { &*p }.get_children().get(edge).store(*child);
                    }
                }
                parents = layer.to_vec();
            }
            for (order, candidate) in [false, true, true, false].into_iter().enumerate() {
                let started = Instant::now();
                if candidate { record_once(&s); }
                else { s.recursively_record_eval_cache(unsafe { &mut *root }); }
                println!("MOVE_PROBE {}", serde_json::json!({"kind":"shared_cache",
                    "depth":depth,"unique_nodes":1+depth*2,"baseline_recursive_calls":(1u64<<(depth+1))-1,
                    "order":order,"variant":if candidate {"dedup"} else {"baseline"},
                    "elapsed_ms":started.elapsed().as_secs_f64()*1000.0}));
            }
        }
    }

    #[test]
    #[ignore = "explicit retirement timing; no background worker simulated"]
    fn retirement_abba() {
        for count in [100000, 300000] {
            for keep in [10, 90] {
                let mut expected_keys = None;
                for (order, deferred) in [false, true, true, false].into_iter().enumerate() {
                    let mut s = fixture(count, keep, 8);
                    let root = s.root_node.as_deref().unwrap() as *const SearchNode;
                    let started = Instant::now();
                    s.apply_recursively_any_order_multithreaded(&[unsafe { &*root }], &|_, _| {});
                    let mark_ms = started.elapsed().as_secs_f64() * 1000.0;
                    let mut retired = Vec::new();
                    let started = Instant::now();
                    if deferred {
                        for shard in &s.node_table.as_ref().unwrap().entries {
                            shard.lock().retain(|_, ptr| {
                                let n = unsafe { &**ptr };
                                if n.node_age.load(Ordering::Acquire) >= s.search_node_age { return true; }
                                s.remove_subtree_value_bias(Some(n));
                                retired.push(unsafe { Box::from_raw(*ptr) });
                                false
                            });
                        }
                    } else {
                        s.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_multithreaded(true);
                    }
                    let foreground_ms = started.elapsed().as_secs_f64() * 1000.0 + mark_ms;
                    let retired_count = retired.len();
                    let started = Instant::now();
                    drop(retired);
                    let destruction_ms = started.elapsed().as_secs_f64() * 1000.0;
                    let mut keys = Vec::new();
                    let mut live = HashSet::new();
                    for shard in &s.node_table.as_ref().unwrap().entries {
                        for (key, ptr) in shard.lock().iter() {
                            keys.push(key.hash0);
                            live.insert(*ptr as usize);
                        }
                    }
                    keys.sort_unstable();
                    if let Some(ref expected) = expected_keys { assert_eq!(&keys, expected); }
                    else { expected_keys = Some(keys); }
                    live.insert(root as usize);
                    for address in &live {
                        let n = unsafe { &*(*address as *const SearchNode) };
                        assert_eq!(n.stats.visits.load(Ordering::Acquire), 1);
                        for i in 0..n.get_children().get_capacity() {
                            let child = n.get_children().get(i).get_raw_ptr();
                            if child.is_null() { break; }
                            assert!(live.contains(&(child as usize)), "retained edge must remain live");
                        }
                    }
                    println!("MOVE_PROBE {}", serde_json::json!({"kind":"retirement",
                        "nodes":count,"keep_pct":keep,"order":order,
                        "variant":if deferred {"deferred_drop"} else {"baseline"},
                        "foreground_ms":foreground_ms,"mark_ms":mark_ms,
                        "destruction_ms":destruction_ms,"retired":retired_count,
                        "retained_keys_and_edges_verified":true}));
                }
            }
        }
    }

    #[test]
    #[ignore = "actual CPU dummy search lifecycle; no strength claim"]
    fn actual_dummy_search() {
        let mut s = super::tests::search_with_dummy();
        s.search_params.num_threads = 8;
        s.search_params.max_visits = 50000;
        s.search_params.max_playouts = 50000;
        s.search_params.use_graph_search = true;
        s.search_params.use_eval_cache = false;
        // These two differ from SearchParams::new(): use the CLI setup defaults.
        s.search_params.dynamic_score_utility_factor = 0.3;
        s.search_params.subtree_value_bias_factor = 0.45;
        s.search_params.subtree_value_bias_free_prop = 0.8;
        s.search_params.subtree_value_bias_weight_exponent = 0.85;
        s.subtree_value_bias_table = Some(Box::new(SubtreeValueBiasTable::new(
            s.search_params.subtree_value_bias_table_num_shards)));
        let board = Board::new(19, 19);
        let history = BoardHistory::new(board.clone(), P_BLACK, Rules::default(), 0);
        s.set_position(P_BLACK, &board, &history);
        s.run_whole_search(P_BLACK);
        let count = s.node_table.as_ref().unwrap().entries.iter().map(|m|m.lock().len()).sum::<usize>();
        s.eval_cache = Some(Arc::new(EvalCacheTable::new(65536)));
        s.search_params.eval_cache_min_visits = 100;
        let root = s.root_node.as_deref_mut().unwrap() as *mut SearchNode;
        let started = Instant::now();
        s.recursively_record_eval_cache(unsafe { &mut *root });
        let cache_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        s.begin_search(false);
        let begin_same_root_ms = started.elapsed().as_secs_f64() * 1000.0;
        let chosen = s.get_chosen_move_loc();
        let started = Instant::now();
        assert!(s.make_move(chosen, P_BLACK));
        let make_move_ms = started.elapsed().as_secs_f64() * 1000.0;
        let kept = s.node_table.as_ref().unwrap().entries.iter().map(|m|m.lock().len()).sum::<usize>();
        let started = Instant::now();
        s.begin_search(false);
        let begin_ms = started.elapsed().as_secs_f64() * 1000.0;
        println!("MOVE_PROBE {}", serde_json::json!({"kind":"actual_dummy_search",
            "nodes":count,"kept":kept,"cache_ms":cache_ms,"make_move_ms":make_move_ms,
            "begin_search_ms":begin_ms,"begin_same_root_ms":begin_same_root_ms,"move":chosen,
            "dynamic_score_utility_factor":0.3,"subtree_value_bias_factor":0.45}));
    }
}
