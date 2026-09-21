// Included only in an isolated COPY of go-core by probe_server.py.
#[cfg(test)]
mod move_latency_probe {
    use super::*;
    use std::time::Instant;

    fn fixture(count: usize, keep_pct: usize, slots: usize) -> Search {
        let position = Position::new(19, 7.5).unwrap();
        let mut s = Search::new(position.clone(), SearchConfig::default()).unwrap();
        s.nodes.clear();
        s.index.clear();
        s.root = count - count * keep_pct / 100;
        let template = Node::new(&position);
        for i in 0..count {
            let mut n = Node::new(&position);
            n.key = template.key;
            n.key[..8].copy_from_slice(&(i as u64).to_le_bytes());
            n.state = State::Expanded;
            n.stats.visits = (count - i) as u64;
            n.stats.weight_sum = n.stats.visits as f64;
            n.raw = Some(n.stats);
            n.ownership = vec![0.125; 361];
            n.edges = (0..slots).map(|j| Edge {
                point: Some(j as u16), prior: 1.0 / slots as f64,
                child: None, visits: 0, in_flight: 0,
            }).collect();
            s.index.insert(n.key, i);
            s.nodes.push(n);
        }
        // Two components separated at the new root. Add transpositions and
        // cycles to exercise graph semantics, with stale parents across cut.
        for i in 0..count {
            let start = if i < s.root { 0 } else { s.root };
            let end = if i < s.root { s.root } else { count };
            let mut children = Vec::new();
            for k in 1..=4 {
                let child = start + (i - start) * 4 + k;
                if child < end { children.push(child); }
            }
            if i + 2 < end { children.push(i + 2); }
            if i % 97 == 0 && i > start { children.push(start); }
            if i + 1 == s.root { children.push(s.root); }
            for (e, c) in children.into_iter().enumerate() {
                s.nodes[i].set_child(e, c);
                s.nodes[i].edges[e].visits = 1;
                s.nodes[c].parents.insert(i);
            }
        }
        s
    }

    // Candidate keeps compact storage and original node order, but replaces
    // hash lookups for dense node IDs with indexed arrays. This is still O(N+E).
    fn dense_collect(s: &mut Search) {
        let mut reachable = vec![false; s.nodes.len()];
        let mut stack = vec![s.root];
        reachable[s.root] = true;
        while let Some(n) = stack.pop() {
            for c in s.nodes[n].edges.iter().filter_map(|e| e.child) {
                if !reachable[c] {
                    reachable[c] = true;
                    stack.push(c);
                }
            }
        }
        let kept = reachable.iter().filter(|&&r| r).count();
        let mut remap = vec![usize::MAX; s.nodes.len()];
        let mut nodes = Vec::with_capacity(kept);
        for (old, n) in std::mem::take(&mut s.nodes).into_iter().enumerate() {
            if reachable[old] {
                remap[old] = nodes.len();
                nodes.push(n);
            }
        }
        for n in &mut nodes {
            for e in &mut n.edges {
                e.child = e.child.and_then(|c| (remap[c] != usize::MAX).then_some(remap[c]));
            }
            #[cfg(feature = "sparse-child-iteration")]
            { n.linked_edges = n.edges.iter().enumerate()
                .filter_map(|(i, e)| e.child.map(|_| i as u16)).collect(); }
            n.parents.clear();
        }
        for n in 0..nodes.len() {
            let children: Vec<_> = nodes[n].edges.iter().filter_map(|e| e.child).collect();
            for c in children { nodes[c].parents.insert(n); }
        }
        s.root = remap[s.root];
        s.index = nodes.iter().enumerate().map(|(i, n)| (n.key, i)).collect();
        s.nodes = nodes;
    }

    fn signature(s: &Search) -> String {
        let mut h = Sha256::new();
        h.update(s.root.to_le_bytes());
        h.update(s.nodes.len().to_le_bytes());
        assert_eq!(s.nodes.len(), s.index.len());
        for (i, n) in s.nodes.iter().enumerate() {
            assert_eq!(s.index[&n.key], i);
            h.update(n.key);
            h.update(serde_json::to_vec(&n.stats).unwrap());
            h.update(serde_json::to_vec(&n.raw).unwrap());
            h.update(format!("{:?}", n.state));
            for x in &n.ownership { h.update(x.to_bits().to_le_bytes()); }
            for e in &n.edges {
                h.update(e.point.unwrap_or(u16::MAX).to_le_bytes());
                h.update(e.prior.to_bits().to_le_bytes());
                h.update(e.visits.to_le_bytes());
                h.update(e.child.unwrap_or(usize::MAX).to_le_bytes());
                h.update(e.in_flight.to_le_bytes());
            }
            let mut parents: Vec<_> = n.parents.iter().copied().collect();
            parents.sort_unstable();
            h.update(parents.len().to_le_bytes());
            for p in parents {
                assert!(s.nodes[p].edges.iter().any(|e| e.child == Some(i)));
                h.update(p.to_le_bytes());
            }
        }
        format!("{:x}", h.finalize())
    }

    fn numbers(name: &str, fallback: &str) -> Vec<usize> {
        std::env::var(name).unwrap_or(fallback.into()).split(',')
            .map(|n| n.parse().unwrap()).collect()
    }

    #[test]
    #[ignore = "explicit offline CPU benchmark"]
    fn abba_collection() {
        let slots = numbers("MOVE_PROBE_EDGES", "256")[0];
        let rounds = numbers("MOVE_PROBE_ROUNDS", "2")[0];
        assert!((8..=362).contains(&slots));
        for count in numbers("MOVE_PROBE_NODES", "10000,100000,300000") {
            for keep in numbers("MOVE_PROBE_KEEP", "10,50,90") {
                assert!((1..=99).contains(&keep));
                let mut reference = None;
                for round in 0..rounds {
                    for (order, candidate) in [false, true, true, false].into_iter().enumerate() {
                        let mut s = fixture(count, keep, slots);
                        let started = Instant::now();
                        if candidate { dense_collect(&mut s); } else { s.collect_unreachable(); }
                        let ms = started.elapsed().as_secs_f64() * 1000.0;
                        let sig = signature(&s);
                        if let Some(ref expected) = reference { assert_eq!(&sig, expected); }
                        else { reference = Some(sig.clone()); }
                        println!("MOVE_PROBE {}", serde_json::json!({
                            "nodes":count,"keep_pct":keep,"edge_slots":slots,
                            "round":round,"order":order,"variant":if candidate {"dense"} else {"baseline"},
                            "elapsed_ms":ms,"remaining":s.nodes.len(),"signature":sig
                        }));
                    }
                }
            }
        }
    }
}
