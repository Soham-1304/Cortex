use std::collections::{BTreeMap, HashSet};

pub const LN2: f64 = std::f64::consts::LN_2;

/// Exponential recency decay. lambda = LN2 / half_life_secs.
pub fn decay(weight: f64, dt_secs: f64, lambda: f64) -> f64 {
    weight * (-lambda * dt_secs).exp()
}

pub fn reinforce(weight: f64, bump: f64) -> f64 {
    weight + bump
}

/// Maps an unbounded raw count (e.g. co-mention count) to (0, 1); k = 1.0 is a
/// typical softener. Use this when building adjacency lists for [`spread`].
pub fn normalize_edge_weight(raw: f64, k: f64) -> f64 {
    raw / (raw + k)
}

/// Spreading activation: BFS from seeds over adjacency list `adj`
/// (node -> [(neighbor, base_w)]).
///
/// Contract:
/// - Edge weights MUST be normalized multiplicative factors in (0, 1]; use
///   [`normalize_edge_weight`] when building the adjacency list.
/// - Adjacency indices must be dense 0..n (remap SQLite rowids before calling).
/// - Traversal follows out-edges only; build symmetric adjacency for undirected
///   relations.
/// - Activation multiplies by base_w each hop and accumulates additively when
///   paths converge; activations below `eps` are pruned. Termination is
///   guaranteed by `max_hops` regardless of cycles in the graph.
/// - Seeds are excluded from the result.
///
/// Returns node -> accumulated activation, ordered deterministically for
/// downstream tie-breaking.
pub fn spread(
    adj: &[Vec<(usize, f64)>],
    seeds: &[(usize, f64)],
    max_hops: usize,
    eps: f64,
) -> BTreeMap<usize, f64> {
    let mut act: BTreeMap<usize, f64> = BTreeMap::new();
    let seed_ids: HashSet<usize> = seeds.iter().map(|&(s, _)| s).collect();
    let mut frontier: Vec<(usize, f64)> = seeds.to_vec();
    for _ in 0..max_hops {
        let mut next: Vec<(usize, f64)> = Vec::new();
        for (node, a) in frontier.drain(..) {
            for &(nb, w) in &adj[node] {
                if seed_ids.contains(&nb) { continue; }
                let na = a * w;
                if na < eps { continue; }
                *act.entry(nb).or_insert(0.0) += na;
                next.push((nb, na));
            }
        }
        if next.is_empty() { break; }
        frontier = next;
    }
    act
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn half_life_exact() {
        let lam = LN2 / (14.0 * 86400.0);
        assert!((decay(1.0, 14.0 * 86400.0, lam) - 0.5).abs() < 1e-9);
        assert!((decay(1.0, 0.0, lam) - 1.0).abs() < 1e-9);
    }
    #[test]
    fn reinforcement_linear_accumulation() {
        let mut w = 1.0;
        for _ in 0..10 { w = reinforce(w, 1.0); }
        assert!((w - 11.0).abs() < 1e-9);
    }
    #[test]
    fn normalize_edge_weight_maps_counts_to_unit_interval() {
        assert!((normalize_edge_weight(0.0, 1.0) - 0.0).abs() < 1e-9);
        assert!((normalize_edge_weight(1.0, 1.0) - 0.5).abs() < 1e-9);
        let mut prev = -1.0;
        for raw in [0.1f64, 0.5, 1.0, 2.0, 10.0, 1000.0] {
            let n = normalize_edge_weight(raw, 1.0);
            assert!(n > prev && n > 0.0 && n < 1.0);
            prev = n;
        }
        assert!(normalize_edge_weight(1e12, 1.0) < 1.0);
    }
    #[test]
    fn spreading_activation_two_hops_exact_values() {
        let adj = vec![vec![(1usize, 0.9)], vec![(0usize, 0.9), (2usize, 0.4)], vec![(1usize, 0.4)]];
        let got = spread(&adj, &[(0, 1.0)], 2, 0.1);
        assert!((got[&1] - 0.9).abs() < 1e-9);
        assert!((got[&2] - 0.36).abs() < 1e-9);
        assert!(!got.contains_key(&0));
    }
    #[test]
    fn spread_deterministic_and_prunes() {
        let adj = vec![vec![(1usize, 0.01)]];
        assert!(spread(&adj, &[(0, 1.0)], 3, 1.0).is_empty());
    }
    #[test]
    fn cycle_terminates_and_stays_bounded() {
        let adj = vec![vec![(1usize, 0.5)], vec![(0usize, 0.5)]];
        let got = spread(&adj, &[(0, 1.0)], 8, 1e-6);
        let v = got[&1];
        assert!(v > 0.0 && v <= 1.0);
    }
    #[test]
    fn converging_paths_accumulate_additively() {
        // diamond: 0->1->3 contributes 0.25, 0->2->3 contributes 0.40
        let adj = vec![
            vec![(1usize, 0.5), (2usize, 0.8)],
            vec![(3usize, 0.5)],
            vec![(3usize, 0.5)],
            vec![],
        ];
        let got = spread(&adj, &[(0, 1.0)], 2, 1e-6);
        assert!((got[&3] - 0.65).abs() < 1e-9);
    }
    #[test]
    fn repeated_runs_identical_maps() {
        let adj = vec![vec![(1usize, 0.7)], vec![(2usize, 0.6)], vec![]];
        let a: BTreeMap<usize, f64> = spread(&adj, &[(0, 1.0)], 2, 1e-6);
        let b: BTreeMap<usize, f64> = spread(&adj, &[(0, 1.0)], 2, 1e-6);
        assert_eq!(a, b);
    }
}
