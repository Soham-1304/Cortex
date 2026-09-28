use std::collections::{HashMap, HashSet};

pub fn bcy_at_budget(ranked: &[String], costs: &[i64], gold: &[String], budget: i64) -> f64 {
    assert_eq!(ranked.len(), costs.len(), "ranked/cost length mismatch");
    if gold.is_empty() {
        return 0.0;
    }
    let g: HashSet<&String> = gold.iter().collect();
    let total_of: HashMap<&String, i64> =
        ranked.iter().zip(costs.iter()).map(|(id, c)| (id, *c)).collect();
    let mut remaining = budget.max(0);
    let mut exposed: HashMap<&String, i64> = HashMap::new();
    for (id, c) in ranked.iter().zip(costs.iter()) {
        if remaining <= 0 {
            break;
        }
        let take = (*c).min(remaining);
        if g.contains(id) {
            *exposed.entry(id).or_insert(0) += take;
        }
        remaining -= take;
    }
    let mut sum = 0.0;
    for gid in gold {
        let l = total_of.get(gid).copied().unwrap_or(0);
        if l <= 0 {
            continue;
        }
        let e = exposed.get(gid).copied().unwrap_or(0).min(l);
        sum += e as f64 / l as f64;
    }
    sum / gold.len() as f64
}

pub fn recall_curve(ranked: &[String], gold: &[String], k_max: usize) -> Vec<f64> {
    (1..=k_max).map(|k| recall_at_k(ranked, gold, k)).collect()
}

pub fn knee_index(curve: &[f64]) -> usize {
    let n = curve.len();
    if n < 3 {
        return n.saturating_sub(1);
    }
    let (y0, yn) = (curve[0], curve[n - 1]);
    let mut best = 1;
    let mut best_d = -1.0f64;
    for (i, &y) in curve.iter().enumerate().skip(1).take(n - 2) {
        let t = i as f64 / (n - 1) as f64;
        let d = y - (y0 + t * (yn - y0));
        if d > best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

pub fn precision_at_k(ranked: &[String], gold: &[String], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let g: HashSet<&String> = gold.iter().collect();
    ranked.iter().take(k).filter(|d| g.contains(d)).count() as f64 / k as f64
}

pub fn recall_at_k(ranked: &[String], gold: &[String], k: usize) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    let g: HashSet<&String> = gold.iter().collect();
    ranked.iter().take(k).filter(|d| g.contains(d)).count() as f64 / gold.len() as f64
}

/// Orientation check (orientation-first objective): true when the top-k
/// window holds BOTH a gold span (the definition) and a gold file (the
/// neighborhood) — the minimum for an agent to get oriented. Callers are
/// not covered (no call data in bench corpora); documented limitation.
pub fn oriented_at_k(
    ranked: &[String],
    gold: &[String],
    id_to_path: &HashMap<String, String>,
    gold_files: &HashSet<String>,
    k: usize,
) -> bool {
    if gold.is_empty() || gold_files.is_empty() {
        return false;
    }
    let g: HashSet<&String> = gold.iter().collect();
    let mut span = false;
    let mut file = false;
    for id in ranked.iter().take(k) {
        if g.contains(id) {
            span = true;
        }
        if let Some(p) = id_to_path.get(id) {
            if gold_files.contains(p) {
                file = true;
            }
        }
        if span && file {
            return true;
        }
    }
    false
}

/// File-level recall: fraction of gold *files* with at least one symbol doc
/// in the top-k ranks. The file-first accuracy metric: a query whose gold
/// file is found can be completed by within-file search, even when the exact
/// gold span is buried. `id_to_path` maps doc id -> file path.
pub fn file_recall_at_k(
    ranked: &[String],
    id_to_path: &HashMap<String, String>,
    gold_files: &HashSet<String>,
    k: usize,
) -> f64 {
    if gold_files.is_empty() {
        return 0.0;
    }
    let mut hit = HashSet::new();
    for id in ranked.iter().take(k) {
        if let Some(p) = id_to_path.get(id) {
            if gold_files.contains(p) {
                hit.insert(p.clone());
            }
        }
    }
    hit.len() as f64 / gold_files.len() as f64
}

pub fn mrr(ranked: &[String], gold: &[String]) -> f64 {
    let g: HashSet<&String> = gold.iter().collect();
    for (i, d) in ranked.iter().enumerate() {
        if g.contains(d) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

pub fn ndcg_at_k(ranked: &[String], gold: &[String], k: usize) -> f64 {
    let g: HashSet<&String> = gold.iter().collect();
    let dcg: f64 = ranked
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, d)| if g.contains(d) { 1.0 / ((i + 2) as f64).log2() } else { 0.0 })
        .sum();
    let hits = g.len().min(k);
    let idcg: f64 = (0..hits).map(|i| 1.0 / ((i + 2) as f64).log2()).sum();
    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

pub fn set_precision(sel: &[String], gold: &[String]) -> f64 {
    if sel.is_empty() {
        return 0.0;
    }
    let g: HashSet<&String> = gold.iter().collect();
    sel.iter().filter(|d| g.contains(*d)).count() as f64 / sel.len() as f64
}

pub fn set_recall(sel: &[String], gold: &[String]) -> f64 {
    if gold.is_empty() {
        return 0.0;
    }
    let g: HashSet<&String> = gold.iter().collect();
    sel.iter().filter(|d| g.contains(*d)).count() as f64 / gold.len() as f64
}

pub fn minmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().cloned().fold(f64::MIN, f64::max);
    let min = scores.iter().cloned().fold(f64::MAX, f64::min);
    if max - min < 1e-12 {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|s| (s - min) / (max - min)).collect()
}

pub fn bootstrap_ci(values: &[f64], iters: usize, seed: u64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut rng = crate::Rng::new(seed);
    let n = values.len();
    let mut means = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut s = 0.0;
        for _ in 0..n {
            s += values[(rng.next_u64() as usize) % n];
        }
        means.push(s / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo_i = ((iters as f64 * 0.025) as usize).min(iters - 1);
    let hi_i = ((iters as f64 * 0.975) as usize).min(iters - 1);
    (means[lo_i], means[hi_i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oriented_needs_span_and_file() {
        use std::collections::HashMap;
        use std::collections::HashSet;
        let ranked = vec!["w".to_string(), "g1".to_string(), "v".to_string()];
        let gold = vec!["g1".to_string(), "g2".to_string()];
        let mut idp = HashMap::new();
        idp.insert("w".to_string(), "other.py".to_string());
        idp.insert("g1".to_string(), "gold.py".to_string());
        idp.insert("v".to_string(), "gold.py".to_string());
        let gf: HashSet<String> = HashSet::from(["gold.py".to_string()]);
        // g1 is both span and file hit.
        assert!(oriented_at_k(&ranked, &gold, &idp, &gf, 3));
        assert!(!oriented_at_k(&ranked, &gold, &idp, &gf, 1));
        // Same-file non-gold alone is not orientation.
        let ranked2 = vec!["w".to_string(), "v".to_string()];
        assert!(!oriented_at_k(&ranked2, &gold, &idp, &gf, 3));
        // Empty gold never orients.
        assert!(!oriented_at_k(&ranked, &[], &idp, &gf, 3));
    }

    #[test]
    fn ndcg_hand_case() {
        let ranked = vec!["a".to_string(), "x".to_string(), "b".to_string()];
        let gold = vec!["a".to_string(), "b".to_string()];
        let ndcg = ndcg_at_k(&ranked, &gold, 3);
        let expected = 1.5 / (1.0 + 1.0 / 3.0f64.log2());
        assert!((ndcg - expected).abs() < 1e-12);
    }

    #[test]
    fn recall_mrr_precision_basics() {
        let ranked = vec!["a".to_string(), "x".to_string(), "b".to_string()];
        let gold = vec!["a".to_string(), "b".to_string()];
        assert_eq!(recall_at_k(&ranked, &gold, 3), 1.0);
        assert_eq!(recall_at_k(&ranked, &gold, 1), 0.5);
        assert!((mrr(&ranked, &gold) - 1.0).abs() < 1e-12);
        assert_eq!(precision_at_k(&ranked, &gold, 2), 0.5);
    }

    #[test]
    fn file_recall_counts_files_not_docs() {
        use std::collections::{HashMap, HashSet};
        let id_to_path: HashMap<String, String> = [
            ("s1", "a.py"),
            ("s2", "a.py"),
            ("s3", "b.py"),
            ("s4", "c.py"),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        let gold_files: HashSet<String> =
            ["a.py".to_string(), "b.py".to_string()].into_iter().collect();
        // Two a.py docs in top-2 still cover only 1 of 2 gold files.
        let ranked = vec!["s1".to_string(), "s2".to_string(), "s3".to_string()];
        assert_eq!(file_recall_at_k(&ranked, &id_to_path, &gold_files, 2), 0.5);
        assert_eq!(file_recall_at_k(&ranked, &id_to_path, &gold_files, 3), 1.0);
        // Unknown ids never count; empty gold is 0.
        let ranked_unknown = vec!["zzz".to_string()];
        assert_eq!(
            file_recall_at_k(&ranked_unknown, &id_to_path, &gold_files, 1),
            0.0
        );
        let empty: HashSet<String> = HashSet::new();
        assert_eq!(file_recall_at_k(&ranked, &id_to_path, &empty, 3), 0.0);
    }

    #[test]
    fn bootstrap_constant_values_exact_ci() {
        let vals = vec![0.5f64; 50];
        let (lo, hi) = bootstrap_ci(&vals, 10000, 7);
        assert!((lo - 0.5).abs() < 1e-12);
        assert!((hi - 0.5).abs() < 1e-12);
    }

    #[test]
    fn bootstrap_seeded_reproducible() {
        let vals: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let a = bootstrap_ci(&vals, 5000, 42);
        let b = bootstrap_ci(&vals, 5000, 42);
        assert_eq!(a, b);
        assert!(a.0 <= a.1);
    }

    #[test]
    fn minmax_bounds() {
        let m = minmax(&[2.0, 4.0, 6.0]);
        assert_eq!(m, vec![0.0, 0.5, 1.0]);
    }

    #[test]
    fn bcy_hand_case_partial_exposure() {
        let ranked: Vec<String> = ["a", "b", "c", "x"].iter().map(|s| s.to_string()).collect();
        let costs = vec![10i64, 10, 10, 10];
        let gold = vec!["a".to_string(), "c".to_string()];
        let bcy = bcy_at_budget(&ranked, &costs, &gold, 25);
        assert!((bcy - 0.75).abs() < 1e-12);
        assert_eq!(bcy_at_budget(&ranked, &costs, &gold, 100), 1.0);
        assert_eq!(bcy_at_budget(&ranked, &costs, &gold, 5), 0.25);
    }

    #[test]
    fn knee_detects_elbow() {
        let curve = vec![0.0, 0.2, 0.9, 1.0, 1.0];
        assert_eq!(knee_index(&curve), 2);
        assert_eq!(knee_index(&[0.5]), 0);
        assert_eq!(knee_index(&[0.1, 0.9]), 1);
    }

    #[test]
    fn recall_curve_monotone_nondecreasing() {
        let ranked: Vec<String> = ["a", "x", "b", "y"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let gold = vec!["a".to_string(), "b".to_string()];
        let c = recall_curve(&ranked, &gold, 4);
        for w in c.windows(2) {
            assert!(w[1] >= w[0] - 1e-12);
        }
    }
}
