pub const RRF_K: usize = 60;

pub fn rrf(rankings: &[Vec<usize>], ndocs: usize) -> Vec<f64> {
    let mut scores = vec![0.0; ndocs];
    for ranking in rankings {
        for (pos, &doc) in ranking.iter().enumerate() {
            scores[doc] += 1.0 / (RRF_K + pos + 1) as f64;
        }
    }
    scores
}

pub fn rrf_weighted(rankings: &[Vec<usize>], weights: &[f64], ndocs: usize) -> Vec<f64> {
    assert_eq!(rankings.len(), weights.len(), "rankings/weights length mismatch");
    let mut scores = vec![0.0; ndocs];
    for (ranking, &w) in rankings.iter().zip(weights.iter()) {
        for (pos, &doc) in ranking.iter().enumerate() {
            scores[doc] += w * 1.0 / (RRF_K + pos + 1) as f64;
        }
    }
    scores
}

pub fn rank_order(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .total_cmp(&scores[a])
    });
    idx
}

/// Exact-identifier refs leg (mirrors the cortex compile refs arm):
/// query tokens that EXACTLY match a doc's signature tokens (name/path
/// terms, not TF-weighted prose) vote with BM25-style IDF weight.
/// Generic-identifier gate: tokens in >2% of docs are skipped (same ratio
/// rule as `compile.rs` — repo-agnostic, no per-repo constants).
/// Scores are the IDF-weighted voting fraction in [0,1]; docs with no
/// exact hit score 0.
pub fn exact_idf_scores(
    query_toks: &[String],
    sig_toks: &[std::collections::HashSet<String>],
    doc_freq: &dyn Fn(&str) -> usize,
    n_docs: usize,
    generic_frac: f64,
) -> Vec<f64> {
    let n = n_docs as f64;
    let mut seen = std::collections::HashSet::new();
    let mut votes: Vec<(f64, Vec<usize>)> = Vec::new();
    let mut idf_sum = 0.0f64;
    for t in query_toks {
        if t.len() < 2 || !seen.insert(t.as_str()) {
            continue;
        }
        let df = doc_freq(t) as f64;
        if df / n > generic_frac {
            continue;
        }
        let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
        if idf <= 0.0 {
            continue;
        }
        let mut hit = Vec::new();
        for (i, sig) in sig_toks.iter().enumerate() {
            if sig.contains(t) {
                hit.push(i);
            }
        }
        if hit.is_empty() {
            continue;
        }
        idf_sum += idf;
        votes.push((idf, hit));
    }
    let mut scores = vec![0.0f64; sig_toks.len()];
    if idf_sum > 0.0 {
        for (idf, hit) in votes {
            for i in hit {
                scores[i] += idf / idf_sum;
            }
        }
    }
    scores
}

/// Rank docs with positive score, descending (ties by lower index).
/// Empty when the leg has no exact hits — absent leg = no contribution.
pub fn positive_rank_order(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).filter(|&i| scores[i] > 0.0).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .total_cmp(&scores[a])
            .then_with(|| a.cmp(&b))
    });
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_computed_rrf_values() {
        let r1 = vec![0usize, 1, 2];
        let r2 = vec![1usize, 0, 2];
        let s = rrf(&[r1, r2], 3);
        assert!((s[0] - (1.0 / 61.0 + 1.0 / 62.0)).abs() < 1e-12);
        assert!((s[1] - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-12);
        assert!((s[2] - (1.0 / 63.0 + 1.0 / 63.0)).abs() < 1e-12);
        assert!(s[0] > s[2]);
    }

    #[test]
    fn weighted_rrf_scales_by_weight() {
        let r1 = vec![0usize, 1];
        let r2 = vec![1usize, 0];
        let s = rrf_weighted(&[r1.clone(), r2.clone()], &[1.0, 0.0], 2);
        let s1 = rrf(&[r1], 2);
        assert!((s[0] - s1[0]).abs() < 1e-12);
        assert!((s[1] - s1[1]).abs() < 1e-12);
        let s2 = rrf_weighted(&[vec![0,1], vec![1,0]], &[0.5, 0.5], 2);
        // weighted 0.5+0.5 should equal 0.5*rrf(both)
        let both = rrf(&[vec![0,1], vec![1,0]], 2);
        assert!((s2[0] - 0.5*both[0]).abs() < 1e-12);
    }

    #[test]
    fn rank_order_descending() {
        let order = rank_order(&[0.5, 2.0, 1.0]);
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn exact_idf_rare_token_votes_common_gated() {
        use std::collections::HashSet;
        // 100 docs: "plot_wcs" in 1 sig, "self" in 90 sigs.
        let mut sigs: Vec<HashSet<String>> = (0..100)
            .map(|_| HashSet::from(["self".to_string()]))
            .collect();
        sigs[7].insert("plot_wcs".to_string());
        let df = |t: &str| match t {
            "plot_wcs" => 1,
            "self" => 90,
            _ => 0,
        };
        let q = vec!["plot_wcs".to_string(), "self".to_string()];
        let s = exact_idf_scores(&q, &sigs, &df, 100, 0.02);
        assert_eq!(s[7], 1.0);
        assert!(s.iter().enumerate().all(|(i, &v)| i == 7 || v == 0.0));
        // Query-side dedup: repeated token votes once.
        let q2 = vec!["plot_wcs".to_string(), "plot_wcs".to_string()];
        assert_eq!(exact_idf_scores(&q2, &sigs, &df, 100, 0.02), s);
    }

    #[test]
    fn exact_idf_no_hits_gives_empty_ranking() {
        use std::collections::HashSet;
        let sigs = vec![HashSet::from(["aaa".to_string()])];
        let s = exact_idf_scores(&["zzz".to_string()], &sigs, &|_| 0, 1, 0.02);
        assert_eq!(s, vec![0.0]);
        assert!(positive_rank_order(&s).is_empty());
    }
}
