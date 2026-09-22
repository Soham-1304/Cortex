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
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
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
}
