use crate::Rng;

#[derive(Clone, Debug)]
pub struct Isotonic {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

struct Block {
    x_max: f64,
    sy: f64,
    n: f64,
}

impl Isotonic {
    pub fn fit(scores: &[f64], labels: &[f64]) -> Isotonic {
        assert_eq!(scores.len(), labels.len(), "score/label length mismatch");
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            scores[a]
                .total_cmp(&scores[b])
        });
        let mut blocks: Vec<Block> = Vec::with_capacity(scores.len());
        for &i in &order {
            let mut b = Block { x_max: scores[i], sy: labels[i], n: 1.0 };
            while let Some(top) = blocks.last() {
                if top.sy / top.n > b.sy / b.n + 1e-15 {
                    let t = blocks.pop().unwrap();
                    b = Block { x_max: b.x_max, sy: t.sy + b.sy, n: t.n + b.n };
                } else {
                    break;
                }
            }
            blocks.push(b);
        }
        let mut xs = Vec::with_capacity(blocks.len());
        let mut ys = Vec::with_capacity(blocks.len());
        for blk in blocks {
            xs.push(blk.x_max);
            ys.push(blk.sy / blk.n);
        }
        Isotonic { xs, ys }
    }

    pub fn predict(&self, s: f64) -> f64 {
        if self.xs.is_empty() {
            return 0.0;
        }
        if s <= self.xs[0] {
            return self.ys[0];
        }
        let last = self.xs.len() - 1;
        if s >= self.xs[last] {
            return self.ys[last];
        }
        let idx = self.xs.partition_point(|&x| x < s).max(1);
        let (x0, x1) = (self.xs[idx - 1], self.xs[idx]);
        let (y0, y1) = (self.ys[idx - 1], self.ys[idx]);
        if x1 == x0 {
            return y1;
        }
        y0 + (y1 - y0) * ((s - x0) / (x1 - x0))
    }
}

pub fn expected_calibration_error(preds: &[f64], labels: &[f64], bins: usize) -> f64 {
    assert_eq!(preds.len(), labels.len());
    if preds.is_empty() {
        return 0.0;
    }
    let mut bin_n = vec![0usize; bins];
    let mut bin_conf = vec![0.0f64; bins];
    let mut bin_acc = vec![0.0f64; bins];
    for (p, y) in preds.iter().zip(labels.iter()) {
        let b = ((p * bins as f64) as usize).min(bins - 1);
        bin_n[b] += 1;
        bin_conf[b] += p;
        bin_acc[b] += y;
    }
    let n = preds.len() as f64;
    let mut ece = 0.0;
    for b in 0..bins {
        if bin_n[b] == 0 {
            continue;
        }
        let conf = bin_conf[b] / bin_n[b] as f64;
        let acc = bin_acc[b] / bin_n[b] as f64;
        ece += (bin_n[b] as f64 / n) * (acc - conf).abs();
    }
    ece
}

pub fn brier_score(preds: &[f64], labels: &[f64]) -> f64 {
    assert_eq!(preds.len(), labels.len());
    if preds.is_empty() {
        return 0.0;
    }
    preds
        .iter()
        .zip(labels.iter())
        .map(|(p, y)| (p - y) * (p - y))
        .sum::<f64>()
        / preds.len() as f64
}

pub fn bootstrap_rho_ci(
    cal_scores: &[f64],
    cal_labels: &[f64],
    ctx_scores: &[Vec<f64>],
    ctx_costs: &[Vec<i64>],
    iters: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let point = mean_rho(&Isotonic::fit(cal_scores, cal_labels), ctx_scores, ctx_costs);
    let n = cal_scores.len();
    let mut rng = Rng::new(seed);
    let mut rhos = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut s = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for _ in 0..n {
            let i = (rng.next_u64() as usize) % n;
            s.push(cal_scores[i]);
            y.push(cal_labels[i]);
        }
        let iso = Isotonic::fit(&s, &y);
        rhos.push(mean_rho(&iso, ctx_scores, ctx_costs));
    }
    rhos.sort_by(|a, b| a.total_cmp(b));
    let lo_i = ((iters as f64 * 0.025) as usize).min(iters - 1);
    let hi_i = ((iters as f64 * 0.975) as usize).min(iters - 1);
    (point, rhos[lo_i], rhos[hi_i])
}

fn mean_rho(iso: &Isotonic, ctx_scores: &[Vec<f64>], ctx_costs: &[Vec<i64>]) -> f64 {
    let mut total = 0.0;
    let mut count = 0.0;
    for (scores, costs) in ctx_scores.iter().zip(ctx_costs.iter()) {
        let b: i64 = costs.iter().sum();
        if b == 0 {
            continue;
        }
        let num: f64 = scores
            .iter()
            .zip(costs.iter())
            .map(|(s, c)| iso.predict(*s) * (*c as f64))
            .sum();
        total += num / b as f64;
        count += 1.0;
    }
    if count == 0.0 {
        0.0
    } else {
        total / count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separated_data_recovers_zero_one() {
        let iso = Isotonic::fit(&[1.0, 2.0, 3.0, 4.0], &[0.0, 0.0, 1.0, 1.0]);
        assert!(iso.predict(1.5) < 0.01);
        assert!(iso.predict(3.5) > 0.99);
    }

    #[test]
    fn violation_pools_to_mean() {
        let iso = Isotonic::fit(&[1.0, 2.0], &[1.0, 0.0]);
        assert!((iso.predict(1.0) - 0.5).abs() < 1e-12);
        assert!((iso.predict(2.0) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn predictions_monotone_on_random_data() {
        let mut rng = Rng::new(31337);
        for _ in 0..100 {
            let n = 2 + (rng.next_u64() % 30) as usize;
            let xs: Vec<f64> = (0..n).map(|_| rng.unit()).collect();
            let ys: Vec<f64> = (0..n).map(|_| (rng.unit() > 0.5) as i32 as f64).collect();
            let iso = Isotonic::fit(&xs, &ys);
            let grid: Vec<f64> = (0..=50).map(|i| i as f64 / 50.0).collect();
            for w in grid.windows(2) {
                let a = iso.predict(w[0]);
                let b = iso.predict(w[1]);
                assert!(
                    a <= b + 1e-12,
                    "non-monotone: p({})={} > p({})={}",
                    w[0], a, w[1], b
                );
            }
        }
    }

    #[test]
    fn clamps_outside_training_range() {
        let iso = Isotonic::fit(&[2.0, 4.0, 6.0, 8.0], &[0.0, 0.0, 1.0, 1.0]);
        assert_eq!(iso.predict(-10.0), 0.0);
        assert_eq!(iso.predict(100.0), 1.0);
    }

    #[test]
    fn ece_perfect_calibration_zero() {
        let preds = vec![1.0, 0.0];
        let labels = vec![1.0, 0.0];
        assert!(expected_calibration_error(&preds, &labels, 10) < 1e-12);
        let mixed = vec![0.9, 0.1];
        assert!((expected_calibration_error(&mixed, &labels, 10) - 0.1).abs() < 1e-12);
    }

    #[test]
    fn ece_hand_case() {
        let preds = vec![0.5, 0.5];
        let labels = vec![1.0, 0.0];
        assert!(expected_calibration_error(&preds, &labels, 10).abs() < 1e-12);
        let preds = vec![0.95, 0.95];
        let labels = vec![1.0, 0.0];
        assert!((expected_calibration_error(&preds, &labels, 10) - 0.45).abs() < 1e-12);
    }

    #[test]
    fn brier_basics() {
        assert!((brier_score(&[1.0, 0.0], &[1.0, 0.0])).abs() < 1e-12);
        assert!((brier_score(&[0.5, 0.5], &[1.0, 0.0]) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn rho_bootstrap_reproducible_and_ordered() {
        let cs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let cl = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let ctx_s = vec![vec![0.75, 0.35]];
        let ctx_c = vec![vec![10i64, 20i64]];
        let (p1, lo1, hi1) = bootstrap_rho_ci(&cs, &cl, &ctx_s, &ctx_c, 5000, 42);
        let (_, lo2, hi2) = bootstrap_rho_ci(&cs, &cl, &ctx_s, &ctx_c, 5000, 42);
        assert_eq!(lo1, lo2);
        assert_eq!(hi1, hi2);
        assert!(lo1 <= p1 + 1e-9 && p1 <= hi1 + 1e-9);
        assert!(lo1 > 0.0 && hi1 < 1.0);
    }

    #[test]
    fn rho_higher_for_gold_heavy_context() {
        let cs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let cl = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let iso = Isotonic::fit(&cs, &cl);
        let gold_ctx = vec![0.78, 0.72];
        let junk_ctx = vec![0.22, 0.18];
        let costs = vec![10i64, 10];
        let rg = mean_rho(&iso, &[gold_ctx], std::slice::from_ref(&costs));
        let rj = mean_rho(&iso, &[junk_ctx], std::slice::from_ref(&costs));
        assert!(rg > rj + 0.3);
    }
}
