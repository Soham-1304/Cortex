#[derive(Clone, Debug)]
pub struct Item {
    pub idx: usize,
    pub value: f64,
    pub cost: i64,
}

pub fn knapsack(items: &[Item], budget: i64) -> Vec<usize> {
    let b = budget.max(0) as usize;
    let feas: Vec<&Item> = items
        .iter()
        .filter(|it| it.cost >= 1 && (it.cost as usize) <= b)
        .collect();
    let n = feas.len();
    if n == 0 || b == 0 {
        return Vec::new();
    }
    let mut dp = vec![0.0f64; b + 1];
    let mut take = vec![false; n * (b + 1)];
    for (i, it) in feas.iter().enumerate() {
        let c = it.cost as usize;
        for j in (c..=b).rev() {
            let cand = dp[j - c] + it.value;
            if cand > dp[j] {
                dp[j] = cand;
                take[i * (b + 1) + j] = true;
            }
        }
    }
    let mut sel = Vec::new();
    let mut j = b;
    for i in (0..n).rev() {
        let c = feas[i].cost as usize;
        if c <= j && take[i * (b + 1) + j] {
            sel.push(feas[i].idx);
            j -= c;
        }
    }
    sel.reverse();
    sel
}

pub fn greedy(items: &[Item], budget: i64) -> Vec<usize> {
    let mut order: Vec<&Item> = items
        .iter()
        .filter(|it| it.cost >= 1 && it.cost <= budget)
        .collect();
    order.sort_by(|a, b| {
        let ra = a.value / a.cost as f64;
        let rb = b.value / b.cost as f64;
        rb.partial_cmp(&ra).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sel = Vec::new();
    let mut used = 0i64;
    for it in order {
        if used + it.cost <= budget {
            used += it.cost;
            sel.push(it.idx);
        }
    }
    sel
}

pub fn total_value(items: &[Item], sel: &[usize]) -> f64 {
    sel.iter().map(|i| items[*i].value).sum()
}

pub fn total_cost(items: &[Item], sel: &[usize]) -> i64 {
    sel.iter().map(|i| items[*i].cost).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    fn brute_force(items: &[Item], budget: i64) -> f64 {
        let n = items.len();
        let mut best = 0.0f64;
        for mask in 0..(1u32 << n) {
            let mut v = 0.0f64;
            let mut c = 0i64;
            let mut ok = true;
            for (i, it) in items.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    v += it.value;
                    c += it.cost;
                    if c > budget {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && v > best {
                best = v;
            }
        }
        best
    }

    #[test]
    fn dp_equals_brute_force_on_random_instances() {
        let mut rng = Rng::new(12345);
        for _ in 0..200 {
            let n = 1 + (rng.next_u64() % 10) as usize;
            let items: Vec<Item> = (0..n)
                .map(|i| Item {
                    idx: i,
                    value: rng.unit(),
                    cost: 1 + (rng.next_u64() % 8) as i64,
                })
                .collect();
            let budget = 1 + (rng.next_u64() % 40) as i64;
            let dp_val = total_value(&items, &knapsack(&items, budget));
            let bf_val = brute_force(&items, budget);
            assert!(
                (dp_val - bf_val).abs() < 1e-9,
                "dp={} bf={} n={} budget={}",
                dp_val,
                bf_val,
                n,
                budget
            );
        }
    }

    #[test]
    fn dp_never_below_greedy_and_budget_respected() {
        let mut rng = Rng::new(999);
        for _ in 0..500 {
            let n = 1 + (rng.next_u64() % 30) as usize;
            let items: Vec<Item> = (0..n)
                .map(|i| Item {
                    idx: i,
                    value: rng.unit(),
                    cost: 1 + (rng.next_u64() % 12) as i64,
                })
                .collect();
            let budget = 1 + (rng.next_u64() % 60) as i64;
            let sel_dp = knapsack(&items, budget);
            let sel_gr = greedy(&items, budget);
            assert!(total_cost(&items, &sel_dp) <= budget);
            assert!(total_cost(&items, &sel_gr) <= budget);
            assert!(total_value(&items, &sel_dp) >= total_value(&items, &sel_gr) - 1e-9);
        }
    }

    #[test]
    fn empty_when_budget_too_small() {
        let items = vec![Item { idx: 0, value: 1.0, cost: 10 }];
        assert!(knapsack(&items, 5).is_empty());
        assert!(greedy(&items, 5).is_empty());
    }
}
