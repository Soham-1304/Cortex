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
        rb.total_cmp(&ra)
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

/// Candidate symbol inside a file node of the hierarchical budget tree.
#[derive(Clone, Debug)]
pub struct TreeSymbol {
    pub sym_idx: usize,
    pub relevance: f64,
    /// L2 signature cost in tokens ($\ge 1$).
    pub sig_cost: i64,
    /// Extra tokens required to expand L2 signature into an L3 source snippet (`0` if unavailable).
    pub snippet_extra_cost: i64,
}

/// File node in the two-stage File -> Symbol budget tree.
#[derive(Clone, Debug)]
pub struct TreeFile {
    pub file_idx: usize,
    /// One-time token overhead of opening this file in the context block (path + graph edges).
    pub header_cost: i64,
    /// Stage-1 file-level relevance score.
    pub file_prior: f64,
    pub symbols: Vec<TreeSymbol>,
}

/// Output decision for a packed symbol.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedSymbol {
    pub sym_idx: usize,
    /// Whether this symbol was upgraded from L2 signature to L3 implementation snippet.
    pub expand_snippet: bool,
}

/// Output decision for a packed file and its chosen symbols.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedFile {
    pub file_idx: usize,
    pub symbols: Vec<PackedSymbol>,
}

/// Submodular two-stage hierarchical budget allocator (File -> Span -> Progressive L3 Snippet).
///
/// Stage 1+2 (Breadth Localization):
/// Greedily opens files and selects symbol spans by maximizing marginal submodular gain per
/// smoothed token cost:
/// $$\Delta V(F, r) = \gamma^r \cdot \left(\alpha \cdot P(F) + (1 - \alpha) \cdot R(F, r)\right), \quad
///   \text{eff\_density} = \frac{\Delta V(F, r)}{(\Delta c + c_0)^\beta}$$
/// where $\gamma = 0.85$ penalizes redundant symbols within the same file, $\Delta c$ includes
/// `header_cost` on the first symbol of file $F$ ($r=0$), and $(c + 6)^{0.65}$ eliminates the
/// classic knapsack pathology where 2-token getter methods outrank 12-token core functions.
///
/// Stage 3 (Depth Expansion):
/// Once top candidate spans are localized (or localization reaches diminishing marginal returns),
/// upgrades the highest-value selected symbols from L2 signature to L3 code snippets within the
/// remaining integer `budget`, strictly guaranteeing `tokens_used <= budget`.
pub fn pack_hierarchical(files: &[TreeFile], budget: i64) -> (Vec<PackedFile>, i64) {
    if budget <= 0 || files.is_empty() {
        return (Vec::new(), 0);
    }

    const GAMMA: f64 = 0.85;
    const ALPHA: f64 = 0.35;
    const COST_SMOOTH: f64 = 6.0;
    const COST_EXP: f64 = 0.65;
    // Reserve up to 55% of generous budgets (>=1200 tokens) for L3 source snippets on top hits
    // while always allowing unused snippet reserve to spill back into L2 localization if needed.
    let loc_soft_cap = if budget >= 1200 {
        (budget * 55) / 100
    } else {
        budget
    };

    // Sort symbols inside each file by descending relevance once.
    let mut sorted_files: Vec<(usize, i64, f64, Vec<TreeSymbol>)> = files
        .iter()
        .filter(|f| !f.symbols.is_empty() && f.header_cost >= 0)
        .map(|f| {
            let mut syms = f.symbols.clone();
            syms.retain(|s| s.sig_cost >= 1);
            syms.sort_by(|a, b| b.relevance.total_cmp(&a.relevance));
            (f.file_idx, f.header_cost, f.file_prior, syms)
        })
        .filter(|(_, _, _, syms)| !syms.is_empty())
        .collect();

    let nf = sorted_files.len();
    let mut cursor = vec![0usize; nf];
    let mut chosen: Vec<Vec<PackedSymbol>> = vec![Vec::new(); nf];
    let mut used = 0i64;

    // Pass 1: Localize files & L2 symbol spans up to `loc_soft_cap`.
    run_l2_pass(
        &sorted_files,
        &mut cursor,
        &mut chosen,
        &mut used,
        loc_soft_cap,
        GAMMA,
        ALPHA,
        COST_SMOOTH,
        COST_EXP,
    );

    // Pass 2: Upgrade highest-scoring selected symbols from L2 signature -> L3 code snippet
    // using remaining budget. Prioritize rank-0 / top-relevance symbols across top files.
    let mut upgrade_candidates: Vec<(f64, usize, usize, i64)> = Vec::new();
    for (fi, (_, _, f_prior, syms)) in sorted_files.iter().enumerate() {
        for (rank_in_file, ps) in chosen[fi].iter().enumerate() {
            let sym = &syms[rank_in_file];
            if sym.snippet_extra_cost > 0 {
                let rank_decay = GAMMA.powi(rank_in_file as i32);
                let utility = rank_decay * (ALPHA * *f_prior + (1.0 - ALPHA) * sym.relevance);
                let density = utility / ((sym.snippet_extra_cost as f64) + 12.0).powf(0.45);
                let _ = ps;
                upgrade_candidates.push((density, fi, rank_in_file, sym.snippet_extra_cost));
            }
        }
    }
    upgrade_candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

    for (_, fi, rank_in_file, extra) in upgrade_candidates {
        if used + extra <= budget {
            used += extra;
            chosen[fi][rank_in_file].expand_snippet = true;
        }
    }

    // Pass 3: Spill any remaining budget back into additional L2 symbol localization!
    if used < budget {
        run_l2_pass(
            &sorted_files,
            &mut cursor,
            &mut chosen,
            &mut used,
            budget,
            GAMMA,
            ALPHA,
            COST_SMOOTH,
            COST_EXP,
        );
    }

    // Order output files by their peak realized score (file_prior + top symbol relevance).
    let mut out_order: Vec<(f64, usize)> = sorted_files
        .iter()
        .enumerate()
        .filter(|(fi, _)| !chosen[*fi].is_empty())
        .map(|(fi, (_, _, fp, syms))| (*fp * ALPHA + syms[0].relevance * (1.0 - ALPHA), fi))
        .collect();
    out_order.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut result = Vec::with_capacity(out_order.len());
    for (_, fi) in out_order {
        let file_idx = sorted_files[fi].0;
        let syms = std::mem::take(&mut chosen[fi]);
        result.push(PackedFile {
            file_idx,
            symbols: syms,
        });
    }
    sorted_files.clear();
    (result, used)
}

#[allow(clippy::too_many_arguments)]
fn run_l2_pass(
    sorted_files: &[(usize, i64, f64, Vec<TreeSymbol>)],
    cursor: &mut [usize],
    chosen: &mut [Vec<PackedSymbol>],
    used: &mut i64,
    cap: i64,
    gamma: f64,
    alpha: f64,
    cost_smooth: f64,
    cost_exp: f64,
) {
    loop {
        let mut best_fi: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_cost = 0i64;

        for (fi, (_, hdr_cost, f_prior, syms)) in sorted_files.iter().enumerate() {
            let r = cursor[fi];
            if r >= syms.len() {
                continue;
            }
            let sym = &syms[r];
            let step_cost = if r == 0 {
                *hdr_cost + sym.sig_cost
            } else {
                sym.sig_cost
            };
            if step_cost <= 0 || *used + step_cost > cap {
                continue;
            }
            // Also peek 2-step amortization for unopened files whose top-2 symbols both fit:
            // prevents a high header_cost from unfairly penalizing a file with two strong methods.
            let gain1 = gamma.powi(r as i32) * (alpha * *f_prior + (1.0 - alpha) * sym.relevance);
            let mut eff = gain1 / ((step_cost as f64) + cost_smooth).powf(cost_exp);
            if r == 0 && syms.len() >= 2 {
                let c2 = step_cost + syms[1].sig_cost;
                if *used + c2 <= cap {
                    let gain2 = gain1
                        + gamma * (alpha * *f_prior + (1.0 - alpha) * syms[1].relevance);
                    let eff2 = gain2 / ((c2 as f64) + cost_smooth).powf(cost_exp);
                    if eff2 > eff {
                        eff = eff2;
                    }
                }
            }
            if eff > best_score {
                best_score = eff;
                best_fi = Some(fi);
                best_cost = step_cost;
            }
        }

        let Some(fi) = best_fi else {
            break;
        };
        let r = cursor[fi];
        let sym_idx = sorted_files[fi].3[r].sym_idx;
        *used += best_cost;
        chosen[fi].push(PackedSymbol {
            sym_idx,
            expand_snippet: false,
        });
        cursor[fi] = r + 1;
    }
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

    #[test]
    fn hierarchical_pack_respects_budget_and_expands_top_snippets() {
        let files = vec![
            TreeFile {
                file_idx: 0,
                header_cost: 6,
                file_prior: 1.0,
                symbols: vec![
                    TreeSymbol {
                        sym_idx: 10,
                        relevance: 0.95,
                        sig_cost: 8,
                        snippet_extra_cost: 40,
                    },
                    TreeSymbol {
                        sym_idx: 11,
                        relevance: 0.60,
                        sig_cost: 6,
                        snippet_extra_cost: 35,
                    },
                ],
            },
            TreeFile {
                file_idx: 1,
                header_cost: 6,
                file_prior: 0.70,
                symbols: vec![TreeSymbol {
                    sym_idx: 20,
                    relevance: 0.75,
                    sig_cost: 7,
                    snippet_extra_cost: 30,
                }],
            },
        ];

        // Tight budget (35 fits 14 + 6 + 13 = 33 L2 tokens across both files, no room for snippets):
        let (packed_tight, used_tight) = pack_hierarchical(&files, 35);
        assert!(used_tight <= 35, "used_tight={used_tight}");
        assert_eq!(packed_tight.len(), 2);
        assert!(packed_tight[0].symbols.iter().all(|s| !s.expand_snippet));

        // Generous budget (>=1200): localizes all symbols AND upgrades top symbols to L3 snippets.
        let (packed_rich, used_rich) = pack_hierarchical(&files, 1500);
        assert!(used_rich <= 1500, "used_rich={used_rich}");
        assert_eq!(packed_rich.len(), 2);
        assert!(packed_rich[0].symbols[0].expand_snippet);
    }
}
