use std::path::PathBuf;

use serde::Serialize;

use bench::bm25::Bm25Index;
use bench::calib::{brier_score, expected_calibration_error, Isotonic};
use bench::corpus::Corpus;
use bench::eval::{
    bcy_at_budget, bootstrap_ci, knee_index, minmax, mrr, ndcg_at_k, recall_at_k,
    recall_curve, set_precision, set_recall,
};
use bench::fuse;
use bench::pack::{self, Item};
use bench::provider::pick_provider;

#[derive(Clone)]
struct QueryRun {
    ndcg10: f64,
    recall20: f64,
    mrr: f64,
    packed_precision: f64,
    packed_recall: f64,
    rho_naive: f64,
    rho_insample: f64,
    rho_holdout: f64,
    bcy: f64,
    knee_k: usize,
    saturated: bool,
    advisory: String,
    packed_tokens: i64,
    fused_all: Vec<f64>,
    label_all: Vec<f64>,
    packed_scores: Vec<f64>,
    packed_costs: Vec<i64>,
}

#[derive(Serialize)]
struct QueryRow {
    id: String,
    ndcg10: f64,
    recall20: f64,
    mrr: f64,
    packed_precision: f64,
    packed_recall: f64,
    rho_naive: f64,
    rho_insample: f64,
    rho_holdout: f64,
    bcy: f64,
    knee_k: usize,
    saturated: bool,
    advisory: String,
    packed_tokens: i64,
}

#[derive(Serialize)]
struct Aggregate {
    ndcg10: f64,
    recall20: f64,
    mrr: f64,
    packed_precision: f64,
    packed_recall: f64,
    rho_naive: f64,
    rho_insample: f64,
    rho_holdout: f64,
    rho_holdout_lo: f64,
    rho_holdout_hi: f64,
    bcy: f64,
    avg_packed_tokens: f64,
}

#[derive(Serialize)]
struct JsonOut {
    docs: usize,
    queries: usize,
    budget: i64,
    tau_low: f64,
    ece_uncalibrated: f64,
    ece_calibrated: f64,
    brier_uncalibrated: f64,
    brier_calibrated: f64,
    aggregate: Aggregate,
    queries_detail: Vec<QueryRow>,
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn report(name: &str, vals: &[f64], extra: &str) {
    let (lo, hi) = bootstrap_ci(vals, 10_000, 42);
    println!(
        "  {:<18} mean {:.4}  [95% CI {:.4} - {:.4}] {}",
        name,
        mean(vals),
        lo,
        hi,
        extra
    );
}

fn rho_of(scores: &[f64], costs: &[i64], iso: &Isotonic) -> f64 {
    let b: i64 = costs.iter().sum();
    if b == 0 {
        return 0.0;
    }
    scores
        .iter()
        .zip(costs.iter())
        .map(|(s, c)| iso.predict(*s) * (*c as f64))
        .sum::<f64>()
        / b as f64
}

fn cache_key(texts: &[String], prov_name: &str, dim: usize) -> u64 {
    // FNV-1a over provider + dim + each text + count. Any corpus change
    // (or provider/dim change) yields a different key, so a stale cache
    // can never silently misalign embeddings.
    let mut h: u64 = 0xcbf29ce484222325;
    for b in prov_name.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= dim as u64;
    h = h.wrapping_mul(0x100000001b3);
    h ^= texts.len() as u64;
    h = h.wrapping_mul(0x100000001b3);
    for t in texts {
        h ^= t.len() as u64;
        h = h.wrapping_mul(0x100000001b3);
        for b in t.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}

fn cached_doc_vecs(corpus: &Corpus, prov: &dyn bench::provider::Provider, fixtures: &std::path::Path) -> Vec<Vec<f32>> {
    let texts: Vec<String> = corpus.docs.iter().map(|d| d.text.clone()).collect();
    let cache_path = fixtures.join(format!("embed_cache_{}.json", prov.name().replace('/', "_")));
    if cache_path.exists() {
        if let Ok(raw) = std::fs::read_to_string(&cache_path) {
            // New format: {"key":u64,"dim":usize,"vectors":[...]}.
            // Legacy bare-vector format (positional — known unsafe) falls
            // through to re-embed below.
            if let Ok(w) = serde_json::from_str::<serde_json::Value>(&raw) {
                if w.get("vectors").is_some() {
                    let ok = w.get("vectors").and_then(|v| v.as_array()).map(|a| a.len() == texts.len()).unwrap_or(false);
                    if ok {
                        if let (Some(k), Some(vecs)) = (w.get("key").and_then(|v| v.as_u64()), w.get("vectors")) {
                            if let Ok(vecs) = serde_json::from_value::<Vec<Vec<f32>>>(vecs.clone()) {
                                // Verify key with actual dim from cached vectors.
                                let dim = vecs.first().map(|v| v.len()).unwrap_or(0);
                                if k == cache_key(&texts, prov.name(), dim) {
                                    eprintln!("[bench] loaded {} cached embeddings from {}", vecs.len(), cache_path.display());
                                    return vecs;
                                }
                                eprintln!("[bench] cache key mismatch (corpus changed) — re-embedding");
                            }
                        }
                    }
                } else {
                    eprintln!("[bench] legacy cache (positional, unverified) — re-embedding to content-keyed format");
                }
            }
        }
    }
    let v = prov.embed_batch(&texts).unwrap_or_else(|e| {
        eprintln!("[bench] embed_batch failed ({}), falling back to hash", e);
        texts.iter().map(|t| bench::embed::embed(t)).collect()
    });
    let dim = v.first().map(|vv| vv.len()).unwrap_or(0);
    let wrapper = serde_json::json!({"key": cache_key(&texts, prov.name(), dim), "dim": dim, "vectors": v});
    let _ = std::fs::write(&cache_path, serde_json::to_string(&wrapper).unwrap());
    eprintln!("[bench] cached {} embeddings to {}", v.len(), cache_path.display());
    v
}

/// Miss-rank diagnostic: bucket every gold doc by its absolute rank in the
/// fused ranking (1-based). Buckets: found (1-19), fusion loss (20-49),
/// near-tail (50-199, GAK co-file territory), deep tail (200-999),
/// vocab gap (1000+). Decides which recall lever to pull.
fn miss_rank_histogram(corpus: &Corpus, runs: &[QueryRun]) {
    let mut buckets = [0usize; 5];
    let mut total_gold = 0usize;
    for (r, q) in runs.iter().zip(corpus.queries.iter()) {
        let order = fuse::rank_order(&r.fused_all);
        for gid in &q.gold {
            total_gold += 1;
            if let Some(pos) = order.iter().position(|&i| &corpus.docs[i].id == gid) {
                match pos + 1 {
                    1..=19 => buckets[0] += 1,
                    20..=49 => buckets[1] += 1,
                    50..=199 => buckets[2] += 1,
                    200..=999 => buckets[3] += 1,
                    _ => buckets[4] += 1,
                }
            } else {
                buckets[4] += 1;
            }
        }
    }
    eprintln!("\n[miss-rank diagnostic] over {} gold spans (all queries):", total_gold);
    let names = [
        "rank 1-19    (found)",
        "rank 20-49   (fusion loss)",
        "rank 50-199  (near-tail, GAK pull-in)",
        "rank 200-999 (deep tail)",
        "rank 1000+   (vocab/semantic gap)",
    ];
    for (name, &c) in names.iter().zip(buckets.iter()) {
        let pct = if total_gold > 0 { 100.0 * c as f64 / total_gold as f64 } else { 0.0 };
        eprintln!("  {:<36} {:>5}  {:>5.1}%", name, c, pct);
    }
}

fn build_runs(corpus: &Corpus, budget: i64, prov: &dyn bench::provider::Provider, fuse_mode: &str, fixtures: &std::path::Path) -> Vec<QueryRun> {
    let texts: Vec<String> = corpus.docs.iter().map(|d| d.text.clone()).collect();
    let costs: Vec<i64> = corpus.docs.iter().map(|d| bench::count_tokens(&d.text)).collect();
    let bm25 = Bm25Index::build(&texts);
    let doc_vecs = cached_doc_vecs(corpus, prov, fixtures);
    let k_max = texts.len().min(20);
    let crrf_weights: Option<(f64, f64)> = if fuse_mode == "crrf" {
        let mut bm25_scores_all: Vec<f64> = Vec::new();
        let mut cos_scores_all: Vec<f64> = Vec::new();
        let mut labels_all: Vec<f64> = Vec::new();
        for q in &corpus.queries {
            let bm25_s = bm25.score(&q.text);
            let qv = prov.embed_one(&q.text).unwrap_or_else(|_| bench::embed::embed(&q.text));
            let cos_s: Vec<f64> = doc_vecs.iter().map(|d| bench::provider::cosine_generic(&qv, d) as f64).collect();
            for (i, doc) in corpus.docs.iter().enumerate() {
                let label = if q.gold.contains(&doc.id) { 1.0 } else { 0.0 };
                bm25_scores_all.push(bm25_s[i]);
                cos_scores_all.push(cos_s[i]);
                labels_all.push(label);
            }
        }
        let bm25_norm = minmax(&bm25_scores_all);
        let cos_norm = minmax(&cos_scores_all);
        let iso_bm25 = Isotonic::fit(&bm25_norm, &labels_all);
        let iso_cos = Isotonic::fit(&cos_norm, &labels_all);
        let pred_bm25: Vec<f64> = bm25_norm.iter().map(|s| iso_bm25.predict(*s)).collect();
        let pred_cos: Vec<f64> = cos_norm.iter().map(|s| iso_cos.predict(*s)).collect();
        let ece_bm25 = expected_calibration_error(&pred_bm25, &labels_all, 10);
        let ece_cos = expected_calibration_error(&pred_cos, &labels_all, 10);
        let w_bm25 = (1.0 - ece_bm25).clamp(0.1, 1.0);
        let w_cos = (1.0 - ece_cos).clamp(0.1, 1.0);
        eprintln!("[bench] CRRF weights: bm25 ECE={:.3} w={:.3} | cos ECE={:.3} w={:.3}", ece_bm25, w_bm25, ece_cos, w_cos);
        Some((w_bm25, w_cos))
    } else { None };

    let mut runs = Vec::new();
    for q in &corpus.queries {
        let r_bm25 = bm25.ranking(&q.text);
        let qvec = prov.embed_one(&q.text).unwrap_or_else(|_| bench::embed::embed(&q.text));
        let r_cos = {
            let mut sims: Vec<(usize, f32)> = doc_vecs.iter().enumerate().map(|(i, d)| (i, bench::provider::cosine_generic(&qvec, d))).collect();
            sims.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sims.into_iter().map(|(i,_)| i).collect::<Vec<usize>>()
        };
        let fused = if let Some((w_bm25, w_cos)) = crrf_weights {
            fuse::rrf_weighted(&[r_bm25, r_cos], &[w_bm25, w_cos], texts.len())
        } else {
            fuse::rrf(&[r_bm25, r_cos], texts.len())
        };
        let s_tilde = minmax(&fused);
        let order = fuse::rank_order(&fused);

        let ranked_ids: Vec<String> =
            order.iter().map(|&i| corpus.docs[i].id.clone()).collect();
        let gold = q.gold.clone();

        let items: Vec<Item> = (0..texts.len())
            .map(|i| Item { idx: i, value: s_tilde[i], cost: costs[i] })
            .collect();
        let sel = pack::knapsack(&items, budget);
        let sel_ids: Vec<String> = sel.iter().map(|&i| corpus.docs[i].id.clone()).collect();
        let b_packed = pack::total_cost(&items, &sel);
        let rho_naive = if b_packed > 0 {
            sel.iter()
                .map(|&i| s_tilde[i] * costs[i] as f64)
                .sum::<f64>()
                / b_packed as f64
        } else {
            0.0
        };
        let packed_scores: Vec<f64> = sel.iter().map(|&i| fused[i]).collect();
        let packed_costs: Vec<i64> = sel.iter().map(|&i| costs[i]).collect();

        let label_all: Vec<f64> = texts
            .iter()
            .enumerate()
            .map(|(i, _)| if gold.contains(&corpus.docs[i].id) { 1.0 } else { 0.0 })
            .collect();

        let curve = recall_curve(&ranked_ids, &gold, k_max.max(3));
        let knee = knee_index(&curve);
        let saturated =
            curve[knee] > 0.99 && curve[curve.len() - 1] - curve[knee] < 0.01;
        let ndcg10 = ndcg_at_k(&ranked_ids, &q.gold, 10);
        let recall20 = recall_at_k(&ranked_ids, &q.gold, 20);
        let mrr_val = mrr(&ranked_ids, &q.gold);
        let pack_p = set_precision(&sel_ids, &q.gold);
        let pack_r = set_recall(&sel_ids, &q.gold);

        runs.push(QueryRun {
            bcy: bcy_at_budget(&ranked_ids, &costs, &gold, budget),
            ndcg10,
            recall20,
            mrr: mrr_val,
            packed_precision: pack_p,
            packed_recall: pack_r,            rho_naive,
            rho_insample: 0.0,
            rho_holdout: 0.0,
            knee_k: knee + 1,
            saturated,
            advisory: String::new(),
            packed_tokens: b_packed,
            fused_all: fused,
            label_all,
            packed_scores,
            packed_costs,
        });
    }
    runs
}

fn finalize(runs: &mut [QueryRun], tau_low: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
    let n = runs.len();

    let cal_scores: Vec<f64> = runs.iter().flat_map(|r| r.fused_all.clone()).collect();
    let cal_labels: Vec<f64> = runs.iter().flat_map(|r| r.label_all.clone()).collect();
    let iso_all = Isotonic::fit(&cal_scores, &cal_labels);

    let pair_scores: Vec<Vec<f64>> = runs.iter().map(|r| r.fused_all.clone()).collect();
    let pair_labels: Vec<Vec<f64>> = runs.iter().map(|r| r.label_all.clone()).collect();

    for (i, r) in runs.iter_mut().enumerate() {
        r.rho_insample = rho_of(&r.packed_scores, &r.packed_costs, &iso_all);

        if n >= 2 {
            let mut s = Vec::new();
            let mut y = Vec::new();
            for j in 0..n {
                if j != i {
                    s.extend(pair_scores[j].iter());
                    y.extend(pair_labels[j].iter());
                }
            }
            let iso_ho = Isotonic::fit(&s, &y);
            r.rho_holdout = rho_of(&r.packed_scores, &r.packed_costs, &iso_ho);
        } else {
            r.rho_holdout = r.rho_insample;
        }

        r.advisory = if !r.saturated && r.rho_holdout < tau_low {
            "RETRIEVE_MORE".to_string()
        } else if r.saturated && r.rho_holdout < tau_low {
            "ESCALATE_MODEL".to_string()
        } else {
            "PROCEED".to_string()
        };
    }

    let preds_flat: Vec<f64> = cal_scores.iter().map(|s| iso_all.predict(*s)).collect();
    let naive_conf: Vec<f64> = minmax(&cal_scores);
    let ece_u = expected_calibration_error(&naive_conf, &cal_labels, 10);
    let ece_c = expected_calibration_error(&preds_flat, &cal_labels, 10);
    let br_u = brier_score(&naive_conf, &cal_labels);
    let br_c = brier_score(&preds_flat, &cal_labels);

    let holdout_vals: Vec<f64> = runs.iter().map(|r| r.rho_holdout).collect();
    let n_it = 10_000usize;
    let mut rng = bench::Rng::new(42);
    let mut means = Vec::with_capacity(n_it);
    for _ in 0..n_it {
        let mut s = 0.0;
        for _ in 0..n {
            s += holdout_vals[(rng.next_u64() as usize) % n];
        }
        means.push(s / n as f64);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = means[((n_it as f64 * 0.025) as usize).min(n_it - 1)];
    let hi = means[((n_it as f64 * 0.975) as usize).min(n_it - 1)];

    (
        mean(&holdout_vals),
        lo,
        hi,
        ece_u,
        ece_c,
        br_u,
        br_c,
    )
}

fn main() {
    let mut fixtures = PathBuf::from("fixtures");
    let mut budgets: Vec<i64> = vec![120];
    let mut json_out = false;
    let mut embed_flag = "auto".to_string();
    let mut fuse_flag = "rrf".to_string();
    let mut tau_low = 0.35f64;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--fixtures" => {
                fixtures = PathBuf::from(args.next().unwrap_or_else(|| "fixtures".into()))
            }
            "--budget" => {
                budgets = args
                    .next()
                    .map(|s| {
                        s.split(',')
                            .filter_map(|x| x.parse().ok())
                            .collect::<Vec<i64>>()
                    })
                    .unwrap_or(budgets)
            }
            "--embed" => embed_flag = args.next().unwrap_or_else(|| "auto".into()),
            "--fuse" => fuse_flag = args.next().unwrap_or_else(|| "rrf".into()),
            "--tau" => tau_low = args.next().and_then(|s| s.parse().ok()).unwrap_or(tau_low),
            "--json" => json_out = true,
            _ => {}
        }
    }
    let provider = pick_provider(&embed_flag);
    eprintln!("[bench] embed provider: {} dim={} | fuse: {}", provider.name(), provider.dim(), fuse_flag);

    let corpus = if fixtures.join("tasks.jsonl").exists() {
        bench::contextbench::load_dir(&fixtures).expect("failed to load contextbench export")
    } else {
        Corpus::load(&fixtures.join("corpus.json")).expect("failed to load corpus.json")
    };

    let mut outs = Vec::new();

    for budget in budgets {
        let mut runs = build_runs(&corpus, budget, provider.as_ref(), &fuse_flag, &fixtures);
        miss_rank_histogram(&corpus, &runs);
        let (rho_pt, rho_lo, rho_hi, ece_u, ece_c, br_u, br_c) =
            finalize(&mut runs, tau_low);

        let rows: Vec<QueryRow> = runs
            .iter()
            .zip(corpus.queries.iter())
            .map(|(r, q)| QueryRow {
                id: q.id.clone(),
                ndcg10: r.ndcg10,
                recall20: r.recall20,
                mrr: r.mrr,
                packed_precision: r.packed_precision,
                packed_recall: r.packed_recall,
                rho_naive: r.rho_naive,
                rho_insample: r.rho_insample,
                rho_holdout: r.rho_holdout,
                bcy: r.bcy,
                knee_k: r.knee_k,
                saturated: r.saturated,
                advisory: r.advisory.clone(),
                packed_tokens: r.packed_tokens,
            })
            .collect();

        let ndcg_vals: Vec<f64> = runs.iter().map(|r| r.ndcg10).collect();
        let recall_vals: Vec<f64> = runs.iter().map(|r| r.recall20).collect();
        let mrr_vals: Vec<f64> = runs.iter().map(|r| r.mrr).collect();
        let pack_p_vals: Vec<f64> = runs.iter().map(|r| r.packed_precision).collect();
        let pack_r_vals: Vec<f64> = runs.iter().map(|r| r.packed_recall).collect();
        let rho_naive_vals: Vec<f64> = runs.iter().map(|r| r.rho_naive).collect();
        let bcy_vals: Vec<f64> = runs.iter().map(|r| r.bcy).collect();

        let agg = Aggregate {
            ndcg10: mean(&ndcg_vals),
            recall20: mean(&recall_vals),
            mrr: mean(&mrr_vals),
            packed_precision: mean(&pack_p_vals),
            packed_recall: mean(&pack_r_vals),
            rho_naive: mean(&rho_naive_vals),
            rho_insample: mean(&runs.iter().map(|r| r.rho_insample).collect::<Vec<_>>()),
            rho_holdout: rho_pt,
            rho_holdout_lo: rho_lo,
            rho_holdout_hi: rho_hi,
            bcy: mean(&bcy_vals),
            avg_packed_tokens: runs.iter().map(|r| r.packed_tokens).sum::<i64>() as f64
                / runs.len() as f64,
        };

        if json_out {
            outs.push(JsonOut {
                docs: corpus.docs.len(),
                queries: corpus.queries.len(),
                budget,
                tau_low,
                ece_uncalibrated: ece_u,
                ece_calibrated: ece_c,
                brier_uncalibrated: br_u,
                brier_calibrated: br_c,
                aggregate: agg,
                queries_detail: rows,
            });
            continue;
        }

        println!("\n=== budget: {} tokens | tau_low: {} ===", budget, tau_low);
        println!("Ranking quality (BM25 + cosine -> RRF):");
        report("nDCG@10", &ndcg_vals, "");
        report("Recall@20", &recall_vals, "");
        report("MRR", &mrr_vals, "");
        println!("Packed-context quality (knapsack DP):");
        report("Precision", &pack_p_vals, "");
        report("Recall", &pack_r_vals, "");
        report("BCY@B", &bcy_vals, "<- baseline");
        report("rho-naive", &rho_naive_vals, "");
        report(
            "rho-insample",
            &[agg.rho_insample],
            "(leaky, reference only)",
        );
        report(
            "rho-HOLDOUT",
            &[agg.rho_holdout],
            &format!("[CI {:.4}-{:.4}] <- honest", rho_lo, rho_hi),
        );
        println!("Calibration trust evidence:");
        println!(
            "  {:<22} ECE {:.4}  Brier {:.4}",
            "uncalibrated (minmax)", ece_u, br_u
        );
        println!(
            "  {:<22} ECE {:.4}  Brier {:.4}",
            "isotonic-calibrated", ece_c, br_c
        );
        println!(
            "  avg packed tokens {:.1}",
            agg.avg_packed_tokens
        );
        println!("Per-query detail (decision rule @ tau={:.2}):", tau_low);
        for (r, row) in runs.iter().zip(rows.iter()) {
            println!(
                "  {:<6} packP={:.2} packR={:.2} BCY={:.2} rho~{:.3} ho~{:.3} knee@k={} sat={} -> {}",
                row.id,
                r.packed_precision,
                r.packed_recall,
                r.bcy,
                r.rho_naive,
                r.rho_holdout,
                r.knee_k,
                r.saturated,
                r.advisory
            );
        }
    }

    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&outs).expect("serialize results")
        );
    }
}

