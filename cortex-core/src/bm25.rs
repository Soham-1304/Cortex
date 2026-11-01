use std::collections::HashMap;

use crate::{tokenize, visit_tokens};

pub const K1: f64 = 1.2;
pub const B: f64 = 0.75;
/// Okapi BM25 query-term saturation constant ($k_3$).
/// Note: for $qtf = 1$, $\frac{(K3 + 1) \cdot 1}{K3 + 1} = 1.0$ (exact single-term parity),
/// while repeated prose words in long issue descriptions saturate cleanly at $K3 + 1 = 2.2$.
pub const K3: f64 = 1.2;

pub struct Bm25Index {
    postings: HashMap<String, Vec<(u32, u32)>>,
    doc_len: Vec<u32>,
    avgdl: f64,
    n: usize,
}

impl Bm25Index {
    pub fn build(texts: &[String]) -> Bm25Index {
        let n = texts.len();
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        let mut doc_len = Vec::with_capacity(n);
        let mut total_len = 0usize;
        let mut local_tf: HashMap<String, u32> = HashMap::new();

        for (doc_id, text) in texts.iter().enumerate() {
            local_tf.clear();
            let mut dl = 0u32;
            visit_tokens(text, |tok| {
                dl = dl.saturating_add(1);
                if let Some(c) = local_tf.get_mut(tok) {
                    *c = c.saturating_add(1);
                } else {
                    local_tf.insert(tok.to_string(), 1);
                }
            });
            total_len += dl as usize;
            doc_len.push(dl);
            for (term, tf) in local_tf.drain() {
                postings.entry(term).or_default().push((doc_id as u32, tf));
            }
        }

        let avgdl = if n > 0 { total_len as f64 / n as f64 } else { 0.0 };
        Bm25Index {
            postings,
            doc_len,
            avgdl,
            n,
        }
    }

    pub fn score(&self, query: &str) -> Vec<f64> {
        let mut scores = vec![0.0; self.n];
        if self.n == 0 {
            return scores;
        }
        let mut qtf: HashMap<String, usize> = HashMap::new();
        let mut qorder: Vec<String> = Vec::new();
        for qt in tokenize(query) {
            if let Some(c) = qtf.get_mut(&qt) {
                *c += 1;
            } else {
                qtf.insert(qt.clone(), 1);
                qorder.push(qt);
            }
        }
        let inv_avgdl = if self.avgdl > 0.0 { 1.0 / self.avgdl } else { 0.0 };
        for qt in &qorder {
            let Some(plist) = self.postings.get(qt) else {
                continue;
            };
            let n_t = plist.len();
            if n_t == 0 {
                continue;
            }
            let idf = (1.0 + (self.n as f64 - n_t as f64 + 0.5) / (n_t as f64 + 0.5)).ln();
            let qf = qtf[qt] as f64;
            let qw = ((K3 + 1.0) * qf) / (K3 + qf);
            let term_weight = idf * qw;
            for &(doc_id, tf) in plist {
                let f = tf as f64;
                let dl = self.doc_len[doc_id as usize] as f64;
                let len_norm = if inv_avgdl > 0.0 { dl * inv_avgdl } else { 1.0 };
                scores[doc_id as usize] +=
                    term_weight * (f * (K1 + 1.0)) / (f + K1 * (1.0 - B + B * len_norm));
            }
        }
        scores
    }

    pub fn ranking(&self, query: &str) -> Vec<usize> {
        let scores = self.score(query);
        let mut idx: Vec<usize> = (0..self.n).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        idx
    }

    /// Document frequency of a term over the indexed corpus (for the
    /// exact-identifier refs leg in `fuse`). Same df universe as scoring.
    pub fn doc_freq(&self, term: &str) -> usize {
        self.postings.get(term).map_or(0, |v| v.len())
    }

    pub fn doc_count(&self) -> usize {
        self.n
    }
}

/// Multi-field Robertson BM25F query scorer with zero heap allocation per scored symbol.
///
/// Combines length-normalized term frequencies across structured code fields
/// (`name`, `path`, `signature`, `body`) into a single pseudo-frequency $\tilde{f}_t$
/// BEFORE applying Okapi $K_1$ saturation:
/// $$\tilde{f}_t = \sum_c w_c \frac{f_{t,c}}{1 - b_c + b_c \cdot (dl_c / \text{avgdl}_c)}, \quad
///   S(d, Q) = \sum_{t \in Q} \text{idf}(t) \cdot q_w(t) \cdot \frac{\tilde{f}_t (K_1 + 1)}{\tilde{f}_t + K_1}$$
pub struct Bm25fQuery {
    /// Maps query token -> `(slot_idx, idf * qweight)`.
    terms: HashMap<String, (usize, f64)>,
    weights: Vec<f64>,
}

/// Structured fields for scoring a code symbol with [`Bm25fQuery`].
#[derive(Clone, Copy)]
pub struct SymbolFields<'a> {
    pub name: &'a str,
    pub path: &'a str,
    pub signature: &'a str,
    pub body: &'a str,
}

/// Corpus-average token lengths per field for BM25F length normalization.
#[derive(Clone, Copy, Debug)]
pub struct FieldLengths {
    pub avg_name: f64,
    pub avg_path: f64,
    pub avg_sig: f64,
    pub avg_body: f64,
}

impl Default for FieldLengths {
    fn default() -> Self {
        Self {
            avg_name: 3.0,
            avg_path: 6.0,
            avg_sig: 12.0,
            avg_body: 80.0,
        }
    }
}

impl Bm25fQuery {
    /// Build a reusable query plan from raw query text and a corpus IDF callback `idf_fn(term)`.
    pub fn new<F: FnMut(&str) -> f64>(query: &str, mut idf_fn: F) -> Self {
        let mut qtf: HashMap<String, usize> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        visit_tokens(query, |tok| {
            if tok.len() < 2 && !tok.chars().all(|c| c.is_ascii_digit()) {
                return;
            }
            if let Some(c) = qtf.get_mut(tok) {
                *c += 1;
            } else {
                let owned = tok.to_string();
                qtf.insert(owned.clone(), 1);
                order.push(owned);
            }
        });
        let mut terms = HashMap::with_capacity(order.len());
        let mut weights = Vec::with_capacity(order.len());
        for tok in order {
            let idf = idf_fn(&tok);
            if idf <= 0.0 {
                continue;
            }
            let qf = qtf[&tok] as f64;
            let qw = ((K3 + 1.0) * qf) / (K3 + qf);
            let slot = weights.len();
            weights.push(idf * qw);
            terms.insert(tok, (slot, idf * qw));
        }
        Self { terms, weights }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    #[inline]
    pub fn term_count(&self) -> usize {
        self.weights.len()
    }

    /// Score one symbol across `(name, path, signature, body)` using scratch buffer `pseudo_tf`
    /// (resized automatically; zero allocations after first call).
    pub fn score_symbol(
        &self,
        fields: SymbolFields<'_>,
        avgs: FieldLengths,
        pseudo_tf: &mut Vec<f64>,
    ) -> f64 {
        let m = self.weights.len();
        if m == 0 {
            return 0.0;
        }
        pseudo_tf.clear();
        pseudo_tf.resize(m, 0.0);

        // Field priors tuned for code localization:
        // symbol identifier > module path > signature > body/docstring.
        self.accum_field(fields.name, 4.0, 0.30, avgs.avg_name.max(1.0), pseudo_tf);
        self.accum_field(fields.path, 2.5, 0.40, avgs.avg_path.max(1.0), pseudo_tf);
        self.accum_field(fields.signature, 2.0, 0.50, avgs.avg_sig.max(1.0), pseudo_tf);
        if !fields.body.is_empty() {
            self.accum_field(fields.body, 1.0, 0.75, avgs.avg_body.max(1.0), pseudo_tf);
        }

        let mut total = 0.0;
        let mut matched_terms = 0usize;
        for (i, &ft) in pseudo_tf.iter().enumerate() {
            if ft > 0.0 {
                matched_terms += 1;
                total += self.weights[i] * (ft * (K1 + 1.0)) / (ft + K1);
            }
        }
        // Subtle multi-term coordination bonus when multiple distinct query concepts co-occur
        // in the same symbol's fields (rewards specific joint hits over single-term matches).
        if matched_terms >= 2 {
            let coord = 1.0 + 0.06 * ((matched_terms - 1).min(5) as f64);
            total *= coord;
        }
        total
    }

    #[inline]
    fn accum_field(
        &self,
        text: &str,
        weight: f64,
        b: f64,
        avgdl: f64,
        pseudo_tf: &mut [f64],
    ) {
        if text.is_empty() {
            return;
        }
        // Two-pass avoided by recording hits in a tiny stack array for typical fields,
        // falling back to direct accumulation when >64 query hits occur in a long body.
        let mut hits = [0usize; 64];
        let mut n_hits = 0usize;
        let mut dl = 0usize;
        visit_tokens(text, |tok| {
            dl += 1;
            if let Some(&(slot, _)) = self.terms.get(tok) {
                if n_hits < hits.len() {
                    hits[n_hits] = slot;
                }
                n_hits += 1;
            }
        });
        if n_hits == 0 {
            return;
        }
        let denom = (1.0 - b + b * (dl as f64 / avgdl)).max(0.25);
        let scale = weight / denom;
        if n_hits <= hits.len() {
            for &slot in &hits[..n_hits] {
                pseudo_tf[slot] += scale;
            }
        } else {
            visit_tokens(text, |tok| {
                if let Some(&(slot, _)) = self.terms.get(tok) {
                    pseudo_tf[slot] += scale;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_doc_single_term_exact_value() {
        let idx = Bm25Index::build(&["alpha beta".to_string()]);
        let scores = idx.score("alpha");
        assert!((scores[0] - (4.0f64 / 3.0).ln()).abs() < 1e-12);
    }

    #[test]
    fn nonmatching_query_scores_zero() {
        let idx = Bm25Index::build(&["alpha beta".to_string(), "gamma delta".to_string()]);
        for s in idx.score("epsilon") {
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn matching_doc_ranks_above_nonmatching() {
        let idx = Bm25Index::build(&[
            "unrelated content here".to_string(),
            "jwt token expiry validation".to_string(),
        ]);
        let r = idx.ranking("jwt expiry");
        assert_eq!(r[0], 1);
    }

    #[test]
    fn all_empty_corpus_no_nan() {
        let idx = Bm25Index::build(&["".to_string(), "".to_string()]);
        for s in idx.score("anything") {
            assert!(s.is_finite());
            assert_eq!(s, 0.0);
        }
    }

    #[test]
    fn bm25f_prefers_symbol_name_over_body_mention_and_saturates() {
        let q = Bm25fQuery::new("URLValidator ipv6", |tok| match tok {
            "urlvalidator" => 5.5,
            "ipv" => 4.2,
            "url" => 2.5,
            "validator" => 2.8,
            _ => 1.0,
        });
        let avgs = FieldLengths::default();
        let mut scratch = Vec::new();
        let exact_sym = q.score_symbol(
            SymbolFields {
                name: "URLValidator",
                path: "django/core/validators.py",
                signature: "class URLValidator(RegexValidator):",
                body: "Checks ipv6 hostnames and schemes.",
            },
            avgs,
            &mut scratch,
        );
        let noisy_body = q.score_symbol(
            SymbolFields {
                name: "test_other",
                path: "tests/misc/test_helpers.py",
                signature: "def test_other(self):",
                body: "url url url validator validator urlvalidator mentioned in passing",
            },
            avgs,
            &mut scratch,
        );
        assert!(
            exact_sym > noisy_body,
            "exact_sym ({exact_sym}) should beat noisy_body ({noisy_body})"
        );
    }
}
