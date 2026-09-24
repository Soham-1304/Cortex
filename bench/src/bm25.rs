use std::collections::HashMap;

use crate::tokenize;

pub const K1: f64 = 1.2;
pub const B: f64 = 0.75;

pub struct Bm25Index {
    tf: Vec<HashMap<String, usize>>,
    df: HashMap<String, usize>,
    avgdl: f64,
    n: usize,
}

impl Bm25Index {
    pub fn build(texts: &[String]) -> Bm25Index {
        let docs_tokens: Vec<Vec<String>> = texts.iter().map(|t| tokenize(t)).collect();
        let mut tf = Vec::with_capacity(docs_tokens.len());
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;
        for toks in &docs_tokens {
            total_len += toks.len();
            let mut m: HashMap<String, usize> = HashMap::new();
            for t in toks {
                *m.entry(t.clone()).or_insert(0) += 1;
            }
            for t in m.keys() {
                *df.entry(t.clone()).or_insert(0) += 1;
            }
            tf.push(m);
        }
        let n = docs_tokens.len();
        let avgdl = if n > 0 { total_len as f64 / n as f64 } else { 0.0 };
        Bm25Index { tf, df, avgdl, n }
    }

    pub fn score(&self, query: &str) -> Vec<f64> {
        let qtoks = tokenize(query);
        let mut scores = vec![0.0; self.n];
        for (i, tfm) in self.tf.iter().enumerate() {
            let dl = tfm.values().sum::<usize>() as f64;
            let mut s = 0.0;
            for qt in &qtoks {
                let n_t = *self.df.get(qt).unwrap_or(&0);
                if n_t == 0 {
                    continue;
                }
                let idf = (1.0 + (self.n as f64 - n_t as f64 + 0.5) / (n_t as f64 + 0.5)).ln();
                let f = *tfm.get(qt).unwrap_or(&0) as f64;
                if f == 0.0 {
                    continue;
                }
                let len_norm = if self.avgdl > 0.0 { dl / self.avgdl } else { 1.0 };
                s += idf * (f * (K1 + 1.0))
                    / (f + K1 * (1.0 - B + B * len_norm));
            }
            scores[i] = s;
        }
        scores
    }

    pub fn ranking(&self, query: &str) -> Vec<usize> {
        let scores = self.score(query);
        let mut idx: Vec<usize> = (0..self.n).collect();
        idx.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    }

    /// Document frequency of a term over the indexed corpus (for the
    /// exact-identifier refs leg in `fuse`). Same df universe as scoring.
    pub fn doc_freq(&self, term: &str) -> usize {
        self.df.get(term).copied().unwrap_or(0)
    }

    pub fn doc_count(&self) -> usize {
        self.n
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
}
