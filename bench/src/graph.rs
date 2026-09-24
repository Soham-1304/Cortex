//! Graph-Augmented Knapsack (GAK) support: sparse file-locality graph.
//!
//! Nodes are corpus docs (index = position in `corpus.docs`). Edges encode
//! file-locality: co-file = 1.0, same-dir = 0.3.
//!
//! History: the GAK packing objective (graph bonus in knapsack values,
//! `--pack gak`) was refuted 2026-09-24 on astropy — effective R@20 0.03
//! vs raw 0.356 at every λ ∈ {0.1,0.2,0.3,0.5}, ρ-HOLDOUT broken. Mechanism:
//! score-mass bonus pools concentrate on big wrong-file neighborhoods
//! (packs the biggest cluster, not the gold cluster). Removed per
//! refutation discipline; this graph stays as the file-locality primitive
//! for the within-file second stage (co-file sets = the within-file unit).

/// CSR adjacency: neighbors of node `i` are
/// `indices[indptr[i]..indptr[i+1]]` with `weights` in the same slots.
#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub weights: Vec<f64>,
}

pub const W_COFILE: f64 = 1.0;
pub const W_SAMEDIR: f64 = 0.3;
/// Max neighbors per node — bounds storage at ≤ 8N edges.
pub const TOP_M: usize = 8;

fn parent_dir(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => "",
    }
}

/// Build the file-locality graph over doc paths.
///
/// Self-edges excluded. Grouped by path/dir (linear), not pairwise: co-file
/// neighbors (w=1.0) fill a node's slots first, same-dir (w=0.3) fill the
/// rest, both in index order for determinism; capped at `TOP_M`.
///
/// Storage ≤ 8N edges of `(u32, f32)`-scale data — kilobytes at 15k docs.
pub fn build_graph(paths: &[String]) -> Graph {
    use std::collections::HashMap;
    let n = paths.len();
    let mut by_path: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut by_dir: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, p) in paths.iter().enumerate() {
        by_path.entry(p.as_str()).or_default().push(i);
        by_dir.entry(parent_dir(p)).or_default().push(i);
    }
    let mut indptr = Vec::with_capacity(n + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    indptr.push(0);
    for i in 0..n {
        let mut count = 0usize;
        if let Some(same) = by_path.get(paths[i].as_str()) {
            for &j in same {
                if j != i && count < TOP_M {
                    indices.push(j);
                    weights.push(W_COFILE);
                    count += 1;
                }
            }
        }
        if count < TOP_M {
            if let Some(same) = by_dir.get(parent_dir(&paths[i])) {
                for &j in same {
                    if j != i && paths[j] != paths[i] && count < TOP_M {
                        indices.push(j);
                        weights.push(W_SAMEDIR);
                        count += 1;
                    }
                }
            }
        }
        indptr.push(indices.len());
    }
    Graph { n, indptr, indices, weights }
}

impl Graph {
    /// `(neighbor, weight)` pairs for node `i`.
    pub fn neighbors(&self, i: usize) -> Vec<(usize, f64)> {
        let (s, e) = (self.indptr[i], self.indptr[i + 1]);
        self.indices[s..e]
            .iter()
            .zip(self.weights[s..e].iter())
            .map(|(&j, &w)| (j, w))
            .collect()
    }

    pub fn edge_count(&self) -> usize {
        self.indices.len()
    }
}

/// File → member doc indices. The within-file unit for the second stage:
/// stage 1 finds files, stage 2 ranks symbols inside them.
pub fn file_members(paths: &[String]) -> std::collections::HashMap<String, Vec<usize>> {
    let mut m: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, p) in paths.iter().enumerate() {
        m.entry(p.clone()).or_default().push(i);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cofile_and_samedir_weights() {
        let paths = vec![
            "pkg/a.py".to_string(),
            "pkg/a.py".to_string(),
            "pkg/b.py".to_string(),
            "other/c.py".to_string(),
        ];
        let g = build_graph(&paths);
        let n0: std::collections::HashMap<usize, f64> =
            g.neighbors(0).into_iter().collect();
        assert_eq!(n0.get(&1), Some(&W_COFILE));
        assert_eq!(n0.get(&2), Some(&W_SAMEDIR));
        assert!(!n0.contains_key(&3));
        assert!(!n0.contains_key(&0));
    }

    #[test]
    fn root_level_files_share_empty_dir() {
        let paths = vec!["a.py".to_string(), "b.py".to_string()];
        let g = build_graph(&paths);
        assert_eq!(g.neighbors(0), vec![(1, W_SAMEDIR)]);
    }

    #[test]
    fn top_m_capping_prefers_cofile() {
        // 3 co-file + 8 same-dir candidates, cap 8 -> all 3 co-file + 5 same-dir.
        let mut paths = vec![
            "d/f.py".to_string(),
            "d/f.py".to_string(),
            "d/f.py".to_string(),
            "d/f.py".to_string(),
        ];
        for i in 0..8 {
            paths.push(format!("d/g{i}.py"));
        }
        let g = build_graph(&paths);
        let n0 = g.neighbors(0);
        assert_eq!(n0.len(), TOP_M);
        let cofile = n0.iter().filter(|(_, w)| *w == W_COFILE).count();
        assert_eq!(cofile, 3);
    }

    #[test]
    fn isolated_node_has_no_neighbors() {
        let paths = vec!["a/x.py".to_string(), "b/y.py".to_string()];
        let g = build_graph(&paths);
        assert!(g.neighbors(0).is_empty());
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn storage_bounded_by_8n() {
        let paths: Vec<String> =
            (0..50).map(|i| format!("pkg/mod{}.py", i % 5)).collect();
        let g = build_graph(&paths);
        assert!(g.edge_count() <= 8 * g.n);
    }

    #[test]
    fn file_members_groups_indices() {
        let paths = vec![
            "a/x.py".to_string(),
            "a/x.py".to_string(),
            "b/y.py".to_string(),
        ];
        let m = file_members(&paths);
        assert_eq!(m.len(), 2);
        assert_eq!(m["a/x.py"], vec![0, 1]);
        assert_eq!(m["b/y.py"], vec![2]);
    }
}
