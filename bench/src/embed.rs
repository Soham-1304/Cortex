use crate::tokenize;

pub const DIM: usize = 64;

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub fn embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for tok in tokenize(text) {
        let h = splitmix64(fnv1a64(&tok));
        let idx = (h % DIM as u64) as usize;
        let sign = if h >> 63 == 0 { 1.0f32 } else { -1.0f32 };
        v[idx] += sign;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn ranking(query_vec: &[f32], doc_vecs: &[Vec<f32>]) -> Vec<usize> {
    let mut sims: Vec<(usize, f32)> =
        doc_vecs.iter().enumerate().map(|(i, d)| (i, cosine(query_vec, d))).collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        assert_eq!(embed("alpha beta gamma"), embed("alpha beta gamma"));
    }

    #[test]
    fn unit_norm() {
        let v = embed("alpha beta gamma delta");
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn same_text_similarity_one() {
        let s = cosine(&embed("jwt token expiry"), &embed("jwt token expiry"));
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn different_text_less_similar_than_same() {
        let a = embed("jwt token expiry validation");
        let same = embed("jwt token expiry validation");
        let diff = embed("render pdf invoice tax breakdown");
        assert!(cosine(&a, &same) > cosine(&a, &diff));
    }

    #[test]
    fn empty_text_zero_vector() {
        let v = embed("");
        assert!(v.iter().all(|x| *x == 0.0));
    }
}
