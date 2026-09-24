pub mod bm25;
pub mod calib;
pub mod contextbench;
pub mod corpus;
pub mod embed;
pub mod eval;
pub mod fuse;
pub mod graph;
pub mod pack;
pub mod provider;
pub mod symbols;

pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.is_empty() {
            continue;
        }
        // Keep the full compound (exact-match behavior preserved), plus
        // identifier parts for recall (snake/camel/digit splits).
        let full = raw.to_lowercase();
        out.push(full.clone());
        for part in split_ident_parts(raw) {
            if part != full {
                out.push(part);
            }
        }
    }
    out
}

/// Identifier parts of one alphanumeric+underscore run (original case):
/// snake segments, camel humps (with acronym-run handling), and
/// letterdigit boundaries. All output lowercase, nonempty, deduped.
fn split_ident_parts(raw: &str) -> Vec<String> {
    let mut parts = Vec::new();
    for seg in raw.split('_') {
        if seg.is_empty() {
            continue;
        }
        parts.extend(split_camel(seg));
    }
    let mut out = Vec::new();
    for p in parts {
        for piece in split_digit_bounds(&p) {
            let l = piece.to_lowercase();
            if !l.is_empty() && !out.contains(&l) {
                out.push(l);
            }
        }
    }
    out
}

/// Split camel humps: boundary before an uppercase letter following a
/// lowercase/digit, or before the last capital of an acronym run when a
/// lowercase follows (HTTPServer -> HTTP | Server). Operates on the
/// original case; caller lowercases.
fn split_camel(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut pieces = Vec::new();
    let mut start = 0usize;
    for i in 1..chars.len() {
        let prev = chars[i - 1];
        let cur = chars[i];
        let next = chars.get(i + 1).copied();
        let boundary = cur.is_uppercase()
            && (prev.is_lowercase()
                || prev.is_numeric()
                || (prev.is_uppercase() && next.is_some_and(|n| n.is_lowercase())));
        if boundary {
            pieces.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    pieces.push(chars[start..].iter().collect());
    pieces
}

/// Split letterdigit runs: col1 -> [col, 1], utf8 -> [utf, 8].
fn split_digit_bounds(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut pieces = Vec::new();
    let mut start = 0usize;
    for i in 1..chars.len() {
        if chars[i].is_numeric() != chars[i - 1].is_numeric() {
            pieces.push(chars[start..i].iter().collect());
            start = i;
        }
    }
    pieces.push(chars[start..].iter().collect());
    pieces
}

pub fn count_tokens(text: &str) -> i64 {
    tokenize(text).len().max(1) as i64
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        tokenize(s)
    }

    fn has(t: &[String], w: &str) -> bool {
        t.iter().any(|x| x == w)
    }

    #[test]
    fn plain_text_unchanged() {
        assert_eq!(toks("hello world"), vec!["hello", "world"]);
        assert!(toks("").is_empty());
        assert_eq!(toks("READ SERR 3"), vec!["read", "serr", "3"]);
    }

    #[test]
    fn snake_keeps_full_and_parts() {
        let t = toks("separability_matrix");
        assert!(has(&t, "separability_matrix"));
        assert!(has(&t, "separability"));
        assert!(has(&t, "matrix"));
    }

    #[test]
    fn camel_splits_humps() {
        let t = toks("readSerr");
        assert!(has(&t, "read"));
        assert!(has(&t, "serr"));
        let t = toks("SkyCoord");
        assert!(has(&t, "sky"));
        assert!(has(&t, "coord"));
    }

    #[test]
    fn acronym_run_splits_before_last_capital() {
        let t = toks("HTTPServer");
        assert!(has(&t, "http"));
        assert!(has(&t, "server"));
        // All-caps acronyms stay whole.
        assert_eq!(toks("SERR"), vec!["serr"]);
        assert_eq!(toks("FITS"), vec!["fits"]);
    }

    #[test]
    fn digit_boundaries_split() {
        let t = toks("col1");
        assert!(has(&t, "col1"));
        assert!(has(&t, "col"));
        assert!(has(&t, "1"));
    }

    #[test]
    fn mixed_snake_camel_digits() {
        let t = toks("read_serr1");
        assert!(has(&t, "read_serr1"));
        assert!(has(&t, "read"));
        assert!(has(&t, "serr"));
        assert!(has(&t, "1"));
    }
}
