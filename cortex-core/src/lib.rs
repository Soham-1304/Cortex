//! cortex-core: retrieval primitives shared by the `cortex` binary and the
//! `bench` research harness (tokenizer, BM25, fusion, packing, symbol
//! extraction, evaluation statistics).

pub mod bm25;
pub mod eval;
pub mod fuse;
pub mod pack;
pub mod symbols;

pub fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    visit_tokens(text, |tok| out.push(tok.to_string()));
    out
}

/// Zero-allocation (for ASCII identifiers <=128 bytes) token visitor that emits
/// the exact same sequence as [`tokenize`]: full lowercased compound first,
/// followed by deduplicated snake/camel/digit subword parts.
pub fn visit_tokens<F: FnMut(&str)>(text: &str, mut emit: F) {
    let mut lower_buf = [0u8; 128];
    let mut spans = [(0usize, 0usize); 32];
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.is_empty() {
            continue;
        }
        if raw.is_ascii() && raw.len() <= 128 {
            let b = raw.as_bytes();
            let len = b.len();
            let mut has_underscore = false;
            let mut has_upper = false;
            let mut has_lower = false;
            let mut has_digit = false;
            for i in 0..len {
                let c = b[i];
                lower_buf[i] = c.to_ascii_lowercase();
                if c == b'_' {
                    has_underscore = true;
                } else if c.is_ascii_uppercase() {
                    has_upper = true;
                } else if c.is_ascii_lowercase() {
                    has_lower = true;
                } else if c.is_ascii_digit() {
                    has_digit = true;
                }
            }
            // SAFETY: `raw` is ASCII and `to_ascii_lowercase` preserves UTF-8 ASCII.
            let full = std::str::from_utf8(&lower_buf[..len]).unwrap_or("");
            emit(full);
            // Fast exit when no subword boundary can exist: no underscore, no mixed
            // case (e.g. all-lower or all-upper), and not mixed alpha+digit.
            let mixed_case = has_upper && has_lower;
            let mixed_alnum = has_digit && (has_upper || has_lower);
            if !has_underscore && !mixed_case && !mixed_alnum {
                continue;
            }
            let mut n_spans = 0usize;
            let mut seg_start = 0usize;
            for idx in 0..=len {
                if idx == len || b[idx] == b'_' {
                    if idx > seg_start {
                        // Camel split over b[seg_start..idx]
                        let mut c_start = seg_start;
                        for i in (seg_start + 1)..idx {
                            let prev = b[i - 1];
                            let cur = b[i];
                            let next = if i + 1 < idx { Some(b[i + 1]) } else { None };
                            let boundary = cur.is_ascii_uppercase()
                                && (prev.is_ascii_lowercase()
                                    || prev.is_ascii_digit()
                                    || (prev.is_ascii_uppercase()
                                        && next.is_some_and(|n| n.is_ascii_lowercase())));
                            if boundary {
                                push_digit_spans(b, c_start, i, full, &mut spans, &mut n_spans, &mut emit);
                                c_start = i;
                            }
                        }
                        push_digit_spans(b, c_start, idx, full, &mut spans, &mut n_spans, &mut emit);
                    }
                    seg_start = idx + 1;
                }
            }
            continue;
        }
        // Unicode or >128-byte identifier fallback.
        let full = raw.to_lowercase();
        emit(&full);
        for part in split_ident_parts(raw) {
            if part != full {
                emit(&part);
            }
        }
    }
}

#[inline]
fn push_digit_spans<F: FnMut(&str)>(
    b: &[u8],
    start: usize,
    end: usize,
    full: &str,
    seen: &mut [(usize, usize); 32],
    n_seen: &mut usize,
    emit: &mut F,
) {
    let mut d_start = start;
    for i in (start + 1)..end {
        if b[i].is_ascii_digit() != b[i - 1].is_ascii_digit() {
            emit_subspan(d_start, i, full, seen, n_seen, emit);
            d_start = i;
        }
    }
    emit_subspan(d_start, end, full, seen, n_seen, emit);
}

#[inline]
fn emit_subspan<F: FnMut(&str)>(
    s: usize,
    e: usize,
    full: &str,
    seen: &mut [(usize, usize); 32],
    n_seen: &mut usize,
    emit: &mut F,
) {
    if e <= s {
        return;
    }
    let sub = &full[s..e];
    if sub == full {
        return;
    }
    for &(ps, pe) in &seen[..*n_seen] {
        if &full[ps..pe] == sub {
            return;
        }
    }
    if *n_seen < seen.len() {
        seen[*n_seen] = (s, e);
        *n_seen += 1;
    }
    emit(sub);
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
    let mut n = 0i64;
    visit_tokens(text, |_| n += 1);
    n.max(1)
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

    #[test]
    fn visit_tokens_matches_split_reference_and_count() {
        let samples = [
            "",
            "URLValidator",
            "HTTPServer_v2_connectionPool",
            "__init__ _private_method",
            "sympy/matrices/expressions/matexpr.py",
            " unicode_αβγ_Name2 ",
        ];
        for s in samples {
            let mut visited = Vec::new();
            visit_tokens(s, |t| visited.push(t.to_string()));
            assert_eq!(visited, tokenize(s), "mismatch on {s:?}");
            let expected_count = visited.len().max(1) as i64;
            assert_eq!(count_tokens(s), expected_count);
        }
    }
}
