use std::collections::BTreeMap;

use crate::Result;

/// CORTEX.md — the Tier-CORE root memory file (spec §3).
/// Sections are `## Name` headers; anything before the first header is preserved
/// verbatim as the preamble (minus the regenerated `# CORTEX` title).
///
/// Writes go ONLY through permission-gated callers (CLI `core edit`,
/// MCP approval flow); this layer guarantees atomic replacement so a crash
/// never corrupts the file.
#[derive(Default)]
pub struct CoreFile {
    pub path: String,
    pub preamble: Option<String>,
    pub sections: BTreeMap<String, String>,
}

pub const SKELETON: &str =
    "# CORTEX\n\n## Profile\n(describe yourself, your stack, preferences)\n\n## Rules\n(one rule per line)\n";

impl CoreFile {
    pub fn load(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut sections = BTreeMap::new();
        let mut preamble_lines: Vec<&str> = Vec::new();
        let mut cur: Option<String> = None;
        let mut buf = String::new();
        for line in text.lines() {
            if let Some(name) = line.strip_prefix("## ") {
                if let Some(prev) = cur.take() {
                    sections.insert(prev, buf.trim_end().to_string() + "\n");
                    buf.clear();
                }
                cur = Some(name.trim().to_string());
            } else if cur.is_some() {
                buf.push_str(line);
                buf.push('\n');
            } else if line.trim() != "# CORTEX" || !preamble_lines.is_empty() {
                preamble_lines.push(line);
            }
        }
        let preamble = if preamble_lines.is_empty() {
            None
        } else {
            let joined = preamble_lines.join("\n");
            let trimmed = joined.trim_end();
            if trimmed.is_empty() { None } else { Some(trimmed.to_string() + "\n") }
        };
        if let Some(last) = cur.take() {
            sections.insert(last, buf.trim_end().to_string() + "\n");
        }
        Ok(Self { path: path.to_string(), preamble, sections })
    }

    /// Writes SKELETON atomically, then loads it.
    pub fn init_skeleton(path: &str) -> Result<Self> {
        let c = Self { path: path.to_string(), preamble: None, sections: BTreeMap::new() };
        c.save_with(SKELETON)?;
        Self::load(path)
    }

    pub fn section(&self, name: &str) -> Option<&String> {
        self.sections.get(name)
    }

    pub fn set_section(&mut self, name: &str, body: &str) {
        self.sections.insert(name.to_string(), body.to_string());
    }

    /// Section bodies concatenated in sorted order, for context injection.
    pub fn render(&self) -> String {
        let mut out = String::new();
        for body in self.sections.values() {
            out.push_str(body);
        }
        out
    }

    /// Permission-gating lives in callers (CLI `core edit`, MCP approval flow).
    /// This method is atomic: writes to `{path}.tmp` then renames, so a crash
    /// never leaves a corrupted CORTEX.md.
    pub fn save(&self) -> Result<()> {
        let mut out = String::from("# CORTEX\n");
        if let Some(pre) = &self.preamble {
            out.push('\n');
            out.push_str(pre);
            out.push('\n');
        }
        for (name, body) in &self.sections {
            out.push_str("\n## ");
            out.push_str(name);
            out.push('\n');
            out.push_str(body);
        }
        self.save_with(&out)
    }

    fn save_with(&self, contents: &str) -> Result<()> {
        let tmp = format!("{}.tmp", self.path);
        if let Err(e) = std::fs::write(&tmp, contents) {
            let _ = std::fs::remove_file(&tmp);
            return Err(e.into());
        }
        std::fs::rename(&tmp, &self.path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_path(name: &str) -> String {
        let p = std::env::temp_dir().join(name).to_string_lossy().to_string();
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn roundtrip_preserves_unknown_sections() {
        let p = tmp_path("cx_test_core_rt");
        std::fs::write(&p, "# CORTEX\n\n## Profile\nlikes rust\n\n## Rules\nno secrets\n").unwrap();
        let mut c = CoreFile::load(&p).unwrap();
        assert_eq!(c.section("Profile").map(|s| s.as_str()), Some("likes rust\n"));
        c.set_section("Rules", "no secrets\nbe terse\n");
        c.save().unwrap();
        let out = std::fs::read_to_string(&p).unwrap();
        assert!(out.contains("likes rust"));
        assert!(out.contains("be terse"));
    }

    #[test]
    fn save_is_atomic_on_failure() {
        let p = "/tmp/cx_test_core_missing_dir/x.md".to_string();
        let _ = std::fs::remove_dir_all("/tmp/cx_test_core_missing_dir");
        let c = CoreFile { path: p.clone(), preamble: None, sections: Default::default() };
        assert!(c.save().is_err());
        assert!(!std::path::Path::new("/tmp/cx_test_core_missing_dir/x.md").exists());
    }

    #[test]
    fn skeleton_created_by_init() {
        let p = tmp_path("cx_test_core_init");
        CoreFile::init_skeleton(&p).unwrap();
        let c = CoreFile::load(&p).unwrap();
        assert!(c.section("Profile").is_some() && c.section("Rules").is_some());
    }

    #[test]
    fn render_concatenates_sections() {
        let mut c = CoreFile { path: "x".into(), preamble: None, sections: Default::default() };
        c.set_section("B", "beta\n");
        c.set_section("A", "alpha\n");
        let r = c.render();
        assert!(r.contains("alpha") && r.contains("beta"));
        // A sorts before B (BTreeMap): alpha appears before beta
        assert!(r.find("alpha").unwrap() < r.find("beta").unwrap());
    }

    #[test]
    fn content_before_first_section_preserved() {
        let p = tmp_path("cx_test_core_preamble");
        std::fs::write(&p, "# CORTEX\nfree-form preamble line\n\n## Profile\nbody\n").unwrap();
        let c = CoreFile::load(&p).unwrap();
        c.save().unwrap();
        let out = std::fs::read_to_string(&p).unwrap();
        assert!(out.contains("free-form preamble line"));
        assert!(out.contains("body"));
    }
}
