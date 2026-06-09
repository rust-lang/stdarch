//! Shared check/bless harness for stdarch generators.

use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Write,
    Check,
    Bless,
}

impl Mode {
    pub fn from_env() -> Self {
        match std::env::var("STDARCH_GEN_MODE").as_deref() {
            Ok("check") => Mode::Check,
            Ok("bless") => Mode::Bless,
            _ => Mode::Write,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Mismatch { path: PathBuf, kind: MismatchKind },
    Generator(Box<dyn StdError + Send + Sync>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchKind {
    MissingInCommitted,
    ExtraInCommitted,
    ContentsDiffer,
    MissingFromGenerated,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O error: {e}"),
            Error::Mismatch { path, kind } => match kind {
                MismatchKind::MissingInCommitted => {
                    write!(f, "{}: generated but not committed", path.display())
                }
                MismatchKind::ExtraInCommitted => {
                    write!(f, "{}: committed but no longer generated", path.display())
                }
                MismatchKind::ContentsDiffer => write!(f, "{}: contents differ", path.display()),
                MismatchKind::MissingFromGenerated => write!(
                    f,
                    "{}: declared owned but generator did not produce it",
                    path.display()
                ),
            },
            Error::Generator(e) => write!(f, "generator failed: {e}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Generator(e) => Some(&**e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

// owned is the list of relative paths the generator manages. Files in
// committed not listed here are left untouched by Bless and ignored by Check.
pub fn run<F, E>(committed: &Path, owned: &[&str], mode: Mode, generate: F) -> Result<()>
where
    F: FnOnce(&Path) -> std::result::Result<(), E>,
    E: Into<Box<dyn StdError + Send + Sync>>,
{
    match mode {
        Mode::Write => {
            fs::create_dir_all(committed)?;
            generate(committed).map_err(|e| Error::Generator(e.into()))
        }
        Mode::Check => {
            let scratch = tempfile::tempdir()?;
            generate(scratch.path()).map_err(|e| Error::Generator(e.into()))?;
            compare(scratch.path(), committed, owned)
        }
        Mode::Bless => {
            let scratch = tempfile::tempdir()?;
            generate(scratch.path()).map_err(|e| Error::Generator(e.into()))?;
            apply_bless(scratch.path(), committed, owned)
        }
    }
}

fn compare(generated: &Path, committed: &Path, owned: &[&str]) -> Result<()> {
    for rel in owned {
        let rel_path = PathBuf::from(rel);
        let gen_path = generated.join(&rel_path);
        let comm_path = committed.join(&rel_path);
        match (gen_path.exists(), comm_path.exists()) {
            (true, false) => {
                return Err(Error::Mismatch {
                    path: rel_path,
                    kind: MismatchKind::MissingInCommitted,
                });
            }
            (false, true) => {
                return Err(Error::Mismatch {
                    path: rel_path,
                    kind: MismatchKind::ExtraInCommitted,
                });
            }
            (false, false) => continue,
            (true, true) => {
                let g = fs::read(&gen_path)?;
                let c = fs::read(&comm_path)?;
                if g != c {
                    return Err(Error::Mismatch {
                        path: rel_path,
                        kind: MismatchKind::ContentsDiffer,
                    });
                }
            }
        }
    }
    Ok(())
}

fn apply_bless(scratch: &Path, committed: &Path, owned: &[&str]) -> Result<()> {
    fs::create_dir_all(committed)?;
    for rel in owned {
        let rel_path = PathBuf::from(rel);
        let from = scratch.join(&rel_path);
        if !from.exists() {
            return Err(Error::Mismatch {
                path: rel_path,
                kind: MismatchKind::MissingFromGenerated,
            });
        }
        let to = committed.join(&rel_path);
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&from, &to)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(p: &Path, b: &[u8]) {
        if let Some(d) = p.parent() {
            fs::create_dir_all(d).unwrap();
        }
        fs::write(p, b).unwrap();
    }

    #[test]
    fn write_mode_creates_committed() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        run(
            &committed,
            &["a.txt"],
            Mode::Write,
            |out| -> std::result::Result<(), io::Error> {
                write(&out.join("a.txt"), b"hi");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(fs::read(committed.join("a.txt")).unwrap(), b"hi");
    }

    #[test]
    fn check_passes_when_identical() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write(&committed.join("a.txt"), b"hi");
        run(
            &committed,
            &["a.txt"],
            Mode::Check,
            |out| -> std::result::Result<(), io::Error> {
                write(&out.join("a.txt"), b"hi");
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn check_fails_on_byte_diff() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write(&committed.join("a.txt"), b"hi");
        let e = run(
            &committed,
            &["a.txt"],
            Mode::Check,
            |out| -> std::result::Result<(), io::Error> {
                write(&out.join("a.txt"), b"HI");
                Ok(())
            },
        )
        .unwrap_err();
        assert!(matches!(
            e,
            Error::Mismatch {
                kind: MismatchKind::ContentsDiffer,
                ..
            }
        ));
    }

    #[test]
    fn check_ignores_unowned_committed_files() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write(&committed.join("mod.rs"), b"hand-written");
        write(&committed.join("a.txt"), b"hi");
        run(
            &committed,
            &["a.txt"],
            Mode::Check,
            |out| -> std::result::Result<(), io::Error> {
                write(&out.join("a.txt"), b"hi");
                Ok(())
            },
        )
        .unwrap();
    }

    #[test]
    fn check_fails_when_owned_file_missing_from_generated() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write(&committed.join("a.txt"), b"hi");
        let e = run(
            &committed,
            &["a.txt"],
            Mode::Check,
            |_| -> std::result::Result<(), io::Error> { Ok(()) },
        )
        .unwrap_err();
        assert!(matches!(
            e,
            Error::Mismatch {
                kind: MismatchKind::ExtraInCommitted,
                ..
            }
        ));
    }

    #[test]
    fn bless_preserves_unowned_files() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        write(&committed.join("mod.rs"), b"hand-written");
        write(&committed.join("old.txt"), b"old");
        run(
            &committed,
            &["new.txt"],
            Mode::Bless,
            |out| -> std::result::Result<(), io::Error> {
                write(&out.join("new.txt"), b"new");
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(fs::read(committed.join("mod.rs")).unwrap(), b"hand-written");
        assert_eq!(fs::read(committed.join("old.txt")).unwrap(), b"old");
        assert_eq!(fs::read(committed.join("new.txt")).unwrap(), b"new");
    }

    #[test]
    fn bless_fails_when_generator_drops_owned_file() {
        let tmp = tempfile::tempdir().unwrap();
        let committed = tmp.path().join("c");
        let e = run(
            &committed,
            &["a.txt"],
            Mode::Bless,
            |_| -> std::result::Result<(), io::Error> { Ok(()) },
        )
        .unwrap_err();
        assert!(matches!(
            e,
            Error::Mismatch {
                kind: MismatchKind::MissingFromGenerated,
                ..
            }
        ));
    }
}
