//! Shared check/bless harness for stdarch generators.

use std::error::Error as StdError;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Controls what `run` does with the generator's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Write the generator's output directly into `committed`.
    ///
    /// This is the default for standalone runs. `owned` is not consulted
    /// whatever the generator emits lands on disk as-is.
    Write,
    /// Verify that the `committed` matches the generator's output for `owned`.
    ///
    /// Runs the generator into a temp directory, then compares the produced file
    /// against the committed copy. Returns an error on the first mismatch.
    Check,
    /// Update the `committed` to match the generator's output for `owned`.
    ///
    /// Runs the generator into a temp directory and copies the produced file
    /// into `committed`. If the generator no longer produces `owned`, the
    /// committed copy is deleted. Files in `committed` that are not `owned`
    /// are left untouched.
    Bless,
}

impl Mode {
    /// Read the mode from the `STDARCH_GEN_MODE` environment variable.
    ///
    /// Recognized values:
    /// - `"check"` → [`Mode::Check`]
    /// - `"bless"` → [`Mode::Bless`]
    /// - anything else, including unset → [`Mode::Write`]
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
    /// Owned file produced by the generator but absent from the `committed`.
    /// Means the `committed` needs to be regenerated.
    MissingInCommitted,
    /// Owned file present in the `committed` but the generator no longer
    /// produces it. The file must be removed from the `committed` .
    ExtraInCommitted,
    /// Owned file exists on both sides but contents differ.
    ContentsDiffer,
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

/// Run a generator under the chosen `mode`, reconciling its output with `committed`.
///
/// Arguments:
/// - `committed` — the directory holding the in-tree (committed) source files.
/// - `owned` — the file inside `committed` that the generator produces.
///   Anything else in `committed` is treated as hand-written and is
///   left untouched by `Bless` and ignored by `Check`. So generated files
///   coexist with hand-written files in the same  directory.
/// - `mode` — what to do with the generator's output.
/// - `generate` — closure that writes the generator's output into the
///   directory it is given. Its error is wrapped in [`Error::Generator`].
///
/// Behavior per mode:
/// - [`Mode::Write`]: invokes `generate(committed)` directly. owned is
///   not consulted.
/// - [`Mode::Check`]: runs the generator into a temp dir and compares
///   `owned` against the committed copy. Mismatch returns [`Error::Mismatch`].
/// - [`Mode::Bless`]:  runs the generator into a temp dir and copies `owned`
///   into `committed`, or removes `committed`'s copy if the generator no
///   longer produces it.
pub fn run<F, E>(committed: &Path, owned: &str, mode: Mode, generate: F) -> Result<()>
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

fn compare(generated: &Path, committed: &Path, owned: &str) -> Result<()> {
    let rel_path = PathBuf::from(owned);
    let gen_path = generated.join(&rel_path);
    let comm_path = committed.join(&rel_path);
    match (gen_path.exists(), comm_path.exists()) {
        (true, false) => Err(Error::Mismatch {
            path: rel_path,
            kind: MismatchKind::MissingInCommitted,
        }),
        (false, true) => Err(Error::Mismatch {
            path: rel_path,
            kind: MismatchKind::ExtraInCommitted,
        }),
        (false, false) => Ok(()),
        (true, true) => {
            if fs::read(&gen_path)? != fs::read(&comm_path)? {
                Err(Error::Mismatch {
                    path: rel_path,
                    kind: MismatchKind::ContentsDiffer,
                })
            } else {
                Ok(())
            }
        }
    }
}

fn apply_bless(scratch: &Path, committed: &Path, owned: &str) -> Result<()> {
    fs::create_dir_all(committed)?;
    let rel_path = PathBuf::from(owned);
    let from = scratch.join(&rel_path);
    let to = committed.join(&rel_path);
    if from.exists() {
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&from, &to)?;
    } else if to.exists() {
        fs::remove_file(&to)?;
    }
    Ok(())
}

#[cfg(all(test, not(target_os = "ios")))]
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
            "a.txt",
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
            "a.txt",
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
            "a.txt",
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
            "a.txt",
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
            "a.txt",
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
            "new.txt",
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
}
