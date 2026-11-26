use clap::Parser;
use std::path::PathBuf;
use stdarch_gen_common::{GeneratorCtx, Mode, run_generator};

mod generate;

/// RISC-V intrinsic spec generator.
///
/// Produces the *.spec files in this directory by parsing the C headers in which they are defined.
///
/// Run in check or bless mode via `STDARCH_GEN_MODE`.
#[derive(clap::Parser)]
struct Args {
    /// Generation mode.
    #[arg(long, env = "STDARCH_GEN_MODE")]
    mode: Option<Mode>,
    /// Path to a rustfmt binary that will be used to reformat the generated code.
    /// If unset, it will just use "rustfmt" from the environment.
    #[arg(long)]
    rustfmt_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let crate_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap());

    // Either "check" to check the output versus the committed output, or "bless"
    // to update the output.
    let mode = args.mode.unwrap_or_default();
    let ctx = GeneratorCtx::new(args.rustfmt_path);

    run_generator(&ctx, &crate_dir, mode, |out_dir| -> anyhow::Result<()> {
        generate::generate(&crate_dir, out_dir, "bitmanip")?;
        generate::generate(&crate_dir, out_dir, "crypto")?;
        Ok(())
    })?;

    Ok(())
}
