use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// The Rust source file path argument
    #[arg(short, long)]
    pub c: String,

    /// The Rust source file path argument
    #[arg(short, long)]
    pub rust: String,
}
