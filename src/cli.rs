use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "mget",
    version,
    about = "Intelligent multi-source LLM downloader"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn parses_download_alias_and_options() {
        let cli = Cli::parse_from([
            "mget",
            "dl",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "--source",
            "hf-mirror",
            "-t",
            "16",
            "--file",
            "README.md",
        ]);
        let Commands::Download(args) = cli.command else {
            panic!("expected download command");
        };
        assert_eq!(args.source, SourceChoice::HfMirror);
        assert_eq!(args.threads, Some(16));
        assert_eq!(args.files, vec!["README.md"]);
    }

    #[test]
    fn parses_ping_command() {
        let cli = Cli::parse_from(["mget", "ping"]);
        assert!(matches!(cli.command, Commands::Ping));
    }
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Diagnose current network latency and recommend a source.
    Ping,
    /// Download a model repository or selected files.
    #[command(alias = "dl")]
    Download(Box<DownloadArgs>),
}

#[derive(Debug, Clone, Args)]
pub struct DownloadArgs {
    /// Hugging Face style model id, e.g. meta-llama/Meta-Llama-3-8B-Instruct.
    pub model: String,
    /// Select source explicitly or use smart routing.
    #[arg(short, long, value_enum, default_value_t = SourceChoice::Auto)]
    pub source: SourceChoice,
    /// Output directory. Defaults to the unified mget cache.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Number of concurrent file/chunk downloads.
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
    /// Model revision, branch, or commit.
    #[arg(long, default_value = "main")]
    pub revision: String,
    /// Download only these exact repository paths. Can be repeated.
    #[arg(long = "file")]
    pub files: Vec<String>,
    /// Include glob patterns. Can be repeated.
    #[arg(long = "include")]
    pub includes: Vec<String>,
    /// Exclude glob patterns. Can be repeated.
    #[arg(long = "exclude")]
    pub excludes: Vec<String>,
    /// Create compatibility symlinks after download.
    #[arg(long, value_enum)]
    pub link: Option<LinkChoice>,
    /// Allow interactive ModelScope candidate selection in a TTY.
    #[arg(long)]
    pub interactive: bool,
    /// Hugging Face token. Overrides env and config file.
    #[arg(long)]
    pub hf_token: Option<String>,
    /// ModelScope token. Overrides env and config file.
    #[arg(long)]
    pub modelscope_token: Option<String>,
    /// Explicit ModelScope repository id to use when source is ModelScope.
    #[arg(long)]
    pub modelscope_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SourceChoice {
    Auto,
    Hf,
    HfMirror,
    Modelscope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LinkChoice {
    Hf,
    Modelscope,
    All,
}
