use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

use crate::source::RepoType;

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
        assert_eq!(
            args.repo_id().as_deref(),
            Some("meta-llama/Meta-Llama-3-8B-Instruct")
        );
        assert_eq!(args.repo_type(), RepoType::Model);
        assert_eq!(args.source, SourceChoice::HfMirror);
        assert_eq!(args.threads, Some(16));
        assert_eq!(args.files, vec!["README.md"]);
    }

    #[test]
    fn parses_modelscope_dataset_forms() {
        let positional = Cli::parse_from([
            "mget",
            "download",
            "modelscope/test-dataset",
            "--repo-type",
            "dataset",
        ]);
        let Commands::Download(args) = positional.command else {
            panic!("expected download command");
        };
        assert_eq!(args.repo_id().as_deref(), Some("modelscope/test-dataset"));
        assert_eq!(args.repo_type(), RepoType::Dataset);

        let explicit = Cli::parse_from(["mget", "download", "--datasets", "owner/data"]);
        let Commands::Download(args) = explicit.command else {
            panic!("expected download command");
        };
        assert_eq!(args.repo_id().as_deref(), Some("owner/data"));
        assert_eq!(args.repo_type(), RepoType::Dataset);
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
    /// Download a model or dataset repository, or selected files.
    #[command(alias = "dl")]
    Download(Box<DownloadArgs>),
}

#[derive(Debug, Clone, Args)]
pub struct DownloadArgs {
    /// Hugging Face style repository id, e.g. meta-llama/Meta-Llama-3-8B-Instruct.
    #[arg(value_name = "REPO_ID", required_unless_present_any = ["model_id", "dataset_id"])]
    pub repo: Option<String>,
    /// Explicit model repository id. Useful for ModelScope CLI compatibility.
    #[arg(
        long = "model",
        value_name = "MODEL_ID",
        conflicts_with_all = ["repo", "dataset_id"]
    )]
    pub model_id: Option<String>,
    /// Explicit dataset repository id. Useful for ModelScope CLI compatibility.
    #[arg(
        long = "dataset",
        visible_alias = "datasets",
        value_name = "DATASET_ID",
        conflicts_with_all = ["repo", "model_id"]
    )]
    pub dataset_id: Option<String>,
    /// Repository type for positional REPO_ID.
    #[arg(long = "repo-type", value_enum, default_value_t = RepoTypeChoice::Model)]
    pub repo_type: RepoTypeChoice,
    /// Select source explicitly or use smart routing.
    #[arg(short, long, value_enum, default_value_t = SourceChoice::Auto)]
    pub source: SourceChoice,
    /// Output directory. Defaults to the unified mget cache.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Number of concurrent file/chunk downloads.
    #[arg(short = 't', long)]
    pub threads: Option<usize>,
    /// Repository revision, branch, or commit.
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
    /// Explicit ModelScope model id to use for model-id mapping.
    #[arg(long)]
    pub modelscope_id: Option<String>,
}

impl DownloadArgs {
    pub fn repo_id(&self) -> Option<String> {
        self.model_id
            .clone()
            .or_else(|| self.dataset_id.clone())
            .or_else(|| self.repo.clone())
    }

    pub fn repo_type(&self) -> RepoType {
        if self.dataset_id.is_some() {
            RepoType::Dataset
        } else if self.model_id.is_some() {
            RepoType::Model
        } else {
            self.repo_type.into()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SourceChoice {
    Auto,
    Hf,
    HfMirror,
    Modelscope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum RepoTypeChoice {
    Model,
    Dataset,
}

impl From<RepoTypeChoice> for RepoType {
    fn from(value: RepoTypeChoice) -> Self {
        match value {
            RepoTypeChoice::Model => Self::Model,
            RepoTypeChoice::Dataset => Self::Dataset,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LinkChoice {
    Hf,
    Modelscope,
    All,
}
