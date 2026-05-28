use std::{io, path::PathBuf};

use thiserror::Error;

pub type Result<T> = std::result::Result<T, MgetError>;

#[derive(Debug, Error)]
pub enum MgetError {
    #[error("network request failed: {0}")]
    Network(#[from] reqwest::Error),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("failed to parse response: {0}")]
    Json(#[from] serde_json::Error),
    #[error("failed to parse config file: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid glob pattern: {0}")]
    Glob(#[from] globset::Error),
    #[error("no available source after network probing")]
    NoAvailableSource,
    #[error(
        "model mapping is ambiguous for '{input}'. Pass --modelscope-id or rerun with --interactive in a TTY. Candidates: {candidates}"
    )]
    AmbiguousModelMapping { input: String, candidates: String },
    #[error(
        "no ModelScope mapping found for '{0}'. Pass --modelscope-id to choose the target repository."
    )]
    ModelMappingNotFound(String),
    #[error("source returned no downloadable files for '{0}'")]
    EmptyRepository(String),
    #[error(
        "not enough free disk space at {path}: need {needed} bytes, available {available} bytes"
    )]
    NotEnoughSpace {
        path: PathBuf,
        needed: u64,
        available: u64,
    },
    #[error("checksum mismatch for {path}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("download failed after retries: {0}")]
    DownloadFailed(String),
    #[error("symlink target already exists and points elsewhere: {0}")]
    SymlinkConflict(PathBuf),
    #[error("{0}")]
    Message(String),
}

impl From<dialoguer::Error> for MgetError {
    fn from(value: dialoguer::Error) -> Self {
        Self::Message(value.to_string())
    }
}
