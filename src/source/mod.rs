pub mod huggingface;
pub mod modelscope;

use std::fmt;

use async_trait::async_trait;
use reqwest::header::HeaderMap;

use crate::error::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepoType {
    Model,
    Dataset,
}

impl RepoType {
    pub fn label(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Dataset => "dataset",
        }
    }

    pub fn cache_segment(self) -> &'static str {
        match self {
            Self::Model => "models",
            Self::Dataset => "datasets",
        }
    }
}

impl fmt::Display for RepoType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceKind {
    HuggingFace,
    HfMirror,
    ModelScope,
}

impl SourceKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "Hugging Face",
            Self::HfMirror => "HF Mirror",
            Self::ModelScope => "ModelScope",
        }
    }

    pub fn cache_segment(self) -> &'static str {
        match self {
            Self::HuggingFace => "huggingface",
            Self::HfMirror => "hf-mirror",
            Self::ModelScope => "modelscope",
        }
    }
}

impl fmt::Display for SourceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub requested_id: String,
    pub source_id: String,
    pub repo_type: RepoType,
    pub revision: String,
}

#[derive(Debug, Clone)]
pub struct RemoteFile {
    pub path: String,
    pub size: Option<u64>,
    pub sha256: Option<String>,
    pub md5: Option<String>,
}

#[async_trait]
pub trait ModelSource: Send + Sync {
    fn source_kind(&self) -> SourceKind;
    fn auth_headers(&self) -> HeaderMap;
    async fn resolve_model(
        &self,
        model: &str,
        repo_type: RepoType,
        revision: &str,
    ) -> Result<ResolvedModel>;
    async fn list_files(&self, model: &ResolvedModel) -> Result<Vec<RemoteFile>>;
    fn download_url(&self, model: &ResolvedModel, file: &RemoteFile) -> String;
}
