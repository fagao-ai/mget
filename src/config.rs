use std::{env, fs, path::PathBuf};

use directories::{BaseDirs, ProjectDirs};
use serde::Deserialize;

use crate::{cli::DownloadArgs, error::Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub hf_token: Option<String>,
    pub modelscope_token: Option<String>,
    pub default_threads: usize,
    pub cache_dir: PathBuf,
}

#[derive(Debug, Default, Deserialize)]
struct FileConfig {
    hf_token: Option<String>,
    modelscope_token: Option<String>,
    default_threads: Option<usize>,
    cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct EffectiveConfig {
    pub hf_token: Option<String>,
    pub modelscope_token: Option<String>,
    pub threads: usize,
    pub cache_dir: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self> {
        let file = read_file_config()?;
        let default_cache = BaseDirs::new()
            .map(|dirs| dirs.home_dir().join(".cache").join("mget"))
            .unwrap_or_else(|| PathBuf::from(".mget-cache"));

        Ok(Self {
            hf_token: env_first(&["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"]).or(file.hf_token),
            modelscope_token: env_first(&["MODELSCOPE_API_TOKEN"]).or(file.modelscope_token),
            default_threads: file.default_threads.unwrap_or(6).max(1),
            cache_dir: file.cache_dir.unwrap_or(default_cache),
        })
    }

    pub fn effective_for(&self, args: &DownloadArgs) -> EffectiveConfig {
        EffectiveConfig {
            hf_token: args.hf_token.clone().or_else(|| self.hf_token.clone()),
            modelscope_token: args
                .modelscope_token
                .clone()
                .or_else(|| self.modelscope_token.clone()),
            threads: args.threads.unwrap_or(self.default_threads).max(1),
            cache_dir: self.cache_dir.clone(),
        }
    }
}

fn read_file_config() -> Result<FileConfig> {
    let Some(project_dirs) = ProjectDirs::from("", "", "mget") else {
        return Ok(FileConfig::default());
    };
    let path = project_dirs.config_dir().join("config.toml");
    if !path.exists() {
        return Ok(FileConfig::default());
    }
    let content = fs::read_to_string(path)?;
    Ok(toml::from_str(&content)?)
}

fn env_first(keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| env::var(key).ok())
        .filter(|value| !value.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{DownloadArgs, RepoTypeChoice, SourceChoice};

    #[test]
    fn cli_values_override_config_defaults() {
        let config = Config {
            hf_token: Some("file-hf".into()),
            modelscope_token: Some("file-ms".into()),
            default_threads: 4,
            cache_dir: PathBuf::from("/tmp/mget"),
        };
        let args = DownloadArgs {
            repo: Some("org/model".into()),
            model_id: None,
            dataset_id: None,
            repo_type: RepoTypeChoice::Model,
            source: SourceChoice::Auto,
            output: None,
            threads: Some(16),
            revision: "main".into(),
            files: vec![],
            includes: vec![],
            excludes: vec![],
            link: None,
            interactive: false,
            hf_token: Some("cli-hf".into()),
            modelscope_token: None,
            modelscope_id: None,
        };

        let effective = config.effective_for(&args);
        assert_eq!(effective.hf_token.as_deref(), Some("cli-hf"));
        assert_eq!(effective.modelscope_token.as_deref(), Some("file-ms"));
        assert_eq!(effective.threads, 16);
    }
}
