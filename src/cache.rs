use std::path::{Component, Path, PathBuf};

use directories::BaseDirs;
use tokio::fs;

use crate::{
    cli::LinkChoice,
    error::{MgetError, Result},
    source::{RepoType, ResolvedModel, SourceKind},
};

pub fn default_output_root(
    explicit_output: Option<&Path>,
    cache_dir: &Path,
    source: SourceKind,
    model: &ResolvedModel,
) -> PathBuf {
    explicit_output.map_or_else(
        || {
            let root = cache_dir.join(source.cache_segment());
            let root = if matches!(model.repo_type, RepoType::Dataset) {
                root.join(model.repo_type.cache_segment())
            } else {
                root
            };
            root.join(sanitize_repo_id(&model.source_id))
                .join(sanitize_component(&model.revision))
        },
        Path::to_path_buf,
    )
}

pub async fn create_compat_links(
    choice: Option<LinkChoice>,
    source: SourceKind,
    model: &ResolvedModel,
    target: &Path,
) -> Result<()> {
    let Some(choice) = choice else {
        return Ok(());
    };

    let mut destinations = Vec::new();
    if matches!(choice, LinkChoice::Hf | LinkChoice::All) {
        destinations.push(hf_cache_path(model));
    }
    if matches!(choice, LinkChoice::Modelscope | LinkChoice::All) {
        destinations.push(modelscope_cache_path(model));
    }

    for destination in destinations.into_iter().flatten() {
        create_one_link(&destination, target).await?;
    }

    tracing::debug!(?source, "compatibility links created");
    Ok(())
}

pub fn sanitize_repo_id(repo: &str) -> PathBuf {
    repo.split('/')
        .filter(|part| !part.is_empty())
        .map(sanitize_component)
        .collect()
}

fn sanitize_component(component: &str) -> String {
    component
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            ch => ch,
        })
        .collect()
}

fn hf_cache_path(model: &ResolvedModel) -> Option<PathBuf> {
    let repo_prefix = match model.repo_type {
        RepoType::Model => "models",
        RepoType::Dataset => "datasets",
    };
    BaseDirs::new().map(|dirs| {
        dirs.home_dir()
            .join(".cache")
            .join("huggingface")
            .join("hub")
            .join(format!(
                "{repo_prefix}--{}",
                model.requested_id.replace('/', "--")
            ))
            .join("snapshots")
            .join(sanitize_component(&model.revision))
    })
}

fn modelscope_cache_path(model: &ResolvedModel) -> Option<PathBuf> {
    BaseDirs::new().map(|dirs| {
        let root = dirs
            .home_dir()
            .join(".cache")
            .join("modelscope")
            .join("hub")
            .join(sanitize_repo_id(&model.source_id));
        if matches!(model.repo_type, RepoType::Dataset) {
            dirs.home_dir()
                .join(".cache")
                .join("modelscope")
                .join("hub")
                .join("datasets")
                .join(sanitize_repo_id(&model.source_id))
        } else {
            root.join(sanitize_component(&model.revision))
        }
    })
}

async fn create_one_link(destination: &Path, target: &Path) -> Result<()> {
    match fs::symlink_metadata(destination).await {
        Ok(_) => {
            let existing = fs::read_link(destination).await.ok();
            if existing.as_deref() == Some(target) {
                return Ok(());
            }
            return Err(MgetError::SymlinkConflict(destination.to_path_buf()));
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => return Err(err.into()),
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).await?;
    }
    symlink_dir(target, destination).await
}

#[cfg(unix)]
async fn symlink_dir(target: &Path, link: &Path) -> Result<()> {
    let target = target.to_path_buf();
    let link = link.to_path_buf();
    tokio::task::spawn_blocking(move || std::os::unix::fs::symlink(target, link))
        .await
        .map_err(|err| MgetError::Message(err.to_string()))??;
    Ok(())
}

#[cfg(windows)]
async fn symlink_dir(target: &Path, link: &Path) -> Result<()> {
    let target = target.to_path_buf();
    let link = link.to_path_buf();
    tokio::task::spawn_blocking(move || std::os::windows::fs::symlink_dir(target, link))
        .await
        .map_err(|err| MgetError::Message(err.to_string()))??;
    Ok(())
}

pub fn ensure_relative_repo_path(path: &str) -> Result<PathBuf> {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute()
        || candidate
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::Prefix(_)))
    {
        return Err(MgetError::Message(format!(
            "repository file path is unsafe: {path}"
        )));
    }
    Ok(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_repo_into_nested_path() {
        assert_eq!(
            sanitize_repo_id("meta-llama/Meta-Llama-3").to_string_lossy(),
            "meta-llama/Meta-Llama-3"
        );
    }

    #[test]
    fn rejects_parent_dir_paths() {
        assert!(ensure_relative_repo_path("../secret").is_err());
        assert!(ensure_relative_repo_path("ok/file.txt").is_ok());
    }

    #[test]
    fn hf_link_uses_requested_id_after_modelscope_mapping() {
        let model = ResolvedModel {
            requested_id: "meta-llama/Model".into(),
            source_id: "LLM-Research/Model".into(),
            repo_type: RepoType::Model,
            revision: "main".into(),
        };
        let path = hf_cache_path(&model).unwrap();
        assert!(path.to_string_lossy().contains("models--meta-llama--Model"));
        assert!(!path.to_string_lossy().contains("LLM-Research"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dangling_symlink_is_reported_as_conflict() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("target");
        let link = temp.path().join("link");
        std::os::unix::fs::symlink(temp.path().join("missing"), &link).unwrap();

        assert!(matches!(
            create_one_link(&link, &target).await,
            Err(MgetError::SymlinkConflict(path)) if path == link
        ));
    }
}
