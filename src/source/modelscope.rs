use crate::{
    download::USER_AGENT as MGET_USER_AGENT,
    error::Result,
    source::{ModelSource, RemoteFile, ResolvedModel, SourceKind},
};
use async_trait::async_trait;
use reqwest::{
    Client,
    header::{AUTHORIZATION, HeaderMap, HeaderValue, USER_AGENT},
};

#[derive(Debug, Clone)]
pub struct ModelScopeSource {
    client: Client,
    base_url: String,
    token: Option<String>,
}

impl ModelScopeSource {
    pub fn new(token: Option<String>) -> Self {
        Self::with_base_url("https://modelscope.cn", token)
    }

    pub fn with_base_url(base_url: impl Into<String>, token: Option<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token,
        }
    }

    pub async fn search_models(&self, query: &str) -> Result<Vec<String>> {
        let response = self
            .client
            .get(format!("{}/api/v1/models", self.base_url))
            .query(&[("Name", query), ("PageSize", "5")])
            .headers(self.auth_headers())
            .send()
            .await?
            .error_for_status()?;
        let value: serde_json::Value = response.json().await?;
        Ok(extract_model_ids(&value))
    }

    fn repo_files_url(&self, model: &ResolvedModel) -> String {
        format!(
            "{}/api/v1/models/{}/repo/files",
            self.base_url, model.source_id
        )
    }
}

#[async_trait]
impl ModelSource for ModelScopeSource {
    fn source_kind(&self) -> SourceKind {
        SourceKind::ModelScope
    }

    fn auth_headers(&self) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static(MGET_USER_AGENT));
        if let Some(token) = &self.token
            && let Ok(value) = HeaderValue::from_str(&format!("Bearer {token}"))
        {
            headers.insert(AUTHORIZATION, value);
        }
        headers
    }

    async fn resolve_model(&self, model: &str, revision: &str) -> Result<ResolvedModel> {
        Ok(ResolvedModel {
            requested_id: model.to_string(),
            source_id: model.to_string(),
            revision: revision.to_string(),
        })
    }

    async fn list_files(&self, model: &ResolvedModel) -> Result<Vec<RemoteFile>> {
        let response = self
            .client
            .get(self.repo_files_url(model))
            .query(&[("Revision", model.revision.as_str()), ("Recursive", "true")])
            .headers(self.auth_headers())
            .send()
            .await?
            .error_for_status()?;
        let value: serde_json::Value = response.json().await?;
        Ok(extract_files(&value))
    }

    fn download_url(&self, model: &ResolvedModel, file: &RemoteFile) -> String {
        format!(
            "{}/api/v1/models/{}/repo?Revision={}&FilePath={}",
            self.base_url,
            model.source_id,
            urlencoding::encode(&model.revision),
            urlencoding::encode(&file.path)
        )
    }
}

fn extract_model_ids(value: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    collect_model_ids(value, &mut out);
    out.sort();
    out.dedup();
    out
}

fn collect_model_ids(value: &serde_json::Value, out: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for key in ["ModelId", "modelId", "Path", "path", "Name", "name"] {
                if let Some(serde_json::Value::String(id)) = map.get(key)
                    && id.contains('/')
                {
                    out.push(id.clone());
                }
            }
            for child in map.values() {
                collect_model_ids(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_model_ids(item, out);
            }
        }
        _ => {}
    }
}

fn extract_files(value: &serde_json::Value) -> Vec<RemoteFile> {
    let mut out = Vec::new();
    collect_files(value, &mut out);
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn collect_files(value: &serde_json::Value, out: &mut Vec<RemoteFile>) {
    match value {
        serde_json::Value::Object(map) => {
            let path = ["Path", "path", "Name", "name", "FilePath", "filePath"]
                .iter()
                .find_map(|key| map.get(*key).and_then(|value| value.as_str()));
            let is_dir = ["Type", "type"]
                .iter()
                .find_map(|key| map.get(*key).and_then(|value| value.as_str()))
                .is_some_and(|kind| {
                    kind.eq_ignore_ascii_case("tree") || kind.eq_ignore_ascii_case("dir")
                });
            if let Some(path) = path
                && !is_dir
                && !path.ends_with('/')
                && path.contains('.')
            {
                out.push(RemoteFile {
                    path: path.to_string(),
                    size: ["Size", "size"]
                        .iter()
                        .find_map(|key| map.get(*key).and_then(|value| value.as_u64())),
                    sha256: ["Sha256", "sha256", "SHA256"]
                        .iter()
                        .find_map(|key| map.get(*key).and_then(|value| value.as_str()))
                        .map(ToOwned::to_owned),
                    md5: ["Md5", "md5", "MD5"]
                        .iter()
                        .find_map(|key| map.get(*key).and_then(|value| value.as_str()))
                        .map(ToOwned::to_owned),
                });
            }
            for child in map.values() {
                collect_files(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_files(item, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_model_ids_from_nested_search_payload() {
        let value = serde_json::json!({
            "Data": {
                "Model": [
                    { "ModelId": "org/model-a" },
                    { "Path": "org/model-b" },
                    { "Name": "plain-name" }
                ]
            }
        });
        let ids = extract_model_ids(&value);
        assert_eq!(ids, vec!["org/model-a", "org/model-b"]);
    }

    #[test]
    fn extracts_files_from_nested_payload() {
        let value = serde_json::json!({
            "Data": {
                "Files": [
                    { "Path": "README.md", "Size": 10, "Sha256": "abc" },
                    { "Path": "folder", "Type": "tree" }
                ]
            }
        });
        let files = extract_files(&value);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "README.md");
        assert_eq!(files[0].size, Some(10));
        assert_eq!(files[0].sha256.as_deref(), Some("abc"));
    }
}
