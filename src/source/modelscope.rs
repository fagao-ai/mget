use crate::{
    download::USER_AGENT as MGET_USER_AGENT,
    error::Result,
    source::{ModelSource, RemoteFile, ResolvedModel, SourceKind},
};
use async_trait::async_trait;
use reqwest::{
    Client,
    header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue, USER_AGENT},
};

const DEFAULT_MODELSCOPE_REVISION: &str = "master";
const REQUEST_ID_HEADER: &str = "X-Request-ID";

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
        let owner_or_group = query.split('/').next().unwrap_or(query);
        let response = self
            .client
            .put(format!("{}/api/v1/models/", self.base_url))
            .headers(self.auth_headers())
            .header(CONTENT_TYPE, "application/json")
            .body(format!(
                r#"{{"Path":"{}","PageNumber":1,"PageSize":100}}"#,
                owner_or_group
            ))
            .send()
            .await?
            .error_for_status()?;
        let value: serde_json::Value = response.json().await?;
        let mut ids = extract_model_ids(&value);
        if query.contains('/') {
            ids.sort_by_key(|id| if id == query { 0 } else { 1 });
        }
        Ok(ids)
    }

    async fn list_files_for_revision(
        &self,
        model: &ResolvedModel,
        revision: Option<&str>,
    ) -> Result<Vec<RemoteFile>> {
        let mut request = self
            .client
            .get(self.repo_files_url(model))
            .query(&[("Recursive", "true")])
            .headers(self.auth_headers());
        if let Some(revision) = revision {
            request = request.query(&[("Revision", revision)]);
        }
        let response = request.send().await?.error_for_status()?;
        let value: serde_json::Value = response.json().await?;
        Ok(extract_files(&value))
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
        if let Ok(request_id) = HeaderValue::from_str(&uuid::Uuid::new_v4().simple().to_string()) {
            headers.insert(REQUEST_ID_HEADER, request_id);
        }
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
            revision: if revision == "main" {
                DEFAULT_MODELSCOPE_REVISION.to_string()
            } else {
                revision.to_string()
            },
        })
    }

    async fn list_files(&self, model: &ResolvedModel) -> Result<Vec<RemoteFile>> {
        let files = self
            .list_files_for_revision(model, Some(&model.revision))
            .await?;
        if files.is_empty() && model.revision == "master" {
            return self.list_files_for_revision(model, None).await;
        }
        Ok(files)
    }

    fn download_url(&self, model: &ResolvedModel, file: &RemoteFile) -> String {
        format!(
            "{}/api/v1/models/{}/repo?Revision={}&FilePath={}",
            self.base_url,
            model.source_id,
            urlencoding::encode(&model.revision),
            urlencoding::encode_binary(file.path.as_bytes()).replace("%20", "+")
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
            for key in [
                "ModelId", "modelId", "model_id", "Path", "path", "Name", "name",
            ] {
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
            let name = ["Name", "name"]
                .iter()
                .find_map(|key| map.get(*key).and_then(|value| value.as_str()));
            let path = ["Path", "path", "FilePath", "filePath", "Name", "name"]
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
                && !matches!(name, Some(".gitignore" | ".gitattributes"))
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
