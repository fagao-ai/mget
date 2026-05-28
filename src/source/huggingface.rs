use async_trait::async_trait;
use reqwest::{
    Client,
    header::{AUTHORIZATION, HeaderMap, HeaderValue, USER_AGENT},
};
use serde::Deserialize;

use crate::{
    download::USER_AGENT as MGET_USER_AGENT,
    error::Result,
    source::{ModelSource, RemoteFile, ResolvedModel, SourceKind},
};

#[derive(Debug, Clone)]
pub struct HuggingFaceSource {
    client: Client,
    kind: SourceKind,
    base_url: String,
    token: Option<String>,
}

impl HuggingFaceSource {
    pub fn new(kind: SourceKind, token: Option<String>) -> Self {
        let base_url = match kind {
            SourceKind::HuggingFace => "https://huggingface.co",
            SourceKind::HfMirror => "https://hf-mirror.com",
            SourceKind::ModelScope => unreachable!("use ModelScopeSource"),
        };
        Self::with_base_url(kind, base_url, token)
    }

    pub fn with_base_url(
        kind: SourceKind,
        base_url: impl Into<String>,
        token: Option<String>,
    ) -> Self {
        Self {
            client: Client::new(),
            kind,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            token,
        }
    }

    fn model_api_url(&self, model: &str, revision: &str) -> String {
        format!(
            "{}/api/models/{}/revision/{}",
            self.base_url,
            model,
            urlencoding::encode(revision)
        )
    }
}

#[async_trait]
impl ModelSource for HuggingFaceSource {
    fn source_kind(&self) -> SourceKind {
        self.kind
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
            .get(self.model_api_url(&model.source_id, &model.revision))
            .headers(self.auth_headers())
            .send()
            .await?
            .error_for_status()?;
        let model_info: HfModelInfo = response.json().await?;
        Ok(model_info
            .siblings
            .into_iter()
            .filter(|sibling| !sibling.rfilename.ends_with('/'))
            .map(|sibling| RemoteFile {
                path: sibling.rfilename,
                size: sibling.size,
                sha256: sibling
                    .lfs
                    .as_ref()
                    .and_then(|lfs| lfs.sha256.clone())
                    .or(sibling.sha256),
                md5: None,
            })
            .collect())
    }

    fn download_url(&self, model: &ResolvedModel, file: &RemoteFile) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.base_url,
            model.source_id,
            urlencoding::encode(&model.revision),
            file.path
                .split('/')
                .map(urlencoding::encode)
                .collect::<Vec<_>>()
                .join("/")
        )
    }
}

#[derive(Debug, Deserialize)]
struct HfModelInfo {
    #[serde(default)]
    siblings: Vec<HfSibling>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    rfilename: String,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    sha256: Option<String>,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Debug, Deserialize)]
struct HfLfs {
    #[serde(default)]
    sha256: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ModelSource;

    #[tokio::test]
    async fn lists_hf_siblings_from_api_response() {
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/models/org/model/revision/main"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "siblings": [
                    {
                        "rfilename": "README.md",
                        "size": 5,
                        "lfs": { "sha256": "abc" }
                    }
                ]
            })))
            .mount(&server)
            .await;

        let source = HuggingFaceSource::with_base_url(
            SourceKind::HuggingFace,
            server.uri(),
            Some("secret".into()),
        );
        let model = source.resolve_model("org/model", "main").await.unwrap();
        let files = source.list_files(&model).await.unwrap();

        assert_eq!(files.len(), 1);
        assert_eq!(files[0].path, "README.md");
        assert_eq!(files[0].size, Some(5));
        assert_eq!(files[0].sha256.as_deref(), Some("abc"));
        assert!(
            source
                .download_url(&model, &files[0])
                .ends_with("/org/model/resolve/main/README.md")
        );
    }
}
