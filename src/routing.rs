use std::time::{Duration, Instant};

use futures_util::future::join_all;
use reqwest::{Client, StatusCode};

use crate::{
    cli::SourceChoice,
    error::{MgetError, Result},
    source::{RepoType, SourceKind},
};

const PROBE_TIMEOUT: Duration = Duration::from_millis(2500);
// Cloudflare TLS handshakes through a local proxy can exceed one second.
const MIRROR_PROBE_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub source: SourceKind,
    pub latency: Option<Duration>,
    pub error: Option<String>,
}

impl ProbeResult {
    pub fn is_available(&self) -> bool {
        self.latency.is_some()
    }
}

pub async fn print_ping() -> Result<()> {
    let results = probe_all().await;
    println!("[1/1] Diagnosing local network environment...");
    for result in &results {
        match result.latency {
            Some(latency) => println!(
                "* {:<18}: {}ms (available)",
                result.source,
                latency.as_millis()
            ),
            None => println!(
                "* {:<18}: unavailable ({})",
                result.source,
                result.error.as_deref().unwrap_or("unknown error")
            ),
        }
    }
    let recommended = choose_best(&results)?;
    println!("Recommended source: {recommended}");
    Ok(())
}

/// Returns every usable source in preference order. The first entry is what
/// the downloader tries first; the rest are fallbacks when a repository lookup
/// fails (for example a HF-only model that ModelScope does not have).
pub async fn candidate_sources(
    choice: SourceChoice,
    repo_type: RepoType,
) -> Result<Vec<SourceKind>> {
    match choice {
        SourceChoice::Hf => Ok(vec![SourceKind::HuggingFace]),
        SourceChoice::HfMirror => {
            // hf-mirror.com has had extended outages. Probe both mirrors at
            // once, but keep the Cloudflare-backed .net domain as a fallback
            // when .com answers the probe but its repository API fails.
            let results = probe_endpoints_with_timeout(
                &[
                    (SourceKind::HfMirror, "https://hf-mirror.com"),
                    (SourceKind::HfMirrorNet, "https://hf-mirror.net"),
                ],
                MIRROR_PROBE_TIMEOUT,
            )
            .await;
            let candidates = results
                .iter()
                .filter(|result| result.is_available())
                .map(|result| result.source)
                .collect::<Vec<_>>();
            if !candidates.is_empty() {
                return Ok(candidates);
            }
            Err(MgetError::Message(format!(
                "no HF mirror available: hf-mirror.com ({}) and hf-mirror.net ({}) both failed",
                results[0].error.as_deref().unwrap_or("unknown error"),
                results[1].error.as_deref().unwrap_or("unknown error"),
            )))
        }
        SourceChoice::HfMirrorNet => Ok(vec![SourceKind::HfMirrorNet]),
        SourceChoice::Modelscope => Ok(vec![SourceKind::ModelScope]),
        SourceChoice::Auto => {
            let mut results = probe_all().await;
            if repo_type == RepoType::Dataset {
                let modelscope_available = results
                    .iter()
                    .any(|result| result.source == SourceKind::ModelScope && result.is_available());
                results.retain(|result| {
                    matches!(
                        result.source,
                        SourceKind::HuggingFace | SourceKind::HfMirror | SourceKind::HfMirrorNet
                    )
                });
                if modelscope_available && !results.iter().any(ProbeResult::is_available) {
                    return Err(MgetError::Message(
                        "dataset auto only probes Hugging Face/HF Mirror; pass --source modelscope for ModelScope datasets".into(),
                    ));
                }
            }
            let candidates = rank_available(&results);
            if candidates.is_empty() {
                return Err(MgetError::NoAvailableSource);
            }
            Ok(candidates)
        }
    }
}

pub async fn probe_all() -> Vec<ProbeResult> {
    probe_endpoints(&[
        (SourceKind::HuggingFace, "https://huggingface.co"),
        (SourceKind::HfMirror, "https://hf-mirror.com"),
        (SourceKind::HfMirrorNet, "https://hf-mirror.net"),
        (SourceKind::ModelScope, "https://modelscope.cn"),
    ])
    .await
}

async fn probe_endpoints(endpoints: &[(SourceKind, &str)]) -> Vec<ProbeResult> {
    probe_endpoints_with_timeout(endpoints, PROBE_TIMEOUT).await
}

async fn probe_endpoints_with_timeout(
    endpoints: &[(SourceKind, &str)],
    timeout: Duration,
) -> Vec<ProbeResult> {
    let client = Client::builder()
        .timeout(timeout)
        .user_agent(crate::download::USER_AGENT)
        .build()
        .expect("valid reqwest client");
    join_all(
        endpoints
            .iter()
            .copied()
            .map(|(source, endpoint)| probe_one(client.clone(), source, endpoint)),
    )
    .await
}

pub fn choose_best(results: &[ProbeResult]) -> Result<SourceKind> {
    rank_available(results)
        .into_iter()
        .next()
        .ok_or(MgetError::NoAvailableSource)
}

fn rank_available(results: &[ProbeResult]) -> Vec<SourceKind> {
    let mut available = results
        .iter()
        .filter_map(|result| result.latency.map(|latency| (result.source, latency)))
        .collect::<Vec<_>>();

    if results.iter().any(|result| {
        result.source == SourceKind::HuggingFace
            && result
                .latency
                .is_some_and(|latency| latency <= Duration::from_millis(200))
            && result.is_available()
    }) {
        available.sort_by_key(|(source, latency)| {
            (u8::from(*source != SourceKind::HuggingFace), *latency)
        });
    } else {
        available.sort_by_key(|(_, latency)| *latency);
    }

    available.into_iter().map(|(source, _)| source).collect()
}

async fn probe_one(client: Client, source: SourceKind, endpoint: &str) -> ProbeResult {
    let start = Instant::now();
    let response = client.head(endpoint).send().await;
    match response {
        Ok(resp) if is_probe_success(resp.status()) => ProbeResult {
            source,
            latency: Some(start.elapsed()),
            error: None,
        },
        Ok(resp) => ProbeResult {
            source,
            latency: None,
            error: Some(format!("HTTP {}", resp.status())),
        },
        Err(err) => ProbeResult {
            source,
            latency: None,
            error: Some(err.to_string()),
        },
    }
}

fn is_probe_success(status: StatusCode) -> bool {
    status.is_success() || status.is_redirection() || status == StatusCode::METHOD_NOT_ALLOWED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_fast_hugging_face_under_threshold() {
        let results = vec![
            ProbeResult {
                source: SourceKind::HuggingFace,
                latency: Some(Duration::from_millis(100)),
                error: None,
            },
            ProbeResult {
                source: SourceKind::ModelScope,
                latency: Some(Duration::from_millis(10)),
                error: None,
            },
        ];
        assert_eq!(choose_best(&results).unwrap(), SourceKind::HuggingFace);
    }

    #[test]
    fn chooses_lowest_latency_when_hf_is_slow() {
        let results = vec![
            ProbeResult {
                source: SourceKind::HuggingFace,
                latency: Some(Duration::from_millis(300)),
                error: None,
            },
            ProbeResult {
                source: SourceKind::HfMirror,
                latency: Some(Duration::from_millis(80)),
                error: None,
            },
        ];
        assert_eq!(choose_best(&results).unwrap(), SourceKind::HfMirror);
    }

    #[test]
    fn chooses_hf_mirror_net_when_it_is_the_only_available_source() {
        let results = vec![
            ProbeResult {
                source: SourceKind::HfMirror,
                latency: None,
                error: Some("connection timed out".into()),
            },
            ProbeResult {
                source: SourceKind::HfMirrorNet,
                latency: Some(Duration::from_millis(900)),
                error: None,
            },
        ];
        assert_eq!(choose_best(&results).unwrap(), SourceKind::HfMirrorNet);
    }

    #[test]
    fn ranks_all_available_sources_for_fallback() {
        let results = vec![
            ProbeResult {
                source: SourceKind::HuggingFace,
                latency: Some(Duration::from_millis(150)),
                error: None,
            },
            ProbeResult {
                source: SourceKind::HfMirrorNet,
                latency: Some(Duration::from_millis(50)),
                error: None,
            },
            ProbeResult {
                source: SourceKind::ModelScope,
                latency: Some(Duration::from_millis(10)),
                error: None,
            },
        ];
        assert_eq!(
            rank_available(&results),
            vec![
                SourceKind::HuggingFace,
                SourceKind::ModelScope,
                SourceKind::HfMirrorNet
            ]
        );
    }

    #[test]
    fn slow_hugging_face_is_ranked_by_latency() {
        let results = vec![
            ProbeResult {
                source: SourceKind::HuggingFace,
                latency: Some(Duration::from_millis(300)),
                error: None,
            },
            ProbeResult {
                source: SourceKind::HfMirrorNet,
                latency: Some(Duration::from_millis(50)),
                error: None,
            },
        ];
        assert_eq!(
            rank_available(&results),
            vec![SourceKind::HfMirrorNet, SourceKind::HuggingFace]
        );
    }

    #[tokio::test]
    async fn explicit_modelscope_still_allowed_for_datasets() {
        assert_eq!(
            candidate_sources(SourceChoice::Modelscope, RepoType::Dataset)
                .await
                .unwrap(),
            vec![SourceKind::ModelScope]
        );
    }
}
