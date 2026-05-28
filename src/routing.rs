use std::time::{Duration, Instant};

use futures_util::future::join_all;
use reqwest::{Client, StatusCode};

use crate::{
    cli::SourceChoice,
    error::{MgetError, Result},
    source::SourceKind,
};

const PROBE_TIMEOUT: Duration = Duration::from_millis(2500);

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

pub async fn select_source(choice: SourceChoice) -> Result<SourceKind> {
    match choice {
        SourceChoice::Hf => Ok(SourceKind::HuggingFace),
        SourceChoice::HfMirror => Ok(SourceKind::HfMirror),
        SourceChoice::Modelscope => Ok(SourceKind::ModelScope),
        SourceChoice::Auto => {
            let results = probe_all().await;
            choose_best(&results)
        }
    }
}

pub async fn probe_all() -> Vec<ProbeResult> {
    let client = Client::builder()
        .timeout(PROBE_TIMEOUT)
        .user_agent(crate::download::USER_AGENT)
        .build()
        .expect("valid reqwest client");
    let probes = [
        (SourceKind::HuggingFace, "https://huggingface.co"),
        (SourceKind::HfMirror, "https://hf-mirror.com"),
        (SourceKind::ModelScope, "https://modelscope.cn"),
    ];
    join_all(
        probes
            .into_iter()
            .map(|(source, endpoint)| probe_one(client.clone(), source, endpoint)),
    )
    .await
}

pub fn choose_best(results: &[ProbeResult]) -> Result<SourceKind> {
    if results.iter().any(|result| {
        result.source == SourceKind::HuggingFace
            && result
                .latency
                .is_some_and(|latency| latency <= Duration::from_millis(200))
            && result.is_available()
    }) {
        return Ok(SourceKind::HuggingFace);
    }

    results
        .iter()
        .filter_map(|result| result.latency.map(|latency| (result.source, latency)))
        .min_by_key(|(_, latency)| *latency)
        .map(|(source, _)| source)
        .ok_or(MgetError::NoAvailableSource)
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
}
