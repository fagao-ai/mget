use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_util::{StreamExt, stream};
use globset::{Glob, GlobSet, GlobSetBuilder};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use md5::Md5;
use reqwest::{
    Client, StatusCode,
    header::{ACCEPT_RANGES, CONTENT_LENGTH, CONTENT_RANGE, HeaderMap, RANGE},
};
use sha2::{Digest, Sha256};
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom},
    sync::{Mutex, Semaphore},
    time::sleep,
};

use crate::{
    cache,
    cli::DownloadArgs,
    config::{Config, EffectiveConfig},
    error::{MgetError, Result},
    mapping, routing,
    source::{
        ModelSource, RemoteFile, RepoType, ResolvedModel, SourceKind,
        huggingface::HuggingFaceSource, modelscope::ModelScopeSource,
    },
};

pub const USER_AGENT: &str = concat!("mget/", env!("CARGO_PKG_VERSION"));
const LARGE_FILE_CHUNK_THRESHOLD: u64 = 256 * 1024 * 1024;
const DEFAULT_CHUNK_SIZE: u64 = 64 * 1024 * 1024;
const MAX_RETRIES: usize = 3;
const DETAIL_BAR_THRESHOLD: u64 = 8 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
struct DownloadTuning {
    chunk_size: u64,
    range_threshold: u64,
    chunk_threads: usize,
}

const DEFAULT_TUNING: DownloadTuning = DownloadTuning {
    chunk_size: DEFAULT_CHUNK_SIZE,
    range_threshold: LARGE_FILE_CHUNK_THRESHOLD,
    chunk_threads: 4,
};

pub async fn run_download(args: DownloadArgs, config: Config) -> Result<()> {
    let effective = config.effective_for(&args);
    let repo_type = args.repo_type();
    let candidates = routing::candidate_sources(args.source, repo_type).await?;
    let mut last_error = None;
    let mut selected = None;
    for source_kind in candidates {
        let source = build_source(source_kind, &effective);
        let resolved =
            match resolve_model(&args, repo_type, &*source, source_kind, &effective).await {
                Ok(resolved) => resolved,
                Err(err) => {
                    eprintln!("warning: {source_kind} lookup failed: {err}; trying next source");
                    last_error = Some(err);
                    continue;
                }
            };
        match source.list_files(&resolved).await {
            Ok(files) => {
                selected = Some((source_kind, source, resolved, files));
                break;
            }
            Err(err) => {
                eprintln!(
                    "warning: {source_kind} cannot list {}: {err}; trying next source",
                    resolved.source_id
                );
                last_error = Some(err);
            }
        }
    }
    let (source_kind, source, resolved, files) =
        selected.ok_or_else(|| last_error.unwrap_or(MgetError::NoAvailableSource))?;
    if resolved.requested_id != resolved.source_id {
        println!(
            "Mapped repo id: {} -> {}",
            resolved.requested_id, resolved.source_id
        );
    }
    let files = filter_files(files, &args)?;
    if files.is_empty() {
        return Err(MgetError::EmptyRepository(resolved.source_id));
    }
    let files = fill_missing_sizes(&*source, &resolved, files).await?;

    let output_root = cache::default_output_root(
        args.output.as_deref(),
        &effective.cache_dir,
        source_kind,
        &resolved,
    );
    let plan = DownloadPlan::new(&*source, &resolved, files, output_root)?;
    preflight_disk_space(&plan, DEFAULT_CHUNK_SIZE).await?;
    print_download_summary(&resolved, source.source_kind(), &plan);
    execute_plan(
        &plan,
        &*source,
        effective.threads,
        effective.chunk_threads,
        args.http2,
    )
    .await?;
    cache::create_compat_links(args.link, source_kind, &resolved, &plan.output_root).await?;
    println!("Done: {}", plan.output_root.display());
    Ok(())
}

fn build_source(source_kind: SourceKind, config: &EffectiveConfig) -> Box<dyn ModelSource> {
    match source_kind {
        SourceKind::HuggingFace | SourceKind::HfMirror | SourceKind::HfMirrorNet => {
            Box::new(HuggingFaceSource::new(source_kind, config.hf_token.clone()))
        }
        SourceKind::ModelScope => Box::new(ModelScopeSource::new(config.modelscope_token.clone())),
    }
}

async fn resolve_model(
    args: &DownloadArgs,
    repo_type: RepoType,
    source: &dyn ModelSource,
    source_kind: SourceKind,
    config: &EffectiveConfig,
) -> Result<ResolvedModel> {
    let repo_id = args
        .repo_id()
        .ok_or_else(|| MgetError::Message("repository id is required".to_string()))?;
    let model_id = if source_kind == SourceKind::ModelScope && repo_type == RepoType::Model {
        let ms = ModelScopeSource::new(config.modelscope_token.clone());
        mapping::resolve_modelscope_id(
            &repo_id,
            args.modelscope_id.as_deref(),
            args.interactive,
            &ms,
        )
        .await?
    } else {
        repo_id
    };
    source
        .resolve_model(&model_id, repo_type, &args.revision)
        .await
}

#[derive(Debug)]
struct DownloadPlan {
    output_root: PathBuf,
    files: Vec<FileTask>,
}

impl DownloadPlan {
    fn new(
        source: &dyn ModelSource,
        model: &ResolvedModel,
        files: Vec<RemoteFile>,
        output_root: PathBuf,
    ) -> Result<Self> {
        let files = files
            .into_iter()
            .map(|file| {
                let relative = cache::ensure_relative_repo_path(&file.path)?;
                let target = output_root.join(&relative);
                let part = target.with_extension(part_extension(target.extension()));
                let state = target.with_extension(state_extension(target.extension()));
                Ok(FileTask {
                    url: source.download_url(model, &file),
                    remote: file,
                    target,
                    part,
                    state,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { output_root, files })
    }
}

#[derive(Debug, Clone)]
struct FileTask {
    remote: RemoteFile,
    url: String,
    target: PathBuf,
    part: PathBuf,
    state: PathBuf,
}

fn part_extension(existing: Option<&std::ffi::OsStr>) -> String {
    existing.map_or_else(
        || "mget-part".to_string(),
        |ext| format!("{}.mget-part", ext.to_string_lossy()),
    )
}

fn state_extension(existing: Option<&std::ffi::OsStr>) -> String {
    existing.map_or_else(
        || "mget-state.json".to_string(),
        |ext| format!("{}.mget-state.json", ext.to_string_lossy()),
    )
}

fn filter_files(files: Vec<RemoteFile>, args: &DownloadArgs) -> Result<Vec<RemoteFile>> {
    let include_set = build_glob_set(&args.includes)?;
    let exclude_set = build_glob_set(&args.excludes)?;
    let exact_files = &args.files;

    Ok(files
        .into_iter()
        .filter(|file| {
            exact_files.is_empty() || exact_files.iter().any(|wanted| wanted == &file.path)
        })
        .filter(|file| {
            include_set
                .as_ref()
                .is_none_or(|set| set.is_match(&file.path))
        })
        .filter(|file| {
            exclude_set
                .as_ref()
                .is_none_or(|set| !set.is_match(&file.path))
        })
        .collect())
}

fn build_glob_set(patterns: &[String]) -> Result<Option<GlobSet>> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut builder = GlobSetBuilder::new();
    for pattern in patterns {
        builder.add(Glob::new(pattern)?);
    }
    Ok(Some(builder.build()?))
}

async fn fill_missing_sizes(
    source: &dyn ModelSource,
    model: &ResolvedModel,
    files: Vec<RemoteFile>,
) -> Result<Vec<RemoteFile>> {
    if files.iter().all(|file| file.size.is_some()) {
        return Ok(files);
    }

    let client = Client::builder()
        .user_agent(USER_AGENT)
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(MgetError::Network)?;
    let headers = source.auth_headers();

    stream::iter(files)
        .map(|mut file| {
            let client = client.clone();
            let headers = headers.clone();
            let url = source.download_url(model, &file);
            async move {
                if file.size.is_none() {
                    match probe_content_length(&client, headers, &url).await {
                        Ok(Some(size)) => file.size = Some(size),
                        Ok(None) => {}
                        Err(err) => {
                            tracing::debug!(%url, error = %err, "failed to probe content length")
                        }
                    }
                }
                Ok(file)
            }
        })
        .buffer_unordered(8)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()
}

async fn probe_content_length(
    client: &Client,
    headers: HeaderMap,
    url: &str,
) -> Result<Option<u64>> {
    let response = client.head(url).headers(headers).send().await?;
    if !response.status().is_success() && response.status() != StatusCode::PARTIAL_CONTENT {
        return Ok(None);
    }
    Ok(response
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .or_else(|| {
            response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                .and_then(parse_content_range_total)
        }))
}

fn parse_content_range_total(value: &str) -> Option<u64> {
    value.rsplit_once('/').and_then(|(_, total)| {
        if total == "*" {
            None
        } else {
            total.parse::<u64>().ok()
        }
    })
}

fn parse_content_range_bounds(value: &str) -> Option<(u64, u64)> {
    let value = value.strip_prefix("bytes ")?;
    let (range, _) = value.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    Some((start.parse().ok()?, end.parse().ok()?))
}

async fn preflight_disk_space(plan: &DownloadPlan, chunk_size: u64) -> Result<()> {
    fs::create_dir_all(&plan.output_root).await?;
    let mut needed = 0;
    for task in &plan.files {
        needed += task_remaining_size(task, chunk_size).await?;
    }
    if needed == 0 {
        return Ok(());
    }
    let available = fs2::available_space(&plan.output_root)?;
    if available < needed {
        return Err(MgetError::NotEnoughSpace {
            path: plan.output_root.clone(),
            needed,
            available,
        });
    }
    Ok(())
}

async fn task_remaining_size(task: &FileTask, chunk_size: u64) -> Result<u64> {
    let Some(size) = task.remote.size else {
        return Ok(0);
    };

    if let Ok(meta) = fs::metadata(&task.target).await {
        return Ok(if meta.len() == size { 0 } else { size });
    }

    let part_meta = fs::metadata(&task.part).await.ok();
    if part_meta.as_ref().is_some_and(|meta| meta.len() == size)
        && fs::metadata(&task.state).await.is_ok()
    {
        let chunks = make_chunks(size, chunk_size);
        let state = RangeState::load(&task.state, chunks.len()).await?;
        return Ok(chunks
            .iter()
            .enumerate()
            .filter(|(index, _)| !state.completed.get(*index).copied().unwrap_or(false))
            .map(|(_, chunk)| chunk.len())
            .sum());
    }

    let part_len = part_meta.map(|meta| meta.len()).unwrap_or(0);
    Ok(size.saturating_sub(part_len.min(size)))
}

async fn execute_plan(
    plan: &DownloadPlan,
    source: &dyn ModelSource,
    threads: usize,
    chunk_threads: usize,
    http2: bool,
) -> Result<()> {
    let tuning = DownloadTuning {
        chunk_threads: chunk_threads.min(threads).max(1),
        ..DEFAULT_TUNING
    };
    let mut builder = Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(30));
    if !http2 {
        // HTTP/1.1 is the default: large range downloads are bandwidth-bound,
        // and some mirrors reset long HTTP/2 streams. HTTP/2 can be enabled
        // explicitly for high-latency links where multiplexing helps.
        builder = builder.http1_only();
    }
    let client = builder.build().map_err(MgetError::Network)?;
    let progress = MultiProgress::with_draw_target(ProgressDrawTarget::stderr_with_hz(12));
    let total_bar = progress.add(ProgressBar::new(total_known_size(&plan.files)));
    total_bar.set_style(total_style());
    total_bar.set_message("starting");
    total_bar.enable_steady_tick(Duration::from_millis(250));
    let progress_state = Arc::new(DownloadProgress {
        multi: progress,
        total: total_bar,
        completed: AtomicUsize::new(0),
        total_files: plan.files.len(),
    });

    let semaphore = Arc::new(Semaphore::new(threads));
    stream::iter(plan.files.clone())
        .map(|task| {
            let client = client.clone();
            let headers = source.auth_headers();
            let semaphore = semaphore.clone();
            let progress_state = progress_state.clone();
            async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|err| MgetError::Message(err.to_string()))?;
                download_file_with_retry(&client, headers, task, &progress_state, tuning).await
            }
        })
        .buffer_unordered(threads)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    progress_state.total.finish_with_message("complete");
    Ok(())
}

struct DownloadProgress {
    multi: MultiProgress,
    total: ProgressBar,
    completed: AtomicUsize,
    total_files: usize,
}

#[derive(Clone, Copy)]
struct ProgressBars<'a> {
    total: &'a ProgressBar,
    detail: Option<&'a ProgressBar>,
}

fn total_known_size(files: &[FileTask]) -> u64 {
    files.iter().filter_map(|task| task.remote.size).sum()
}

fn print_download_summary(model: &ResolvedModel, source: SourceKind, plan: &DownloadPlan) {
    let total = total_known_size(&plan.files);
    println!(
        "mget {} {}  {}",
        source.label(),
        model.repo_type,
        model.source_id
    );
    println!(
        "files: {}  size: {}  target: {}",
        plan.files.len(),
        format_bytes(total),
        plan.output_root.display()
    );
}

fn display_path(path: &str) -> String {
    const MAX_CHARS: usize = 42;
    let count = path.chars().count();
    if count <= MAX_CHARS {
        return path.to_string();
    }
    let tail = path
        .chars()
        .rev()
        .take(MAX_CHARS - 1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    format!("…{tail}")
}

fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn total_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] {bar:36.cyan/blue} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}  {wide_msg}",
    )
    .unwrap()
    .progress_chars("━━╾")
    .tick_strings(&["🌘", "🌗", "🌖", "🌕", "🌔", "🌓", "🌒", "🌑"])
}

fn detail_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.magenta} [{elapsed_precise}] {bar:36.magenta/black} {bytes:>9}/{total_bytes:<9} {bytes_per_sec:>10}  {wide_msg}",
    )
    .unwrap()
    .progress_chars("━━╾")
    .tick_strings(&["◐", "◓", "◑", "◒"])
}

async fn download_file_with_retry(
    client: &Client,
    headers: HeaderMap,
    task: FileTask,
    progress: &DownloadProgress,
    tuning: DownloadTuning,
) -> Result<()> {
    let mut last_error = None;
    for attempt in 0..=MAX_RETRIES {
        match download_file(client, headers.clone(), task.clone(), progress, tuning).await {
            Ok(()) => return Ok(()),
            Err(err) => {
                last_error = Some(err);
                if attempt < MAX_RETRIES {
                    sleep(Duration::from_millis(300 * 2_u64.pow(attempt as u32))).await;
                }
            }
        }
    }
    Err(MgetError::DownloadFailed(
        last_error
            .map(|err| err.to_string())
            .unwrap_or_else(|| task.remote.path),
    ))
}

async fn download_file(
    client: &Client,
    headers: HeaderMap,
    task: FileTask,
    progress: &DownloadProgress,
    tuning: DownloadTuning,
) -> Result<()> {
    if target_is_complete(&task).await? {
        if let Some(size) = task.remote.size {
            exclude_existing_bytes(&progress.total, None, size);
        }
        mark_file_done(progress, &task.remote.path);
        return Ok(());
    }

    if let Some(parent) = task.target.parent() {
        fs::create_dir_all(parent).await?;
    }

    progress.total.set_message(format!(
        "{}/{} downloading {}",
        progress.completed.load(Ordering::Relaxed),
        progress.total_files,
        display_path(&task.remote.path)
    ));
    let detail_bar = make_detail_bar(&progress.multi, &task);

    let accept_ranges = supports_ranges(client, &headers, &task)
        .await
        .unwrap_or(false);
    let size = task.remote.size.unwrap_or(0);
    let part_len = fs::metadata(&task.part)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    let has_range_state = fs::metadata(&task.state).await.is_ok();
    // A range download preallocates the part to the full size before the
    // first chunk is recorded. Such a part must not be mistaken for a
    // resumable stream part, or it can be renamed into place while sparse.
    let has_stream_part = part_len > 0 && part_len < size && !has_range_state;
    // A range state file proves this part was preallocated for ranged
    // download. Keep using ranges even if a transient HEAD failure stops
    // Accept-Ranges from being advertised on this retry.
    if (accept_ranges || has_range_state) && size >= tuning.range_threshold && !has_stream_part {
        sanitize_range_resume_state(&task, size).await?;
        let completed_before = range_completed_bytes(&task, size, tuning.chunk_size)
            .await
            .unwrap_or(0);
        let range_attempt_bytes = Arc::new(AtomicU64::new(0));
        let range_result = download_range_file(
            client,
            headers.clone(),
            &task,
            ProgressBars {
                total: &progress.total,
                detail: detail_bar.as_ref(),
            },
            size,
            tuning,
            range_attempt_bytes.clone(),
        )
        .await;
        if let Err(err) = range_result {
            eprintln!(
                "warning: ranged download interrupted for {}; keeping completed chunks \
                 ({:.1} MiB were already complete) and retrying: {err}",
                task.remote.path,
                completed_before as f64 / 1024.0 / 1024.0
            );
            restore_range_progress(
                &progress.total,
                detail_bar.as_ref(),
                completed_before,
                range_attempt_bytes.load(Ordering::Relaxed),
            );
            if let Some(bar) = detail_bar.as_ref() {
                bar.finish_and_clear();
            }
            // Do not fall back to a stream restart here. A ranged download may
            // have written tens of gigabytes already, and truncating the part
            // file (or cleaning up its state) throws that work away. The outer
            // retry loop resumes from the untouched chunk state instead.
            return Err(err);
        }
    } else {
        download_stream_file(client, headers, &task, &progress.total, detail_bar.as_ref()).await?;
    }

    verify_file(&task).await?;
    cleanup_temp_files(&task).await?;
    if let Some(detail_bar) = detail_bar {
        detail_bar.finish_and_clear();
    }
    mark_file_done(progress, &task.remote.path);
    Ok(())
}

fn make_detail_bar(progress: &MultiProgress, task: &FileTask) -> Option<ProgressBar> {
    let size = task.remote.size?;
    if size < DETAIL_BAR_THRESHOLD {
        return None;
    }
    let bar = progress.add(ProgressBar::new(size));
    bar.set_style(detail_style());
    bar.set_message(display_path(&task.remote.path));
    bar.enable_steady_tick(Duration::from_millis(250));
    Some(bar)
}

fn mark_file_done(progress: &DownloadProgress, path: &str) {
    let done = progress.completed.fetch_add(1, Ordering::Relaxed) + 1;
    progress.total.set_message(format!(
        "{done}/{} done {}",
        progress.total_files,
        display_path(path)
    ));
}

async fn target_is_complete(task: &FileTask) -> Result<bool> {
    if fs::metadata(&task.target).await.is_err() {
        return Ok(false);
    }
    match verify_file(task).await {
        Ok(()) => Ok(true),
        Err(MgetError::ChecksumMismatch { .. } | MgetError::SizeMismatch { .. }) => {
            eprintln!(
                "warning: existing target failed integrity check for {}; re-downloading",
                task.remote.path
            );
            fs::remove_file(&task.target).await?;
            Ok(false)
        }
        Err(err) => Err(err),
    }
}

async fn supports_ranges(client: &Client, headers: &HeaderMap, task: &FileTask) -> Result<bool> {
    let mut request = client.head(&task.url).headers(headers.clone());
    request = request.header(RANGE, "bytes=0-0");
    let response = request.send().await?;
    if response.status() == StatusCode::PARTIAL_CONTENT {
        return Ok(true);
    }
    Ok(response
        .headers()
        .get(ACCEPT_RANGES)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.eq_ignore_ascii_case("bytes")))
}

async fn range_completed_bytes(task: &FileTask, size: u64, chunk_size: u64) -> Result<u64> {
    let chunks = make_chunks(size, chunk_size);
    let state = RangeState::load(&task.state, chunks.len()).await?;
    Ok(chunks
        .iter()
        .enumerate()
        .filter(|(index, _)| state.completed.get(*index).copied().unwrap_or(false))
        .map(|(_, chunk)| chunk.len())
        .sum())
}

async fn sanitize_range_resume_state(task: &FileTask, size: u64) -> Result<()> {
    if fs::metadata(&task.state).await.is_err() {
        return Ok(());
    }
    let part_is_valid = fs::metadata(&task.part)
        .await
        .map(|meta| meta.len() == size)
        .unwrap_or(false);
    if part_is_valid {
        return Ok(());
    }

    remove_if_exists(&task.state).await?;
    remove_if_exists(&task.part).await?;
    Ok(())
}

async fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err.into()),
    }
}

fn restore_range_progress(
    total_bar: &ProgressBar,
    detail_bar: Option<&ProgressBar>,
    excluded_bytes: u64,
    attempted_bytes: u64,
) {
    restore_existing_bytes(total_bar, detail_bar, excluded_bytes);
    dec_bars(total_bar, detail_bar, attempted_bytes);
}

async fn download_stream_file(
    client: &Client,
    headers: HeaderMap,
    task: &FileTask,
    total_bar: &ProgressBar,
    detail_bar: Option<&ProgressBar>,
) -> Result<()> {
    let mut existing = fs::metadata(&task.part)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    if let Some(size) = task.remote.size {
        if existing == size {
            if fs::metadata(&task.state).await.is_ok() {
                return Err(MgetError::Message(format!(
                    "refusing to finalize preallocated range part for {} as a stream download",
                    task.remote.path
                )));
            }
            fs::rename(&task.part, &task.target).await?;
            exclude_existing_bytes(total_bar, detail_bar, size);
            return Ok(());
        }
        if existing > size {
            fs::remove_file(&task.part).await?;
            existing = 0;
        }
    }

    let mut request = client.get(&task.url).headers(headers);
    if existing > 0 {
        request = request.header(RANGE, format!("bytes={existing}-"));
    }

    let response = request.send().await?.error_for_status()?;
    let append = if existing > 0 && response.status() == StatusCode::PARTIAL_CONTENT {
        let content_range = response
            .headers()
            .get(CONTENT_RANGE)
            .and_then(|value| value.to_str().ok())
            .and_then(parse_content_range_bounds);
        if content_range.is_some_and(|(start, _)| start == existing) {
            true
        } else {
            return Err(MgetError::Message(format!(
                "server returned unexpected content-range for {}",
                task.remote.path
            )));
        }
    } else {
        if existing > 0 {
            eprintln!(
                "warning: server ignored resume for {}; restarting partial download",
                task.remote.path
            );
            existing = 0;
        }
        false
    };
    exclude_existing_bytes(total_bar, detail_bar, existing);

    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut file = options.open(&task.part).await?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        inc_bars(total_bar, detail_bar, chunk.len() as u64);
    }
    file.flush().await?;
    fs::rename(&task.part, &task.target).await?;
    Ok(())
}

async fn download_range_file(
    client: &Client,
    headers: HeaderMap,
    task: &FileTask,
    bars: ProgressBars<'_>,
    size: u64,
    tuning: DownloadTuning,
    attempt_bytes: Arc<AtomicU64>,
) -> Result<()> {
    let chunks = make_chunks(size, tuning.chunk_size);
    let chunk_count = chunks.len();
    // Write an all-false state before preallocating the part. Its presence is
    // what distinguishes a range download from a stream partial file.
    RangeState::ensure(&task.state, chunk_count).await?;
    let state = RangeState::load(&task.state, chunks.len()).await?;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .read(true)
        .truncate(false)
        .open(&task.part)
        .await?;
    file.set_len(size).await?;
    drop(file);

    let completed_bytes = chunks
        .iter()
        .enumerate()
        .filter(|(index, _)| state.completed.get(*index).copied().unwrap_or(false))
        .map(|(_, chunk)| chunk.len())
        .sum::<u64>();
    exclude_existing_bytes(bars.total, bars.detail, completed_bytes);

    let state_lock = Arc::new(Mutex::new(()));
    stream::iter(
        chunks
            .iter()
            .copied()
            .enumerate()
            .filter(|(index, _)| !state.completed[*index])
            .collect::<Vec<_>>(),
    )
    .map(|(index, chunk)| {
        let headers = headers.clone();
        let state_lock = state_lock.clone();
        let attempt_bytes = attempt_bytes.clone();
        async move {
            let mut last_error = None;
            for attempt in 0..=MAX_RETRIES {
                match download_one_chunk(client, headers.clone(), task, chunk, bars, &attempt_bytes)
                    .await
                {
                    Ok(()) => {
                        last_error = None;
                        break;
                    }
                    Err(err) => {
                        last_error = Some(err);
                        if attempt < MAX_RETRIES {
                            sleep(Duration::from_millis(300 * 2_u64.pow(attempt as u32))).await;
                        }
                    }
                }
            }
            if let Some(err) = last_error {
                return Err(err);
            }
            let _guard = state_lock.lock().await;
            RangeState::mark_complete(&task.state, chunk_count, index).await
        }
    })
    .buffer_unordered(tuning.chunk_threads)
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    fs::rename(&task.part, &task.target).await?;
    Ok(())
}

async fn download_one_chunk(
    client: &Client,
    headers: HeaderMap,
    task: &FileTask,
    chunk: ByteRange,
    bars: ProgressBars<'_>,
    attempt_bytes: &AtomicU64,
) -> Result<()> {
    let response = client
        .get(&task.url)
        .headers(headers)
        .header(RANGE, format!("bytes={}-{}", chunk.start, chunk.end))
        .send()
        .await?;
    if response.status() != StatusCode::PARTIAL_CONTENT {
        return Err(MgetError::Message(format!(
            "server did not honor range request for {}: expected 206, got {}",
            task.remote.path,
            response.status()
        )));
    }
    let content_range = response
        .headers()
        .get(CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(parse_content_range_bounds);
    if content_range != Some((chunk.start, chunk.end)) {
        return Err(MgetError::Message(format!(
            "server returned unexpected content-range for {}",
            task.remote.path
        )));
    }
    let mut file = OpenOptions::new().write(true).open(&task.part).await?;
    file.seek(SeekFrom::Start(chunk.start)).await?;
    let mut stream = response.bytes_stream();
    let mut written = 0;
    let result: Result<()> = async {
        while let Some(bytes) = stream.next().await {
            let bytes = bytes?;
            let remaining = chunk.len() - written;
            if bytes.len() as u64 > remaining {
                // Never let an oversized body spill into the next chunk. Write
                // only the requested bytes and fail the chunk so it is fetched
                // again from a clean boundary.
                if remaining > 0 {
                    file.write_all(&bytes[..remaining as usize]).await?;
                    written += remaining;
                    attempt_bytes.fetch_add(remaining, Ordering::Relaxed);
                    inc_bars(bars.total, bars.detail, remaining);
                }
                return Err(MgetError::Message(format!(
                    "long body for {} chunk {}-{}: got at least {} bytes, want {}",
                    task.remote.path,
                    chunk.start,
                    chunk.end,
                    written + (bytes.len() as u64 - remaining),
                    chunk.len()
                )));
            }
            file.write_all(&bytes).await?;
            written += bytes.len() as u64;
            attempt_bytes.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            inc_bars(bars.total, bars.detail, bytes.len() as u64);
        }
        file.flush().await?;
        if written != chunk.len() {
            return Err(MgetError::Message(format!(
                "short body for {} chunk {}-{}: got {written} bytes, want {}",
                task.remote.path,
                chunk.start,
                chunk.end,
                chunk.len()
            )));
        }
        Ok(())
    }
    .await;
    if result.is_err() && written > 0 {
        // A failed chunk is retried from its start. Roll back its partial
        // progress so the bars and the retry accounting stay accurate.
        dec_bars(bars.total, bars.detail, written);
        attempt_bytes.fetch_sub(written, Ordering::Relaxed);
    }
    result
}

fn inc_bars(total_bar: &ProgressBar, detail_bar: Option<&ProgressBar>, amount: u64) {
    total_bar.inc(amount);
    if let Some(detail_bar) = detail_bar {
        detail_bar.inc(amount);
    }
}

fn dec_bars(total_bar: &ProgressBar, detail_bar: Option<&ProgressBar>, amount: u64) {
    if amount == 0 {
        return;
    }
    total_bar.dec(amount);
    if let Some(detail_bar) = detail_bar {
        detail_bar.dec(amount);
    }
}

fn exclude_existing_bytes(total_bar: &ProgressBar, detail_bar: Option<&ProgressBar>, amount: u64) {
    if amount == 0 {
        return;
    }
    total_bar.dec_length(amount);
    total_bar.reset_eta();
    if let Some(detail_bar) = detail_bar {
        detail_bar.dec_length(amount);
        detail_bar.reset_eta();
    }
}

fn restore_existing_bytes(total_bar: &ProgressBar, detail_bar: Option<&ProgressBar>, amount: u64) {
    if amount == 0 {
        return;
    }
    total_bar.inc_length(amount);
    total_bar.reset_eta();
    if let Some(detail_bar) = detail_bar {
        detail_bar.inc_length(amount);
        detail_bar.reset_eta();
    }
}

#[derive(Debug, Clone, Copy)]
struct ByteRange {
    start: u64,
    end: u64,
}

impl ByteRange {
    fn len(self) -> u64 {
        self.end - self.start + 1
    }
}

fn make_chunks(size: u64, chunk_size: u64) -> Vec<ByteRange> {
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < size {
        let end = (start + chunk_size - 1).min(size - 1);
        chunks.push(ByteRange { start, end });
        start = end + 1;
    }
    chunks
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct RangeState {
    completed: Vec<bool>,
}

impl RangeState {
    async fn load(path: &Path, chunk_count: usize) -> Result<Self> {
        match fs::read_to_string(path).await {
            Ok(content) => match serde_json::from_str::<Self>(&content) {
                Ok(mut state) => {
                    state.completed.resize(chunk_count, false);
                    Ok(state)
                }
                Err(err) => {
                    eprintln!(
                        "warning: corrupt resume state {} ({err}); restarting this file",
                        path.display()
                    );
                    let backup = path.with_extension("json.corrupt");
                    remove_if_exists(&backup).await?;
                    fs::rename(path, &backup).await?;
                    let state = Self {
                        completed: vec![false; chunk_count],
                    };
                    Self::save(path, &state).await?;
                    Ok(state)
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self {
                completed: vec![false; chunk_count],
            }),
            Err(err) => Err(err.into()),
        }
    }

    async fn ensure(path: &Path, chunk_count: usize) -> Result<()> {
        match fs::metadata(path).await {
            Ok(_) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                Self::save(
                    path,
                    &Self {
                        completed: vec![false; chunk_count],
                    },
                )
                .await
            }
            Err(err) => Err(err.into()),
        }
    }

    async fn save(path: &Path, state: &Self) -> Result<()> {
        let content = serde_json::to_vec_pretty(state)?;
        // Write-then-rename keeps the resume state valid if the process is
        // killed while a chunk is being recorded.
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, content).await?;
        fs::rename(&tmp, path).await?;
        Ok(())
    }

    async fn mark_complete(path: &Path, chunk_count: usize, index: usize) -> Result<()> {
        let mut state = Self::load(path, chunk_count).await?;
        if let Some(item) = state.completed.get_mut(index) {
            *item = true;
        }
        Self::save(path, &state).await
    }
}

async fn verify_file(task: &FileTask) -> Result<()> {
    if let Some(expected) = &task.remote.sha256 {
        let actual = hash_file_sha256(&task.target).await?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(MgetError::ChecksumMismatch {
                path: task.target.clone(),
                expected: expected.clone(),
                actual,
            });
        }
        return Ok(());
    }

    if let Some(expected) = &task.remote.md5 {
        let actual = hash_file_md5(&task.target).await?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(MgetError::ChecksumMismatch {
                path: task.target.clone(),
                expected: expected.clone(),
                actual,
            });
        }
        return Ok(());
    }

    if let Some(size) = task.remote.size {
        let actual = fs::metadata(&task.target).await?.len();
        if actual != size {
            return Err(MgetError::SizeMismatch {
                path: task.target.clone(),
                expected: size,
                actual,
            });
        }
    } else {
        eprintln!(
            "warning: no checksum or size metadata for {}; skipped integrity check",
            task.remote.path
        );
    }
    Ok(())
}

async fn hash_file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let read = file.read(&mut buf).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    let digest = hex::encode(hasher.finalize());
    drop_page_cache(&file);
    Ok(digest)
}

async fn hash_file_md5(path: &Path) -> Result<String> {
    let mut file = File::open(path).await?;
    let mut hasher = Md5::new();
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let read = file.read(&mut buf).await?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    let digest = hex::encode(hasher.finalize());
    drop_page_cache(&file);
    Ok(digest)
}

/// Verification reads a whole model file sequentially. Without this, those
/// pages stay charged to the service cgroup as page cache and make `mget`
/// look like it is using tens of gigabytes of RAM.
#[cfg(target_os = "linux")]
fn drop_page_cache(file: &File) {
    use std::os::unix::io::AsRawFd;

    // SAFETY: the fd is owned by `file` and stays valid for the call.
    unsafe {
        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
    }
}

#[cfg(not(target_os = "linux"))]
fn drop_page_cache(_: &File) {}

async fn cleanup_temp_files(task: &FileTask) -> Result<()> {
    for path in [&task.part, &task.state] {
        remove_if_exists(path).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{DownloadArgs, RepoTypeChoice, SourceChoice};

    fn test_tuning() -> DownloadTuning {
        DownloadTuning {
            chunk_size: 8,
            range_threshold: 16,
            chunk_threads: 2,
        }
    }

    fn args() -> DownloadArgs {
        DownloadArgs {
            repo: Some("org/model".into()),
            model_id: None,
            dataset_id: None,
            repo_type: RepoTypeChoice::Model,
            source: SourceChoice::Auto,
            output: None,
            threads: None,
            chunk_threads: None,
            http2: false,
            revision: "main".into(),
            files: vec![],
            includes: vec![],
            excludes: vec![],
            link: None,
            interactive: false,
            hf_token: None,
            modelscope_token: None,
            modelscope_id: None,
        }
    }

    #[test]
    fn filters_exact_file() {
        let mut args = args();
        args.files = vec!["a.bin".into()];
        let files = vec![
            RemoteFile {
                path: "a.bin".into(),
                size: None,
                sha256: None,
                md5: None,
            },
            RemoteFile {
                path: "b.bin".into(),
                size: None,
                sha256: None,
                md5: None,
            },
        ];
        let filtered = filter_files(files, &args).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].path, "a.bin");
    }

    #[test]
    fn filters_include_and_exclude_globs() {
        let mut args = args();
        args.includes = vec!["*.safetensors".into()];
        args.excludes = vec!["model-00002*".into()];
        let files = vec![
            RemoteFile {
                path: "model-00001.safetensors".into(),
                size: None,
                sha256: None,
                md5: None,
            },
            RemoteFile {
                path: "model-00002.safetensors".into(),
                size: None,
                sha256: None,
                md5: None,
            },
            RemoteFile {
                path: "README.md".into(),
                size: None,
                sha256: None,
                md5: None,
            },
        ];
        let filtered = filter_files(files, &args).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].path, "model-00001.safetensors");
    }

    #[test]
    fn makes_expected_chunks() {
        let chunks = make_chunks(10, 4);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 4);
        assert_eq!(chunks[2].start, 8);
        assert_eq!(chunks[2].end, 9);
    }

    #[tokio::test]
    async fn stream_download_resumes_existing_part_when_range_is_honored() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{header, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .and(header("range", "bytes=6-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 6-10/11")
                    .set_body_bytes("world".as_bytes()),
            )
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        tokio::fs::write(&part, b"hello ").await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(11),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target: target.clone(),
            part,
            state: target.with_extension("bin.mget-state.json"),
        };

        download_stream_file(
            &Client::new(),
            HeaderMap::new(),
            &task,
            &ProgressBar::hidden(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"hello world");
    }

    #[tokio::test]
    async fn stream_download_restarts_when_server_ignores_range() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes("fresh".as_bytes()))
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        tokio::fs::write(&part, b"stale ").await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: None,
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target: target.clone(),
            part,
            state: target.with_extension("bin.mget-state.json"),
        };

        download_stream_file(
            &Client::new(),
            HeaderMap::new(),
            &task,
            &ProgressBar::hidden(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"fresh");
    }

    #[tokio::test]
    async fn stream_download_rejects_wrong_content_range() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{header, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .and(header("range", "bytes=6-"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 0-4/11")
                    .set_body_bytes("wrong".as_bytes()),
            )
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        tokio::fs::write(&part, b"hello ").await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(11),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target,
            part,
            state: temp.path().join("file.bin.mget-state.json"),
        };

        let err = download_stream_file(
            &Client::new(),
            HeaderMap::new(),
            &task,
            &ProgressBar::hidden(),
            None,
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("unexpected content-range"));
    }

    #[tokio::test]
    async fn range_chunk_rejects_non_partial_response() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes("full file".as_bytes()))
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&part)
            .await
            .unwrap();
        file.set_len(8).await.unwrap();
        drop(file);
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(8),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target,
            part,
            state: temp.path().join("file.bin.mget-state.json"),
        };

        let total_bar = ProgressBar::hidden();
        let err = download_one_chunk(
            &Client::new(),
            HeaderMap::new(),
            &task,
            ByteRange { start: 0, end: 3 },
            ProgressBars {
                total: &total_bar,
                detail: None,
            },
            &AtomicU64::new(0),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("expected 206"));
    }

    #[tokio::test]
    async fn range_chunk_rejects_short_body() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 0-7/16")
                    .set_body_bytes(b"abcd".to_vec()),
            )
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&part)
            .await
            .unwrap();
        file.set_len(16).await.unwrap();
        drop(file);
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(16),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target,
            part,
            state: temp.path().join("file.bin.mget-state.json"),
        };
        let total_bar = ProgressBar::hidden();

        let err = download_one_chunk(
            &Client::new(),
            HeaderMap::new(),
            &task,
            ByteRange { start: 0, end: 7 },
            ProgressBars {
                total: &total_bar,
                detail: None,
            },
            &AtomicU64::new(0),
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("short body"));
    }

    #[tokio::test]
    async fn range_chunk_rejects_long_body_without_spilling() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 0-7/16")
                    .set_body_bytes(b"abcdefghijkl".to_vec()),
            )
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        tokio::fs::write(&part, vec![0xAA; 16]).await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(16),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target,
            part: part.clone(),
            state: temp.path().join("file.bin.mget-state.json"),
        };
        let total_bar = ProgressBar::hidden();

        let err = download_one_chunk(
            &Client::new(),
            HeaderMap::new(),
            &task,
            ByteRange { start: 0, end: 7 },
            ProgressBars {
                total: &total_bar,
                detail: None,
            },
            &AtomicU64::new(0),
        )
        .await
        .unwrap_err();

        assert!(err.to_string().contains("long body"));
        let data = tokio::fs::read(&part).await.unwrap();
        assert_eq!(&data[..8], b"abcdefgh");
        assert_eq!(&data[8..], &[0xAA; 8]);
    }

    #[tokio::test]
    async fn corrupt_state_is_quarantined_and_reset() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("file.bin.mget-state.json");
        tokio::fs::write(&state, b"{not valid json").await.unwrap();

        let loaded = RangeState::load(&state, 3).await.unwrap();
        assert_eq!(loaded.completed, vec![false, false, false]);
        assert!(
            tokio::fs::metadata(state.with_extension("json.corrupt"))
                .await
                .is_ok()
        );

        RangeState::mark_complete(&state, 3, 1).await.unwrap();
        let reloaded = RangeState::load(&state, 3).await.unwrap();
        assert_eq!(reloaded.completed, vec![false, true, false]);
    }

    #[tokio::test]
    async fn preallocated_part_without_state_uses_range_download() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{header, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/file.bin"))
            .respond_with(ResponseTemplate::new(206).insert_header("content-range", "bytes 0-0/16"))
            .mount(&server)
            .await;
        for (range, content_range, body) in [
            ("bytes=0-7", "bytes 0-7/16", &b"abcdefgh"[..]),
            ("bytes=8-15", "bytes 8-15/16", &b"ijklmnop"[..]),
        ] {
            Mock::given(method("GET"))
                .and(path("/file.bin"))
                .and(header("range", range))
                .respond_with(
                    ResponseTemplate::new(206)
                        .insert_header("content-range", content_range)
                        .set_body_bytes(body.to_vec()),
                )
                .mount(&server)
                .await;
        }

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&part)
            .await
            .unwrap();
        file.set_len(16).await.unwrap();
        drop(file);
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(16),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target: target.clone(),
            part,
            state: target.with_extension("bin.mget-state.json"),
        };
        let progress = DownloadProgress {
            multi: MultiProgress::new(),
            total: ProgressBar::hidden(),
            completed: AtomicUsize::new(0),
            total_files: 1,
        };

        download_file(
            &Client::new(),
            HeaderMap::new(),
            task,
            &progress,
            test_tuning(),
        )
        .await
        .unwrap();

        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"abcdefghijklmnop");
    }

    #[tokio::test]
    async fn range_state_prevents_sparse_part_from_being_finalized_as_stream() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{header, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/file.bin"))
            .respond_with(ResponseTemplate::new(200))
            .mount(&server)
            .await;
        for (range, content_range, body) in [
            ("bytes=0-7", "bytes 0-7/16", &b"abcdefgh"[..]),
            ("bytes=8-15", "bytes 8-15/16", &b"ijklmnop"[..]),
        ] {
            Mock::given(method("GET"))
                .and(path("/file.bin"))
                .and(header("range", range))
                .respond_with(
                    ResponseTemplate::new(206)
                        .insert_header("content-range", content_range)
                        .set_body_bytes(body.to_vec()),
                )
                .mount(&server)
                .await;
        }

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        let state = target.with_extension("bin.mget-state.json");
        tokio::fs::write(&part, vec![0; 16]).await.unwrap();
        RangeState::save(
            &state,
            &RangeState {
                completed: vec![false, false],
            },
        )
        .await
        .unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(16),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target: target.clone(),
            part,
            state,
        };
        let progress = DownloadProgress {
            multi: MultiProgress::new(),
            total: ProgressBar::hidden(),
            completed: AtomicUsize::new(0),
            total_files: 1,
        };

        download_file(
            &Client::new(),
            HeaderMap::new(),
            task,
            &progress,
            test_tuning(),
        )
        .await
        .unwrap();

        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"abcdefghijklmnop");
    }

    #[tokio::test]
    async fn corrupt_existing_target_is_removed_for_redownload() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        tokio::fs::write(&target, b"short").await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(10),
                sha256: None,
                md5: None,
            },
            url: "http://example.com/file.bin".into(),
            target: target.clone(),
            part: target.with_extension("bin.mget-part"),
            state: target.with_extension("bin.mget-state.json"),
        };

        assert!(!target_is_complete(&task).await.unwrap());
        assert!(tokio::fs::metadata(&target).await.is_err());
    }

    #[tokio::test]
    async fn preflight_counts_only_remaining_part_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        tokio::fs::write(&part, vec![0; 4]).await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(10),
                sha256: None,
                md5: None,
            },
            url: "http://example.com/file.bin".into(),
            target,
            part,
            state: temp.path().join("file.bin.mget-state.json"),
        };

        assert_eq!(task_remaining_size(&task, 8).await.unwrap(), 6);
    }

    #[tokio::test]
    async fn preflight_ignores_state_when_part_is_missing() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let state = target.with_extension("bin.mget-state.json");
        RangeState::mark_complete(&state, 1, 0).await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(10),
                sha256: None,
                md5: None,
            },
            url: "http://example.com/file.bin".into(),
            target: target.clone(),
            part: target.with_extension("bin.mget-part"),
            state,
        };

        assert_eq!(task_remaining_size(&task, 8).await.unwrap(), 10);
    }

    #[tokio::test]
    async fn range_resume_state_is_removed_when_part_is_missing() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let state = target.with_extension("bin.mget-state.json");
        RangeState::mark_complete(&state, 1, 0).await.unwrap();
        let part = target.with_extension("bin.mget-part");
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(10),
                sha256: None,
                md5: None,
            },
            url: "http://example.com/file.bin".into(),
            target,
            part: part.clone(),
            state: state.clone(),
        };

        sanitize_range_resume_state(&task, 10).await.unwrap();

        assert!(tokio::fs::metadata(&state).await.is_err());
        assert!(tokio::fs::metadata(&part).await.is_err());
    }

    #[tokio::test]
    async fn range_resume_state_is_removed_when_part_len_is_wrong() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let state = target.with_extension("bin.mget-state.json");
        let part = target.with_extension("bin.mget-part");
        RangeState::mark_complete(&state, 1, 0).await.unwrap();
        tokio::fs::write(&part, b"short").await.unwrap();
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(10),
                sha256: None,
                md5: None,
            },
            url: "http://example.com/file.bin".into(),
            target,
            part: part.clone(),
            state: state.clone(),
        };

        sanitize_range_resume_state(&task, 10).await.unwrap();

        assert!(tokio::fs::metadata(&state).await.is_err());
        assert!(tokio::fs::metadata(&part).await.is_err());
    }

    #[tokio::test]
    async fn range_failure_keeps_completed_chunks_for_retry() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{header, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("HEAD"))
            .and(path("/file.bin"))
            .respond_with(ResponseTemplate::new(206).insert_header("content-range", "bytes 0-0/16"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .and(header("range", "bytes=0-7"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 0-7/16")
                    .set_body_bytes(b"abcdefgh".to_vec()),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .and(header("range", "bytes=8-15"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        let state = target.with_extension("bin.mget-state.json");
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(16),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target,
            part: part.clone(),
            state: state.clone(),
        };
        let progress = DownloadProgress {
            multi: MultiProgress::new(),
            total: ProgressBar::hidden(),
            completed: AtomicUsize::new(0),
            total_files: 1,
        };

        download_file(
            &Client::new(),
            HeaderMap::new(),
            task.clone(),
            &progress,
            test_tuning(),
        )
        .await
        .unwrap_err();

        assert!(tokio::fs::metadata(&part).await.is_ok());
        assert!(tokio::fs::metadata(&state).await.is_ok());
        let state = RangeState::load(&state, 2).await.unwrap();
        assert_eq!(state.completed, vec![true, false]);
        let data = tokio::fs::read(&part).await.unwrap();
        assert_eq!(data.len(), 16);
        assert_eq!(&data[..8], b"abcdefgh");
    }

    #[tokio::test]
    async fn range_chunk_retries_transient_failure() {
        use reqwest::header::HeaderMap;
        use wiremock::{
            Mock, MockServer, ResponseTemplate,
            matchers::{header, method, path},
        };

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .and(header("range", "bytes=0-7"))
            .respond_with(ResponseTemplate::new(500))
            .up_to_n_times(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .and(header("range", "bytes=0-7"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 0-7/16")
                    .set_body_bytes(b"abcdefgh".to_vec()),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/file.bin"))
            .and(header("range", "bytes=8-15"))
            .respond_with(
                ResponseTemplate::new(206)
                    .insert_header("content-range", "bytes 8-15/16")
                    .set_body_bytes(b"ijklmnop".to_vec()),
            )
            .mount(&server)
            .await;

        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("file.bin");
        let part = target.with_extension("bin.mget-part");
        let task = FileTask {
            remote: RemoteFile {
                path: "file.bin".into(),
                size: Some(16),
                sha256: None,
                md5: None,
            },
            url: format!("{}/file.bin", server.uri()),
            target: target.clone(),
            part,
            state: target.with_extension("bin.mget-state.json"),
        };
        let total_bar = ProgressBar::hidden();

        download_range_file(
            &Client::new(),
            HeaderMap::new(),
            &task,
            ProgressBars {
                total: &total_bar,
                detail: None,
            },
            16,
            test_tuning(),
            Arc::new(AtomicU64::new(0)),
        )
        .await
        .unwrap();

        assert_eq!(tokio::fs::read(&target).await.unwrap(), b"abcdefghijklmnop");
    }
}
