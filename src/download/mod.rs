use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use futures_util::{StreamExt, stream};
use globset::{Glob, GlobSet, GlobSetBuilder};
use indicatif::{MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};
use md5::Md5;
use reqwest::{
    Client, StatusCode,
    header::{ACCEPT_RANGES, HeaderMap, RANGE},
};
use sha2::{Digest, Sha256};
use tokio::{
    fs::{self, File, OpenOptions},
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, SeekFrom},
    sync::Semaphore,
    time::sleep,
};

use crate::{
    cache,
    cli::DownloadArgs,
    config::{Config, EffectiveConfig},
    error::{MgetError, Result},
    mapping, routing,
    source::{
        ModelSource, RemoteFile, ResolvedModel, SourceKind, huggingface::HuggingFaceSource,
        modelscope::ModelScopeSource,
    },
};

pub const USER_AGENT: &str = concat!("mget/", env!("CARGO_PKG_VERSION"));
const LARGE_FILE_CHUNK_THRESHOLD: u64 = 256 * 1024 * 1024;
const DEFAULT_CHUNK_SIZE: u64 = 64 * 1024 * 1024;
const DEFAULT_CHUNK_THREADS: usize = 2;
const MAX_RETRIES: usize = 3;
const DETAIL_BAR_THRESHOLD: u64 = 8 * 1024 * 1024;

pub async fn run_download(args: DownloadArgs, config: Config) -> Result<()> {
    let effective = config.effective_for(&args);
    let source_kind = routing::select_source(args.source).await?;
    let source = build_source(source_kind, &effective);
    let resolved = resolve_model(&args, &*source, source_kind, &effective).await?;
    if resolved.requested_id != resolved.source_id {
        println!(
            "Mapped model id: {} -> {}",
            resolved.requested_id, resolved.source_id
        );
    }
    let files = filter_files(source.list_files(&resolved).await?, &args)?;
    if files.is_empty() {
        return Err(MgetError::EmptyRepository(resolved.source_id));
    }

    let output_root = cache::default_output_root(
        args.output.as_deref(),
        &effective.cache_dir,
        source_kind,
        &resolved,
    );
    let plan = DownloadPlan::new(&*source, &resolved, files, output_root)?;
    preflight_disk_space(&plan).await?;
    print_download_summary(&resolved, source.source_kind(), &plan);
    execute_plan(&plan, &*source, effective.threads).await?;
    cache::create_compat_links(args.link, source_kind, &resolved, &plan.output_root).await?;
    println!("Done: {}", plan.output_root.display());
    Ok(())
}

fn build_source(source_kind: SourceKind, config: &EffectiveConfig) -> Box<dyn ModelSource> {
    match source_kind {
        SourceKind::HuggingFace | SourceKind::HfMirror => {
            Box::new(HuggingFaceSource::new(source_kind, config.hf_token.clone()))
        }
        SourceKind::ModelScope => Box::new(ModelScopeSource::new(config.modelscope_token.clone())),
    }
}

async fn resolve_model(
    args: &DownloadArgs,
    source: &dyn ModelSource,
    source_kind: SourceKind,
    config: &EffectiveConfig,
) -> Result<ResolvedModel> {
    let model_id = if source_kind == SourceKind::ModelScope {
        let ms = ModelScopeSource::new(config.modelscope_token.clone());
        mapping::resolve_modelscope_id(
            &args.model,
            args.modelscope_id.as_deref(),
            args.interactive,
            &ms,
        )
        .await?
    } else {
        args.model.clone()
    };
    source.resolve_model(&model_id, &args.revision).await
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

async fn preflight_disk_space(plan: &DownloadPlan) -> Result<()> {
    fs::create_dir_all(&plan.output_root).await?;
    let needed = plan
        .files
        .iter()
        .filter_map(|task| task.remote.size)
        .sum::<u64>();
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

async fn execute_plan(plan: &DownloadPlan, source: &dyn ModelSource, threads: usize) -> Result<()> {
    let client = Client::builder()
        .user_agent(USER_AGENT)
        .build()
        .map_err(MgetError::Network)?;
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
                download_file_with_retry(&client, headers, task, &progress_state, threads).await
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

fn total_known_size(files: &[FileTask]) -> u64 {
    files.iter().filter_map(|task| task.remote.size).sum()
}

fn print_download_summary(model: &ResolvedModel, source: SourceKind, plan: &DownloadPlan) {
    let total = total_known_size(&plan.files);
    println!("mget {}  {}", source.label(), model.source_id);
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
    threads: usize,
) -> Result<()> {
    let mut last_error = None;
    for attempt in 0..=MAX_RETRIES {
        match download_file(client, headers.clone(), task.clone(), progress, threads).await {
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
    threads: usize,
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
    let has_stream_part = fs::metadata(&task.part)
        .await
        .map(|meta| meta.len() > 0)
        .unwrap_or(false)
        && fs::metadata(&task.state).await.is_err();
    if accept_ranges && size >= LARGE_FILE_CHUNK_THRESHOLD && !has_stream_part {
        let chunk_threads = DEFAULT_CHUNK_THREADS.min(threads).max(1);
        download_range_file(
            client,
            headers,
            &task,
            &progress.total,
            detail_bar.as_ref(),
            size,
            chunk_threads,
        )
        .await?;
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
    verify_file(task).await?;
    Ok(true)
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

async fn download_stream_file(
    client: &Client,
    headers: HeaderMap,
    task: &FileTask,
    total_bar: &ProgressBar,
    detail_bar: Option<&ProgressBar>,
) -> Result<()> {
    let existing = fs::metadata(&task.part)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut request = client.get(&task.url).headers(headers);
    if existing > 0 {
        request = request.header(RANGE, format!("bytes={existing}-"));
        exclude_existing_bytes(total_bar, detail_bar, existing);
    }

    let response = request.send().await?.error_for_status()?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(existing > 0)
        .write(true)
        .open(&task.part)
        .await?;
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
    total_bar: &ProgressBar,
    detail_bar: Option<&ProgressBar>,
    size: u64,
    threads: usize,
) -> Result<()> {
    let chunks = make_chunks(size, DEFAULT_CHUNK_SIZE);
    let chunk_count = chunks.len();
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
    exclude_existing_bytes(total_bar, detail_bar, completed_bytes);

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
        async move {
            download_one_chunk(client, headers, task, index, chunk, total_bar, detail_bar).await?;
            RangeState::mark_complete(&task.state, chunk_count, index).await
        }
    })
    .buffer_unordered(threads.max(1))
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
    _index: usize,
    chunk: ByteRange,
    total_bar: &ProgressBar,
    detail_bar: Option<&ProgressBar>,
) -> Result<()> {
    let response = client
        .get(&task.url)
        .headers(headers)
        .header(RANGE, format!("bytes={}-{}", chunk.start, chunk.end))
        .send()
        .await?
        .error_for_status()?;
    let mut file = OpenOptions::new().write(true).open(&task.part).await?;
    file.seek(SeekFrom::Start(chunk.start)).await?;
    let mut stream = response.bytes_stream();
    while let Some(bytes) = stream.next().await {
        let bytes = bytes?;
        file.write_all(&bytes).await?;
        inc_bars(total_bar, detail_bar, bytes.len() as u64);
    }
    file.flush().await?;
    Ok(())
}

fn inc_bars(total_bar: &ProgressBar, detail_bar: Option<&ProgressBar>, amount: u64) {
    total_bar.inc(amount);
    if let Some(detail_bar) = detail_bar {
        detail_bar.inc(amount);
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
            Ok(content) => {
                let mut state: Self = serde_json::from_str(&content)?;
                state.completed.resize(chunk_count, false);
                Ok(state)
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Self {
                completed: vec![false; chunk_count],
            }),
            Err(err) => Err(err.into()),
        }
    }

    async fn mark_complete(path: &Path, chunk_count: usize, index: usize) -> Result<()> {
        let mut state = Self::load(path, chunk_count).await?;
        if let Some(item) = state.completed.get_mut(index) {
            *item = true;
        }
        let content = serde_json::to_vec_pretty(&state)?;
        fs::write(path, content).await?;
        Ok(())
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
            return Err(MgetError::Message(format!(
                "size mismatch for {}: expected {size}, got {actual}",
                task.target.display()
            )));
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
    Ok(hex::encode(hasher.finalize()))
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
    Ok(hex::encode(hasher.finalize()))
}

async fn cleanup_temp_files(task: &FileTask) -> Result<()> {
    for path in [&task.part, &task.state] {
        match fs::remove_file(path).await {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{DownloadArgs, SourceChoice};

    fn args() -> DownloadArgs {
        DownloadArgs {
            model: "org/model".into(),
            source: SourceChoice::Auto,
            output: None,
            threads: None,
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
}
