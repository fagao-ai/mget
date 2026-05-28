use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use futures_util::{StreamExt, stream};
use globset::{Glob, GlobSet, GlobSetBuilder};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
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
const LARGE_FILE_CHUNK_THRESHOLD: u64 = 64 * 1024 * 1024;
const DEFAULT_CHUNK_SIZE: u64 = 16 * 1024 * 1024;
const MAX_RETRIES: usize = 3;

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
    println!(
        "Downloading {} [{}]",
        resolved.source_id,
        source.source_kind().label()
    );

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
    let progress = MultiProgress::new();
    let root = progress.add(ProgressBar::new(total_known_size(&plan.files)));
    root.set_style(progress_style());
    root.set_message(format!("{} files", plan.files.len()));

    let semaphore = Arc::new(Semaphore::new(threads));
    stream::iter(plan.files.clone())
        .map(|task| {
            let client = client.clone();
            let headers = source.auth_headers();
            let semaphore = semaphore.clone();
            let root = root.clone();
            let progress = progress.clone();
            async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|err| MgetError::Message(err.to_string()))?;
                download_file_with_retry(&client, headers, task, &progress, &root, threads).await
            }
        })
        .buffer_unordered(threads)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    root.finish_with_message("complete");
    Ok(())
}

fn total_known_size(files: &[FileTask]) -> u64 {
    files.iter().filter_map(|task| task.remote.size).sum()
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "[{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta} {msg}",
    )
    .unwrap()
    .progress_chars("=>-")
}

async fn download_file_with_retry(
    client: &Client,
    headers: HeaderMap,
    task: FileTask,
    progress: &MultiProgress,
    root: &ProgressBar,
    threads: usize,
) -> Result<()> {
    let mut last_error = None;
    for attempt in 0..=MAX_RETRIES {
        match download_file(
            client,
            headers.clone(),
            task.clone(),
            progress,
            root,
            threads,
        )
        .await
        {
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
    progress: &MultiProgress,
    root: &ProgressBar,
    threads: usize,
) -> Result<()> {
    if target_is_complete(&task).await? {
        if let Some(size) = task.remote.size {
            root.inc(size);
        }
        return Ok(());
    }

    if let Some(parent) = task.target.parent() {
        fs::create_dir_all(parent).await?;
    }

    let size = task.remote.size.unwrap_or(0);
    let bar = progress.add(ProgressBar::new(size));
    bar.set_style(progress_style());
    bar.set_message(task.remote.path.clone());

    let accept_ranges = supports_ranges(client, &headers, &task)
        .await
        .unwrap_or(false);
    if accept_ranges && size >= LARGE_FILE_CHUNK_THRESHOLD {
        download_range_file(client, headers, &task, &bar, root, size, threads).await?;
    } else {
        download_stream_file(client, headers, &task, &bar, root).await?;
    }

    verify_file(&task).await?;
    cleanup_temp_files(&task).await?;
    bar.finish_with_message(format!("{} complete", task.remote.path));
    Ok(())
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
    bar: &ProgressBar,
    root: &ProgressBar,
) -> Result<()> {
    let existing = fs::metadata(&task.part)
        .await
        .map(|meta| meta.len())
        .unwrap_or(0);
    let mut request = client.get(&task.url).headers(headers);
    if existing > 0 {
        request = request.header(RANGE, format!("bytes={existing}-"));
        bar.set_position(existing);
        root.inc(existing);
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
        bar.inc(chunk.len() as u64);
        root.inc(chunk.len() as u64);
    }
    file.flush().await?;
    fs::rename(&task.part, &task.target).await?;
    Ok(())
}

async fn download_range_file(
    client: &Client,
    headers: HeaderMap,
    task: &FileTask,
    bar: &ProgressBar,
    root: &ProgressBar,
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
    bar.set_position(completed_bytes);
    root.inc(completed_bytes);

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
            download_one_chunk(client, headers, task, index, chunk, bar, root).await?;
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
    bar: &ProgressBar,
    root: &ProgressBar,
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
        bar.inc(bytes.len() as u64);
        root.inc(bytes.len() as u64);
    }
    file.flush().await?;
    Ok(())
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
