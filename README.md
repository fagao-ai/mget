<p align="center">
  <img src="assert/icon.svg" alt="mget icon" width="120" height="120">
</p>

<h1 align="center">mget</h1>

<p align="center">
  An intelligent multi-source downloader for model and dataset repositories.
</p>

<p align="center">
  <a href="#installation">Installation</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#command-reference">Command Reference</a> |
  <a href="#development">Development</a>
</p>

`mget` can download from Hugging Face, HF Mirror, and ModelScope, with source
probing, resumable downloads, optional chunked transfers, and cache compatibility
links.

## Features

- Download model repositories or selected files.
- Download dataset repositories from Hugging Face/HF Mirror and ModelScope.
- Probe available sources and choose a fast source automatically.
- Resume interrupted downloads with `.mget-part` files.
- Use HTTP range requests and chunked downloads for large files when supported.
- Verify checksums or file sizes when metadata is available.
- Filter files with exact paths, include globs, and exclude globs.
- Create compatibility symlinks for Hugging Face and ModelScope cache layouts.
- Support tokens from CLI flags, environment variables, or config files.

## Installation

Install the latest release on macOS/Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/fagao-ai/mget/main/scripts/install.sh | bash
```

Install the latest release on Windows PowerShell:

```powershell
iwr https://raw.githubusercontent.com/fagao-ai/mget/main/scripts/install.ps1 -useb | iex
```

From a local checkout:

```bash
cargo install --path .
```

Or build a release binary:

```bash
cargo build --release
./target/release/mget --help
```

## Quick Start

Diagnose source latency:

```bash
mget ping
```

Download a model repository:

```bash
mget download Qwen/Qwen2.5-0.5B-Instruct
```

Use a specific source:

```bash
mget download Qwen/Qwen2.5-0.5B-Instruct --source hf
mget download Qwen/Qwen2.5-0.5B-Instruct --source hf-mirror
mget download Qwen/Qwen2.5-0.5B-Instruct --source hf-mirror-net
mget download Qwen/Qwen2.5-0.5B-Instruct --source modelscope
```

Download a Hugging Face dataset:

```bash
mget download --dataset SwarmSageGuru/Minist --source hf
```

Download a ModelScope dataset:

```bash
mget download modelscope/test-dataset --repo-type dataset --source modelscope
```

Download selected files:

```bash
mget download Qwen/Qwen2.5-0.5B-Instruct --file README.md
mget download Qwen/Qwen2.5-0.5B-Instruct --include "*.safetensors" --exclude "*00002*"
```

HTTP/1.1 is used by default. Enable HTTP/2 when multiplexing helps on a
high-latency link:

```bash
mget download Qwen/Qwen2.5-0.5B-Instruct --http2
```

Choose an output directory:

```bash
mget download Qwen/Qwen2.5-0.5B-Instruct --output ./Qwen2.5-0.5B-Instruct
```

## Command Reference

```text
mget <COMMAND>

Commands:
  ping      Diagnose current network latency and recommend a source
  download  Download a model or dataset repository, or selected files
```

`download` accepts:

```text
mget download [OPTIONS] [REPO_ID]

Options:
      --model <MODEL_ID>
      --dataset <DATASET_ID>
      --repo-type <model|dataset>
  -s, --source <auto|hf|hf-mirror|hf-mirror-net|modelscope>
  -o, --output <OUTPUT>
  -t, --threads <THREADS>
      --chunk-threads <N>
      --http2
      --revision <REVISION>
      --file <PATH>
      --include <GLOB>
      --exclude <GLOB>
      --link <hf|modelscope|all>
      --interactive
      --hf-token <TOKEN>
      --modelscope-token <TOKEN>
      --modelscope-id <REPO_ID>
```

## Sources

`mget` supports these source choices:

- `auto`: probe sources and choose a fast available source.
- `hf`: use `https://huggingface.co`.
- `hf-mirror`: use `https://hf-mirror.com`, falling back to
  `https://hf-mirror.net` when the primary mirror is unreachable.
- `hf-mirror-net`: force `https://hf-mirror.net` (Cloudflare-backed mirror).
- `modelscope`: use `https://modelscope.cn`.

For datasets, `auto` only probes Hugging Face and the HF Mirrors to avoid
routing a Hugging Face dataset id to ModelScope accidentally. Use
`--source modelscope` explicitly for ModelScope datasets.

`auto` ranks every reachable source and falls back to the next one if a
repository lookup fails, so a HF-only model is not lost just because
ModelScope answered the latency probe first.

`-t/--threads` controls how many files are downloaded at once, while
`--chunk-threads` controls how many range chunks are fetched per large file
(default 4, capped by `--threads`). For a single multi-hundred-GB file,
`--threads 16 --chunk-threads 16` gives the most parallelism.

After SHA256/MD5 verification, mget asks the OS to drop the verified file from
the page cache. This keeps verification from being counted as tens of GB of
process RAM, but the first model load afterwards may need to read the file from
disk again.

## Resuming Downloads

Interrupted downloads leave temporary files next to the target file:

- `*.mget-part`
- `*.mget-state.json` for chunked range downloads

Run the same command again to resume. If the server ignores range requests, mget
falls back safely to a fresh stream download instead of appending invalid data.
If a chunked range download is interrupted, completed chunks and their state
are kept and only the missing chunks are fetched on the next attempt; mget
never discards a partial ranged download to restart it as a stream.
If the state file itself is corrupt, it is moved aside as
`*.mget-state.json.corrupt` and the file restarts safely instead of failing
forever.

## Cache and Compatibility Links

By default, downloads are stored under:

```text
~/.cache/mget/<source>/<repo>/<revision>
```

Use `--output` to choose another location.

Use `--link` to create compatibility symlinks:

```bash
mget download Qwen/Qwen2.5-0.5B-Instruct --link hf
mget download Qwen/Qwen2.5-0.5B-Instruct --link modelscope
mget download Qwen/Qwen2.5-0.5B-Instruct --link all
```

## Authentication

CLI flags:

```bash
mget download private/repo --hf-token "$HF_TOKEN"
mget download private/repo --source modelscope --modelscope-token "$MODELSCOPE_API_TOKEN"
```

Environment variables:

```bash
export HF_TOKEN=...
export HUGGINGFACE_HUB_TOKEN=...
export MODELSCOPE_API_TOKEN=...
```

Config file values are also supported:

```toml
hf_token = "..."
modelscope_token = "..."
default_threads = 6
default_chunk_threads = 4
cache_dir = "/path/to/cache"
```

The config file is read from the platform-specific config directory for `mget`.

## Development

Useful checks:

```bash
cargo fmt --all
cargo test
cargo clippy --all-targets -- -D warnings
```

## Release Versioning

Releases are tag-driven. Pushing a tag such as `v0.2.0` makes the release workflow rewrite
`Cargo.toml` and `Cargo.lock` from the tag before building, so `mget --version` reports the
tagged version.

Long tags are supported too. `v0.2.0.20261022` is shown as `mget 0.2.0.20261022`; Cargo uses
the semver-compatible form `0.2.0-20261022` internally. Local builds use
`0.2.0+YYYYMMDD.<commit>` so they are distinguishable from release binaries.

## License

MIT
