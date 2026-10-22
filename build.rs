use std::env;
use std::path::Path;
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn main() {
    println!("cargo:rerun-if-env-changed=MGET_VERSION");
    for path in [".git/HEAD", ".git/refs/heads/main"] {
        if Path::new(path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }

    let package_version = env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
    let release_version = env::var("MGET_VERSION")
        .ok()
        .map(|value| value.trim().trim_start_matches('v').to_string())
        .filter(|value| !value.is_empty());

    let version = release_version.unwrap_or_else(|| {
        // Local builds carry a long, reproducible-ish version: 0.2.0+20261022.0392b25.
        match (
            git(&["show", "-s", "--format=%cd", "--date=format:%Y%m%d", "HEAD"]),
            git(&["rev-parse", "--short=7", "HEAD"]),
        ) {
            (Some(date), Some(sha)) => format!("{package_version}+{date}.{sha}"),
            _ => package_version,
        }
    });

    println!("cargo:rustc-env=MGET_VERSION={version}");
}
