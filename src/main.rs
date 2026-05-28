mod cache;
mod cli;
mod config;
mod download;
mod error;
mod mapping;
mod routing;
mod source;

use clap::Parser;
use cli::{Cli, Commands};
use error::Result;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mget=warn".into()),
        )
        .without_time()
        .init();

    if let Err(err) = run().await {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

async fn run() -> Result<()> {
    let cli = Cli::parse();
    let config = config::Config::load()?;

    match cli.command {
        Commands::Ping => routing::print_ping().await,
        Commands::Download(args) => download::run_download(*args, config).await,
    }
}
