#!/bin/bash
set -e

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)
    OS_TYPE="linux"
    ;;
  Darwin)
    OS_TYPE="darwin"
    ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64)
    ARCH_TYPE="amd64"
    ;;
  aarch64|arm64)
    ARCH_TYPE="arm64"
    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

LATEST_VERSION=$(curl -s https://api.github.com/repos/fagao-ai/mget/releases/latest | grep '"tag_name"' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_VERSION" ]; then
  echo "Failed to fetch latest version"
  exit 1
fi

echo "Installing mget $LATEST_VERSION for $OS_TYPE-$ARCH_TYPE..."

BINARY_NAME="mget-$OS_TYPE-$ARCH_TYPE"
DOWNLOAD_URL="https://github.com/fagao-ai/mget/releases/download/${LATEST_VERSION}/${BINARY_NAME}"
echo "Downloading from $DOWNLOAD_URL..."

TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"
curl -L "$DOWNLOAD_URL" -o mget

if [ -w "/usr/local/bin" ]; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="$HOME/.local/bin"
  mkdir -p "$BIN_DIR"
fi

echo "Installing to $BIN_DIR..."
chmod +x mget
mv mget "$BIN_DIR/mget"

cd -
rm -rf "$TMP_DIR"

echo "mget has been installed to $BIN_DIR/mget"
echo ""
echo "Add to PATH if needed:"
if [ "$BIN_DIR" = "$HOME/.local/bin" ]; then
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi
echo ""
echo "Verify installation:"
echo "  mget --version"
