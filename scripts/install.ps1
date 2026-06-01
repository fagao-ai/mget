#!/usr/bin/env pwsh

$ErrorActionPreference = "Stop"

$latestRelease = Invoke-RestMethod -Uri "https://api.github.com/repos/fagao-ai/mget/releases/latest"
$version = $latestRelease.tag_name
Write-Host "Installing mget $version for Windows..."

$downloadUrl = "https://github.com/fagao-ai/mget/releases/download/$version/mget-windows-amd64.exe"
$exePath = "$env:TEMP\mget.exe"

Write-Host "Downloading from $downloadUrl..."
Invoke-WebRequest -Uri $downloadUrl -OutFile $exePath

$binDir = "$env:USERPROFILE\.local\bin"
New-Item -ItemType Directory -Force -Path $binDir | Out-Null

Write-Host "Installing to $binDir..."
Move-Item -Path $exePath -Destination "$binDir\mget.exe" -Force

Remove-Item -Path $exePath -Force -ErrorAction SilentlyContinue

Write-Host "mget has been installed to $binDir\mget.exe"
Write-Host ""
Write-Host "Add to PATH if needed:"
Write-Host "  [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';$binDir', 'User')"
Write-Host ""
Write-Host "Then restart your terminal and verify installation:"
Write-Host "  mget --version"
