param(
    [switch]$PrepareOnly,
    [string]$FactsRoot,
    [string]$WorkspaceRoot
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runtimeRoot = Join-Path $projectRoot '.local/docling-studio'
$commit = 'e95680a0f0d921a112883091ff078ea13b535e65'
$zipHash = '87B379BA705113E2CE5052C13554AA87D18F90E630C52097B036C44D63779DE4'
$sourceRoot = Join-Path $runtimeRoot "docling-Studio-$commit"
if (-not $FactsRoot) { $FactsRoot = Join-Path $projectRoot '.local/facts-v2-release' }
if (-not [IO.Path]::IsPathRooted($FactsRoot)) { $FactsRoot = Join-Path $projectRoot $FactsRoot }
if (-not $WorkspaceRoot) { $WorkspaceRoot = Join-Path $runtimeRoot 'workspace-v2' }
if (-not [IO.Path]::IsPathRooted($WorkspaceRoot)) { $WorkspaceRoot = Join-Path $projectRoot $WorkspaceRoot }
$backend = Join-Path $sourceRoot 'document-parser'
$frontend = Join-Path $sourceRoot 'frontend'
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
foreach ($command in @('uv','npm.cmd','node.exe','curl.exe','pdftoppm','pdfinfo')) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) { throw "Missing prerequisite: $command" }
}
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    $archivePath = Join-Path $runtimeRoot 'source.zip'
    & curl.exe --fail --silent --show-error --max-time 120 "https://codeload.github.com/scub-france/Docling-Studio/zip/$commit" -o $archivePath
    if ($LASTEXITCODE -ne 0) { throw 'Studio download failed.' }
    if ((Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash -ne $zipHash) { throw 'Studio archive checksum mismatch.' }
    Expand-Archive -LiteralPath $archivePath -DestinationPath $runtimeRoot
}
Push-Location $backend
try {
    & uv sync --frozen --no-dev --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'Studio Python setup failed.' }
} finally { Pop-Location }
Push-Location $frontend
try {
    if (-not (Test-Path -LiteralPath 'node_modules/.package-lock.json')) {
        & npm.cmd ci --no-audit --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'Studio frontend setup failed.' }
    }
} finally { Pop-Location }
$python = Join-Path $backend '.venv/Scripts/python.exe'
& $python (Join-Path $PSScriptRoot 'patch_upstream.py') $sourceRoot
if ($LASTEXITCODE -ne 0) { throw 'Compatibility patch failed.' }
& $python (Join-Path $PSScriptRoot 'import_facts.py') --studio $sourceRoot --facts $FactsRoot --output $WorkspaceRoot
if ($LASTEXITCODE -ne 0) { throw 'Fact import failed.' }
if ($PrepareOnly) { return }
foreach ($port in @(57569,57570)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is occupied. Use the running Studio, or stop its process before restarting."
    }
}
# Studio receives no project .env. Parsing/reasoning/index services are not connected.
$env:DB_PATH = Join-Path $WorkspaceRoot 'studio.sqlite'
$env:UPLOAD_DIR = Join-Path $WorkspaceRoot 'uploads'
$env:CONVERSION_ENGINE = 'remote'
$env:DOCLING_SERVE_URL = 'http://127.0.0.1:9'
$env:DOCLING_SERVE_API_KEY = ''
$env:REASONING_ENABLED = 'false'
$env:EMBEDDING_URL = ''
$env:OPENSEARCH_URL = ''
$env:NEO4J_URI = ''
$env:CORS_ORIGINS = 'http://127.0.0.1:57569'
$env:APP_VERSION = "0.7.1-$($commit.Substring(0,7))"
$env:VITE_APP_VERSION = $env:APP_VERSION
$env:VITE_API_PROXY_TARGET = 'http://127.0.0.1:57570'
$apiProcess = Start-Process -FilePath $python -ArgumentList '-m uvicorn main:app --host 127.0.0.1 --port 57570' -WorkingDirectory $backend -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeRoot 'backend.log') -RedirectStandardError (Join-Path $runtimeRoot 'backend-error.log')
$uiProcess = Start-Process -FilePath (Get-Command node.exe).Source -ArgumentList 'node_modules/vite/bin/vite.js --host 127.0.0.1 --port 57569 --strictPort' -WorkingDirectory $frontend -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeRoot 'frontend.log') -RedirectStandardError (Join-Path $runtimeRoot 'frontend-error.log')
@{ backend_pid=$apiProcess.Id; frontend_pid=$uiProcess.Id; source_commit=$commit; url='http://127.0.0.1:57569/analyses' } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $runtimeRoot 'server.json') -Encoding utf8
Write-Output 'Docling Studio: http://127.0.0.1:57569/analyses'
Write-Output "Logs and process IDs: $runtimeRoot"
