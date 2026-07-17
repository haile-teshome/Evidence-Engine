# Build a SELF-CONTAINED Evidence Engine bundle for Windows x64.
# End users need no system Node.js or Python: this bundles a Node runtime, a
# relocatable python-build-standalone interpreter, and vendored pip wheels next
# to the prebuilt frontend. Run on a Windows x64 machine (or CI runner) so the
# vendored wheels' compiled extensions match.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build-windows.ps1
$ErrorActionPreference = "Stop"

# --- pinned versions (keep in sync with packaging/build.sh) -------------------
$NodeVersion = "v22.11.0"
$PbsTag      = "20241016"
$PyVersion   = "3.12.7"

$Root  = (Resolve-Path "$PSScriptRoot\..").Path
$Out   = Join-Path $Root "dist-bundle\win-x64"
$Stage = Join-Path $Out "Evidence Engine"
$Cache = Join-Path $Root ".build-cache"
New-Item -ItemType Directory -Force -Path $Cache | Out-Null

$NodeUrl = "https://nodejs.org/dist/$NodeVersion/node-$NodeVersion-win-x64.zip"
$PyUrl   = "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsTag/cpython-$PyVersion+$PbsTag-x86_64-pc-windows-msvc-install_only.tar.gz"
$NodeZip = Join-Path $Cache "node-$NodeVersion-win-x64.zip"
$PyTgz   = Join-Path $Cache "cpython-$PyVersion-win.tar.gz"

Write-Host "==> [1/6] Building the frontend (dist/)"
Push-Location $Root; npm run build | Out-Null; Pop-Location

Write-Host "==> [2/6] Fetching Node $NodeVersion (win-x64)"
if (-not (Test-Path $NodeZip)) { Invoke-WebRequest -Uri $NodeUrl -OutFile $NodeZip }

Write-Host "==> [3/6] Fetching python-build-standalone $PyVersion (windows-msvc)"
if (-not (Test-Path $PyTgz)) { Invoke-WebRequest -Uri $PyUrl -OutFile $PyTgz }

Write-Host "==> [4/6] Assembling bundle -> $Stage"
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
# Copy the project, excluding dev/build cruft and user-side regenerated dirs.
$exclude = @(".git", ".build-cache", "dist-bundle", "node_modules", "runtime", "wheels", ".venv", "__pycache__", ".env.local", ".env")
robocopy $Root $Stage /E /XD ($exclude | ForEach-Object { Join-Path $Root $_ }) /XF ".env.local" | Out-Null

New-Item -ItemType Directory -Force -Path (Join-Path $Stage "runtime\node") | Out-Null
Expand-Archive -Path $NodeZip -DestinationPath (Join-Path $Stage "runtime\node-tmp") -Force
# node zip extracts to node-vX-win-x64\... ; flatten one level into runtime\node
$inner = Get-ChildItem (Join-Path $Stage "runtime\node-tmp") | Select-Object -First 1
Move-Item (Join-Path $inner.FullName "*") (Join-Path $Stage "runtime\node") -Force
Remove-Item -Recurse -Force (Join-Path $Stage "runtime\node-tmp")
# python-build-standalone tar.gz -> runtime\python\python.exe
tar -xzf $PyTgz -C (Join-Path $Stage "runtime")

Write-Host "==> [5/6] Vendoring backend wheels with the bundled Python"
$Bpy = Join-Path $Stage "runtime\python\python.exe"
& $Bpy -m pip download -r (Join-Path $Stage "Backend\requirements.txt") -d (Join-Path $Stage "Backend\wheels") | Out-Null

Write-Host "==> [6/6] Zipping"
Compress-Archive -Path $Stage -DestinationPath (Join-Path $Out "..\EvidenceEngine-win-x64.zip") -Force
Write-Host "==> Done: dist-bundle\EvidenceEngine-win-x64.zip"
Write-Host "    Users unzip it and double-click launch.bat."
