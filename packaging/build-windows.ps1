# Build a SELF-CONTAINED Evidence Engine for Windows x64.
# End users need no system Node.js or Python: this bundles a Node runtime, a
# relocatable python-build-standalone interpreter, and vendored pip wheels next
# to the prebuilt frontend. Run on a Windows x64 machine (or CI runner) so the
# vendored wheels' compiled extensions match.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build-windows.ps1
#
# Output, in order of preference:
#   dist-bundle\EvidenceEngine-<version>-win-x64-Setup.exe   (if Inno Setup is installed)
#   dist-bundle\EvidenceEngine-<version>-win-x64.zip         (always, as a fallback)
#
# Install Inno Setup to get the one-double-click installer:
#   winget install -e --id JRSoftware.InnoSetup
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

# package.json is the single place the version is edited.
$Version = (Get-Content (Join-Path $Root "package.json") -Raw | ConvertFrom-Json).version
if (-not $Version) { throw "Could not read the version from package.json" }

$NodeUrl = "https://nodejs.org/dist/$NodeVersion/node-$NodeVersion-win-x64.zip"
$PyUrl   = "https://github.com/astral-sh/python-build-standalone/releases/download/$PbsTag/cpython-$PyVersion+$PbsTag-x86_64-pc-windows-msvc-install_only.tar.gz"
$NodeZip = Join-Path $Cache "node-$NodeVersion-win-x64.zip"
$PyTgz   = Join-Path $Cache "cpython-$PyVersion-win.tar.gz"

Write-Host "==> Evidence Engine $Version (win-x64)"

Write-Host "==> [1/7] Building the frontend (dist/)"
Push-Location $Root; npm run build | Out-Null; Pop-Location

Write-Host "==> [2/7] Fetching Node $NodeVersion (win-x64)"
if (-not (Test-Path $NodeZip)) { Invoke-WebRequest -Uri $NodeUrl -OutFile $NodeZip }

Write-Host "==> [3/7] Fetching python-build-standalone $PyVersion (windows-msvc)"
if (-not (Test-Path $PyTgz)) { Invoke-WebRequest -Uri $PyUrl -OutFile $PyTgz }

Write-Host "==> [4/7] Assembling bundle -> $Stage"
if (Test-Path $Out) { Remove-Item -Recurse -Force $Out }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
# Excludes decide whether this ships the app or the developer's machine. Kept in
# step with packaging/build.sh: user caches, the video project, benchmark and
# report trees, and anything that could hold a credential.
$excludeDirs = @(
  ".git", ".build-cache", "dist-bundle", "node_modules", "runtime", "wheels",
  ".venv", "__pycache__", ".rank_cache", ".hf_cache", "textcache", "pdfcache",
  "video", "benchmark", "reports", "eval", ".pytest_cache", "htmlcov",
  ".claude", ".githooks", "Evidence Engine.app"
) | ForEach-Object { Join-Path $Root $_ }
# .env.example is deliberately NOT excluded: the launcher needs it to write the
# user's env file on first run. Real env files are pruned straight after.
$excludeFiles = @(".env", ".env.local", "*.bak-*", ".coverage", "coverage.json", ".DS_Store")
robocopy $Root $Stage /E /XD $excludeDirs /XF $excludeFiles /NFL /NDL /NJH /NJS | Out-Null
Get-ChildItem -Path $Stage -Recurse -Force -File -Filter ".env.*" -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -ne ".env.example" } | Remove-Item -Force

Set-Content -Path (Join-Path $Stage "VERSION") -Value $Version -NoNewline

Write-Host "==> [5/7] Auditing the staged payload"
# Fail the build rather than ship a credential: a leaked key costs more than a
# failed build. Mirrors the audit in packaging/build.sh.
$leaks = Get-ChildItem -Path $Stage -Recurse -Force -File -Include ".env", ".env.*", "*.bak-*" `
         -ErrorAction SilentlyContinue | Where-Object { $_.Name -ne ".env.example" }
if ($leaks) {
  Write-Host "REFUSING TO BUILD: environment files staged:"
  $leaks | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.FullName)" }
  exit 1
}
$keyish = Get-ChildItem -Path $Stage -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Length -lt 2MB } |
  Select-String -Pattern 'sk-ant-[A-Za-z0-9_-]{20}', 'AIza[0-9A-Za-z_-]{30}', 'sk-[A-Za-z0-9]{40}' `
                -List -ErrorAction SilentlyContinue
if ($keyish) {
  Write-Host "REFUSING TO BUILD: something in the bundle looks like an API key:"
  $keyish | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.Path)" }
  exit 1
}
Write-Host "    no credentials staged"

Write-Host "==> [6/7] Adding the runtimes and vendoring backend wheels"
New-Item -ItemType Directory -Force -Path (Join-Path $Stage "runtime\node") | Out-Null
Expand-Archive -Path $NodeZip -DestinationPath (Join-Path $Stage "runtime\node-tmp") -Force
# node zip extracts to node-vX-win-x64\... ; flatten one level into runtime\node
$inner = Get-ChildItem (Join-Path $Stage "runtime\node-tmp") | Select-Object -First 1
Move-Item (Join-Path $inner.FullName "*") (Join-Path $Stage "runtime\node") -Force
Remove-Item -Recurse -Force (Join-Path $Stage "runtime\node-tmp")
# python-build-standalone tar.gz -> runtime\python\python.exe
tar -xzf $PyTgz -C (Join-Path $Stage "runtime")

$Bpy = Join-Path $Stage "runtime\python\python.exe"
& $Bpy -m pip download -r (Join-Path $Stage "Backend\requirements.txt") -d (Join-Path $Stage "Backend\wheels") | Out-Null

Write-Host "==> [7/7] Packaging"
$Zip = Join-Path $Root "dist-bundle\EvidenceEngine-$Version-win-x64.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path $Stage -DestinationPath $Zip -Force
Write-Host "==> Done: $Zip"

# The installer is the route we actually want people to take. Built when Inno
# Setup is present; the zip above remains for anyone who would rather unzip.
$iscc = @(
  "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
  "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) { $iscc = (Get-Command ISCC.exe -ErrorAction SilentlyContinue).Source }

if ($iscc) {
  & $iscc "/DAppVersion=$Version" "/DPayloadDir=$Stage" `
          "/DOutDir=$(Join-Path $Root 'dist-bundle')" `
          (Join-Path $PSScriptRoot "installer.iss") | Out-Null
  Write-Host "==> Done: dist-bundle\EvidenceEngine-$Version-win-x64-Setup.exe"
  Write-Host "    Users download it, double-click, and open Evidence Engine from the Start Menu."
} else {
  Write-Host "==> Inno Setup not found, so no installer was built (the zip above still works)."
  Write-Host "    Install it with: winget install -e --id JRSoftware.InnoSetup"
}
