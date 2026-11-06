# Packaging and releases

These scripts produce a **download-and-run** Evidence Engine: the end user needs
no system Node.js or Python. Each build ships:

- `runtime/node/` — a Node runtime (runs the launcher + serves the built frontend)
- `runtime/python/` — a relocatable [python-build-standalone](https://github.com/astral-sh/python-build-standalone) interpreter
- `Backend/wheels/` — vendored pip wheels, so the backend venv is built **offline** on first run
- the prebuilt `dist/` frontend and the project source
- a `VERSION` stamp, which is what the app reports and compares against

Only **Ollama + the ~4 GB model** download from the network on first run (too big
to ship in the download; the launcher fetches them automatically).

## Nothing is written inside the install

The venv, the env file and the database all live in `~/.evidence-engine`
(`%USERPROFILE%\.evidence-engine` on Windows). That is what makes the macOS app
installable into `/Applications`, and what makes updating safe: the whole app can
be replaced without touching a single review. A source checkout keeps using
`Backend/.venv` and `Backend/.env` when they already exist, so development is
unaffected.

## Build

Run on (or matching) the target platform so the vendored wheels' compiled
extensions match the target's ABI.

```bash
packaging/build.sh darwin-arm64     # Apple Silicon → .dmg
packaging/build.sh darwin-x64       # Intel Mac     → .dmg
packaging/build.sh linux-x64        # Linux         → .zip
# Windows (needs Inno Setup for the installer; falls back to a zip without it):
powershell -ExecutionPolicy Bypass -File packaging/build-windows.ps1
```

Output lands in `dist-bundle/`:

| Platform | Artifact | How the user installs it |
|---|---|---|
| macOS | `EvidenceEngine-<version>-darwin-{arm64,x64}.dmg` | Open, drag to Applications |
| Windows | `EvidenceEngine-<version>-win-x64-Setup.exe` | Run it, open from the Start Menu |
| Linux | `EvidenceEngine-<version>-linux-x64.zip` | Unzip, run `launch.sh` |

Both build scripts **refuse to finish** if the staged payload contains an
environment file, anything key-shaped, or a developer cache. A leaked key costs
more than a failed build.

## Releasing

The version lives in exactly one place, `package.json`. Everything else reads it:
the `VERSION` stamp, the macOS `Info.plist`, the installer, the artifact names,
and `/api/update`.

```bash
# 1. bump the version
npm version 1.1.0 --no-git-tag-version   # or edit package.json by hand
git commit -am "Release 1.1.0"

# 2. tag and push; CI builds all four targets and publishes the release
git tag v1.1.0
git push origin main v1.1.0
```

`.github/workflows/release.yml` builds each target on its own runner, writes
`SHA256SUMS`, and publishes a GitHub Release. It **fails fast if the tag and
`package.json` disagree**, since that mismatch would either nag every user
forever or never notify them at all.

That published release is what the in-app check reads. `Backend/updates.py` asks
`api.github.com` for the latest release at most once a day, caches the answer in
`~/.evidence-engine/update-check.json`, and the sidebar shows a banner with a
link. Set `EE_UPDATE_CHECK=0` to switch it off entirely; `EE_UPDATE_REPO` points
it at a different repository.

## Code signing

The builds are **unsigned**, so the first open takes one extra step. That is
documented for users in `packaging/INSTALL-macOS.txt` (shipped inside the .dmg)
and in the release notes:

- macOS: control-click the app and choose Open, or System Settings → Privacy &
  Security → "Open Anyway".
- Windows: "More info" → "Run anyway" on the SmartScreen prompt.

macOS builds are ad-hoc signed (`codesign --sign -`). That is not a substitute
for a Developer ID; it is there because on Apple Silicon every binary needs at
least an ad-hoc signature to run at all.

To ship without any warning you need paid certificates. The bundle layout does
not change when signing is added, so this can be bolted on later.

### macOS (needs an Apple Developer ID, about $99/yr)

```bash
# Sign the bundle, hardened runtime on, from the inside out.
codesign --force --deep --options runtime --timestamp \
  --sign "Developer ID Application: YOUR NAME (TEAMID)" \
  "dist-bundle/darwin-arm64/Evidence Engine.app"

# Notarize the .dmg, then staple the ticket into it so it works offline.
xcrun notarytool submit "dist-bundle/EvidenceEngine-1.1.0-darwin-arm64.dmg" \
  --apple-id "you@example.edu" --team-id TEAMID \
  --password "app-specific-password" --wait
xcrun stapler staple "dist-bundle/EvidenceEngine-1.1.0-darwin-arm64.dmg"

# Verify what a user's machine will conclude.
spctl -a -t open --context context:primary-signature -vvv \
  "dist-bundle/EvidenceEngine-1.1.0-darwin-arm64.dmg"
```

In CI, store the certificate as a base64 secret, import it into a temporary
keychain, and keep the notarization password in a secret too. Never commit either.

### Windows (needs an Authenticode certificate)

```powershell
signtool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 `
  /a "dist-bundle\EvidenceEngine-1.1.0-win-x64-Setup.exe"
```

An OV certificate still leaves a SmartScreen prompt until the binary builds
reputation; an EV certificate avoids it from the first download.

## Versions

The runtimes are pinned in each script (`NODE_VERSION`, `PBS_TAG`, `PY_VERSION`)
for reproducible builds. Bump deliberately and keep `build.sh` and
`build-windows.ps1` in sync.
