#!/usr/bin/env python3
"""Does this interpreter have everything the backend needs?

Run by packaging/build.sh and build-windows.ps1 against the interpreter that
ships inside the bundle, before the bundle is wrapped up. A dependency that did
not install is a build failure here rather than something a reviewer discovers
on their own laptop, where there is nothing they can do about it.

requirements.txt is the source of truth, markers and all. A hand-written list
of imports was the first attempt and it failed the Intel build on its first run:
docling is excluded on x86_64 macOS on purpose, because PyTorch stopped
publishing wheels for it, and a second copy of that rule in a build script had
no way to know. Asking the requirements file what applies to THIS interpreter
cannot drift from what pip was told to install.

Distribution names are checked, not import names: `biopython` imports as `Bio`
and `beautifulsoup4` as `bs4`, and a mapping between them is one more thing to
keep right. A short import smoke test follows, to prove the interpreter runs at
all and that the web stack is wired up.
"""
import sys
from importlib.metadata import PackageNotFoundError, distribution

SMOKE = ["fastapi", "uvicorn", "pydantic", "requests"]


def wanted(path):
    """Every requirement whose marker applies to the interpreter running this."""
    from packaging.requirements import Requirement

    out = []
    for raw in open(path, encoding="utf-8"):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            req = Requirement(line)
        except Exception as e:                      # a line we cannot parse is a
            print(f"  cannot read requirement {line!r}: {e}")   # build problem too
            return None
        if req.marker and not req.marker.evaluate():
            continue
        out.append(req.name)
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: verify_bundle.py <requirements.txt>")
        return 2
    names = wanted(sys.argv[1])
    if names is None:
        return 1

    missing = []
    for name in names:
        try:
            distribution(name)
        except PackageNotFoundError:
            missing.append(name)

    broken = []
    for mod in SMOKE:
        try:
            __import__(mod)
        except Exception as e:
            broken.append(f"{mod} ({e.__class__.__name__}: {e})")

    if missing or broken:
        if missing:
            print(f"  not installed: {', '.join(sorted(missing))}")
        if broken:
            print(f"  will not import: {'; '.join(broken)}")
        return 1

    print(f"  {len(names)} requirements present, and the backend's own imports work")
    return 0


if __name__ == "__main__":
    sys.exit(main())
