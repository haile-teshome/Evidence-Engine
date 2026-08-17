# UniTable table-recognition model (stack component)

Image-based table recognition (image -> HTML) used as a higher-fidelity fallback
to pdfplumber in `_extract_pdf_tables`. MIT-licensed, ~0.94-0.96 TEDS on
PubTabNet val. Runs on Apple MPS / CUDA / CPU.

- `src/`, `vocab/` : vendored UniTable inference code (no training deps).
- `weights/`       : place the 3 weight files here (NOT committed, ~1.5 GB):
    unitable_large_structure.pt  unitable_large_bbox.pt  unitable_large_content.pt
  from https://huggingface.co/poloclub/UniTable  (or set EE_UNITABLE_WEIGHTS to
  a directory that contains them).

Enable at runtime with EE_USE_UNITABLE=1. Off by default; when off the platform
falls back to pdfplumber exactly as before.

Extra Python deps (only needed when enabled): torch, torchvision, tokenizers,
einops, beautifulsoup4.
