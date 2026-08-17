"""Image-based table recognition using UniTable (Large), as a higher-fidelity
fallback for the deterministic pdfplumber table extractor.

pdfplumber recovers bordered/aligned tables from the PDF text layer, but fails on
scanned pages and complex spanning tables. UniTable reads a rendered page/table
IMAGE and reconstructs the table as HTML (structure + cell text), reaching
~0.94-0.96 TEDS on PubTabNet val.

Everything here is LAZY and GUARDED: if torch, the vendored model code, or the
weights are unavailable, every entry point returns None and the caller falls back
to the existing behavior. The feature is OFF unless EE_USE_UNITABLE=1 AND the
weights directory exists, so the default shipped behavior is unchanged.

Model stack: local, MIT-licensed, ~1.5 GB of weights (structure/bbox/content),
runs on Apple MPS / CUDA / CPU. Weights are NOT bundled in the repo; point
EE_UNITABLE_WEIGHTS at a directory containing:
    unitable_large_structure.pt, unitable_large_bbox.pt, unitable_large_content.pt
"""
from __future__ import annotations

import os
import re
from functools import lru_cache, partial
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE / "table_model"
_WEIGHTS = Path(os.getenv("EE_UNITABLE_WEIGHTS", str(_MODEL_DIR / "weights")))
_ENABLED = os.getenv("EE_USE_UNITABLE") == "1"

_D_MODEL, _PATCH, _NHEAD, _DROP = 768, 16, 12, 0.2
_WEIGHT_FILES = ("unitable_large_structure.pt", "unitable_large_bbox.pt", "unitable_large_content.pt")


def available() -> bool:
    """True only if the feature is enabled, weights exist, and torch imports."""
    if not _ENABLED:
        return False
    if not all((_WEIGHTS / w).is_file() for w in _WEIGHT_FILES):
        return False
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load_models():
    """Load the three UniTable models once. Returns None on any failure."""
    try:
        import sys
        if str(_MODEL_DIR) not in sys.path:
            sys.path.insert(0, str(_MODEL_DIR))
        import torch
        import tokenizers as tk
        from torch import nn
        from src.model import EncoderDecoder, ImgLinearBackbone, Encoder, Decoder

        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda:0" if torch.cuda.is_available() else "cpu"))

        def build(vocab_path, max_seq_len, weights):
            vocab = tk.Tokenizer.from_file(str(_MODEL_DIR / vocab_path))
            model = EncoderDecoder(
                backbone=ImgLinearBackbone(d_model=_D_MODEL, patch_size=_PATCH),
                encoder=Encoder(d_model=_D_MODEL, nhead=_NHEAD, dropout=_DROP, activation="gelu",
                                norm_first=True, nlayer=12, ff_ratio=4),
                decoder=Decoder(d_model=_D_MODEL, nhead=_NHEAD, dropout=_DROP, activation="gelu",
                                norm_first=True, nlayer=4, ff_ratio=4),
                vocab_size=vocab.get_vocab_size(), d_model=_D_MODEL,
                padding_idx=vocab.token_to_id("<pad>"), max_seq_len=max_seq_len,
                dropout=_DROP, norm_layer=partial(nn.LayerNorm, eps=1e-6))
            model.load_state_dict(torch.load(_WEIGHTS / weights, map_location="cpu"))
            return vocab, model.to(device).eval()

        v_s, m_s = build("vocab/vocab_html.json", 784, _WEIGHT_FILES[0])
        v_b, m_b = build("vocab/vocab_bbox.json", 1024, _WEIGHT_FILES[1])
        v_c, m_c = build("vocab/vocab_cell_6k.json", 200, _WEIGHT_FILES[2])
        return {"device": device, "s": (v_s, m_s), "b": (v_b, m_b), "c": (v_c, m_c)}
    except Exception as e:  # noqa: BLE001
        print(f"[table_recognition] UniTable load failed: {e}")
        return None


def _html_to_rows(html: str) -> List[List[str]]:
    """Flatten a UniTable HTML table into rows of plain-text cells."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        rows = []
        for tr in soup.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                txt = re.sub(r"\s+", " ", td.get_text(" ", strip=True)).strip()
                span = int(td.get("colspan", 1) or 1)
                cells.extend([txt] + [""] * (span - 1))
            if any(c for c in cells):
                rows.append(cells)
        return rows
    except Exception:
        return []


def recognize_table_image(image) -> Optional[List[List[str]]]:
    """PIL.Image -> rows of cell text, or None if UniTable is unavailable/fails."""
    if not available():
        return None
    models = _load_models()
    if not models:
        return None
    try:
        import torch
        from torchvision import transforms
        from src.utils import (subsequent_mask, pred_token_within_range, greedy_sampling,
                               bbox_str_to_token_list, cell_str_to_token_list,
                               html_str_to_token_list, build_table_from_html_and_cell,
                               html_table_template)
        from src.vocab import HTML_TOKENS, TASK_TOKENS, RESERVED_TOKENS, BBOX_TOKENS
        VALID_HTML = ["<eos>"] + HTML_TOKENS
        INVALID_CELL = ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS
        VALID_BBOX = ["<eos>"] + BBOX_TOKENS
        dev = models["device"]

        def to_tensor(img, size):
            T = transforms.Compose([
                transforms.Resize(size), transforms.ToTensor(),
                transforms.Normalize(mean=[0.86597056, 0.88463002, 0.87491087],
                                     std=[0.20686628, 0.18201602, 0.18485524])])
            return T(img).to(dev).unsqueeze(0)

        def decode(model, img, prefix, max_len, eos, white=None, black=None):
            with torch.no_grad():
                mem = model.encode(img)
                ctx = torch.tensor(prefix, dtype=torch.int32).repeat(img.shape[0], 1).to(dev)
                for _ in range(max_len):
                    if all(eos in k for k in ctx):
                        break
                    mask = subsequent_mask(ctx.shape[1]).to(dev)
                    logits = model.generator(model.decode(mem, ctx, tgt_mask=mask, tgt_padding_mask=None))[:, -1, :]
                    logits = pred_token_within_range(logits.detach(), white_list=white, black_list=black)
                    _, nxt = greedy_sampling(logits)
                    ctx = torch.cat([ctx, nxt], dim=1)
            return ctx

        image = image.convert("RGB")
        size = image.size
        it = to_tensor(image, (448, 448))
        v_s, m_s = models["s"]; v_b, m_b = models["b"]; v_c, m_c = models["c"]

        ph = decode(m_s, it, [v_s.token_to_id("[html]")], 512, v_s.token_to_id("<eos>"),
                    white=[v_s.token_to_id(i) for i in VALID_HTML])
        ph = html_str_to_token_list(v_s.decode(ph.cpu().numpy()[0], skip_special_tokens=False))
        pb = decode(m_b, it, [v_b.token_to_id("[bbox]")], 1024, v_b.token_to_id("<eos>"),
                    white=[v_b.token_to_id(i) for i in VALID_BBOX[:449]])
        pb = bbox_str_to_token_list(v_b.decode(pb.cpu().numpy()[0], skip_special_tokens=False))
        ratio = [size[0] / 448, size[1] / 448] * 2
        pb = [[int(round(a * b)) for a, b in zip(e, ratio)] for e in pb]
        if not pb:
            return _html_to_rows(html_table_template("".join(build_table_from_html_and_cell(ph, []))))
        crops = torch.cat([to_tensor(image.crop(b), (112, 448)) for b in pb], dim=0)
        pc = decode(m_c, crops, [v_c.token_to_id("[cell]")], 200, v_c.token_to_id("<eos>"),
                    black=[v_c.token_to_id(i) for i in INVALID_CELL])
        pc = [cell_str_to_token_list(i) for i in v_c.decode_batch(pc.cpu().numpy(), skip_special_tokens=False)]
        pc = [re.sub(r'(\d).\s+(\d)', r'\1.\2', i) for i in pc]
        html = html_table_template("".join(build_table_from_html_and_cell(ph, pc)))
        return _html_to_rows(html)
    except Exception as e:  # noqa: BLE001
        print(f"[table_recognition] recognize failed: {e}")
        return None
