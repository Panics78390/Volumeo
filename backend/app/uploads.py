"""Stockage des images importées et de leur version détourée."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .config import Settings
from .pipeline.background import load_image, remove_background


def save_upload(settings: Settings, data: bytes, filename: str, remove_bg: bool = True) -> dict:
    """Décode l'image, la détoure et enregistre original + détourage + métadonnées."""
    img = load_image(data)
    upload_id = uuid.uuid4().hex[:12]
    d = settings.uploads_dir / upload_id
    d.mkdir(parents=True, exist_ok=True)
    img.save(d / "original.png")
    if remove_bg:
        cutout, method = remove_background(img, use_rembg=settings.use_rembg, model=settings.rembg_model)
    else:
        cutout, method = img, "none"
    cutout.save(d / "cutout.png")
    meta = {
        "id": upload_id,
        "filename": Path(filename or "image").name,
        "width": img.width,
        "height": img.height,
        "background_method": method,
        "created_at": time.time(),
    }
    (d / "meta.json").write_text(json.dumps(meta))
    return meta


def get_upload(settings: Settings, upload_id: str) -> dict | None:
    if not upload_id.isalnum():
        return None
    p = settings.uploads_dir / upload_id / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())
