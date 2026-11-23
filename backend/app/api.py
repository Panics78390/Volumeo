"""Routes HTTP de l'API (préfixe /api)."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .pipeline import presets
from .pipeline.background import ImageError
from .pipeline.device import system_info
from .pipeline.exporters import FORMATS
from .pipeline.generators import REGISTRY, default_generator_name, list_generators
from .pipeline.runner import PipelineParams, export_cached
from .uploads import get_upload, save_upload

router = APIRouter(prefix="/api")
ID_RE = re.compile(r"^[a-f0-9]{12}$")


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------
class ResolutionIn(BaseModel):
    preset: Literal["draft", "standard", "high", "custom"] = "standard"
    target_faces: int | None = Field(None, ge=presets.MIN_FACES, le=presets.MAX_FACES)
    texture_size: int | None = None
    smoothing: int = Field(0, ge=0, le=50, description="Itérations de lissage de Taubin")
    isotropic: bool = Field(False, description="Remaillage isotrope (PyMeshLab) avant décimation")


class JobIn(ResolutionIn):
    upload_id: str
    generator: str | None = None
    name: str | None = Field(None, max_length=100)
    seed: int = 0


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def _queue(request: Request):
    return request.app.state.queue


def _settings(request: Request):
    return request.app.state.settings


def _params(body: ResolutionIn, generator: str, seed: int = 0) -> PipelineParams:
    try:
        faces, tex, mc = presets.resolve(body.preset, body.target_faces, body.texture_size)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return PipelineParams(generator=generator, target_faces=faces, texture_size=tex, mc_resolution=mc,
                          smoothing=body.smoothing, isotropic=body.isotropic, seed=seed)


def _job_out(request: Request, job: dict) -> dict:
    q = _queue(request)
    jid = job["id"]
    base = f"/api/jobs/{jid}"
    done = job["status"] == "done"
    return {
        **job,
        "queue_position": q.position(jid) if job["status"] == "queued" else None,
        "urls": {
            "preview": f"{base}/files/input.png",
            "model": f"{base}/files/model.glb" if done else None,
            "texture": f"{base}/files/texture.png" if done else None,
            "downloads": {fmt: f"{base}/download/{fmt}" for fmt in FORMATS} if done else {},
        },
    }


def _get_job(request: Request, job_id: str) -> dict:
    if not ID_RE.match(job_id):
        raise HTTPException(404, "Génération introuvable")
    job = _queue(request).store.get(job_id)
    if job is None:
        raise HTTPException(404, "Génération introuvable")
    return job


def _safe_name(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", name, flags=re.UNICODE).strip("_")
    return s[:60] or "modele"


# ---------------------------------------------------------------------------
# Système
# ---------------------------------------------------------------------------
@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/system")
def system(request: Request) -> dict:
    s = _settings(request)
    return {
        "device": system_info(s.device),
        "generators": list_generators(),
        "default_generator": default_generator_name(s.default_generator),
        "presets": [p.to_dict() for p in presets.PRESETS.values()],
        "texture_sizes": list(presets.TEXTURE_SIZES),
        "faces_range": [presets.MIN_FACES, presets.MAX_FACES],
        "formats": {k: {"filename": v[0], "description": v[2]} for k, v in FORMATS.items()},
        "max_upload_mb": s.max_upload_mb,
    }


# ---------------------------------------------------------------------------
# Import d'image
# ---------------------------------------------------------------------------
@router.post("/uploads", status_code=201)
async def upload(request: Request, file: UploadFile = File(...), remove_bg: bool = Form(True)) -> dict:
    s = _settings(request)
    data = await file.read(s.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Image trop lourde (max {s.max_upload_mb} Mo)")
    try:
        meta = await asyncio.to_thread(save_upload, s, data, file.filename or "image", remove_bg)
    except ImageError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {**meta, "urls": {k: f"/api/uploads/{meta['id']}/{k}" for k in ("original", "cutout")}}


@router.get("/uploads/{upload_id}/{kind}")
def upload_file(request: Request, upload_id: str, kind: Literal["original", "cutout"]):
    if not ID_RE.match(upload_id) or get_upload(_settings(request), upload_id) is None:
        raise HTTPException(404, "Image introuvable")
    return FileResponse(_settings(request).uploads_dir / upload_id / f"{kind}.png", media_type="image/png")


# ---------------------------------------------------------------------------
# Générations
# ---------------------------------------------------------------------------
@router.post("/jobs", status_code=201)
def create_job(request: Request, body: JobIn) -> dict:
    s = _settings(request)
    meta = get_upload(s, body.upload_id) if ID_RE.match(body.upload_id) else None
    if meta is None:
        raise HTTPException(404, "Image introuvable : importez-la à nouveau")
    generator = body.generator or default_generator_name(s.default_generator)
    if generator not in REGISTRY:
        raise HTTPException(422, f"Générateur inconnu : {generator}")
    ok, reason = REGISTRY[generator].availability()
    if not ok:
        raise HTTPException(422, f"Générateur « {generator} » indisponible : {reason}")
    params = _params(body, generator, body.seed)
    name = body.name or Path(meta["filename"]).stem
    job = _queue(request).submit(name, params, upload_id=body.upload_id)
    return _job_out(request, job)


@router.post("/jobs/{job_id}/reprocess", status_code=201)
def reprocess_job(request: Request, job_id: str, body: ResolutionIn) -> dict:
    """Ré-ajuste une génération terminée (résolution, lissage, texture) sans relancer l'IA."""
    parent = _get_job(request, job_id)
    if parent["status"] != "done":
        raise HTTPException(409, "La génération d'origine n'est pas terminée")
    params = _params(body, parent["params"]["generator"], parent["params"].get("seed", 0))
    job = _queue(request).submit(parent["name"], params, upload_id=parent["upload_id"], parent_id=job_id,
                                 kind="reprocess")
    return _job_out(request, job)


@router.get("/jobs")
def list_jobs(request: Request, limit: int = 50, offset: int = 0) -> dict:
    q = _queue(request)
    jobs = q.store.list(limit=min(max(limit, 1), 200), offset=max(offset, 0))
    return {"total": q.store.count(), "items": [_job_out(request, j) for j in jobs]}


@router.get("/jobs/{job_id}")
def get_job(request: Request, job_id: str) -> dict:
    return _job_out(request, _get_job(request, job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str) -> dict:
    _get_job(request, job_id)
    _queue(request).cancel(job_id)
    return _job_out(request, _get_job(request, job_id))


@router.delete("/jobs/{job_id}", status_code=204)
def delete_job(request: Request, job_id: str) -> None:
    _get_job(request, job_id)
    _queue(request).delete(job_id)


@router.get("/jobs/{job_id}/files/{name}")
def job_file(request: Request, job_id: str, name: Literal["input.png", "model.glb", "texture.png"]):
    _get_job(request, job_id)
    path = _queue(request).job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(404, "Fichier pas encore disponible")
    media = {"png": "image/png", "glb": "model/gltf-binary"}[name.rsplit(".", 1)[1]]
    return FileResponse(path, media_type=media, headers={"Cache-Control": "no-cache"})


@router.get("/jobs/{job_id}/download/{fmt}")
async def download(request: Request, job_id: str, fmt: str):
    job = _get_job(request, job_id)
    if fmt not in FORMATS:
        raise HTTPException(404, f"Format inconnu : {fmt}")
    if job["status"] != "done":
        raise HTTPException(409, "La génération n'est pas terminée")
    path = await asyncio.to_thread(export_cached, _queue(request).job_dir(job_id), fmt)
    filename, media, _ = FORMATS[fmt]
    ext = filename.split(".", 1)[1] if fmt != "obj" else "obj.zip"
    return FileResponse(path, media_type=media, filename=f"{_safe_name(job['name'])}.{ext}")
