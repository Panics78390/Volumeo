"""Orchestration du pipeline complet image → maillage texturé → exports.

Fichiers produits dans le dossier d'une génération :

* ``input.png``    image détourée et cadrée envoyée au générateur
* ``raw.ply``      maillage brut (avec couleurs de sommets), pour les ré-ajustements
* ``final.npz``    maillage final déplié (sommets, faces, UV, normales)
* ``texture.png``  texture bakée
* ``model.glb``    modèle pour la visionneuse ; autres formats exportés à la demande
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import trimesh
from PIL import Image

from . import mesh as meshops
from .background import prepare_foreground
from .exporters import FORMATS, export_format, export_glb
from .generators import GenerationOptions, get_generator
from .texture import UVMesh, bake_texture, color_fn_from_vertex_colors, unwrap

log = logging.getLogger(__name__)

Progress = Callable[[float, str], None]


@dataclass
class PipelineParams:
    generator: str = "procedural"
    target_faces: int = 50_000
    texture_size: int = 2048
    mc_resolution: int = 256
    smoothing: int = 0  # itérations de lissage de Taubin (0 = aucun)
    isotropic: bool = False  # remaillage isotrope PyMeshLab avant décimation
    seed: int = 0
    # Mode « texte vers 3D » : description à partir de laquelle l'image est d'abord générée.
    prompt: str | None = None
    # Image + texte : image de départ que le prompt transforme, et liberté laissée au modèle.
    source_upload_id: str | None = None
    strength: float = 0.6

    def to_dict(self) -> dict:
        return asdict(self)


def _stage(progress: Progress, start: float, end: float) -> Progress:
    """Remappe la progression d'une étape [0,1] sur la plage globale [start,end]."""

    def report(frac: float, message: str) -> None:
        progress(start + (end - start) * max(0.0, min(1.0, frac)), message)

    return report


def _postprocess(
    raw: trimesh.Trimesh,
    color_fn,
    params: PipelineParams,
    out_dir: Path,
    progress: Progress,
    timings: dict,
) -> dict:
    t = time.perf_counter()
    progress(0.0, f"Remaillage vers ~{params.target_faces:,} triangles".replace(",", " "))
    m = meshops.remesh_to_target(raw, params.target_faces, isotropic=params.isotropic)
    if params.smoothing:
        progress(0.25, "Lissage")
        m = meshops.smooth(m, params.smoothing)
    timings["remesh"] = time.perf_counter() - t

    t = time.perf_counter()
    progress(0.3, "Dépliage UV (xatlas)")
    uvm = unwrap(m, params.texture_size)
    timings["uv"] = time.perf_counter() - t

    t = time.perf_counter()
    bake_progress = _stage(progress, 0.45, 0.9)
    bake_progress(0.0, f"Bake de la texture {params.texture_size} px")
    uvm.texture = bake_texture(uvm, color_fn, params.texture_size, progress=lambda f: bake_progress(f, "Bake de la texture"))
    timings["bake"] = time.perf_counter() - t

    t = time.perf_counter()
    progress(0.92, "Export GLB")
    uvm.texture.save(out_dir / "texture.png")
    np.savez_compressed(out_dir / "final.npz", vertices=uvm.vertices, faces=uvm.faces, uvs=uvm.uvs, normals=uvm.normals)
    export_glb(uvm, out_dir / FORMATS["glb"][0])
    timings["export"] = time.perf_counter() - t
    progress(1.0, "Terminé")
    return meshops.stats(m)


def run_pipeline(cutout: Image.Image, out_dir: Path, params: PipelineParams, progress: Progress) -> dict:
    """Pipeline complet à partir d'une image RGBA détourée."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    progress(0.01, "Cadrage du sujet")
    fg = prepare_foreground(cutout, size=512)
    fg.save(out_dir / "input.png")

    t = time.perf_counter()
    gen = get_generator(params.generator)
    result = gen.generate(
        fg,
        GenerationOptions(mc_resolution=params.mc_resolution, seed=params.seed),
        _stage(progress, 0.03, 0.5),
    )
    timings["generation"] = time.perf_counter() - t

    t = time.perf_counter()
    progress(0.5, "Nettoyage du maillage brut")
    raw, color_fn, _ = meshops.normalize(result.mesh, result.color_fn)
    raw = meshops.clean(raw)
    raw.export(out_dir / "raw.ply")
    raw_faces = len(raw.faces)
    timings["clean"] = time.perf_counter() - t

    stats = _postprocess(raw, color_fn, params, out_dir, _stage(progress, 0.55, 1.0), timings)
    timings["total"] = time.perf_counter() - t0
    return {
        **stats,
        "raw_faces": raw_faces,
        "generator_info": result.info,
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }


def reprocess(src_dir: Path, out_dir: Path, params: PipelineParams, progress: Progress) -> dict:
    """Ré-ajuste une génération existante (résolution, lissage, texture) sans relancer le modèle."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    progress(0.02, "Chargement du maillage brut")
    raw = trimesh.load(src_dir / "raw.ply", process=False, force="mesh")
    if (src_dir / "input.png").exists():
        (out_dir / "input.png").write_bytes((src_dir / "input.png").read_bytes())
    (out_dir / "raw.ply").write_bytes((src_dir / "raw.ply").read_bytes())
    color_fn = color_fn_from_vertex_colors(raw)
    stats = _postprocess(raw, color_fn, params, out_dir, _stage(progress, 0.05, 1.0), timings)
    timings["total"] = time.perf_counter() - t0
    return {**stats, "raw_faces": int(len(raw.faces)), "timings": {k: round(v, 2) for k, v in timings.items()}}


def load_uvmesh(job_dir: Path) -> UVMesh:
    data = np.load(job_dir / "final.npz")
    tex = Image.open(job_dir / "texture.png")
    tex.load()
    return UVMesh(vertices=data["vertices"], faces=data["faces"], uvs=data["uvs"], normals=data["normals"], texture=tex)


def export_cached(job_dir: Path, fmt: str) -> Path:
    """Exporte (ou réutilise) le modèle d'une génération dans le format demandé."""
    if fmt not in FORMATS:
        raise ValueError(f"Format inconnu : {fmt}")
    path = job_dir / FORMATS[fmt][0]
    if not path.exists():
        export_format(fmt, load_uvmesh(job_dir), job_dir)
    return path
