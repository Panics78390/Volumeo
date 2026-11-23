"""Adaptateur TripoSR (VAST AI Research & Stability AI, licence MIT).

TripoSR reconstruit en une passe un champ triplan (densité + couleur) à partir
d'une seule image. Nous l'interrogeons sur une grille 3D puis extrayons la
surface avec le marching cubes de scikit-image : cela évite de devoir compiler
``torchmcubes`` (dépendance CUDA capricieuse du dépôt d'origine).

Installation (voir README) :
    git clone https://github.com/VAST-AI-Research/TripoSR /opt/TripoSR
    export PYTHONPATH=/opt/TripoSR:$PYTHONPATH
Les poids (``stabilityai/TripoSR`` : config.yaml + model.ckpt, ~1,7 Go) sont
téléchargés automatiquement depuis Hugging Face au premier lancement, ou lus
depuis ``I3D_TRIPOSR_MODEL`` s'il pointe vers un dossier local.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import threading
import types

import numpy as np
import trimesh
from PIL import Image
from skimage import measure

from ...config import settings
from ..background import composite_on_gray
from ..device import resolve_device
from .base import GenerationOptions, Generator, ProgressFn, RawResult

log = logging.getLogger(__name__)

# Seuil de densité utilisé par TripoSR pour définir la surface.
DENSITY_THRESHOLD = 25.0
# Rotation du repère TripoSR vers le repère glTF (Y vers le haut, face vers +Z),
# équivalente à ``tsr.utils.to_gradio_3d_orientation``.
_RX = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
_RY = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
TO_GLTF = _RY @ _RX


def _ensure_torchmcubes_stub() -> None:
    """``tsr.models.isosurface`` importe torchmcubes au chargement ; nous ne
    l'utilisons pas, un module factice suffit s'il n'est pas installé."""
    if "torchmcubes" not in sys.modules and importlib.util.find_spec("torchmcubes") is None:
        stub = types.ModuleType("torchmcubes")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchmcubes", None)

        def marching_cubes(*_a, **_k):  # pragma: no cover - jamais appelé
            raise RuntimeError("torchmcubes non installé (non utilisé par cette application)")

        stub.marching_cubes = marching_cubes
        sys.modules["torchmcubes"] = stub


class TripoSRGenerator(Generator):
    name = "triposr"
    label = "TripoSR (IA)"
    description = "Reconstruction neuronale en une passe. ~1 s sur GPU, 30 s à quelques minutes sur CPU."
    requires_gpu = True

    def __init__(self) -> None:
        self._model = None
        self._device: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        for mod in ("torch", "omegaconf", "einops", "transformers", "huggingface_hub"):
            if importlib.util.find_spec(mod) is None:
                return False, f"module Python manquant : {mod}"
        if importlib.util.find_spec("tsr") is None:
            return False, "code TripoSR introuvable (ajoutez le dépôt au PYTHONPATH)"
        return True, ""

    # -- chargement ---------------------------------------------------------
    def _load(self, progress: ProgressFn):
        with self._lock:
            if self._model is not None:
                return self._model
            _ensure_torchmcubes_stub()
            from tsr.system import TSR  # type: ignore[import-not-found]

            self._device = resolve_device(settings.device)
            progress(0.02, "Chargement des poids TripoSR (premier lancement : téléchargement ~1,7 Go)")
            model = TSR.from_pretrained(settings.triposr_model, config_name="config.yaml", weight_name="model.ckpt")
            model.renderer.set_chunk_size(settings.chunk_size)
            model.to(self._device)
            model.eval()
            self._model = model
            log.info("TripoSR chargé sur %s", self._device)
            return model

    def unload(self) -> None:
        with self._lock:
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass

    # -- génération ---------------------------------------------------------
    def generate(self, image: Image.Image, options: GenerationOptions, progress: ProgressFn) -> RawResult:
        import torch

        model = self._load(progress)
        device = self._device
        torch.manual_seed(options.seed)

        progress(0.1, "Encodage de l'image (transformer)")
        rgb = composite_on_gray(image.convert("RGBA"))
        with torch.no_grad():
            scene_code = model([rgb], device=device)[0]

        radius = float(model.renderer.cfg.radius)
        n = int(options.mc_resolution)

        def query(points: np.ndarray, key: str) -> np.ndarray:
            out = []
            step = 1 << 20
            for i in range(0, len(points), step):
                pts = torch.from_numpy(np.ascontiguousarray(points[i : i + step], dtype=np.float32)).to(device)
                with torch.no_grad():
                    out.append(model.renderer.query_triplane(model.decoder, pts, scene_code)[key].float().cpu().numpy())
            return np.concatenate(out, axis=0)

        progress(0.3, f"Échantillonnage de la densité (grille {n}³)")
        lin = np.linspace(-radius, radius, n, dtype=np.float32)
        grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), axis=-1).reshape(-1, 3)
        density = query(grid, "density_act").reshape(n, n, n)
        del grid

        progress(0.7, "Extraction de la surface (marching cubes)")
        level = density - DENSITY_THRESHOLD
        if level.max() <= 0:
            raise RuntimeError("Le modèle n'a produit aucune surface (image peu lisible ?)")
        verts, faces, _, _ = measure.marching_cubes(level, level=0.0)
        verts = verts / (n - 1) * (2 * radius) - radius  # indices -> repère TripoSR

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if mesh.volume < 0:
            mesh.invert()
        mesh.apply_transform(TO_GLTF)
        inv = np.linalg.inv(TO_GLTF)

        def color_fn(points: np.ndarray) -> np.ndarray:
            local = trimesh.transform_points(points, inv)
            return np.clip(query(local, "color"), 0.0, 1.0)

        progress(0.9, "Couleurs des sommets")
        vc = color_fn(mesh.vertices)
        mesh.visual.vertex_colors = (np.column_stack([vc, np.ones(len(vc))]) * 255).astype(np.uint8)
        return RawResult(mesh=mesh, color_fn=color_fn, info={"device": device, "grid": n})
