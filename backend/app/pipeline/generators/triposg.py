"""Adaptateur TripoSG (VAST AI Research, licence MIT).

TripoSG est un modèle de diffusion « rectified flow » de 1,5 milliard de
paramètres qui produit une forme 3D bien plus précise que TripoSR, mais sans
couleur. Volumeo complète donc la forme ainsi :

1. **Orientation** : la forme sort dans le repère propre de l'objet, pas
   forcément tournée comme sur la photo. On cherche la rotation (axe vertical,
   cap, élévation de la caméra) dont la silhouette correspond le mieux à celle
   de l'image détourée.
2. **Couleurs** : les faces que l'objectif voit vraiment (test de profondeur)
   reçoivent le pixel net de la photo. Si la forme est symétrique (voiture,
   personnage...), le côté caché reprend la photo de son reflet. Le reste
   (l'arrière, par exemple) vient de TripoSR, qui reconstruit le même objet en
   couleur, ou à défaut d'une simple projection de l'image.

Installation (voir README) :
    git clone https://github.com/VAST-AI-Research/TripoSG /opt/TripoSG
    export PYTHONPATH=/opt/TripoSG:$PYTHONPATH
Les poids (``VAST-AI/TripoSG``) sont téléchargés depuis Hugging Face au premier
lancement, ou lus depuis ``I3D_TRIPOSG_MODEL`` s'il pointe vers un dossier local.
Il faut une carte graphique NVIDIA d'au moins 8 Go ; sur CPU c'est très lent.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import threading
import types
from dataclasses import dataclass, field, replace

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from ...config import settings
from ..background import composite_on_gray
from ..device import resolve_device
from .base import GenerationOptions, Generator, ProgressFn, RawResult

log = logging.getLogger(__name__)

REQUIRED_MODULES = ("torch", "diffusers", "transformers", "einops", "jaxtyping", "typeguard", "omegaconf")


def _ensure_diso_stub() -> None:
    """``triposg.inference_utils`` importe ``diso`` (extension CUDA à compiler)
    pour son décodeur rapide. Nous utilisons le décodeur hiérarchique, basé sur
    scikit-image : un module factice suffit si ``diso`` n'est pas installé."""
    if "diso" in sys.modules or importlib.util.find_spec("diso") is not None:
        return
    stub = types.ModuleType("diso")
    stub.__spec__ = importlib.machinery.ModuleSpec("diso", None)

    class DiffDMC:  # pragma: no cover - jamais appelé
        def __init__(self, *_a, **_k):
            raise RuntimeError("diso non installé (non utilisé par cette application)")

    stub.DiffDMC = DiffDMC
    sys.modules["diso"] = stub


# ---------------------------------------------------------------------------
# Orientation par la silhouette
# ---------------------------------------------------------------------------
def _rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _up_rotations() -> list[np.ndarray]:
    """Six rotations amenant chacun des axes ±X, ±Y, ±Z vers le haut (+Y).
    L'identité (Y déjà vertical, cas attendu) est la première."""
    return [
        np.eye(3),
        _rot_x(np.pi),  # -Y vers le haut
        _rot_x(-np.pi / 2),  # +Z vers le haut
        _rot_x(np.pi / 2),  # -Z vers le haut
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64),  # +X vers le haut
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),  # -X vers le haut
    ]


@dataclass
class ViewFit:
    """Résultat de l'alignement de la forme sur la photo."""

    upright: np.ndarray  # 3x3 : repère TripoSG -> objet debout, tourné comme sur la photo
    elevation: float  # radians : hauteur de la caméra au-dessus de l'objet
    iou: float
    scale: float = 1.0  # repère objet -> coordonnées image normalisées ([-0.5, 0.5])
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))  # (2,) décalage XY

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        """Points du repère « debout » vers le repère de la photo (X droite, Y haut, Z vers l'objectif)."""
        p = points @ _rot_x(self.elevation).T * self.scale
        p[:, :2] += self.offset
        return p


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _splat(xy: np.ndarray, size: int) -> np.ndarray:
    grid = np.zeros((size, size), dtype=bool)
    ij = np.clip(np.round(xy).astype(np.int64), 0, size - 1)
    grid[ij[:, 1], ij[:, 0]] = True
    grid = ndimage.binary_closing(grid, iterations=2, border_value=0)
    return ndimage.binary_fill_holes(grid)


def fit_to_silhouette(
    vertices: np.ndarray,
    alpha: np.ndarray,
    yaw_step: float = 15.0,
    elevations: tuple[float, ...] = (0.0, 15.0, 30.0),
    size: int = 96,
    max_points: int = 30000,
    seed: int = 0,
    top_k: int = 1,
) -> list[ViewFit]:
    """Cherche la rotation de la forme dont la projection orthographique
    ressemble le plus à la silhouette ``alpha`` (image HxW, 0-255).

    L'échelle et le décalage sont ensuite fixés par :func:`place_on_image`,
    une fois la forme tournée et centrée. Renvoie les ``top_k`` meilleures
    orientations, de la plus probable à la moins probable."""
    mask = np.asarray(Image.fromarray(alpha.astype(np.uint8)).resize((size, size), Image.BILINEAR)) > 127
    if mask.sum() < 16:
        raise ValueError("Silhouette trop petite pour orienter le modèle")
    mx0, my0, mx1, my1 = _mask_bbox(mask)
    mw, mh = mx1 - mx0 + 1, my1 - my0 + 1
    mcx, mcy = (mx0 + mx1) / 2, (my0 + my1) / 2

    pts = np.asarray(vertices, dtype=np.float64)
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(seed).choice(len(pts), max_points, replace=False)]
    pts = pts - (pts.min(0) + pts.max(0)) / 2

    def score(k: int, up: np.ndarray, yaw: float, elev: float) -> tuple[float, ViewFit] | None:
        upright = _rot_y(np.radians(yaw)) @ up
        cam = pts @ (_rot_x(np.radians(elev)) @ upright).T
        x, y = cam[:, 0], -cam[:, 1]  # image : y vers le bas
        pw, ph = np.ptp(x), np.ptp(y)
        if pw <= 0 or ph <= 0:
            return None
        s = np.sqrt((mw * mh) / (pw * ph))
        px = (x - (x.min() + x.max()) / 2) * s + mcx
        py = (y - (y.min() + y.max()) / 2) * s + mcy
        sil = _splat(np.column_stack([px, py]), size)
        iou = np.logical_and(sil, mask).sum() / max(np.logical_or(sil, mask).sum(), 1)
        # Légère préférence pour le repère attendu (Y vertical) et une caméra basse.
        value = iou + (0.01 if k == 0 else 0.0) - 0.0005 * elev
        fit = ViewFit(upright=upright, elevation=float(np.radians(elev)), iou=float(iou))
        return value, fit, (k, yaw, elev)

    ups = _up_rotations()
    scored = [r for k, up in enumerate(ups) for yaw in np.arange(0.0, 360.0, yaw_step) for elev in elevations
              if (r := score(k, up, yaw, elev)) is not None]
    # Affinage du cap autour des meilleures orientations grossières (une par piste,
    # pour que les candidates restent variées : avant / arrière, axe vertical...).
    scored.sort(key=lambda t: -t[0])
    refined = []
    for best in scored[:top_k]:
        k, yaw, elev = best[2]
        for dy in np.arange(-yaw_step / 2, yaw_step / 2 + 1e-6, yaw_step / 6):
            if dy and (r := score(k, ups[k], yaw + dy, elev)) is not None and r[0] > best[0]:
                best = r
        refined.append(best)
    scored = [(v, f) for v, f, _ in refined]
    if not scored:
        raise ValueError("Forme dégénérée : impossible de l'orienter")
    scored.sort(key=lambda t: -t[0])
    return [f for _, f in scored[:top_k]]


# ---------------------------------------------------------------------------
# Couleurs
# ---------------------------------------------------------------------------
def projection_colors(image: Image.Image, fit: ViewFit):
    """Projette la photo sur l'avant et l'arrière (secours sans TripoSR)."""
    from .procedural import _bleed_colors

    rgba = np.asarray(image.convert("RGBA"))
    h, w = rgba.shape[:2]
    colors = _bleed_colors(rgba)

    def color_fn(points: np.ndarray) -> np.ndarray:
        cam = fit.to_camera(points)
        px = (cam[:, 0] + 0.5) * (w - 1)
        py = (0.5 - cam[:, 1]) * (h - 1)
        out = np.empty((len(points), 3), dtype=np.float32)
        for c in range(3):
            out[:, c] = ndimage.map_coordinates(colors[..., c], [py, px], order=1, mode="nearest")
        return np.clip(out, 0.0, 1.0)

    return color_fn


def transfer_colors(src_vertices: np.ndarray, src_colors: np.ndarray, fit: ViewFit, target_vertices: np.ndarray):
    """``color_fn`` qui reprend les couleurs d'une autre reconstruction (TripoSR)
    du même objet, exprimée dans le repère de la photo.

    Les deux formes sont superposées boîte englobante contre boîte englobante,
    puis chaque point prend la couleur (moyenne pondérée des 4 sommets les plus
    proches) de la reconstruction colorée.
    """
    src = np.asarray(src_vertices, dtype=np.float64)
    cols = np.asarray(src_colors, dtype=np.float32)[:, :3]
    if cols.max() > 1.0:
        cols = cols / 255.0
    cam = fit.to_camera(np.asarray(target_vertices, dtype=np.float64))
    t_min, t_max = cam.min(0), cam.max(0)
    s_min, s_max = src.min(0), src.max(0)
    ratio = (s_max - s_min) / np.maximum(t_max - t_min, 1e-9)
    tree = cKDTree(src)
    k = min(4, len(src))

    def color_fn(points: np.ndarray) -> np.ndarray:
        q = (fit.to_camera(np.asarray(points, dtype=np.float64)) - t_min) * ratio + s_min
        dist, idx = tree.query(q, k=k)
        if k == 1:
            return np.clip(cols[idx], 0.0, 1.0)
        wgt = 1.0 / np.maximum(dist, 1e-6)
        wgt /= wgt.sum(axis=1, keepdims=True)
        return np.clip((cols[idx] * wgt[..., None]).sum(axis=1), 0.0, 1.0).astype(np.float32)

    return color_fn


def find_symmetry(vertices: np.ndarray, n: int = 6000, seed: int = 0, tol: float = 0.008):
    """Cherche un plan de symétrie miroir parmi les trois plans principaux de la
    boîte englobante (les formes TripoSG sortent alignées sur leurs axes).

    Renvoie ``(normale, point)`` du plan, ou ``None`` si aucun plan ne colle à
    moins de ``tol`` × diagonale (distance moyenne entre la forme et son reflet)."""
    pts = np.asarray(vertices, dtype=np.float64)
    if len(pts) > n:
        pts = pts[np.random.default_rng(seed).choice(len(pts), n, replace=False)]
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    diag = float(np.linalg.norm(hi - lo))
    if diag <= 0:
        return None
    tree = cKDTree(pts)
    scores = []
    for axis in range(3):
        refl = pts.copy()
        refl[:, axis] = 2 * center[axis] - refl[:, axis]
        scores.append((tree.query(refl)[0].mean() / diag, axis))
    scores.sort()
    # Un seul plan franchement meilleur que les autres : un objet presque cubique
    # ne doit pas recopier son avant sur son arrière.
    if scores[0][0] > tol or scores[0][0] > 0.5 * scores[1][0]:
        return None
    normal = np.zeros(3)
    normal[scores[0][1]] = 1.0
    return normal, center


def photo_colors(image: Image.Image, fit: ViewFit, mesh: trimesh.Trimesh, fallback_fn=None,
                 symmetry: tuple[np.ndarray, np.ndarray] | None = None, seed: int = 0):
    """``color_fn`` qui colle la photo nette sur les faces réellement vues par
    l'objectif (test de profondeur), au lieu de couleurs interpolées.

    - face vue et tournée vers l'objectif : pixel de la photo ;
    - face cachée dont le reflet par ``symmetry`` (plan de symétrie, repère du
      maillage) est vu : pixel du reflet (le flanc droit d'une voiture copie le
      flanc gauche) ;
    - sinon : ``fallback_fn`` (couleurs TripoSR), ou la projection simple.
    Les transitions sont fondues selon l'angle entre la face et l'objectif.
    """
    from .procedural import _bleed_colors

    rgba = np.asarray(image.convert("RGBA"))
    h, w = rgba.shape[:2]
    colors = _bleed_colors(rgba)
    inside = ndimage.binary_dilation(rgba[..., 3] > 127, iterations=2)
    if fallback_fn is None:
        fallback_fn = projection_colors(image, fit)

    # Tampon de profondeur à la résolution de la photo (la caméra regarde vers -Z :
    # le point visible d'un pixel est celui de plus grand z).
    samples, _ = trimesh.sample.sample_surface(mesh, 300_000, seed=seed)
    samples = np.vstack([samples, mesh.vertices])
    cam = fit.to_camera(samples)

    def pix(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix = np.clip(np.round((c[:, 0] + 0.5) * (w - 1)).astype(np.int64), 0, w - 1)
        iy = np.clip(np.round((0.5 - c[:, 1]) * (h - 1)).astype(np.int64), 0, h - 1)
        return ix, iy

    zbuf = np.full((h, w), -np.inf)
    ix, iy = pix(cam)
    np.maximum.at(zbuf, (iy, ix), cam[:, 2])
    holes = ~np.isfinite(zbuf)
    zbuf = np.where(holes, ndimage.maximum_filter(np.where(holes, -1e9, zbuf), size=5), zbuf)
    zbuf[zbuf <= -1e8] = -np.inf
    eps = 0.02 * float(np.ptp(cam[:, 2]) + 1e-9)

    normals_cam = np.asarray(mesh.vertex_normals) @ _rot_x(fit.elevation).T
    tree = cKDTree(mesh.vertices)
    if symmetry is not None:
        sym_n = np.asarray(symmetry[0], dtype=np.float64)
        sym_n = sym_n / np.linalg.norm(sym_n)
        sym_c = np.asarray(symmetry[1], dtype=np.float64)

    def view_weight(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Poids photo (0..1) et position dans le repère de la photo de chaque point."""
        c = fit.to_camera(np.asarray(points, dtype=np.float64))
        px_, py_ = pix(c)
        visible = (c[:, 2] >= zbuf[py_, px_] - eps) & inside[py_, px_]
        facing = normals_cam[tree.query(points)[1], 2]
        weight = np.clip((facing - 0.05) / 0.35, 0.0, 1.0) * visible
        return weight, c

    def sample_photo(c: np.ndarray) -> np.ndarray:
        fx = (c[:, 0] + 0.5) * (w - 1)
        fy = (0.5 - c[:, 1]) * (h - 1)
        out = np.empty((len(c), 3), dtype=np.float32)
        for ch in range(3):
            out[:, ch] = ndimage.map_coordinates(colors[..., ch], [fy, fx], order=1, mode="nearest")
        return out

    def color_fn(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        wgt, c = view_weight(points)
        out = np.zeros((len(points), 3), dtype=np.float32)
        out += wgt[:, None] * sample_photo(c)
        rest = 1.0 - wgt
        need = rest > 1e-3
        if symmetry is not None and need.any():
            p = points[need]
            mirrored = p - 2 * ((p - sym_c) @ sym_n)[:, None] * sym_n
            mw, mc = view_weight(mirrored)
            take = rest[need] * mw
            out[need] += take[:, None] * sample_photo(mc)
            rest[need] -= take
            need = rest > 1e-3
        if need.any():
            out[need] += rest[need][:, None] * np.asarray(fallback_fn(points[need]), dtype=np.float32)
        return np.clip(out, 0.0, 1.0)

    return color_fn


def _dtype_for(device: str):
    import torch

    return torch.float16 if device.startswith("cuda") else torch.float32


def model_path(name: str) -> str:
    """Dossier local des poids TripoSG.

    diffusers refuse de charger directement depuis le Hub un pipeline dont les
    composants viennent d'un autre paquet (ici ``triposg.schedulers...``) :
    on télécharge donc d'abord le dépôt complet, puis on charge le dossier local,
    comme le script officiel de TripoSG.
    """
    from pathlib import Path

    if Path(name).is_dir():
        return name
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=name)


# ---------------------------------------------------------------------------
# Générateur
# ---------------------------------------------------------------------------
class TripoSGGenerator(Generator):
    name = "triposg"
    label = "TripoSG (IA, forme précise)"
    description = (
        "Forme nettement plus fidèle que TripoSR, couleurs reprises de TripoSR. "
        "Carte NVIDIA de 8 Go conseillée ; 30 s à 1 min par modèle."
    )
    requires_gpu = True

    def __init__(self) -> None:
        self._pipe = None
        self._device: str | None = None
        self._lock = threading.Lock()

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        for mod in REQUIRED_MODULES:
            if importlib.util.find_spec(mod) is None:
                return False, f"module Python manquant : {mod}"
        if importlib.util.find_spec("triposg") is None:
            return False, "code TripoSG introuvable (ajoutez le dépôt au PYTHONPATH)"
        return True, ""

    def _load(self, progress: ProgressFn):
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            _ensure_diso_stub()
            import torch
            from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # type: ignore[import-not-found]

            self._device = resolve_device(settings.device)
            dtype = _dtype_for(self._device)
            progress(0.02, "Chargement des poids TripoSG (premier lancement : téléchargement de plusieurs Go)")
            # Chargement direct en float16 sur GPU : évite un pic de ~6 Go de RAM en float32.
            pipe = TripoSGPipeline.from_pretrained(model_path(settings.triposg_model), torch_dtype=dtype)
            pipe.set_progress_bar_config(disable=True)
            # ``torch_dtype`` ne convertit que les poids chargés : certains tampons créés à
            # l'initialisation restent en float32 (erreur « Float and Half »). Le ``.to``
            # complet, comme dans le script officiel, convertit tout.
            pipe = pipe.to(self._device, dtype)
            self._pipe = pipe
            log.info("TripoSG chargé sur %s", self._device)
            return pipe

    def unload(self) -> None:
        with self._lock:
            self._pipe = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass

    def generate(self, image: Image.Image, options: GenerationOptions, progress: ProgressFn) -> RawResult:
        import torch

        pipe = self._load(progress)
        device = self._device or "cpu"
        rgba = image.convert("RGBA")
        rgb = Image.fromarray((composite_on_gray(rgba, gray=1.0) * 255).astype(np.uint8))
        steps = max(1, int(settings.triposg_steps))

        def on_step(_pipe, i, _t, kwargs):
            progress(0.05 + 0.5 * (i + 1) / steps, f"Diffusion de la forme (étape {i + 1}/{steps})")
            return kwargs

        progress(0.05, "Encodage de l'image (DINOv2)")
        # Grille finale 256³ (brouillon, standard) ou 512³ (haute) ; le décodeur
        # hiérarchique tourne en partie sur le processeur, 512³ coûte plusieurs minutes.
        depth = 9 if options.mc_resolution > 256 else 8
        with torch.no_grad():
            out = pipe(
                image=rgb,
                num_inference_steps=steps,
                guidance_scale=float(settings.triposg_guidance),
                generator=torch.Generator(device=device).manual_seed(int(options.seed)),
                callback_on_step_end=on_step,
                use_flash_decoder=False,
                dense_octree_depth=min(8, depth),
                hierarchical_octree_depth=depth,
            )
        verts, faces = out.samples[0]
        if verts is None or len(verts) == 0:
            raise RuntimeError("TripoSG n'a produit aucune surface (image peu lisible ?)")
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        progress(0.6, "Nettoyage de la forme")
        mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces), process=True)
        if mesh.volume < 0:
            mesh.invert()

        progress(0.65, "Orientation selon la photo")
        alpha = np.asarray(rgba)[..., 3]
        symmetry = find_symmetry(mesh.vertices, seed=options.seed)
        candidates = fit_to_silhouette(mesh.vertices, alpha, seed=options.seed, top_k=8)

        colored = self._triposr_reference(image, options, progress)
        if colored is not None:
            # La reconstruction TripoSR (repère de la photo) départage les orientations
            # dont la silhouette se ressemble (avant / arrière d'une voiture, par exemple).
            fit = best_by_shape(candidates, mesh.vertices, alpha, colored.vertices)
        else:
            fit = candidates[0]
        log.info("TripoSG : silhouette IoU %.2f, élévation %.0f°", fit.iou, np.degrees(fit.elevation))

        rot = np.eye(4)
        rot[:3, :3] = fit.upright
        mesh.apply_transform(rot)
        shift = -(mesh.bounds[0] + mesh.bounds[1]) / 2
        mesh.apply_translation(shift)
        fit = place_on_image(fit, mesh.vertices, alpha)
        if symmetry is not None:
            symmetry = (fit.upright @ symmetry[0], fit.upright @ symmetry[1] + shift)

        progress(0.9, "Couleurs : photo sur les faces visibles")
        if colored is not None:
            cols = np.asarray(colored.visual.vertex_colors)[:, :3]
            fallback, colors_from = transfer_colors(colored.vertices, cols, fit, mesh.vertices), "photo+triposr"
        else:
            fallback, colors_from = projection_colors(image, fit), "photo+projection"
        color_fn = photo_colors(image, fit, mesh, fallback, symmetry, seed=options.seed)

        progress(0.95, "Couleurs des sommets")
        vc = color_fn(mesh.vertices)
        mesh.visual.vertex_colors = (np.column_stack([vc, np.ones(len(vc))]) * 255).astype(np.uint8)
        info = {"device": device, "steps": steps, "silhouette_iou": round(fit.iou, 3), "colors": colors_from,
                "symmetry": symmetry is not None}
        return RawResult(mesh=mesh, color_fn=color_fn, info=info)

    def _triposr_reference(self, image, options, progress) -> trimesh.Trimesh | None:
        """Reconstruction TripoSR colorée du même objet, ou ``None`` si indisponible."""
        mode = settings.triposg_colors
        if mode == "projection":
            return None
        from . import REGISTRY, get_generator

        ok, reason = REGISTRY["triposr"].availability()
        if not ok:
            if mode == "triposr":
                raise RuntimeError(f"Couleurs TripoSR demandées mais indisponibles : {reason}")
            return None

        def sub(frac, msg):
            progress(0.7 + 0.2 * frac, f"Couleurs : {msg}")

        progress(0.7, "Couleurs : reconstruction TripoSR")
        res = get_generator("triposr").generate(image, GenerationOptions(mc_resolution=160, seed=options.seed), sub)
        return res.mesh


def best_by_shape(candidates: list[ViewFit], vertices: np.ndarray, alpha: np.ndarray, reference: np.ndarray,
                  n: int = 4000, seed: int = 0) -> ViewFit:
    """Parmi des orientations candidates, garde celle dont la forme (dans le
    repère de la photo) est la plus proche de ``reference`` (distance de Chamfer
    après superposition des boîtes englobantes)."""
    rng = np.random.default_rng(seed)
    ref = np.asarray(reference, dtype=np.float64)
    ref = ref[rng.choice(len(ref), min(n, len(ref)), replace=False)]
    r_min, r_max = ref.min(0), ref.max(0)
    ref_n = (ref - r_min) / np.maximum(r_max - r_min, 1e-9)
    ref_tree = cKDTree(ref_n)
    pts = np.asarray(vertices, dtype=np.float64)
    pts = pts[rng.choice(len(pts), min(n, len(pts)), replace=False)]
    best, best_d = candidates[0], np.inf
    for cand in candidates:
        up = pts @ cand.upright.T
        up -= (up.min(0) + up.max(0)) / 2
        cam = place_on_image(cand, up, alpha).to_camera(up)
        c_n = (cam - cam.min(0)) / np.maximum(np.ptp(cam, axis=0), 1e-9)
        d = ref_tree.query(c_n)[0].mean() + cKDTree(c_n).query(ref_n)[0].mean()
        if d < best_d:
            best, best_d = cand, d
    return best


def place_on_image(fit: ViewFit, vertices: np.ndarray, alpha: np.ndarray, size: int = 96) -> ViewFit:
    """Fixe l'échelle et le décalage qui superposent la forme (déjà debout)
    à la silhouette de la photo, en coordonnées image normalisées."""
    mask = np.asarray(Image.fromarray(alpha.astype(np.uint8)).resize((size, size), Image.BILINEAR)) > 127
    mx0, my0, mx1, my1 = _mask_bbox(mask)
    unit = 1.0 / (size - 1)
    cam = np.asarray(vertices, dtype=np.float64) @ _rot_x(fit.elevation).T
    w, h = np.ptp(cam[:, 0]), np.ptp(cam[:, 1])
    scale = np.sqrt(((mx1 - mx0 + 1) * (my1 - my0 + 1)) / max(w * h, 1e-12)) * unit
    cx = (cam[:, 0].min() + cam[:, 0].max()) / 2
    cy = (cam[:, 1].min() + cam[:, 1].max()) / 2
    mcx, mcy = (mx0 + mx1) / 2, (my0 + my1) / 2
    offset = np.array([mcx * unit - 0.5 - cx * scale, 0.5 - mcy * unit - cy * scale])
    return replace(fit, scale=float(scale), offset=offset)
