"""Générateur procédural CPU (sans réseau de neurones, sans poids à télécharger).

Il « gonfle » la silhouette détourée : l'épaisseur en chaque point dépend de
sa distance au contour (profil arrondi), ce qui donne un volume fermé façon
ballon, texturé par projection de l'image sur ses deux faces.

Ce n'est pas une vraie reconstruction 3D : il sert de chemin de secours sans
GPU, de démonstration instantanée et de générateur déterministe pour les
tests. Il emprunte exactement le même chemin (champ implicite → marching
cubes → color_fn) que les modèles neuronaux.
"""

from __future__ import annotations

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage
from skimage import measure

from .base import GenerationOptions, Generator, ProgressFn, RawResult


def _bleed_colors(rgba: np.ndarray) -> np.ndarray:
    """Recopie la couleur des pixels opaques vers les pixels transparents voisins.

    Évite que le noir du fond transparent ne « bave » sur les bords du modèle.
    """
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    opaque = rgba[..., 3] > 127
    if opaque.all() or not opaque.any():
        return rgb
    _, (iy, ix) = ndimage.distance_transform_edt(~opaque, return_indices=True)
    return rgb[iy, ix]


class ProceduralGenerator(Generator):
    name = "procedural"
    label = "Procédural (CPU, sans IA)"
    description = "Gonflage de la silhouette. Instantané, sans GPU : idéal pour tester le pipeline."
    requires_gpu = False

    #: épaisseur maximale relative à la largeur du sujet
    depth_ratio = 0.35

    def generate(self, image: Image.Image, options: GenerationOptions, progress: ProgressFn) -> RawResult:
        rgba = np.asarray(image.convert("RGBA"))
        h, w = rgba.shape[:2]
        n = int(options.mc_resolution)
        progress(0.1, "Analyse de la silhouette")

        alpha = np.asarray(Image.fromarray(rgba[..., 3]).resize((n, n), Image.BILINEAR)).astype(np.float32) / 255.0
        mask = alpha > 0.5
        if mask.sum() < 16:
            raise ValueError("Silhouette trop petite pour générer un volume")

        # Distance au contour (en voxels), positive dedans, négative dehors.
        d_in = ndimage.distance_transform_edt(mask)
        d_out = ndimage.distance_transform_edt(~mask)
        r = float(d_in.max())
        # Profil circulaire : demi-épaisseur d'une sphère de rayon r à la distance d du bord.
        d = np.minimum(d_in, r)
        half = np.sqrt(np.clip(d * (2 * r - d), 0, None))
        # Limiter l'épaisseur totale par rapport à la taille du sujet.
        ys, xs = np.nonzero(mask)
        extent = max(np.ptp(xs), np.ptp(ys)) + 1
        half *= min(1.0, (self.depth_ratio * extent / 2) / max(r, 1e-6))
        half = np.where(mask, half, -d_out)
        half = ndimage.gaussian_filter(half, sigma=1.0)

        progress(0.35, "Construction du champ volumique")
        zmax = int(np.ceil(max(half.max(), 1.0))) + 2
        z = np.arange(-zmax, zmax + 1, dtype=np.float32)
        field = half[:, :, None] - np.abs(z)[None, None, :]
        # Bordure négative pour garantir un maillage fermé.
        field = np.pad(field, 1, mode="constant", constant_values=-1.0)

        progress(0.6, "Extraction de la surface (marching cubes)")
        verts, faces, _, _ = measure.marching_cubes(field, level=0.0)
        verts -= 1.0  # retrait du padding
        s = 1.0 / (n - 1)
        rows, cols, ks = verts[:, 0], verts[:, 1], verts[:, 2]
        xyz = np.column_stack([cols * s - 0.5, 0.5 - rows * s, (ks - zmax) * s])
        mesh = trimesh.Trimesh(vertices=xyz, faces=faces[:, ::-1], process=True)
        if mesh.volume < 0:
            mesh.invert()

        colors = _bleed_colors(rgba)

        def color_fn(points: np.ndarray) -> np.ndarray:
            """Projection orthographique de l'image sur les faces avant et arrière."""
            px = (points[:, 0] + 0.5) * (w - 1)
            py = (0.5 - points[:, 1]) * (h - 1)
            out = np.empty((len(points), 3), dtype=np.float32)
            for c in range(3):
                out[:, c] = ndimage.map_coordinates(colors[..., c], [py, px], order=1, mode="nearest")
            return np.clip(out, 0.0, 1.0)

        mesh.visual.vertex_colors = (np.column_stack([color_fn(mesh.vertices), np.ones(len(mesh.vertices))]) * 255).astype(np.uint8)
        progress(1.0, "Volume généré")
        return RawResult(mesh=mesh, color_fn=color_fn, info={"grid": n})
