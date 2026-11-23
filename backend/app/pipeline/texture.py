"""Dépliage UV (xatlas) et « bake » des couleurs dans une texture.

Le bake se fait sur CPU, sans OpenGL :

1. chaque triangle est rasterisé dans l'espace UV (PIL) avec son identifiant
   encodé en couleur, ce qui donne pour chaque texel le triangle qui le couvre ;
2. les coordonnées barycentriques du centre du texel donnent sa position 3D ;
3. la couleur est demandée au générateur (``color_fn``) en ces positions ;
4. les texels vides sont remplis par la couleur du texel couvert le plus
   proche (« dilatation »), ce qui évite les coutures visibles et les liserés
   noirs au filtrage/mipmapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import trimesh
import xatlas
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

from .generators.base import ColorFn


@dataclass
class UVMesh:
    """Maillage déplié : sommets dupliqués le long des coutures UV."""

    vertices: np.ndarray  # (V, 3) float
    faces: np.ndarray  # (F, 3) int
    uvs: np.ndarray  # (V, 2) float, origine en bas à gauche (convention OBJ)
    normals: np.ndarray  # (V, 3) float
    texture: Image.Image | None = None

    def to_trimesh(self) -> trimesh.Trimesh:
        material = trimesh.visual.material.PBRMaterial(
            name="material0",
            baseColorTexture=self.texture,
            metallicFactor=0.0,
            roughnessFactor=0.85,
            doubleSided=False,
        )
        visual = trimesh.visual.TextureVisuals(uv=self.uvs, material=material)
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            visual=visual,
            process=False,
        )
        return mesh


#: nombre de triangles par bloc envoyé à xatlas (voir ``unwrap``)
FACES_PER_CLUSTER = 6000


def _cluster_faces(mesh: trimesh.Trimesh, k: int) -> np.ndarray:
    """Partitionne les faces en ``k`` blocs spatiaux compacts (k-means sur les centres)."""
    if k <= 1:
        return np.zeros(len(mesh.faces), dtype=np.int64)
    from scipy.cluster.vq import kmeans2

    _, labels = kmeans2(mesh.triangles_center.astype(np.float64), k, seed=0, minit="++")
    return labels


def unwrap(mesh: trimesh.Trimesh, texture_size: int) -> UVMesh:
    """Génère un atlas UV unique avec xatlas (découpe en charts + empaquetage).

    Le temps de calcul des charts de xatlas croît bien plus vite que
    linéairement avec la taille du maillage (≈140 s pour 200 k triangles). On
    découpe donc le maillage en blocs spatiaux d'environ 6 000 triangles,
    ajoutés comme maillages séparés au même atlas : xatlas les traite
    indépendamment (≈10 s pour 200 k) puis les empaquette ensemble dans une
    seule texture. Le prix est quelques coutures supplémentaires, invisibles
    après bake et dilatation.
    """
    k = int(np.clip(round(len(mesh.faces) / FACES_PER_CLUSTER), 1, 64))
    labels = _cluster_faces(mesh, k)
    atlas = xatlas.Atlas()
    parts: list[np.ndarray] = []  # indices des sommets d'origine de chaque bloc
    for i in range(k):
        f = mesh.faces[labels == i]
        if len(f) == 0:
            continue
        used, local = np.unique(f, return_inverse=True)
        atlas.add_mesh(mesh.vertices[used].astype(np.float32), local.reshape(-1, 3).astype(np.uint32))
        parts.append(used)
    pack = xatlas.PackOptions()
    pack.resolution = int(texture_size)
    pack.padding = max(2, texture_size // 512)
    pack.bilinear = True
    atlas.generate(pack_options=pack)
    if atlas.atlas_count != 1:
        # Ne devrait pas arriver (un seul atlas est attendu) : on laisse xatlas choisir l'échelle.
        pack.resolution = 0
        atlas.generate(pack_options=pack)

    vmaps, faces, uvs, offset = [], [], [], 0
    for i, used in enumerate(parts):
        vmapping, indices, uv = atlas[i]
        vmaps.append(used[vmapping])
        faces.append(indices.astype(np.int64) + offset)
        uvs.append(uv)
        offset += len(vmapping)
    vmap = np.concatenate(vmaps)
    return UVMesh(
        vertices=mesh.vertices[vmap].astype(np.float32),
        faces=np.concatenate(faces),
        uvs=np.concatenate(uvs).astype(np.float32),
        normals=mesh.vertex_normals[vmap].astype(np.float32),
    )


def _rasterize_ids(uv_px: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    """Image (size, size) des identifiants de triangle (-1 = texel vide)."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    tri = uv_px[faces]  # (F, 3, 2)
    for i, t in enumerate(tri.tolist()):
        code = i + 1
        color = (code & 0xFF, (code >> 8) & 0xFF, (code >> 16) & 0xFF)
        pts = [tuple(p) for p in t]
        draw.polygon(pts, fill=color, outline=color)
    arr = np.asarray(img).astype(np.int64)
    return (arr[..., 0] | (arr[..., 1] << 8) | (arr[..., 2] << 16)) - 1


def bake_texture(
    uvmesh: UVMesh,
    color_fn: ColorFn,
    size: int,
    progress: Callable[[float], None] | None = None,
    batch: int = 1 << 20,
) -> Image.Image:
    if len(uvmesh.faces) >= (1 << 24) - 1:
        raise ValueError("Trop de triangles pour le bake (max 16 M)")
    # Coordonnées pixel : x = u * size, y = (1 - v) * size (ligne 0 = haut de l'image).
    uv_px = np.column_stack([uvmesh.uvs[:, 0] * size, (1.0 - uvmesh.uvs[:, 1]) * size])
    ids = _rasterize_ids(uv_px, uvmesh.faces, size)
    covered = ids >= 0
    ys, xs = np.nonzero(covered)
    fid = ids[ys, xs]

    # Barycentriques du centre de chaque texel dans son triangle (espace pixel).
    p = np.column_stack([xs + 0.5, ys + 0.5])
    tri = uv_px[uvmesh.faces[fid]]  # (N, 3, 2)
    v0, v1 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    v2 = p - tri[:, 0]
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    b1 = (d11 * d20 - d01 * d21) / denom
    b2 = (d00 * d21 - d01 * d20) / denom
    bary = np.column_stack([1.0 - b1 - b2, b1, b2])
    # Les texels de bordure (contour du polygone) peuvent déborder : on projette dans le triangle.
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-9)
    pos = (uvmesh.vertices[uvmesh.faces[fid]] * bary[:, :, None]).sum(1)

    colors = np.empty((len(pos), 3), dtype=np.float32)
    for i in range(0, len(pos), batch):
        colors[i : i + batch] = color_fn(pos[i : i + batch])
        if progress:
            progress(min(1.0, (i + batch) / max(len(pos), 1)))

    tex = np.zeros((size, size, 3), dtype=np.float32)
    tex[ys, xs] = colors
    # Dilatation : chaque texel vide prend la couleur du texel couvert le plus proche.
    if not covered.all():
        _, (iy, ix) = ndimage.distance_transform_edt(~covered, return_indices=True)
        tex = tex[iy, ix]
    return Image.fromarray((np.clip(tex, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def color_fn_from_vertex_colors(mesh: trimesh.Trimesh, k: int = 4) -> ColorFn:
    """Couleur interpolée depuis les couleurs de sommets d'un maillage dense.

    Sert lorsque le champ couleur du modèle n'est plus disponible (ré-ajustement
    d'une génération passée à partir du maillage brut sauvegardé).
    """
    colors = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
    tree = cKDTree(mesh.vertices)

    def color_fn(points: np.ndarray) -> np.ndarray:
        d, idx = tree.query(points, k=k)
        w = 1.0 / np.maximum(d, 1e-6) ** 2
        w /= w.sum(1, keepdims=True)
        return (colors[idx] * w[:, :, None]).sum(1)

    return color_fn
