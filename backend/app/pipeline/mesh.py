"""Post-traitement géométrique : normalisation, nettoyage, remaillage, lissage."""

from __future__ import annotations

import logging

import fast_simplification
import numpy as np
import trimesh
from trimesh import graph, remesh, smoothing

from .generators.base import ColorFn

log = logging.getLogger(__name__)

# PyMeshLab (Qt) doit être importé depuis le thread principal : s'il est chargé
# pour la première fois dans le thread du worker, le processus plante à l'arrêt
# (« malloc_consolidate(): unaligned fastbin chunk »). On l'importe donc ici,
# au chargement du module, et non paresseusement.
try:
    import pymeshlab
except Exception:  # noqa: BLE001 - dépendance optionnelle
    pymeshlab = None


def normalize(mesh: trimesh.Trimesh, color_fn: ColorFn | None = None) -> tuple[trimesh.Trimesh, ColorFn | None, np.ndarray]:
    """Centre le modèle (pieds au sol, y=0) et ramène sa plus grande dimension à 1.

    Retourne le maillage transformé, la ``color_fn`` adaptée au nouveau repère
    et la matrice 4x4 appliquée.
    """
    lo, hi = mesh.bounds
    scale = 1.0 / max(float((hi - lo).max()), 1e-9)
    center = (lo + hi) / 2.0
    t = np.eye(4)
    t[:3, :3] *= scale
    t[:3, 3] = -scale * np.array([center[0], lo[1], center[2]])
    mesh = mesh.copy()
    mesh.apply_transform(t)
    if color_fn is None:
        return mesh, None, t
    inv = np.linalg.inv(t)

    def wrapped(points: np.ndarray) -> np.ndarray:
        return color_fn(trimesh.transform_points(points, inv))

    return mesh, wrapped, t


def clean(mesh: trimesh.Trimesh, min_component_ratio: float = 0.02) -> trimesh.Trimesh:
    """Fusionne les sommets, retire les faces dégénérées et les petits fragments flottants."""
    mesh = mesh.copy()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("Maillage vide après nettoyage")
    labels = graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    counts = np.bincount(labels)
    if len(counts) > 1:
        keep = counts >= max(1, int(min_component_ratio * counts.max()))
        mesh.update_faces(keep[labels])
        mesh.remove_unreferenced_vertices()
        log.info("Nettoyage : %d composantes, %d conservées", len(counts), int(keep.sum()))
    mesh.fix_normals()
    return mesh


_pymeshlab_ok: bool | None = None


def pymeshlab_available() -> bool:
    """Vrai si PyMeshLab est installé ET que ses filtres de maillage se chargent.

    (Le plugin ``filter_meshing`` a besoin des bibliothèques système OpenGL :
    ``libopengl0`` / ``libgl1`` sous Debian/Ubuntu.)
    """
    global _pymeshlab_ok
    if _pymeshlab_ok is None:
        try:
            _pymeshlab_ok = pymeshlab is not None and hasattr(
                pymeshlab.MeshSet(), "meshing_decimation_quadric_edge_collapse"
            )
        except Exception:  # noqa: BLE001 - plugins natifs manquants
            _pymeshlab_ok = False
        if not _pymeshlab_ok:
            log.warning("PyMeshLab indisponible : décimation de secours avec fast-simplification")
    return _pymeshlab_ok


def _to_meshset(mesh: trimesh.Trimesh):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices.astype(np.float64), face_matrix=mesh.faces.astype(np.int32)))
    return ms


def _from_meshset(ms) -> trimesh.Trimesh:
    m = ms.current_mesh()
    return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)


def isotropic_remesh(mesh: trimesh.Trimesh, target_faces: int, iterations: int = 3) -> trimesh.Trimesh:
    """Remaillage isotrope (triangles réguliers) via PyMeshLab."""
    # Aire d'un triangle équilatéral de côté L : sqrt(3)/4 * L².
    edge = float(np.sqrt(4.0 * mesh.area / (np.sqrt(3.0) * target_faces)))
    ms = _to_meshset(mesh)
    ms.meshing_isotropic_explicit_remeshing(iterations=iterations, targetlen=pymeshlab.PureValue(edge))
    return _from_meshset(ms)


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Décimation par contraction d'arêtes avec métrique d'erreur quadrique.

    PyMeshLab est préféré : il préserve la topologie (un maillage étanche le
    reste, ce qui compte pour l'impression 3D). À défaut, fast-simplification
    est plus rapide mais peut créer quelques arêtes non variétés.
    """
    if pymeshlab_available():
        ms = _to_meshset(mesh)
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            preservetopology=True,
            preserveboundary=True,
            preservenormal=True,
            optimalplacement=True,
            planarquadric=True,
            qualitythr=0.3,
            autoclean=True,
        )
        return _from_meshset(ms)
    v, f = fast_simplification.simplify(
        mesh.vertices.astype(np.float32),
        mesh.faces.astype(np.int32),
        target_reduction=1.0 - target_faces / len(mesh.faces),
        agg=7,
    )
    out = trimesh.Trimesh(vertices=v, faces=f, process=True)
    out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    if not out.is_watertight:
        trimesh.repair.fill_holes(out)  # rebouche les petits trous laissés par la décimation
    return out


def remesh_to_target(mesh: trimesh.Trimesh, target_faces: int, isotropic: bool = False) -> trimesh.Trimesh:
    """Amène le maillage au nombre de triangles visé.

    * pas assez de triangles -> subdivision (×4 par passe) ;
    * trop de triangles -> décimation quadrique.
    """
    if isotropic and pymeshlab_available():
        mesh = isotropic_remesh(mesh, target_faces)
    while len(mesh.faces) < target_faces * 0.95:
        v, f = remesh.subdivide(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    if len(mesh.faces) > target_faces * 1.02:
        mesh = decimate(mesh, target_faces)
    mesh.fix_normals()
    return mesh


def smooth(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """Lissage de Taubin : adoucit le relief sans faire rétrécir le modèle."""
    if iterations <= 0:
        return mesh
    mesh = mesh.copy()
    smoothing.filter_taubin(mesh, lamb=0.5, nu=0.53, iterations=int(iterations))
    return mesh


def stats(mesh: trimesh.Trimesh) -> dict:
    lo, hi = mesh.bounds
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "size": [round(float(x), 4) for x in (hi - lo)],
    }
