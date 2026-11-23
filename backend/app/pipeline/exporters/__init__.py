"""Exports GLB, OBJ (+MTL +PNG, zippé), STL et FBX."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from trimesh.exchange.gltf import export_glb as _trimesh_export_glb

from ..texture import UVMesh
from .fbx import export_fbx

# format -> (nom de fichier, type MIME, description)
FORMATS: dict[str, tuple[str, str, str]] = {
    "glb": ("model.glb", "model/gltf-binary", "glTF binaire texturé (web, Blender, Unity, Unreal)"),
    "obj": ("model_obj.zip", "application/zip", "OBJ + MTL + texture PNG (archive zip)"),
    "stl": ("model.stl", "model/stl", "STL binaire pour l'impression 3D (Z vers le haut, en mm)"),
    "fbx": ("model.fbx", "application/octet-stream", "FBX 7.4 binaire, texture embarquée"),
}


def export_glb(uvmesh: UVMesh, path: Path) -> Path:
    data = _trimesh_export_glb(trimesh.Scene(uvmesh.to_trimesh()), include_normals=True)
    path.write_bytes(data)
    return path


def obj_text(uvmesh: UVMesh, mtl_name: str = "model.mtl", material: str = "material0") -> str:
    """OBJ texte : v / vt / vn / f v/vt/vn (indices 1-based, uv origine bas-gauche)."""
    buf = io.StringIO()
    buf.write("# Généré par Image-to-3D\n")
    buf.write(f"mtllib {mtl_name}\no model\n")
    np.savetxt(buf, uvmesh.vertices, fmt="v %.6f %.6f %.6f")
    np.savetxt(buf, uvmesh.uvs, fmt="vt %.6f %.6f")
    np.savetxt(buf, uvmesh.normals, fmt="vn %.6f %.6f %.6f")
    buf.write(f"usemtl {material}\ns 1\n")
    f = uvmesh.faces + 1
    np.savetxt(buf, np.repeat(f, 3, axis=1), fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
    return buf.getvalue()


def mtl_text(texture_name: str = "texture.png", material: str = "material0") -> str:
    return (
        f"newmtl {material}\n"
        "Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n"
        "Ns 10.0\nillum 1\nd 1.0\n"
        f"map_Kd {texture_name}\n"
    )


def export_obj_zip(uvmesh: UVMesh, path: Path) -> Path:
    tex = io.BytesIO()
    uvmesh.texture.save(tex, format="PNG")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("model.obj", obj_text(uvmesh))
        z.writestr("model.mtl", mtl_text())
        z.writestr("texture.png", tex.getvalue())
    return path


def stl_mesh(uvmesh: UVMesh, size_mm: float = 100.0) -> trimesh.Trimesh:
    """Géométrie soudée, Z vers le haut, posée sur le plateau, plus grande dimension = ``size_mm``."""
    mesh = trimesh.Trimesh(vertices=uvmesh.vertices, faces=uvmesh.faces, process=False)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    # Y-haut (glTF) -> Z-haut (impression 3D) : rotation de +90° autour de X.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    mesh.apply_scale(size_mm / max(float((hi - lo).max()), 1e-9))
    return mesh


def export_stl(uvmesh: UVMesh, path: Path, size_mm: float = 100.0) -> Path:
    path.write_bytes(trimesh.exchange.stl.export_stl(stl_mesh(uvmesh, size_mm)))
    return path


def export_format(fmt: str, uvmesh: UVMesh, out_dir: Path) -> Path:
    if fmt not in FORMATS:
        raise ValueError(f"Format inconnu : {fmt}")
    path = out_dir / FORMATS[fmt][0]
    if fmt == "glb":
        return export_glb(uvmesh, path)
    if fmt == "obj":
        return export_obj_zip(uvmesh, path)
    if fmt == "stl":
        return export_stl(uvmesh, path)
    return export_fbx(uvmesh, path)


__all__ = ["FORMATS", "export_format", "export_glb", "export_obj_zip", "export_stl", "export_fbx", "obj_text", "stl_mesh"]
