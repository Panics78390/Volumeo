"""Tests de bout en bout du pipeline : image → maillage → exports."""

from __future__ import annotations

import io
import shutil
import subprocess
import zipfile

import numpy as np
import pytest
import trimesh
from PIL import Image

from app.pipeline.background import remove_background
from app.pipeline.exporters import FORMATS, obj_text
from app.pipeline.runner import PipelineParams, export_cached, load_uvmesh, reprocess, run_pipeline
from tests.fbx_reader import read_fbx


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    """Génère une fois un modèle « Brouillon » partagé par les tests du module."""
    from tests.conftest import make_photo

    out = tmp_path_factory.mktemp("job")
    cut, _ = remove_background(make_photo().convert("RGBA"), use_rembg=False)
    steps = []
    result = run_pipeline(
        cut, out, PipelineParams(generator="procedural", target_faces=5_000, texture_size=512, mc_resolution=128),
        lambda f, m: steps.append((f, m)),
    )
    return out, result, steps


def test_pipeline_result_and_progress(generated):
    out, result, steps = generated
    assert abs(result["faces"] - 5_000) <= 250
    assert result["watertight"]
    assert result["raw_faces"] > result["faces"]
    fracs = [f for f, _ in steps]
    assert fracs == sorted(fracs), "la progression doit être croissante"
    assert fracs[-1] == pytest.approx(1.0)
    for name in ("input.png", "raw.ply", "final.npz", "texture.png", "model.glb"):
        assert (out / name).exists(), name


def test_texture_is_colored(generated):
    out, _, _ = generated
    tex = np.asarray(Image.open(out / "texture.png").convert("RGB")).reshape(-1, 3).astype(int)
    # La texture doit contenir le rouge, le bleu et le jaune de l'objet.
    red = ((tex[:, 0] > 150) & (tex[:, 1] < 90) & (tex[:, 2] < 90)).mean()
    blue = ((tex[:, 2] > 150) & (tex[:, 0] < 90)).mean()
    yellow = ((tex[:, 0] > 200) & (tex[:, 1] > 180) & (tex[:, 2] < 90)).mean()
    assert red > 0.3 and blue > 0.01 and yellow > 0.001


def test_glb_roundtrip(generated):
    out, result, _ = generated
    scene = trimesh.load(out / "model.glb")
    mesh = next(iter(scene.geometry.values()))
    assert len(mesh.faces) == result["faces"]
    assert mesh.visual.uv is not None and mesh.visual.uv.shape == (len(mesh.vertices), 2)
    assert mesh.visual.material.baseColorTexture.size == (512, 512)
    # Modèle normalisé : plus grande dimension = 1, posé sur y = 0.
    assert np.isclose(mesh.extents.max(), 1.0, atol=1e-3) and np.isclose(mesh.bounds[0][1], 0.0, atol=1e-3)


def test_obj_export(generated):
    out, result, _ = generated
    path = export_cached(out, "obj")
    with zipfile.ZipFile(path) as z:
        assert sorted(z.namelist()) == ["model.mtl", "model.obj", "texture.png"]
        assert "map_Kd texture.png" in z.read("model.mtl").decode()
        obj = z.read("model.obj").decode()
    assert obj.count("\nf ") == result["faces"]
    mesh = trimesh.load(io.BytesIO(obj.encode()), file_type="obj", process=False, force="mesh")
    assert len(mesh.faces) == result["faces"]


def test_stl_export_is_printable(generated):
    out, _, _ = generated
    mesh = trimesh.load(export_cached(out, "stl"))
    assert mesh.is_watertight
    assert np.isclose(mesh.bounds[0][2], 0.0, atol=1e-3), "posé sur le plateau (z=0)"
    assert np.isclose(mesh.extents.max(), 100.0, atol=0.01), "plus grande dimension = 100 mm"


def test_fbx_export_structure(generated):
    out, result, _ = generated
    data = export_cached(out, "fbx").read_bytes()
    version, nodes = read_fbx(data)
    assert version == 7400
    names = [n.name for n in nodes]
    for required in ("FBXHeaderExtension", "GlobalSettings", "Definitions", "Objects", "Connections"):
        assert required in names
    objects = next(n for n in nodes if n.name == "Objects")
    geom = objects.find("Geometry")
    pvi = geom.find("PolygonVertexIndex").props[0]
    assert (pvi < 0).sum() == result["faces"], "un indice négatif termine chaque triangle"
    verts = geom.find("Vertices").props[0].reshape(-1, 3)
    assert pvi[pvi >= 0].max() < len(verts) and (~pvi[pvi < 0]).max() < len(verts)
    uv = geom.find("LayerElementUV")
    assert len(uv.find("UVIndex").props[0]) == len(pvi)
    video = objects.find("Video")
    assert video.find("Content").props[0][:8] == b"\x89PNG\r\n\x1a\n", "texture PNG embarquée"
    conns = next(n for n in nodes if n.name == "Connections").find_all("C")
    assert any(c.props[0] == "OP" and c.props[3] == "DiffuseColor" for c in conns)


@pytest.mark.skipif(shutil.which("assimp") is None, reason="assimp CLI non installé")
@pytest.mark.parametrize("fmt", ["glb", "fbx", "stl"])
def test_exports_open_in_assimp(generated, fmt):
    """Validation croisée avec Open Asset Import Library, si présente."""
    out, result, _ = generated
    path = export_cached(out, fmt)
    info = subprocess.run(["assimp", "info", str(path)], capture_output=True, text=True, timeout=60)
    assert info.returncode == 0, info.stderr
    assert f"Faces:              {result['faces']}" in info.stdout


def test_reprocess_changes_resolution(generated, tmp_path):
    out, _, _ = generated
    res = reprocess(out, tmp_path, PipelineParams(target_faces=2_000, texture_size=512, smoothing=3), lambda f, m: None)
    assert abs(res["faces"] - 2_000) <= 100
    uvm = load_uvmesh(tmp_path)
    assert uvm.texture.size == (512, 512)


def test_formats_registry_complete():
    assert set(FORMATS) == {"glb", "obj", "stl", "fbx"}


def test_obj_text_indices_are_one_based():
    from app.pipeline.texture import UVMesh

    uvm = UVMesh(vertices=np.eye(3, dtype=np.float32), faces=np.array([[0, 1, 2]]),
                 uvs=np.zeros((3, 2), np.float32), normals=np.eye(3, dtype=np.float32))
    assert "f 1/1/1 2/2/2 3/3/3" in obj_text(uvm)
