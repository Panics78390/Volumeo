import numpy as np
import pytest
import trimesh

from app.pipeline import mesh as meshops
from app.pipeline import presets
from app.pipeline.texture import bake_texture, color_fn_from_vertex_colors, unwrap


def test_presets_resolution():
    assert presets.resolve("draft", None, None) == (5_000, 1024, 160)
    assert presets.resolve("standard", None, 4096)[1] == 4096
    assert presets.resolve("custom", 12_345, 512) == (12_345, 512, 256)
    with pytest.raises(ValueError):
        presets.resolve("custom", None, 1024)
    with pytest.raises(ValueError):
        presets.resolve("custom", 10_000, 300)
    with pytest.raises(ValueError):
        presets.resolve("custom", 10, 1024)


@pytest.mark.parametrize("target", [2_000, 20_000, 60_000])
def test_remesh_hits_target(target):
    sphere = trimesh.creation.icosphere(subdivisions=4)  # 5 120 faces
    out = meshops.remesh_to_target(sphere, target)
    assert abs(len(out.faces) - target) <= 0.05 * target
    # La forme est conservée : rayon ~1.
    r = np.linalg.norm(out.vertices, axis=1)
    assert np.allclose(r, 1.0, atol=0.05)


def test_clean_removes_floaters():
    big = trimesh.creation.icosphere(subdivisions=3)
    small = trimesh.creation.icosphere(subdivisions=0, radius=0.01)
    small.apply_translation([3, 0, 0])
    out = meshops.clean(trimesh.util.concatenate([big, small]))
    assert len(out.faces) == len(big.faces)


def test_normalize_and_color_fn_follow_transform():
    m = trimesh.creation.box(extents=[2, 4, 1])
    m.apply_translation([5, 5, 5])
    out, fn, _ = meshops.normalize(m, lambda p: p.copy())
    lo, hi = out.bounds
    assert np.isclose((hi - lo).max(), 1.0) and np.isclose(lo[1], 0.0)
    # La color_fn reçoit les coordonnées d'origine.
    assert np.allclose(fn(np.array([[0.0, 0.0, 0.0]])), [[5, 3, 5]])


def test_smoothing_keeps_volume():
    m = trimesh.creation.icosphere(subdivisions=3)
    m.vertices += np.random.default_rng(0).normal(scale=0.02, size=m.vertices.shape)
    s = meshops.smooth(m, 10)
    assert abs(s.volume - m.volume) / m.volume < 0.05


def test_unwrap_and_bake_reproduce_color_field():
    sphere = trimesh.creation.icosphere(subdivisions=4)
    uvm = unwrap(sphere, 512)
    assert uvm.uvs.min() >= 0 and uvm.uvs.max() <= 1
    assert len(uvm.faces) == len(sphere.faces)

    def gradient(p):  # couleur = position (remise dans [0,1])
        return np.clip((p + 1) / 2, 0, 1)

    tex = np.asarray(bake_texture(uvm, gradient, 512)).astype(np.float32) / 255
    # On relit la texture au centre UV de quelques triangles et compare à la vérité.
    rng = np.random.default_rng(1)
    for f in rng.choice(len(uvm.faces), 50, replace=False):
        uv = uvm.uvs[uvm.faces[f]].mean(0)
        x, y = int(uv[0] * 512), int((1 - uv[1]) * 512)
        expected = gradient(uvm.vertices[uvm.faces[f]].mean(0, keepdims=True))[0]
        assert np.abs(tex[min(y, 511), min(x, 511)] - expected).max() < 0.08


def test_color_from_vertex_colors():
    m = trimesh.creation.icosphere(subdivisions=3)
    m.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(m.vertices), 1))
    fn = color_fn_from_vertex_colors(m)
    assert np.allclose(fn(m.vertices[:10] * 1.01), [[1, 0, 0]] * 10, atol=1e-3)
