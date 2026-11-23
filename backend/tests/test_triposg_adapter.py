"""Tests de l'adaptateur TripoSG avec un faux modèle (pas de GPU ni de poids requis).

Le faux pipeline renvoie une « voiture » faite de boîtes, dans son repère propre
(longueur sur Z, capot vers +Z). L'image d'entrée montre cette voiture de trois
quarts : l'adaptateur doit retrouver l'orientation de la photo, puis colorer la
forme soit par projection, soit à partir d'une reconstruction TripoSR.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import trimesh
from PIL import Image, ImageDraw
from skimage import measure

torch = pytest.importorskip("torch")

from app.pipeline.generators import triposg as T  # noqa: E402

YAW, ELEV = 40.0, 15.0
RED, BLUE = (200, 30, 30), (40, 40, 200)


def _box(p, c, h):
    q = np.abs(p - np.asarray(c)) - np.asarray(h)
    return np.linalg.norm(np.clip(q, 0, None), axis=-1) + np.clip(q.max(-1), None, 0)


def car_sdf(p: np.ndarray) -> np.ndarray:
    """Distance signée (négative dedans) d'une voiture stylisée, repère TripoSG."""
    body = _box(p, [0, -0.15, 0], [0.35, 0.15, 0.8])
    cabin = _box(p, [0, 0.12, -0.15], [0.3, 0.13, 0.4])
    nose = _box(p, [0, -0.05, 0.7], [0.3, 0.05, 0.1])
    return np.minimum(np.minimum(body, cabin), nose)


def car_mesh(n: int = 96) -> trimesh.Trimesh:
    lin = np.linspace(-1, 1, n)
    grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    v, f, _, _ = measure.marching_cubes(-car_sdf(grid).reshape(n, n, n), 0)
    return trimesh.Trimesh(v / (n - 1) * 2 - 1, f, process=True)


def view_rotation() -> np.ndarray:
    return T._rot_x(np.radians(ELEV)) @ T._rot_y(np.radians(YAW))


@pytest.fixture(scope="module")
def car_photo() -> Image.Image:
    """La voiture vue de 3/4 : moitié avant rouge, moitié arrière bleue."""
    mesh = car_mesh()
    cam = mesh.vertices @ view_rotation().T
    order = np.argsort(cam[mesh.faces].mean(1)[:, 2])  # peintre : du fond vers l'avant
    s = 0.85 * 512 / max(np.ptp(cam[:, 0]), np.ptp(cam[:, 1]))
    cx, cy = (cam[:, 0].min() + cam[:, 0].max()) / 2, (cam[:, 1].min() + cam[:, 1].max()) / 2
    im = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    for fi in order:
        f = mesh.faces[fi]
        col = RED if mesh.vertices[f, 2].mean() > 0 else BLUE
        draw.polygon([(256 + (cam[i, 0] - cx) * s, 256 - (cam[i, 1] - cy) * s) for i in f], fill=col + (255,))
    return im


def test_silhouette_fit_finds_photo_orientation(car_photo):
    fits = T.fit_to_silhouette(car_mesh().vertices, np.asarray(car_photo)[..., 3], top_k=6)
    best = fits[0]
    yaw = np.degrees(np.arctan2(best.upright[0, 2], best.upright[2, 2]))
    assert best.iou > 0.9
    assert abs(yaw - YAW) <= 5 and abs(np.degrees(best.elevation) - ELEV) < 1
    assert np.allclose(best.upright[1], [0, 1, 0], atol=1e-6), "la voiture reste debout"
    assert len(fits) == 6 and fits[0].iou >= fits[-1].iou


def test_best_by_shape_resolves_front_back(car_photo):
    """Avec une reconstruction de référence (TripoSR) dans le repère de la photo,
    l'orientation retournée (avant / arrière inversés) est écartée."""
    mesh = car_mesh()
    alpha = np.asarray(car_photo)[..., 3]
    good = T.fit_to_silhouette(mesh.vertices, alpha)[0]
    flipped = T.ViewFit(upright=T._rot_y(np.pi) @ good.upright, elevation=good.elevation, iou=good.iou)
    reference = (mesh.vertices - mesh.bounding_box.centroid) @ view_rotation().T * 0.7  # autre échelle, comme TripoSR
    chosen = T.best_by_shape([flipped, good], mesh.vertices, alpha, reference)
    assert chosen is good


def _fake_triposg(monkeypatch):
    calls = {}

    class Out:
        def __init__(self, samples):
            self.samples = samples

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, torch_dtype):
            calls["path"] = path
            return cls()

        def set_progress_bar_config(self, **kw):
            pass

        def to(self, device, dtype):
            calls["device"] = (device, dtype)
            return self

        def __call__(self, image, num_inference_steps, guidance_scale, generator, callback_on_step_end,
                     use_flash_decoder, dense_octree_depth, hierarchical_octree_depth):
            assert image.mode == "RGB" and image.getpixel((0, 0)) == (255, 255, 255), "fond blanc attendu"
            assert use_flash_decoder is False
            calls["octree"] = (dense_octree_depth, hierarchical_octree_depth)
            for i in range(num_inference_steps):
                callback_on_step_end(self, i, 0, {})
            m = car_mesh(80)
            return Out([(m.vertices.astype(np.float32), m.faces)])

    for name in ("triposg", "triposg.pipelines"):
        mod = types.ModuleType(name)
        mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    pl = types.ModuleType("triposg.pipelines.pipeline_triposg")
    pl.TripoSGPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "triposg.pipelines.pipeline_triposg", pl)
    monkeypatch.setattr(T.TripoSGGenerator, "availability", classmethod(lambda cls: (True, "")))
    monkeypatch.setattr(T.settings, "triposg_steps", 3)
    monkeypatch.setattr(T, "model_path", lambda name: f"/cache/{name}")
    return calls


def _front_back_colors(res):
    """Couleur moyenne de l'avant et de l'arrière de la voiture générée."""
    v = res.mesh.vertices
    heading = np.array([np.sin(np.radians(YAW)), 0, np.cos(np.radians(YAW))])
    along = v @ heading
    c = res.color_fn(v)
    return c[along > np.percentile(along, 85)].mean(0), c[along < np.percentile(along, 15)].mean(0)


def test_triposg_generate_with_projection_colors(monkeypatch, car_photo):
    from app.pipeline.generators.base import GenerationOptions

    calls = _fake_triposg(monkeypatch)
    monkeypatch.setattr(T.settings, "triposg_colors", "projection")
    msgs = []
    res = T.TripoSGGenerator().generate(car_photo, GenerationOptions(mc_resolution=160, seed=0),
                                        lambda f, m: msgs.append((f, m)))
    assert calls["octree"] == (8, 8) and calls["path"] == f"/cache/{T.settings.triposg_model}"
    assert any("étape 3/3" in m for _, m in msgs)
    assert res.info["colors"] == "projection" and res.info["silhouette_iou"] > 0.9
    assert np.allclose(res.mesh.bounding_box.centroid, 0, atol=1e-6)
    assert res.mesh.is_watertight
    front, back = _front_back_colors(res)
    assert front[0] > front[2], "l'avant (rouge sur la photo) est rouge"
    assert res.mesh.visual.vertex_colors.shape == (len(res.mesh.vertices), 4)


def test_triposg_generate_with_triposr_colors(monkeypatch, car_photo):
    """La couleur vient de la reconstruction TripoSR, y compris pour l'arrière
    que la photo ne montre presque pas."""
    from app.pipeline import generators
    from app.pipeline.generators.base import GenerationOptions, RawResult

    _fake_triposg(monkeypatch)
    monkeypatch.setattr(T.settings, "triposg_colors", "auto")

    class FakeTripoSR:
        def generate(self, image, options, progress):
            # Reconstruction « TripoSR » dans le repère de la photo, échelle différente,
            # avant rouge et arrière bleu comme sur l'objet réel.
            m = car_mesh(64)
            front = m.vertices[:, 2] > 0
            m.vertices = (m.vertices - m.bounding_box.centroid) @ view_rotation().T * 0.6
            rgb = np.where(front[:, None], np.array(RED), np.array(BLUE))
            m.visual.vertex_colors = np.column_stack([rgb, np.full(len(rgb), 255)]).astype(np.uint8)
            progress(1.0, "ok")
            return RawResult(mesh=m, color_fn=lambda p: np.zeros((len(p), 3)))

    monkeypatch.setattr(generators.REGISTRY["triposr"], "availability", classmethod(lambda cls: (True, "")))
    monkeypatch.setattr(generators, "get_generator", lambda name: FakeTripoSR())
    res = T.TripoSGGenerator().generate(car_photo, GenerationOptions(mc_resolution=160), lambda f, m: None)
    assert res.info["colors"] == "triposr"
    front, back = _front_back_colors(res)
    assert front[0] > 0.6 and front[2] < 0.3, front
    assert back[2] > 0.6 and back[0] < 0.3, back


def test_triposg_not_default_without_gpu(monkeypatch):
    from app.pipeline import generators

    monkeypatch.setattr(T.TripoSGGenerator, "availability", classmethod(lambda cls: (True, "")))
    monkeypatch.setattr(generators, "_has_cuda", lambda: False)
    assert generators.default_generator_name() != "triposg"
    monkeypatch.setattr(generators, "_has_cuda", lambda: True)
    assert generators.default_generator_name() == "triposg"


def test_model_path_downloads_hub_repo_first(monkeypatch, tmp_path):
    """Un identifiant Hugging Face est d'abord téléchargé : diffusers refuse de charger
    depuis le Hub les composants venant du paquet ``triposg`` (erreur vue sous Windows)."""
    import huggingface_hub

    seen = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: seen.append(repo_id) or str(tmp_path))
    assert T.model_path("VAST-AI/TripoSG") == str(tmp_path) and seen == ["VAST-AI/TripoSG"]
    assert T.model_path(str(tmp_path)) == str(tmp_path) and len(seen) == 1, "dossier local utilisé tel quel"


def test_real_triposg_code_runs_in_float16(monkeypatch, tmp_path):
    """Avec le vrai code TripoSG (s'il est installé) et un mini modèle aléatoire :
    le chargement en float16 (utilisé sur GPU) doit convertir tous les tampons.
    Sans cela : « mat1 and mat2 must have the same dtype, but got Float and Half »."""
    T._ensure_diso_stub()
    pl = pytest.importorskip("triposg.pipelines.pipeline_triposg")
    from diffusers import FlowMatchEulerDiscreteScheduler
    from transformers import BitImageProcessor, Dinov2Config, Dinov2Model
    from triposg.models.autoencoders import TripoSGVAEModel
    from triposg.models.transformers import TripoSGDiTModel

    torch.manual_seed(0)
    pipe = pl.TripoSGPipeline(
        vae=TripoSGVAEModel(latent_channels=8, num_attention_heads=2, width_encoder=32, width_decoder=32,
                            num_layers_encoder=1, num_layers_decoder=1),
        transformer=TripoSGDiTModel(num_attention_heads=2, width=32, in_channels=8, num_layers=3,
                                    cross_attention_dim=32),
        scheduler=FlowMatchEulerDiscreteScheduler(),
        image_encoder_dinov2=Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                                      intermediate_size=37, image_size=56, patch_size=14)),
        feature_extractor_dinov2=BitImageProcessor(size={"shortest_edge": 56}, crop_size={"height": 56, "width": 56}),
    )
    pipe.save_pretrained(tmp_path)

    monkeypatch.setattr(T.settings, "triposg_model", str(tmp_path))
    monkeypatch.setattr(T.settings, "device", "cpu")
    monkeypatch.setattr(T, "_dtype_for", lambda device: torch.float16)
    loaded = T.TripoSGGenerator()._load(lambda f, m: None)
    out = loaded(image=Image.new("RGB", (64, 64), "white"), num_inference_steps=1, use_flash_decoder=False,
                 dense_octree_depth=5, hierarchical_octree_depth=5, generator=torch.Generator().manual_seed(0))
    assert len(out.samples) == 1
