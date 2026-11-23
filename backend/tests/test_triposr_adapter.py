"""Test de l'adaptateur TripoSR avec un faux modèle (pas de GPU ni de poids requis).

Le faux ``tsr.system.TSR`` renvoie un champ de densité sphérique et une couleur
qui dépend de la position : on vérifie que l'adaptateur échantillonne la
grille, extrait la surface, réoriente le repère et fournit une color_fn
cohérente, exactement comme avec le vrai réseau.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")


class _FakeRenderer:
    class cfg:  # noqa: N801 - imite l'attribut OmegaConf
        radius = 0.87

    def set_chunk_size(self, n):
        self.chunk = n

    def query_triplane(self, decoder, positions, scene_code):
        # Sphère de rayon 0.5 centrée en (0, 0, 0.1) (repère TripoSR), densité 50 dedans.
        d = torch.linalg.norm(positions - torch.tensor([0.0, 0.0, 0.1]), dim=-1, keepdim=True)
        density = 50.0 * torch.sigmoid((0.5 - d) * 40)
        color = (positions.clamp(-1, 1) + 1) / 2
        return {"density_act": density, "color": color}


class _FakeTSR:
    calls: list = []

    def __init__(self):
        self.renderer = _FakeRenderer()
        self.decoder = object()

    @classmethod
    def from_pretrained(cls, path, config_name, weight_name):
        cls.calls.append((path, config_name, weight_name))
        return cls()

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, images, device):
        assert images[0].shape == (512, 512, 3) and 0 <= images[0].min() and images[0].max() <= 1
        return torch.zeros(1, 3, 8, 8)


@pytest.fixture
def fake_tsr(monkeypatch):
    tsr = types.ModuleType("tsr")
    tsr.__path__ = []
    system = types.ModuleType("tsr.system")
    system.TSR = _FakeTSR
    monkeypatch.setitem(sys.modules, "tsr", tsr)
    monkeypatch.setitem(sys.modules, "tsr.system", system)
    import importlib.util

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name in ("tsr", "omegaconf", "einops", "transformers",
                                                              "huggingface_hub") else real_find_spec(name, *a))
    yield


def test_triposr_adapter_with_fake_model(fake_tsr, cutout):
    from app.pipeline.background import prepare_foreground
    from app.pipeline.generators.base import GenerationOptions
    from app.pipeline.generators.triposr import TO_GLTF, TripoSRGenerator

    assert TripoSRGenerator.availability() == (True, "")
    gen = TripoSRGenerator()
    res = gen.generate(prepare_foreground(cutout), GenerationOptions(mc_resolution=64), lambda f, m: None)
    mesh = res.mesh
    assert mesh.is_watertight and mesh.volume > 0
    # Rayon ~0.5 et centre = (0,0,0.1) transformé dans le repère glTF.
    center = (TO_GLTF @ np.array([0, 0, 0.1, 1.0]))[:3]
    assert np.allclose(mesh.vertices.mean(0), center, atol=0.02)
    assert np.allclose(np.linalg.norm(mesh.vertices - center, axis=1), 0.5, atol=0.03)
    # color_fn interroge le champ dans le repère d'origine (inverse de la rotation).
    p = mesh.vertices[:5]
    local = (np.linalg.inv(TO_GLTF) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    assert np.allclose(res.color_fn(p), (local + 1) / 2, atol=1e-4)
    assert _FakeTSR.calls[-1][1:] == ("config.yaml", "model.ckpt")


def test_triposr_through_full_pipeline(fake_tsr, cutout, tmp_path):
    from app.pipeline import generators
    from app.pipeline.runner import PipelineParams, run_pipeline

    generators._instances.pop("triposr", None)
    res = run_pipeline(cutout, tmp_path, PipelineParams(generator="triposr", target_faces=3000, texture_size=512,
                                                        mc_resolution=64), lambda f, m: None)
    assert abs(res["faces"] - 3000) <= 150 and (tmp_path / "model.glb").exists()
