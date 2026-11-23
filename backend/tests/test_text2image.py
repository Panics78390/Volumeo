"""Test du module texte → image avec un faux pipeline diffusers (aucun poids téléchargé)."""

from __future__ import annotations

import sys
import types

import pytest
from PIL import Image

torch = pytest.importorskip("torch")


class _FakePipe:
    loaded: list = []
    calls: list = []

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        cls.loaded.append((name, kwargs))
        return cls()

    def set_progress_bar_config(self, **_):
        pass

    def to(self, device):
        self.device = device
        return self

    @classmethod
    def from_pipe(cls, pipe):
        cls.from_pipe_calls = getattr(cls, "from_pipe_calls", 0) + 1
        return pipe

    def __call__(self, callback_on_step_end=None, **kwargs):
        type(self).calls.append(kwargs)
        # Comme diffusers : en image→image, seuls steps × strength pas sont exécutés.
        n = int(kwargs["num_inference_steps"] * kwargs.get("strength", 1.0))
        for step in range(n):
            callback_on_step_end(self, step, 0, {})
        size = kwargs["image"].size if "image" in kwargs else (kwargs["width"], kwargs["height"])
        return types.SimpleNamespace(images=[Image.new("RGB", size, "white")])


@pytest.fixture
def fake_diffusers(monkeypatch):
    from app.pipeline import text2image

    mod = types.ModuleType("diffusers")
    mod.AutoPipelineForText2Image = _FakePipe
    mod.AutoPipelineForImage2Image = _FakePipe
    monkeypatch.setitem(sys.modules, "diffusers", mod)
    monkeypatch.setattr(text2image, "availability", lambda: (True, ""))
    _FakePipe.loaded.clear()
    _FakePipe.calls.clear()
    text2image.unload()
    yield text2image
    text2image.unload()


def test_generate_image_cpu_turbo(fake_diffusers):
    t2i = fake_diffusers
    steps = []
    img = t2i.generate_image("un vase bleu.", seed=3, progress=lambda f, m: steps.append(f))
    assert img.size == (512, 512) and img.mode == "RGB"
    name, kwargs = _FakePipe.loaded[0]
    assert name == "stabilityai/sd-turbo" and kwargs["variant"] == "fp16"  # CPU -> petit modèle
    call = _FakePipe.calls[0]
    assert call["prompt"].startswith("un vase bleu, single object")
    assert call["guidance_scale"] == 0.0 and call["num_inference_steps"] == 2
    assert "negative_prompt" not in call
    assert steps == sorted(steps) and steps[-1] == 1.0
    assert t2i._pipe is None, "le modèle est déchargé après usage pour libérer la mémoire"


def test_custom_model_uses_guidance(fake_diffusers, monkeypatch):
    t2i = fake_diffusers
    monkeypatch.setattr(t2i.settings, "t2i_model", "runwayml/stable-diffusion-v1-5")
    t2i.generate_image("une chaise")
    call = _FakePipe.calls[0]
    assert call["guidance_scale"] > 1 and call["num_inference_steps"] == 25 and "negative_prompt" in call


def test_empty_prompt_rejected(fake_diffusers):
    with pytest.raises(ValueError):
        fake_diffusers.generate_image("   ")


def test_image_to_image(fake_diffusers):
    t2i = fake_diffusers
    src = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
    src.paste((200, 30, 30, 255), (100, 50, 200, 150))
    steps = []
    img = t2i.generate_image("en bois", seed=1, progress=lambda f, m: steps.append(f), init_image=src, strength=0.5)
    call = _FakePipe.calls[0]
    assert img.size == (512, 512)
    init = call["image"]
    assert init.size == (512, 512) and init.mode == "RGB"
    assert init.getpixel((2, 2)) == (255, 255, 255), "transparence aplatie sur fond blanc"
    assert init.getpixel((256, 256))[0] > 150, "sujet conservé au centre"
    assert call["strength"] == 0.5 and "width" not in call
    # 2 pas effectifs demandés sur CPU -> 4 pas programmés × 0,5
    assert call["num_inference_steps"] == 4
    assert steps[-1] == 1.0 and max(steps) <= 1.0
