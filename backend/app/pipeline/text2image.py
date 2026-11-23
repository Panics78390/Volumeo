"""Génération d'une image à partir d'un texte (prompt), première étape du mode « texte vers 3D ».

Le prompt est transformé en image par un modèle de diffusion local, puis
l'image suit le pipeline habituel (détourage → reconstruction 3D → maillage).

Si une image est fournie en plus du texte, le texte la **transforme**
(diffusion « image vers image ») : la forme générale est conservée, le texte
change la matière, la couleur, le style ou ajoute des détails. ``strength``
règle la liberté laissée au modèle (0 = image inchangée, 1 = image ignorée).

Modèles par défaut (famille « Turbo » de Stability AI, 1 à 4 pas de diffusion) :

* GPU CUDA : ``stabilityai/sdxl-turbo`` (≈7 Go à télécharger, 512 px, ~1 s) ;
* CPU      : ``stabilityai/sd-turbo``  (≈2,5 Go, 512 px, 10 à 40 s).

Tout modèle compatible ``diffusers.AutoPipelineForText2Image`` convient :
``I3D_T2I_MODEL=<identifiant Hugging Face ou dossier local>``. Les modèles
Turbo sont sous licence Stability AI Community (usage commercial limité) ;
vérifiez-la si vous comptez vendre vos créations.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import threading
from typing import Callable

from PIL import Image

from ..config import settings
from .device import resolve_device

log = logging.getLogger(__name__)

GPU_MODEL = "stabilityai/sdxl-turbo"
CPU_MODEL = "stabilityai/sd-turbo"

# Le modèle 3D attend UN objet entier, bien détaché du fond : on cadre le prompt en conséquence.
PROMPT_SUFFIX = (
    ", single object, full view, centered, three-quarter view, plain white background, "
    "soft studio lighting, 3d render, high detail"
)
NEGATIVE_PROMPT = "cropped, multiple objects, text, watermark, busy background, blurry"

MAX_PROMPT_CHARS = 500

_lock = threading.Lock()
_pipe = None
_pipe_key: tuple[str, str] | None = None


def availability() -> tuple[bool, str]:
    """(disponible, raison). Ne charge aucun poids."""
    for mod in ("torch", "diffusers", "transformers"):
        if importlib.util.find_spec(mod) is None:
            return False, f"module Python manquant : {mod}"
    return True, ""


def model_name(device: str | None = None) -> str:
    if settings.t2i_model != "auto":
        return settings.t2i_model
    device = device or resolve_device(settings.device)
    return GPU_MODEL if device.startswith("cuda") else CPU_MODEL


def info() -> dict:
    ok, reason = availability()
    return {"available": ok, "reason": reason, "model": model_name(), "max_prompt_chars": MAX_PROMPT_CHARS}


def build_prompt(prompt: str) -> str:
    return prompt.strip().rstrip(".,;") + PROMPT_SUFFIX


def _is_turbo(name: str) -> bool:
    return "turbo" in name.lower()


def _load(device: str, progress: Callable[[float, str], None]):
    """Charge (ou réutilise) le pipeline de diffusion."""
    global _pipe, _pipe_key
    import torch
    from diffusers import AutoPipelineForText2Image

    name = model_name(device)
    if _pipe is not None and _pipe_key == (name, device):
        return _pipe, name
    progress(0.05, f"Chargement du modèle d'image {name.split('/')[-1]} (téléchargement au premier lancement)")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    try:
        # Les poids « fp16 » sont deux fois plus légers à télécharger ; ils sont
        # reconvertis en float32 sur CPU.
        pipe = AutoPipelineForText2Image.from_pretrained(name, torch_dtype=dtype, variant="fp16")
    except (OSError, ValueError):
        pipe = AutoPipelineForText2Image.from_pretrained(name, torch_dtype=dtype)
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    _pipe, _pipe_key = pipe, (name, device)
    log.info("Modèle texte→image %s chargé sur %s", name, device)
    return pipe, name


def unload() -> None:
    """Libère la mémoire (surtout la VRAM, pour laisser la place au modèle 3D)."""
    global _pipe, _pipe_key
    with _lock:
        _pipe, _pipe_key = None, None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass


def prepare_init_image(image: Image.Image, size: int) -> Image.Image:
    """Image de départ carrée sur fond blanc (la transparence éventuelle est aplatie)."""
    rgba = image.convert("RGBA")
    side = max(rgba.size)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.alpha_composite(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2))
    return canvas.convert("RGB").resize((size, size), Image.LANCZOS)


def generate_image(
    prompt: str,
    seed: int = 0,
    progress: Callable[[float, str], None] | None = None,
    init_image: Image.Image | None = None,
    strength: float = 0.6,
) -> Image.Image:
    """Génère une image RGB carrée du sujet décrit par ``prompt``.

    Avec ``init_image``, le prompt transforme cette image au lieu de partir de zéro.
    """
    progress = progress or (lambda f, m: None)
    ok, reason = availability()
    if not ok:
        raise RuntimeError(f"Génération d'image indisponible : {reason}")
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Le prompt est vide")
    import torch

    device = resolve_device(settings.device)
    with _lock:
        pipe, name = _load(device, progress)
        turbo = _is_turbo(name)
        steps = settings.t2i_steps or ((4 if device.startswith("cuda") else 2) if turbo else 25)
        size = 512 if turbo else 768
        kwargs = {
            "prompt": build_prompt(prompt),
            "num_inference_steps": steps,
            # Les modèles Turbo sont distillés pour fonctionner sans guidage (ni prompt négatif).
            "guidance_scale": 0.0 if turbo else 7.0,
            "generator": torch.Generator("cpu").manual_seed(int(seed)),
        }
        if not turbo:
            kwargs["negative_prompt"] = NEGATIVE_PROMPT
        if init_image is None:
            kwargs.update(width=size, height=size)
        else:
            from diffusers import AutoPipelineForImage2Image

            strength = min(max(float(strength), 0.05), 1.0)
            pipe = AutoPipelineForImage2Image.from_pipe(pipe)  # réutilise les poids déjà chargés
            # En image→image seuls steps × strength pas sont réellement exécutés : on
            # ajuste pour en garder au moins autant qu'en texte→image.
            total = max(steps, math.ceil(steps / strength))
            steps = max(1, int(total * strength))
            kwargs.update(image=prepare_init_image(init_image, size), strength=strength, num_inference_steps=total)

        def on_step(_pipe, step, _timestep, callback_kwargs):
            progress(0.2 + 0.75 * (step + 1) / steps, f"Création de l'image (étape {step + 1}/{steps})")
            return callback_kwargs

        progress(0.2, "Création de l'image")
        with torch.inference_mode():
            image = pipe(callback_on_step_end=on_step, **kwargs).images[0]
    if not settings.t2i_keep_loaded:
        unload()
    progress(1.0, "Image créée")
    return image.convert("RGB")
