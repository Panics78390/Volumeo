"""Détection du matériel (GPU CUDA / CPU) sans rendre PyTorch obligatoire."""

from __future__ import annotations

import importlib.util
import logging
import os

log = logging.getLogger(__name__)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def resolve_device(requested: str = "auto") -> str:
    """"auto" -> "cuda" si un GPU est visible, sinon "cpu" (avec avertissement)."""
    if not torch_available():
        return "cpu"
    import torch

    if requested and requested != "auto":
        if requested.startswith("cuda") and not torch.cuda.is_available():
            log.warning("CUDA demandé mais indisponible : bascule sur CPU")
            return "cpu"
        return requested
    if torch.cuda.is_available():
        return "cuda"
    log.warning("Aucun GPU CUDA détecté : la génération neuronale tournera sur CPU (lente)")
    return "cpu"


def system_info(requested: str = "auto") -> dict:
    info: dict = {
        "torch": torch_available(),
        "cuda": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cpu_count": os.cpu_count(),
        "warning": None,
    }
    if info["torch"]:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["device"] = resolve_device(requested)
        if info["cuda"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
    if not info["cuda"]:
        info["warning"] = (
            "Aucun GPU CUDA détecté : les modèles IA tourneront sur CPU (de 30 s à plusieurs minutes "
            "par génération). Le générateur procédural reste instantané."
        )
    return info
