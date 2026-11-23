"""Registre des générateurs image → 3D (modèles interchangeables)."""

from __future__ import annotations

import threading

from .base import GenerationOptions, Generator, RawResult
from .procedural import ProceduralGenerator
from .triposg import TripoSGGenerator
from .triposr import TripoSRGenerator

# Ordre = priorité pour le choix automatique.
REGISTRY: dict[str, type[Generator]] = {
    TripoSGGenerator.name: TripoSGGenerator,
    TripoSRGenerator.name: TripoSRGenerator,
    ProceduralGenerator.name: ProceduralGenerator,
}

_instances: dict[str, Generator] = {}
_lock = threading.Lock()


def list_generators() -> list[dict]:
    out = []
    for name, cls in REGISTRY.items():
        ok, reason = cls.availability()
        out.append(
            {
                "name": name,
                "label": cls.label,
                "description": cls.description,
                "requires_gpu": cls.requires_gpu,
                "available": ok,
                "reason": reason,
            }
        )
    return out


def default_generator_name(configured: str = "auto") -> str:
    if configured != "auto":
        return configured
    for name, cls in REGISTRY.items():
        # TripoSG sur CPU prendrait de longues minutes : on ne le choisit d'office qu'avec un GPU.
        if name == TripoSGGenerator.name and not _has_cuda():
            continue
        if cls.availability()[0]:
            return name
    return ProceduralGenerator.name


def _has_cuda() -> bool:
    from ...config import settings
    from ..device import resolve_device

    try:
        return resolve_device(settings.device).startswith("cuda")
    except Exception:  # pragma: no cover - torch absent ou cassé
        return False


def unload_all() -> None:
    """Libère les modèles 3D chargés (avant d'utiliser le modèle d'image, par exemple)."""
    with _lock:
        instances = list(_instances.values())
    for inst in instances:
        inst.unload()


def get_generator(name: str) -> Generator:
    """Instance unique (les poids ne sont chargés qu'une fois par processus)."""
    if name not in REGISTRY:
        raise ValueError(f"Générateur inconnu : {name}")
    ok, reason = REGISTRY[name].availability()
    if not ok:
        raise RuntimeError(f"Générateur « {name} » indisponible : {reason}")
    with _lock:
        if name not in _instances:
            _instances[name] = REGISTRY[name]()
        return _instances[name]


__all__ = [
    "GenerationOptions",
    "Generator",
    "RawResult",
    "REGISTRY",
    "list_generators",
    "default_generator_name",
    "get_generator",
    "unload_all",
]
