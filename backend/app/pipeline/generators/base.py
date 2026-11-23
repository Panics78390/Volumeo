"""Interface commune des générateurs image → 3D.

Un générateur reçoit l'image détourée et cadrée (RGBA carrée) et renvoie :

* un maillage brut (``trimesh.Trimesh``) en repère Y-haut, sujet face à +Z,
  centré à l'origine et de taille ~1 ;
* une fonction ``color_fn(points) -> rgb`` qui donne la couleur de la surface
  en n'importe quel point 3D. Elle sert au « bake » de la texture après
  remaillage, ce qui permet de changer la résolution sans perdre de couleur.

Pour ajouter un modèle (Stable Fast 3D, InstantMesh, Hunyuan3D-2...), il
suffit d'écrire une sous-classe de :class:`Generator` et de l'enregistrer
dans ``generators/__init__.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import trimesh
from PIL import Image

ColorFn = Callable[[np.ndarray], np.ndarray]
ProgressFn = Callable[[float, str], None]


@dataclass
class GenerationOptions:
    mc_resolution: int = 256
    seed: int = 0


@dataclass
class RawResult:
    mesh: trimesh.Trimesh
    color_fn: ColorFn
    info: dict = field(default_factory=dict)


class Generator(ABC):
    #: identifiant stable utilisé dans l'API
    name: str = "base"
    #: libellé affiché dans l'interface
    label: str = "Générateur"
    description: str = ""
    requires_gpu: bool = False

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        """(disponible, raison). Ne doit PAS charger les poids (appel rapide)."""
        return True, ""

    @abstractmethod
    def generate(self, image: Image.Image, options: GenerationOptions, progress: ProgressFn) -> RawResult:
        """Produit un maillage brut à partir d'une image RGBA détourée et cadrée."""

    def unload(self) -> None:  # pragma: no cover - optionnel
        """Libère la mémoire (GPU) si nécessaire."""
