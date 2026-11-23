"""Téléchargement des poids Hugging Face sans liens symboliques sous Windows.

Le cache Hugging Face relie chaque fichier d'un modèle à son contenu par un lien
symbolique. Sous Windows, créer un lien demande le mode développeur ou des
droits administrateur. ``huggingface_hub`` teste cette possibilité, mais le test
peut réussir alors que la création réelle échoue ensuite avec
``[WinError 1314] Le client ne dispose pas d'un privilège nécessaire``.

Volumeo désactive donc les liens sous Windows : les fichiers sont simplement
déplacés ou copiés dans le cache, ce que ``huggingface_hub`` sait déjà faire.
"""

from __future__ import annotations

import os


def disable_hf_symlinks(force: bool = False) -> bool:
    """Désactive les liens symboliques du cache Hugging Face (Windows, ou ``force``).

    Renvoie ``True`` si la désactivation a été appliquée.
    """
    if os.name != "nt" and not force:
        return False
    try:
        from huggingface_hub import file_download
    except ImportError:  # pragma: no cover - huggingface_hub absent (installation sans IA)
        return False
    file_download.are_symlinks_supported = lambda cache_dir=None: False
    return True
