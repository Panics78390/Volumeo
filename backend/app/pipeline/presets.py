"""Préréglages de résolution du maillage."""

from __future__ import annotations

from dataclasses import asdict, dataclass

TEXTURE_SIZES = (512, 1024, 2048, 4096)
MIN_FACES = 500
MAX_FACES = 500_000


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    description: str
    target_faces: int
    texture_size: int
    # Résolution de la grille du marching cubes : plus elle est fine, plus le
    # maillage brut contient de détails (et plus l'extraction est lente).
    mc_resolution: int

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, Preset] = {
    "draft": Preset("draft", "Brouillon", "~5 000 triangles, rapide", 5_000, 1024, 160),
    "standard": Preset("standard", "Standard", "~50 000 triangles", 50_000, 2048, 256),
    "high": Preset("high", "Haute", "~200 000 triangles", 200_000, 2048, 320),
}


def mc_resolution_for(target_faces: int) -> int:
    """Résolution de marching cubes adaptée à un nombre de triangles personnalisé."""
    if target_faces <= 10_000:
        return 160
    if target_faces <= 80_000:
        return 256
    return 320


def resolve(preset: str, target_faces: int | None, texture_size: int | None) -> tuple[int, int, int]:
    """Retourne (target_faces, texture_size, mc_resolution) validés pour une demande.

    ``preset="custom"`` utilise les valeurs fournies ; un préréglage nommé
    fournit les valeurs par défaut, que l'appelant peut quand même surcharger
    pour la taille de texture.
    """
    if preset == "custom":
        if target_faces is None:
            raise ValueError("target_faces est requis pour une résolution personnalisée")
        faces = int(target_faces)
        tex = int(texture_size or 2048)
        mc = mc_resolution_for(faces)
    else:
        if preset not in PRESETS:
            raise ValueError(f"Préréglage inconnu : {preset}")
        p = PRESETS[preset]
        faces = p.target_faces
        tex = int(texture_size or p.texture_size)
        mc = p.mc_resolution
    if not MIN_FACES <= faces <= MAX_FACES:
        raise ValueError(f"Le nombre de triangles doit être entre {MIN_FACES} et {MAX_FACES}")
    if tex not in TEXTURE_SIZES:
        raise ValueError(f"Taille de texture invalide : {tex} (attendu {TEXTURE_SIZES})")
    return faces, tex, mc
