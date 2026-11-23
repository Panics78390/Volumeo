"""Configuration de l'application, lue depuis les variables d'environnement.

Toutes les variables sont préfixées par ``I3D_`` (ex. ``I3D_DATA_DIR``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"I3D_{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # Dossier où sont stockés les uploads, les résultats et la base SQLite.
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")).resolve())
    # Générateur par défaut : "auto" choisit TripoSG (avec GPU) ou TripoSR s'ils sont installés, sinon "procedural".
    default_generator: str = field(default_factory=lambda: _env("GENERATOR", "auto"))
    # Périphérique PyTorch : "auto", "cuda", "cuda:1", "cpu"...
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    # Dossier local des poids (sinon téléchargement depuis Hugging Face).
    models_dir: Path = field(default_factory=lambda: Path(_env("MODELS_DIR", "./models")).resolve())
    # Identifiant Hugging Face ou chemin local du modèle TripoSR.
    triposr_model: str = field(default_factory=lambda: _env("TRIPOSR_MODEL", "stabilityai/TripoSR"))
    # Identifiant Hugging Face ou chemin local du modèle TripoSG.
    triposg_model: str = field(default_factory=lambda: _env("TRIPOSG_MODEL", "VAST-AI/TripoSG"))
    # Pas de diffusion et force du guidage de TripoSG (valeurs du dépôt officiel).
    triposg_steps: int = field(default_factory=lambda: int(_env("TRIPOSG_STEPS", "50")))
    triposg_guidance: float = field(default_factory=lambda: float(_env("TRIPOSG_GUIDANCE", "7.0")))
    # Couleurs de TripoSG : "auto" (TripoSR s'il est installé, sinon projection), "triposr", "projection".
    triposg_colors: str = field(default_factory=lambda: _env("TRIPOSG_COLORS", "auto"))
    # Taille des lots lors de l'interrogation du réseau (mémoire vs vitesse).
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "8192")))
    # Modèle texte → image : "auto" (SDXL-Turbo sur GPU, SD-Turbo sur CPU) ou identifiant Hugging Face.
    t2i_model: str = field(default_factory=lambda: _env("T2I_MODEL", "auto"))
    # Nombre de pas de diffusion (0 = choix automatique selon le modèle).
    t2i_steps: int = field(default_factory=lambda: int(_env("T2I_STEPS", "0")))
    # Garder le modèle d'image en mémoire entre deux générations (plus rapide, plus de VRAM).
    t2i_keep_loaded: bool = field(default_factory=lambda: _env_bool("T2I_KEEP_LOADED", False))
    # Modèle rembg utilisé pour le détourage (u2net, u2netp, isnet-general-use, birefnet-general...).
    rembg_model: str = field(default_factory=lambda: _env("REMBG_MODEL", "isnet-general-use"))
    # Désactiver rembg force le détourage de secours (couleur de fond estimée).
    use_rembg: bool = field(default_factory=lambda: _env_bool("USE_REMBG", True))
    # Taille maximale d'une image importée (Mo).
    max_upload_mb: int = field(default_factory=lambda: int(_env("MAX_UPLOAD_MB", "20")))
    # Nombre de générations conservées dans l'historique (0 = illimité).
    history_limit: int = field(default_factory=lambda: int(_env("HISTORY_LIMIT", "100")))
    # Dossier du frontend compilé (servi par FastAPI si présent).
    static_dir: Path = field(default_factory=lambda: Path(_env("STATIC_DIR", "./static")).resolve())
    # Origines autorisées (CORS) pour le serveur de dev Vite.
    cors_origins: list[str] = field(
        default_factory=lambda: [o for o in _env("CORS_ORIGINS", "http://localhost:5173").split(",") if o]
    )

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "history.sqlite3"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.jobs_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
