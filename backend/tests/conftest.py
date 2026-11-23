"""Fixtures communes : images synthétiques et client HTTP isolé."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.config import Settings


def make_photo(size=(640, 480), bg=(238, 238, 238)) -> Image.Image:
    """« Photo produit » synthétique : un objet coloré sur fond uni (RGB, sans alpha)."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    d.ellipse((200, 120, 440, 420), fill=(200, 40, 40))  # corps rouge
    d.rectangle((280, 60, 360, 150), fill=(30, 90, 200))  # tête bleue
    d.ellipse((300, 220, 340, 260), fill=(250, 220, 30))  # détail jaune
    return img


def make_cutout(size=256) -> Image.Image:
    """Sujet déjà détouré (PNG RGBA avec transparence)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((size * 0.2, size * 0.1, size * 0.8, size * 0.9), fill=(40, 160, 60, 255))
    return img


def to_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def photo() -> Image.Image:
    return make_photo()


@pytest.fixture
def cutout() -> Image.Image:
    return make_cutout()


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    # rembg est désactivé pour que les tests n'aient pas à télécharger de poids.
    return Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models", static_dir=tmp_path / "static",
                    use_rembg=False, default_generator="procedural", cors_origins=[])


@pytest.fixture
def client(test_settings: Settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(test_settings)) as c:
        yield c


def wait_job(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed", "cancelled"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"La génération {job_id} n'a pas fini en {timeout} s")
