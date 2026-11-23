"""Chargement de l'image, suppression de l'arrière-plan et cadrage du sujet.

La suppression de fond utilise rembg (réseaux U²-Net / IS-Net / BiRefNet).
Si rembg n'est pas disponible (pas installé, poids non téléchargeables), un
détourage de secours estime la couleur du fond à partir des bords de l'image :
cela fonctionne bien pour les photos produit sur fond uni.
"""

from __future__ import annotations

import io
import logging
import threading

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage

log = logging.getLogger(__name__)

MAX_INPUT_SIDE = 2048
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

_rembg_sessions: dict[str, object] = {}
_rembg_lock = threading.Lock()


class ImageError(ValueError):
    """Image invalide ou format non supporté."""


def load_image(data: bytes) -> Image.Image:
    """Décode une image JPG/PNG/WebP, applique l'orientation EXIF et la convertit en RGBA."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.load()
    except Exception as exc:  # noqa: BLE001 - Pillow lève des exceptions variées
        raise ImageError("Impossible de lire l'image (fichier corrompu ?)") from exc
    if fmt not in ALLOWED_FORMATS:
        raise ImageError(f"Format non supporté : {fmt}. Utilisez JPG, PNG ou WebP.")
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGBA")
    if max(img.size) > MAX_INPUT_SIDE:
        img.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.LANCZOS)
    return img


def has_meaningful_alpha(img: Image.Image) -> bool:
    """Vrai si l'image possède déjà un canal alpha qui isole un sujet."""
    if img.mode != "RGBA":
        return False
    alpha = np.asarray(img.getchannel("A"))
    transparent = (alpha < 128).mean()
    return 0.02 < transparent < 0.98


def _rembg_remove(img: Image.Image, model: str) -> Image.Image:
    from rembg import new_session, remove  # import paresseux : dépendance lourde

    with _rembg_lock:
        session = _rembg_sessions.get(model)
        if session is None:
            log.info("Chargement du modèle rembg %s", model)
            session = new_session(model)
            _rembg_sessions[model] = session
    out = remove(img.convert("RGB"), session=session)
    return out.convert("RGBA")


def _fallback_remove(img: Image.Image) -> Image.Image:
    """Détourage de secours : distance à la couleur de fond estimée sur les bords."""
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w, _ = rgb.shape
    b = max(2, min(h, w) // 50)
    border = np.concatenate(
        [rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3), rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)]
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=-1)
    # Seuil : au-dessus du bruit observé sur les bords, avec un minimum absolu.
    border_dist = np.linalg.norm(border - bg, axis=-1)
    thresh = max(25.0, float(np.percentile(border_dist, 99)) * 1.5)
    mask = dist > thresh
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    labels, n = ndimage.label(mask)
    if n == 0:
        raise ImageError("Aucun sujet détecté : essayez une image avec un fond plus uni.")
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    # On garde les composantes significatives (le sujet peut être en plusieurs morceaux).
    keep = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.05 * sizes.max()))
    keep = ndimage.binary_fill_holes(keep)
    alpha = ndimage.gaussian_filter(keep.astype(np.float32), 1.0)
    out = np.dstack([rgb, np.clip(alpha * 255, 0, 255)]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def remove_background(img: Image.Image, use_rembg: bool = True, model: str = "u2net", force: bool = False) -> tuple[Image.Image, str]:
    """Retourne (image RGBA détourée, méthode utilisée).

    Méthodes : ``"alpha"`` (transparence d'origine conservée), ``"rembg"``,
    ``"fallback"`` (estimation de la couleur du fond).
    """
    if not force and has_meaningful_alpha(img):
        return img, "alpha"
    if use_rembg:
        try:
            out = _rembg_remove(img, model)
            if (np.asarray(out.getchannel("A")) > 128).mean() > 0.001:
                return out, "rembg"
            log.warning("rembg n'a détecté aucun sujet, passage au détourage de secours")
        except Exception as exc:  # noqa: BLE001
            log.warning("rembg indisponible (%s), passage au détourage de secours", exc)
    return _fallback_remove(img), "fallback"


def foreground_bbox(rgba: Image.Image, threshold: int = 16) -> tuple[int, int, int, int]:
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.nonzero(alpha > threshold)
    if len(xs) == 0:
        raise ImageError("L'image détourée est vide : aucun sujet trouvé.")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def prepare_foreground(rgba: Image.Image, size: int = 512, ratio: float = 0.85) -> Image.Image:
    """Recadre le sujet, le centre dans un carré et lui fait occuper ``ratio`` du cadre.

    C'est le cadrage attendu par TripoSR / SF3D (sujet centré, marge autour).
    """
    x0, y0, x1, y1 = foreground_bbox(rgba)
    fg = rgba.crop((x0, y0, x1, y1))
    side = max(fg.size)
    canvas_side = max(1, int(round(side / ratio)))
    canvas = Image.new("RGBA", (canvas_side, canvas_side), (0, 0, 0, 0))
    canvas.paste(fg, ((canvas_side - fg.width) // 2, (canvas_side - fg.height) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def composite_on_gray(rgba: Image.Image, gray: float = 0.5) -> np.ndarray:
    """Image float32 HxWx3 dans [0,1], fond gris (entrée attendue par TripoSR)."""
    arr = np.asarray(rgba).astype(np.float32) / 255.0
    alpha = arr[..., 3:4]
    return arr[..., :3] * alpha + (1.0 - alpha) * gray
