import numpy as np
import pytest

from app.pipeline.background import (
    ImageError,
    has_meaningful_alpha,
    load_image,
    prepare_foreground,
    remove_background,
)
from tests.conftest import make_cutout, make_photo, to_bytes


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_load_supported_formats(fmt):
    img = load_image(to_bytes(make_photo(), fmt))
    assert img.mode == "RGBA" and img.size == (640, 480)


def test_load_rejects_other_formats():
    with pytest.raises(ImageError):
        load_image(to_bytes(make_photo(), "BMP"))
    with pytest.raises(ImageError):
        load_image(b"pas une image")


def test_fallback_background_removal_isolates_subject(photo):
    out, method = remove_background(photo.convert("RGBA"), use_rembg=False)
    assert method == "fallback"
    alpha = np.asarray(out.getchannel("A"))
    assert alpha[5, 5] == 0, "le coin (fond) doit être transparent"
    assert alpha[300, 320] == 255, "le centre de l'objet doit être opaque"
    # La surface détourée correspond à peu près à celle des formes dessinées.
    assert 0.15 < (alpha > 127).mean() < 0.35


def test_existing_alpha_is_kept(cutout):
    assert has_meaningful_alpha(cutout)
    out, method = remove_background(cutout, use_rembg=True)
    assert method == "alpha" and out is cutout


def test_prepare_foreground_centers_and_pads():
    fg = prepare_foreground(make_cutout(300), size=512, ratio=0.85)
    assert fg.size == (512, 512)
    ys, xs = np.nonzero(np.asarray(fg.getchannel("A")) > 16)
    # Le sujet occupe ~85 % de la hauteur, centré.
    assert 0.8 < (ys.max() - ys.min()) / 512 < 0.9
    assert abs((xs.min() + xs.max()) / 2 - 256) < 4
