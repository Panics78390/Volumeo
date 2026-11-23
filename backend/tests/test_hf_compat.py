"""Le cache Hugging Face doit fonctionner sans liens symboliques (Windows sans droits)."""

import pytest

file_download = pytest.importorskip("huggingface_hub.file_download")


def test_downloads_copy_files_instead_of_linking(monkeypatch, tmp_path):
    from app.hf_compat import disable_hf_symlinks

    monkeypatch.setattr(file_download, "are_symlinks_supported", file_download.are_symlinks_supported)

    def refuse(*_a, **_k):  # comme Windows sans privilège : WinError 1314
        raise OSError(1314, "Le client ne dispose pas d'un privilège nécessaire")

    monkeypatch.setattr(file_download.os, "symlink", refuse)
    assert disable_hf_symlinks(force=True) is True

    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"poids")
    dst = tmp_path / "snapshots" / "rev" / "model.ckpt"
    dst.parent.mkdir(parents=True)
    file_download._create_symlink(str(blob), str(dst), new_blob=True)
    assert dst.read_bytes() == b"poids" and not dst.is_symlink()


def test_noop_outside_windows(monkeypatch):
    from app import hf_compat

    monkeypatch.setattr(hf_compat.os, "name", "posix")
    assert hf_compat.disable_hf_symlinks() is False
