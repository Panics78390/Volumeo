"""Tests de l'API HTTP : import → génération en file d'attente → historique → téléchargements."""

from tests.conftest import make_photo, to_bytes, wait_job


def _upload(client, fmt="JPEG"):
    r = client.post("/api/uploads", files={"file": (f"jouet.{fmt.lower()}", to_bytes(make_photo(), fmt), "image/jpeg")})
    assert r.status_code == 201, r.text
    return r.json()


def test_system_info(client):
    info = client.get("/api/system").json()
    assert info["default_generator"] == "procedural"
    assert {p["key"] for p in info["presets"]} == {"draft", "standard", "high"}
    assert info["texture_sizes"] == [512, 1024, 2048, 4096]
    assert any(g["name"] == "procedural" and g["available"] for g in info["generators"])
    assert "warning" in info["device"]


def test_upload_and_preview(client):
    up = _upload(client)
    assert up["background_method"] == "fallback"
    r = client.get(up["urls"]["cutout"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"


def test_upload_rejects_bad_file(client):
    r = client.post("/api/uploads", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 422


def test_full_generation_flow(client):
    up = _upload(client, "PNG")
    r = client.post("/api/jobs", json={"upload_id": up["id"], "preset": "custom", "target_faces": 3000,
                                       "texture_size": 512, "generator": "procedural", "smoothing": 2})
    assert r.status_code == 201, r.text
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert abs(job["result"]["faces"] - 3000) <= 150
    assert job["name"] == "jouet"

    glb = client.get(job["urls"]["model"])
    assert glb.status_code == 200 and glb.content[:4] == b"glTF"
    for fmt, magic in {"obj": b"PK", "stl": None, "fbx": b"Kaydara"}.items():
        d = client.get(job["urls"]["downloads"][fmt])
        assert d.status_code == 200, fmt
        assert "attachment" in d.headers["content-disposition"]
        if magic:
            assert d.content.startswith(magic)

    hist = client.get("/api/jobs").json()
    assert hist["total"] == 1 and hist["items"][0]["id"] == job["id"]

    # Ré-ajustement sans relancer le générateur.
    r = client.post(f"/api/jobs/{job['id']}/reprocess", json={"preset": "draft", "texture_size": 512})
    assert r.status_code == 201, r.text
    child = wait_job(client, r.json()["id"])
    assert child["status"] == "done" and child["parent_id"] == job["id"]
    assert abs(child["result"]["faces"] - 5000) <= 250

    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_validation_errors(client):
    up = _upload(client)
    bad = [
        {"upload_id": "000000000000", "preset": "draft"},
        {"upload_id": up["id"], "preset": "custom"},
        {"upload_id": up["id"], "preset": "custom", "target_faces": 10_000, "texture_size": 300},
        {"upload_id": up["id"], "preset": "draft", "generator": "inconnu"},
    ]
    for body in bad:
        assert client.post("/api/jobs", json=body).status_code in (404, 422), body


def test_queue_runs_jobs_in_order_and_cancel(client):
    up = _upload(client)
    body = {"upload_id": up["id"], "preset": "custom", "target_faces": 2000, "texture_size": 512}
    ids = [client.post("/api/jobs", json=body).json()["id"] for _ in range(3)]
    client.post(f"/api/jobs/{ids[2]}/cancel")
    jobs = [wait_job(client, i) for i in ids]
    assert [j["status"] for j in jobs] == ["done", "done", "cancelled"]
    assert jobs[0]["finished_at"] <= jobs[1]["started_at"], "une seule génération à la fois"


# ---------------------------------------------------------------------------
# Texte vers 3D
# ---------------------------------------------------------------------------
def _fake_text2image(monkeypatch, calls):
    """Remplace le modèle de diffusion par une image synthétique (pas de poids requis)."""
    from app.pipeline import text2image

    def fake(prompt, seed=0, progress=None, init_image=None, strength=0.6):
        calls.append((prompt, seed) if init_image is None else (prompt, seed, init_image.size, strength))
        if progress:
            progress(0.5, "Création de l'image")
        return make_photo().convert("RGB")

    monkeypatch.setattr(text2image, "generate_image", fake)
    monkeypatch.setattr(text2image, "availability", lambda: (True, ""))


def test_text_to_3d_flow(client, monkeypatch):
    calls = []
    _fake_text2image(monkeypatch, calls)
    assert client.get("/api/system").json()["text_to_image"]["available"] is True

    r = client.post("/api/jobs", json={"prompt": "  un robot jouet rouge ", "preset": "custom", "target_faces": 2000,
                                       "texture_size": 512, "generator": "procedural", "seed": 7})
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["kind"] == "text" and created["name"] == "un robot jouet rouge"
    job = wait_job(client, created["id"])
    assert job["status"] == "done", job["error"]
    assert calls == [("un robot jouet rouge", 7)]
    assert job["params"]["prompt"] == "un robot jouet rouge"
    assert abs(job["result"]["faces"] - 2000) <= 100
    # L'image générée est conservée comme un import (aperçu + détourage).
    assert client.get(f"/api/uploads/{job['upload_id']}/original").status_code == 200
    assert client.get(job["urls"]["preview"]).status_code == 200

    # Un ré-ajustement ne régénère pas l'image.
    r = client.post(f"/api/jobs/{job['id']}/reprocess", json={"preset": "custom", "target_faces": 1000, "texture_size": 512})
    child = wait_job(client, r.json()["id"])
    assert child["status"] == "done" and len(calls) == 1


def test_text_job_validation(client, monkeypatch):
    up = _upload(client)
    # ni image ni prompt
    assert client.post("/api/jobs", json={"preset": "draft"}).status_code == 422
    assert client.post("/api/jobs", json={"preset": "draft", "upload_id": up["id"], "prompt": "x", "strength": 3}).status_code == 422
    assert client.post("/api/jobs", json={"preset": "draft", "upload_id": "000000000000", "prompt": "x"}).status_code == 404
    assert client.post("/api/jobs", json={"preset": "draft", "prompt": "a" * 501}).status_code == 422
    # modèle d'image non installé -> erreur claire
    from app.pipeline import text2image

    monkeypatch.setattr(text2image, "availability", lambda: (False, "module Python manquant : diffusers"))
    r = client.post("/api/jobs", json={"preset": "draft", "prompt": "une chaise"})
    assert r.status_code == 422 and "diffusers" in r.json()["detail"]


def test_image_plus_text_flow(client, monkeypatch):
    """Image + texte : le prompt transforme l'image importée avant la reconstruction 3D."""
    calls = []
    _fake_text2image(monkeypatch, calls)
    up = _upload(client)
    r = client.post("/api/jobs", json={"upload_id": up["id"], "prompt": "en or massif", "strength": 0.4, "preset": "custom",
                                       "target_faces": 2000, "texture_size": 512, "generator": "procedural"})
    assert r.status_code == 201, r.text
    job = wait_job(client, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert calls == [("en or massif", 0, (640, 480), 0.4)], "l'image importée est passée au modèle"
    assert job["kind"] == "text" and job["params"]["source_upload_id"] == up["id"]
    assert job["upload_id"] != up["id"], "le résultat est un nouvel import, l'original est conservé"
    assert client.get(up["urls"]["original"]).status_code == 200
